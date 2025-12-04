//! OAuth2 token management with secure storage
//!
//! Handles Google OAuth2 flow:
//! 1. Opens browser for user authorization
//! 2. Receives callback on localhost
//! 3. Exchanges code for tokens
//! 4. Stores tokens in macOS Keychain
//! 5. Auto-refreshes expired tokens

use anyhow::{anyhow, Result};
use oauth2::{
    AuthUrl, AuthorizationCode, ClientId, ClientSecret, CsrfToken,
    PkceCodeChallenge, PkceCodeVerifier, RedirectUrl, RefreshToken,
    Scope, TokenResponse, TokenUrl,
    basic::BasicClient,
    reqwest::async_http_client,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

const SERVICE_NAME: &str = "dosa-assistant";
const GOOGLE_AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const GOOGLE_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const REDIRECT_PORT: u16 = 8085;

/// Supported OAuth providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provider {
    Google,
}

impl Provider {
    pub fn name(&self) -> &'static str {
        match self {
            Provider::Google => "google",
        }
    }
}

/// Authentication status for display
#[derive(Debug, Clone)]
pub struct AuthStatus {
    pub provider: Provider,
    pub authenticated: bool,
    pub email: Option<String>,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Stored token data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredToken {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: Option<i64>, // Unix timestamp
    pub email: Option<String>,
}

impl StoredToken {
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            let now = chrono::Utc::now().timestamp();
            // Consider expired if less than 5 minutes remaining
            expires_at - now < 300
        } else {
            false
        }
    }
}

/// Google client secret JSON format (from Google Cloud Console)
#[derive(Debug, Deserialize)]
struct GoogleClientSecretFile {
    web: Option<GoogleClientSecretWeb>,
    installed: Option<GoogleClientSecretWeb>,
}

#[derive(Debug, Deserialize)]
struct GoogleClientSecretWeb {
    client_id: String,
    client_secret: String,
}

/// OAuth2 manager for all providers
pub struct OAuthManager {
    google_client_id: Option<String>,
    google_client_secret: Option<String>,
    tokens: Arc<RwLock<std::collections::HashMap<Provider, StoredToken>>>,
}

impl OAuthManager {
    /// Create a new OAuth manager
    /// 
    /// Loads credentials in this order:
    /// 1. Environment variables (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET)
    /// 2. ~/.config/samantha/google_credentials.json (user config directory)
    /// 3. data/google_credentials.json (relative to binary or current directory)
    /// 4. client_secret*.json files in current directory
    pub fn new() -> Self {
        // First try environment variables
        let mut google_client_id = std::env::var("GOOGLE_CLIENT_ID").ok();
        let mut google_client_secret = std::env::var("GOOGLE_CLIENT_SECRET").ok();

        // If not in env, try to load from credentials file
        if google_client_id.is_none() || google_client_secret.is_none() {
            if let Some((id, secret)) = Self::load_credentials_from_file() {
                google_client_id = Some(id);
                google_client_secret = Some(secret);
            }
        }

        OAuthManager {
            google_client_id,
            google_client_secret,
            tokens: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Load Google credentials from a JSON file
    fn load_credentials_from_file() -> Option<(String, String)> {
        // Try multiple locations in order of preference
        let mut locations = vec![
            // User config directory (highest priority)
            Self::get_user_config_dir().join("google_credentials.json"),
            Self::get_user_config_dir().join("client_secret.json"),
            // Data directory locations (relative to binary)
            Self::get_data_dir().join("google_credentials.json"),
            Self::get_data_dir().join("client_secret.json"),
            // Current directory
            std::path::PathBuf::from("data/google_credentials.json"),
            std::path::PathBuf::from("google_credentials.json"),
            std::path::PathBuf::from("client_secret.json"),
        ];

        // Remove duplicates while preserving order
        locations.dedup();

        for path in &locations {
            if let Some(creds) = Self::try_load_credentials(path) {
                eprintln!("✓ Loaded Google credentials from {:?}", path);
                return Some(creds);
            }
        }

        // Also check for client_secret_*.json pattern in current directory
        if let Ok(entries) = std::fs::read_dir(".") {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("client_secret_") && name.ends_with(".json") {
                        if let Some(creds) = Self::try_load_credentials(&path) {
                            eprintln!("✓ Loaded Google credentials from {:?}", path);
                            return Some(creds);
                        }
                    }
                }
            }
        }

        None
    }

    /// Try to load credentials from a specific file
    fn try_load_credentials(path: &Path) -> Option<(String, String)> {
        let content = std::fs::read_to_string(path).ok()?;
        let parsed: GoogleClientSecretFile = serde_json::from_str(&content).ok()?;
        
        // Try "web" first, then "installed"
        if let Some(web) = parsed.web {
            return Some((web.client_id, web.client_secret));
        }
        if let Some(installed) = parsed.installed {
            return Some((installed.client_id, installed.client_secret));
        }
        
        None
    }

    /// Get data directory path
    fn get_data_dir() -> std::path::PathBuf {
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join("data")))
            .unwrap_or_else(|| std::path::PathBuf::from("./data"))
    }

    /// Get user config directory (~/.config/samantha)
    fn get_user_config_dir() -> std::path::PathBuf {
        dirs::config_dir()
            .map(|p| p.join("samantha"))
            .unwrap_or_else(|| {
                // Fallback to ~/.config/samantha
                dirs::home_dir()
                    .map(|h| h.join(".config").join("samantha"))
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
            })
    }

    /// Create with explicit credentials
    pub fn with_credentials(client_id: &str, client_secret: &str) -> Self {
        OAuthManager {
            google_client_id: Some(client_id.to_string()),
            google_client_secret: Some(client_secret.to_string()),
            tokens: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Check if credentials are configured
    pub fn has_credentials(&self, provider: Provider) -> bool {
        match provider {
            Provider::Google => {
                self.google_client_id.is_some() && self.google_client_secret.is_some()
            }
        }
    }

    /// Start OAuth flow for a provider (opens browser)
    pub async fn authenticate(&self, provider: Provider) -> Result<()> {
        match provider {
            Provider::Google => self.authenticate_google().await,
        }
    }

    /// Google OAuth2 flow
    async fn authenticate_google(&self) -> Result<()> {
        let client_id = self.google_client_id.as_ref()
            .ok_or_else(|| anyhow!("GOOGLE_CLIENT_ID not set. Set it in environment or use /auth config"))?;
        let client_secret = self.google_client_secret.as_ref()
            .ok_or_else(|| anyhow!("GOOGLE_CLIENT_SECRET not set. Set it in environment or use /auth config"))?;

        let client = BasicClient::new(
            ClientId::new(client_id.clone()),
            Some(ClientSecret::new(client_secret.clone())),
            AuthUrl::new(GOOGLE_AUTH_URL.to_string())?,
            Some(TokenUrl::new(GOOGLE_TOKEN_URL.to_string())?),
        )
        .set_redirect_uri(RedirectUrl::new(format!("http://localhost:{}/callback", REDIRECT_PORT))?);

        // Generate PKCE challenge
        let (pkce_challenge, pkce_verifier) = PkceCodeChallenge::new_random_sha256();

        // Build authorization URL
        let (auth_url, csrf_token) = client
            .authorize_url(CsrfToken::new_random)
            .add_scope(Scope::new("https://www.googleapis.com/auth/calendar".to_string())) // Full calendar access for create/edit
            .add_scope(Scope::new("https://www.googleapis.com/auth/gmail.send".to_string())) // Send emails
            .add_scope(Scope::new("https://www.googleapis.com/auth/gmail.readonly".to_string())) // Read emails
            .add_scope(Scope::new("https://www.googleapis.com/auth/userinfo.email".to_string()))
            .set_pkce_challenge(pkce_challenge)
            .url();

        println!("🔐 Opening browser for Google authentication...");
        println!("   If it doesn't open, visit: {}", auth_url);
        
        // Open browser
        if let Err(e) = open::that(auth_url.to_string()) {
            eprintln!("Failed to open browser: {}. Please open the URL manually.", e);
        }

        // Start local server to receive callback
        let code = self.wait_for_callback(csrf_token).await?;

        println!("✓ Received authorization code, exchanging for tokens...");

        // Exchange code for token
        let token_result = client
            .exchange_code(AuthorizationCode::new(code))
            .set_pkce_verifier(pkce_verifier)
            .request_async(async_http_client)
            .await
            .map_err(|e| {
                // Provide more detailed error info
                anyhow!("Token exchange failed: {:?}\n   This can happen if the browser cached the previous auth. Try using an incognito window or clearing Google cookies.", e)
            })?;

        // Calculate expiry
        let expires_at = token_result.expires_in().map(|d| {
            chrono::Utc::now().timestamp() + d.as_secs() as i64
        });

        // Get user email
        let email = self.get_google_user_email(token_result.access_token().secret()).await.ok();

        let stored_token = StoredToken {
            access_token: token_result.access_token().secret().to_string(),
            refresh_token: token_result.refresh_token().map(|t| t.secret().to_string()),
            expires_at,
            email,
        };

        // Store in keychain
        self.save_token_to_keychain(Provider::Google, &stored_token)?;

        // Store in memory
        self.tokens.write().await.insert(Provider::Google, stored_token);

        println!("✓ Google authentication successful!");
        Ok(())
    }

    /// Wait for OAuth callback on local server
    async fn wait_for_callback(&self, expected_state: CsrfToken) -> Result<String> {
        use tiny_http::{Server, Response};
        
        let server = Server::http(format!("127.0.0.1:{}", REDIRECT_PORT))
            .map_err(|e| anyhow!("Failed to start callback server: {}", e))?;

        println!("   Waiting for callback on http://localhost:{}...", REDIRECT_PORT);

        // Wait for a request (with timeout)
        let request = tokio::task::spawn_blocking(move || {
            server.recv_timeout(std::time::Duration::from_secs(120))
        })
        .await?
        .map_err(|_| anyhow!("Timeout waiting for OAuth callback"))?
        .ok_or_else(|| anyhow!("No request received"))?;

        // Parse the callback URL
        let url = request.url();
        let params: std::collections::HashMap<_, _> = url
            .split('?')
            .nth(1)
            .unwrap_or("")
            .split('&')
            .filter_map(|p| {
                let mut parts = p.splitn(2, '=');
                Some((parts.next()?, parts.next()?))
            })
            .collect();

        // Verify state
        let state = params.get("state").ok_or_else(|| anyhow!("Missing state parameter"))?;
        if *state != expected_state.secret() {
            return Err(anyhow!("State mismatch - possible CSRF attack"));
        }

        // Get authorization code
        let code = params.get("code")
            .ok_or_else(|| anyhow!("Missing authorization code"))?
            .to_string();

        // Send success response
        let response = Response::from_string(
            "<html><body><h1>✓ Authentication successful!</h1><p>You can close this window and return to DOSA.</p></body></html>"
        ).with_header(
            tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/html"[..]).unwrap()
        );
        let _ = request.respond(response);

        Ok(urlencoding::decode(&code)?.to_string())
    }

    /// Get user email from Google
    async fn get_google_user_email(&self, access_token: &str) -> Result<String> {
        let client = reqwest::Client::new();
        let resp: serde_json::Value = client
            .get("https://www.googleapis.com/oauth2/v2/userinfo")
            .bearer_auth(access_token)
            .send()
            .await?
            .json()
            .await?;

        resp.get("email")
            .and_then(|e| e.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("No email in response"))
    }

    /// Get a valid access token (auto-refreshes if expired)
    pub async fn get_token(&self, provider: Provider) -> Result<String> {
        let tokens = self.tokens.read().await;
        
        if let Some(token) = tokens.get(&provider) {
            if !token.is_expired() {
                return Ok(token.access_token.clone());
            }
            
            // Token expired, need to refresh
            drop(tokens); // Release read lock
            
            if let Some(refresh_token) = self.tokens.read().await.get(&provider)
                .and_then(|t| t.refresh_token.clone()) 
            {
                return self.refresh_token(provider, &refresh_token).await;
            }
        }
        
        Err(anyhow!("Not authenticated with {}. Use /auth {} to authenticate.", 
            provider.name(), provider.name()))
    }

    /// Refresh an expired token
    async fn refresh_token(&self, provider: Provider, refresh_token: &str) -> Result<String> {
        match provider {
            Provider::Google => self.refresh_google_token(refresh_token).await,
        }
    }

    async fn refresh_google_token(&self, refresh_token: &str) -> Result<String> {
        let client_id = self.google_client_id.as_ref()
            .ok_or_else(|| anyhow!("GOOGLE_CLIENT_ID not set"))?;
        let client_secret = self.google_client_secret.as_ref()
            .ok_or_else(|| anyhow!("GOOGLE_CLIENT_SECRET not set"))?;

        let client = BasicClient::new(
            ClientId::new(client_id.clone()),
            Some(ClientSecret::new(client_secret.clone())),
            AuthUrl::new(GOOGLE_AUTH_URL.to_string())?,
            Some(TokenUrl::new(GOOGLE_TOKEN_URL.to_string())?),
        );

        let token_result = client
            .exchange_refresh_token(&RefreshToken::new(refresh_token.to_string()))
            .request_async(async_http_client)
            .await
            .map_err(|e| anyhow!("Token refresh failed: {}", e))?;

        let expires_at = token_result.expires_in().map(|d| {
            chrono::Utc::now().timestamp() + d.as_secs() as i64
        });

        // Get existing token data
        let mut tokens = self.tokens.write().await;
        if let Some(existing) = tokens.get_mut(&Provider::Google) {
            existing.access_token = token_result.access_token().secret().to_string();
            existing.expires_at = expires_at;
            
            // Save updated token
            self.save_token_to_keychain(Provider::Google, existing)?;
            
            return Ok(existing.access_token.clone());
        }

        Err(anyhow!("No existing token to refresh"))
    }

    /// Check if authenticated with a provider
    pub async fn is_authenticated(&self, provider: Provider) -> bool {
        self.tokens.read().await.contains_key(&provider)
    }

    /// Get authentication status for all providers
    pub async fn get_status(&self) -> Vec<AuthStatus> {
        let tokens = self.tokens.read().await;
        
        vec![
            AuthStatus {
                provider: Provider::Google,
                authenticated: tokens.contains_key(&Provider::Google),
                email: tokens.get(&Provider::Google).and_then(|t| t.email.clone()),
                expires_at: tokens.get(&Provider::Google)
                    .and_then(|t| t.expires_at)
                    .map(|ts| chrono::DateTime::from_timestamp(ts, 0).unwrap_or_default()),
            },
        ]
    }

    /// Revoke access for a provider
    pub async fn revoke(&self, provider: Provider) -> Result<()> {
        // Remove from memory
        self.tokens.write().await.remove(&provider);
        
        // Remove from keychain
        self.delete_token_from_keychain(provider)?;
        
        Ok(())
    }

    /// Load token from macOS Keychain
    fn load_token_from_keychain(&self, provider: Provider) -> Result<StoredToken> {
        let entry = keyring::Entry::new(SERVICE_NAME, provider.name())?;
        let json = entry.get_password()?;
        let token: StoredToken = serde_json::from_str(&json)?;
        Ok(token)
    }

    /// Save token to macOS Keychain
    fn save_token_to_keychain(&self, provider: Provider, token: &StoredToken) -> Result<()> {
        let entry = keyring::Entry::new(SERVICE_NAME, provider.name())?;
        let json = serde_json::to_string(token)?;
        entry.set_password(&json)?;
        Ok(())
    }

    /// Delete token from macOS Keychain
    fn delete_token_from_keychain(&self, provider: Provider) -> Result<()> {
        let entry = keyring::Entry::new(SERVICE_NAME, provider.name())?;
        // Ignore error if entry doesn't exist
        let _ = entry.delete_password();
        Ok(())
    }

    /// Load tokens from keychain on startup
    pub async fn load_stored_tokens(&self) {
        if let Ok(token) = self.load_token_from_keychain(Provider::Google) {
            self.tokens.write().await.insert(Provider::Google, token);
        }
    }
}

impl Default for OAuthManager {
    fn default() -> Self {
        Self::new()
    }
}
