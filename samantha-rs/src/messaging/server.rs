//! Message server that handles multiple frontends
//!
//! This module provides a unified server that can handle messages from
//! both CLI and WhatsApp (and future frontends) simultaneously.

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

use super::types::{IncomingMessage, OutgoingMessage, FrontendType};
use super::whatsapp::{WhatsAppClient, WhatsAppConfig, WebhookPayload};

/// Conversation history entry
#[derive(Debug, Clone)]
pub struct ConversationEntry {
    pub role: String,  // "user" or "assistant"
    pub content: String,
    pub frontend: FrontendType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Conversation state for a user
#[derive(Debug, Default, Clone)]
pub struct ConversationState {
    pub history: Vec<ConversationEntry>,
    pub last_active: Option<chrono::DateTime<chrono::Utc>>,
}

impl ConversationState {
    pub fn add_user_message(&mut self, content: &str, frontend: FrontendType) {
        self.history.push(ConversationEntry {
            role: "user".to_string(),
            content: content.to_string(),
            frontend,
            timestamp: chrono::Utc::now(),
        });
        self.last_active = Some(chrono::Utc::now());
        
        // Keep last 20 messages for context
        if self.history.len() > 20 {
            self.history.remove(0);
        }
    }

    pub fn add_assistant_message(&mut self, content: &str, frontend: FrontendType) {
        self.history.push(ConversationEntry {
            role: "assistant".to_string(),
            content: content.to_string(),
            frontend,
            timestamp: chrono::Utc::now(),
        });
        self.last_active = Some(chrono::Utc::now());
        
        if self.history.len() > 20 {
            self.history.remove(0);
        }
    }

    /// Get conversation history as a string for context
    pub fn get_context(&self) -> String {
        self.history
            .iter()
            .map(|e| format!("{}: {}", e.role, e.content))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Message server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Port for webhook server
    pub webhook_port: u16,
    /// Enable WhatsApp integration
    pub enable_whatsapp: bool,
    /// WhatsApp configuration (if enabled)
    pub whatsapp_config: Option<WhatsAppConfig>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            webhook_port: 8080,
            enable_whatsapp: false,
            whatsapp_config: None,
        }
    }
}

/// Message server that handles multiple frontends
pub struct MessageServer {
    /// WhatsApp client (if enabled)
    whatsapp: Option<Arc<WhatsAppClient>>,
    /// Conversation state per user (keyed by user ID)
    conversations: Arc<RwLock<HashMap<String, ConversationState>>>,
    /// Server configuration
    config: ServerConfig,
}

impl MessageServer {
    /// Create a new message server
    pub fn new(config: ServerConfig) -> Result<Self> {
        let whatsapp = if config.enable_whatsapp {
            let wa_config = config.whatsapp_config.clone()
                .ok_or_else(|| anyhow::anyhow!("WhatsApp config required when WhatsApp is enabled"))?;
            Some(Arc::new(WhatsAppClient::new(wa_config)))
        } else {
            None
        };

        Ok(Self {
            whatsapp,
            conversations: Arc::new(RwLock::new(HashMap::new())),
            config,
        })
    }

    /// Create with WhatsApp enabled from environment
    pub fn with_whatsapp_from_env() -> Result<Self> {
        let wa_config = WhatsAppConfig::from_env()?;
        let config = ServerConfig {
            webhook_port: std::env::var("WEBHOOK_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080),
            enable_whatsapp: true,
            whatsapp_config: Some(wa_config),
        };
        Self::new(config)
    }

    /// Get WhatsApp client reference
    pub fn whatsapp(&self) -> Option<&Arc<WhatsAppClient>> {
        self.whatsapp.as_ref()
    }

    /// Get or create conversation state for a user
    pub async fn get_conversation(&self, user_id: &str) -> ConversationState {
        let conversations = self.conversations.read().await;
        conversations.get(user_id).cloned().unwrap_or_default()
    }

    /// Update conversation state
    pub async fn update_conversation(&self, user_id: &str, state: ConversationState) {
        let mut conversations = self.conversations.write().await;
        conversations.insert(user_id.to_string(), state);
    }

    /// Add a user message to conversation
    pub async fn add_user_message(&self, user_id: &str, content: &str, frontend: FrontendType) {
        let mut conversations = self.conversations.write().await;
        let state = conversations.entry(user_id.to_string()).or_default();
        state.add_user_message(content, frontend);
    }

    /// Add an assistant message to conversation
    pub async fn add_assistant_message(&self, user_id: &str, content: &str, frontend: FrontendType) {
        let mut conversations = self.conversations.write().await;
        let state = conversations.entry(user_id.to_string()).or_default();
        state.add_assistant_message(content, frontend);
    }

    /// Process an incoming message and return the response
    /// This is a placeholder - the actual processing happens in main.rs
    pub async fn handle_message(&self, msg: IncomingMessage) -> Result<String> {
        // Add to conversation history
        self.add_user_message(&msg.sender_id, &msg.text, msg.frontend).await;
        
        // The actual processing will be done by the caller (DOSA core)
        // This just manages the conversation state
        Ok(msg.text)
    }

    /// Send a response and record it in conversation
    pub async fn send_response(&self, user_id: &str, response: &str, frontend: FrontendType) -> Result<()> {
        // Record in conversation
        self.add_assistant_message(user_id, response, frontend).await;
        
        // Send via appropriate frontend
        match frontend {
            FrontendType::WhatsApp => {
                if let Some(wa) = &self.whatsapp {
                    wa.send_message(user_id, response).await?;
                }
            }
            FrontendType::Cli => {
                // CLI output is handled directly in main.rs
            }
        }
        
        Ok(())
    }

    /// Process a WhatsApp webhook payload
    pub async fn process_whatsapp_webhook(&self, payload: WebhookPayload) -> Vec<IncomingMessage> {
        if let Some(wa) = &self.whatsapp {
            wa.process_webhook(payload).await
        } else {
            Vec::new()
        }
    }

    /// Get the webhook verification token
    pub fn webhook_verify_token(&self) -> Option<&str> {
        self.whatsapp.as_ref().map(|wa| wa.verify_token())
    }
}

/// Simple webhook server for receiving WhatsApp messages
/// Uses axum for HTTP handling
#[cfg(feature = "whatsapp")]
pub mod webhook_server {
    use super::*;
    use axum::{
        extract::{Query, State},
        http::StatusCode,
        routing::{get, post},
        Json, Router,
    };
    use serde::Deserialize;
    use std::net::SocketAddr;
    use tokio::sync::mpsc;

    #[derive(Debug, Deserialize)]
    pub struct WebhookVerifyParams {
        #[serde(rename = "hub.mode")]
        mode: Option<String>,
        #[serde(rename = "hub.verify_token")]
        verify_token: Option<String>,
        #[serde(rename = "hub.challenge")]
        challenge: Option<String>,
    }

    pub struct WebhookState {
        pub server: Arc<MessageServer>,
        pub message_tx: mpsc::Sender<IncomingMessage>,
    }

    /// Webhook verification endpoint (GET)
    async fn verify_webhook(
        State(state): State<Arc<WebhookState>>,
        Query(params): Query<WebhookVerifyParams>,
    ) -> Result<String, StatusCode> {
        let expected_token = state.server.webhook_verify_token()
            .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;
        
        match (params.mode.as_deref(), params.verify_token.as_deref(), params.challenge) {
            (Some("subscribe"), Some(token), Some(challenge)) if token == expected_token => {
                log::info!("Webhook verified successfully");
                Ok(challenge)
            }
            _ => {
                log::warn!("Webhook verification failed");
                Err(StatusCode::FORBIDDEN)
            }
        }
    }

    /// Webhook message endpoint (POST)
    async fn receive_webhook(
        State(state): State<Arc<WebhookState>>,
        Json(payload): Json<WebhookPayload>,
    ) -> StatusCode {
        let messages = state.server.process_whatsapp_webhook(payload).await;
        
        for msg in messages {
            if let Err(e) = state.message_tx.send(msg).await {
                log::error!("Failed to queue message: {}", e);
            }
        }
        
        StatusCode::OK
    }

    /// Health check endpoint
    async fn health() -> &'static str {
        "OK"
    }

    /// Start the webhook server
    pub async fn start_webhook_server(
        server: Arc<MessageServer>,
        message_tx: mpsc::Sender<IncomingMessage>,
        port: u16,
    ) -> Result<()> {
        let state = Arc::new(WebhookState { server, message_tx });
        
        let app = Router::new()
            .route("/webhook", get(verify_webhook))
            .route("/webhook", post(receive_webhook))
            .route("/health", get(health))
            .with_state(state);
        
        let addr = SocketAddr::from(([0, 0, 0, 0], port));
        log::info!("Starting webhook server on {}", addr);
        
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
        
        Ok(())
    }
}
