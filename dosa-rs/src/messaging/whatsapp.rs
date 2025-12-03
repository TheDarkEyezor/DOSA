//! WhatsApp Business API client
//!
//! This module provides integration with the WhatsApp Business Cloud API.
//! 
//! Setup:
//! 1. Create a Meta Business account and WhatsApp Business app
//! 2. Get your Phone Number ID and Access Token from the Meta Developer Portal
//! 3. Set up a webhook to receive messages (or use polling mode for dev)
//! 4. Configure WHATSAPP_PHONE_ID and WHATSAPP_TOKEN environment variables

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::Utc;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::types::{FrontendType, IncomingMessage, MessageFrontend, OutgoingMessage};

const WHATSAPP_API_BASE: &str = "https://graph.facebook.com/v18.0";

/// WhatsApp Business API client
pub struct WhatsAppClient {
    /// HTTP client
    http: Client,
    /// WhatsApp Business Phone Number ID
    phone_number_id: String,
    /// Access token for the API
    access_token: String,
    /// Authorized phone numbers that can interact with the bot
    allowed_numbers: HashSet<String>,
    /// Queue of received messages (populated by webhook)
    message_queue: Arc<RwLock<Vec<WebhookMessage>>>,
    /// Webhook verification token
    verify_token: String,
}

/// Configuration for WhatsApp client
#[derive(Debug, Clone)]
pub struct WhatsAppConfig {
    /// WhatsApp Business Phone Number ID
    pub phone_number_id: String,
    /// Access token from Meta Developer Portal
    pub access_token: String,
    /// Phone numbers authorized to use the bot (E.164 format: +1234567890)
    pub allowed_numbers: Vec<String>,
    /// Webhook verification token (you choose this)
    pub verify_token: String,
}

impl WhatsAppConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let phone_number_id = std::env::var("WHATSAPP_PHONE_ID")
            .map_err(|_| anyhow!("WHATSAPP_PHONE_ID environment variable not set"))?;
        
        let access_token = std::env::var("WHATSAPP_TOKEN")
            .map_err(|_| anyhow!("WHATSAPP_TOKEN environment variable not set"))?;
        
        let verify_token = std::env::var("WHATSAPP_VERIFY_TOKEN")
            .unwrap_or_else(|_| "dosa_verify_token".to_string());
        
        // Allowed numbers can be comma-separated
        let allowed_numbers: Vec<String> = std::env::var("WHATSAPP_ALLOWED_NUMBERS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        
        Ok(Self {
            phone_number_id,
            access_token,
            allowed_numbers,
            verify_token,
        })
    }

    /// Load from a config file
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: WhatsAppConfigFile = serde_json::from_str(&content)?;
        
        Ok(Self {
            phone_number_id: config.phone_number_id,
            access_token: config.access_token,
            allowed_numbers: config.allowed_numbers,
            verify_token: config.verify_token.unwrap_or_else(|| "dosa_verify_token".to_string()),
        })
    }
}

#[derive(Debug, Deserialize)]
struct WhatsAppConfigFile {
    phone_number_id: String,
    access_token: String,
    allowed_numbers: Vec<String>,
    verify_token: Option<String>,
}

/// Message received via webhook
#[derive(Debug, Clone, Deserialize)]
pub struct WebhookMessage {
    pub from: String,
    pub id: String,
    pub timestamp: String,
    pub text: Option<TextMessage>,
    #[serde(rename = "type")]
    pub msg_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextMessage {
    pub body: String,
}

/// WhatsApp webhook payload
#[derive(Debug, Deserialize)]
pub struct WebhookPayload {
    pub object: String,
    pub entry: Vec<WebhookEntry>,
}

#[derive(Debug, Deserialize)]
pub struct WebhookEntry {
    pub id: String,
    pub changes: Vec<WebhookChange>,
}

#[derive(Debug, Deserialize)]
pub struct WebhookChange {
    pub value: WebhookValue,
    pub field: String,
}

#[derive(Debug, Deserialize)]
pub struct WebhookValue {
    pub messaging_product: String,
    pub metadata: WebhookMetadata,
    pub contacts: Option<Vec<WebhookContact>>,
    pub messages: Option<Vec<WebhookMessage>>,
    pub statuses: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
pub struct WebhookMetadata {
    pub display_phone_number: String,
    pub phone_number_id: String,
}

#[derive(Debug, Deserialize)]
pub struct WebhookContact {
    pub profile: ContactProfile,
    pub wa_id: String,
}

#[derive(Debug, Deserialize)]
pub struct ContactProfile {
    pub name: String,
}

/// Request body for sending messages
#[derive(Debug, Serialize)]
struct SendMessageRequest {
    messaging_product: String,
    recipient_type: String,
    to: String,
    #[serde(rename = "type")]
    msg_type: String,
    text: SendTextBody,
}

#[derive(Debug, Serialize)]
struct SendTextBody {
    preview_url: bool,
    body: String,
}

/// Response from send message API
#[derive(Debug, Deserialize)]
struct SendMessageResponse {
    messaging_product: String,
    contacts: Vec<ResponseContact>,
    messages: Vec<ResponseMessage>,
}

#[derive(Debug, Deserialize)]
struct ResponseContact {
    input: String,
    wa_id: String,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    id: String,
}

impl WhatsAppClient {
    /// Create a new WhatsApp client from configuration
    pub fn new(config: WhatsAppConfig) -> Self {
        Self {
            http: Client::new(),
            phone_number_id: config.phone_number_id,
            access_token: config.access_token,
            allowed_numbers: config.allowed_numbers.into_iter().collect(),
            message_queue: Arc::new(RwLock::new(Vec::new())),
            verify_token: config.verify_token,
        }
    }

    /// Create from environment variables
    pub fn from_env() -> Result<Self> {
        let config = WhatsAppConfig::from_env()?;
        Ok(Self::new(config))
    }

    /// Get the webhook verification token
    pub fn verify_token(&self) -> &str {
        &self.verify_token
    }

    /// Get reference to message queue for webhook handler
    pub fn message_queue(&self) -> Arc<RwLock<Vec<WebhookMessage>>> {
        Arc::clone(&self.message_queue)
    }

    /// Check if a phone number is authorized
    pub fn is_authorized(&self, phone: &str) -> bool {
        if self.allowed_numbers.is_empty() {
            // If no numbers configured, allow all (dev mode)
            return true;
        }
        
        // Normalize phone number (remove +, spaces, dashes)
        let normalized: String = phone.chars()
            .filter(|c| c.is_ascii_digit())
            .collect();
        
        self.allowed_numbers.iter().any(|allowed| {
            let allowed_normalized: String = allowed.chars()
                .filter(|c| c.is_ascii_digit())
                .collect();
            normalized == allowed_normalized || normalized.ends_with(&allowed_normalized)
        })
    }

    /// Send a text message
    pub async fn send_message(&self, to: &str, text: &str) -> Result<String> {
        let url = format!("{}/{}/messages", WHATSAPP_API_BASE, self.phone_number_id);
        
        let request = SendMessageRequest {
            messaging_product: "whatsapp".to_string(),
            recipient_type: "individual".to_string(),
            to: to.to_string(),
            msg_type: "text".to_string(),
            text: SendTextBody {
                preview_url: false,
                body: text.to_string(),
            },
        };

        let resp = self.http
            .post(&url)
            .bearer_auth(&self.access_token)
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("WhatsApp API error {}: {}", status, body));
        }

        let response: SendMessageResponse = resp.json().await?;
        Ok(response.messages.first()
            .map(|m| m.id.clone())
            .unwrap_or_default())
    }

    /// Process a webhook payload and queue messages
    pub async fn process_webhook(&self, payload: WebhookPayload) -> Vec<IncomingMessage> {
        let mut incoming = Vec::new();

        for entry in &payload.entry {
            for change in &entry.changes {
                if change.field != "messages" {
                    continue;
                }

                if let Some(messages) = &change.value.messages {
                    for msg in messages {
                        // Only process text messages for now
                        if msg.msg_type != "text" {
                            continue;
                        }

                        let text = match &msg.text {
                            Some(t) => t.body.clone(),
                            None => continue,
                        };

                        // Check authorization
                        if !self.is_authorized(&msg.from) {
                            log::warn!("Unauthorized message from: {}", msg.from);
                            continue;
                        }

                        // Get sender name from contacts if available
                        let sender_name = change.value.contacts
                            .as_ref()
                            .and_then(|c| c.first())
                            .map(|c| c.profile.name.clone());

                        let mut incoming_msg = IncomingMessage::new(
                            text,
                            msg.from.clone(),
                            FrontendType::WhatsApp,
                        );
                        incoming_msg.message_id = Some(msg.id.clone());
                        
                        if let Some(name) = sender_name {
                            incoming_msg = incoming_msg.with_metadata("sender_name", name);
                        }

                        incoming.push(incoming_msg);
                    }
                }
            }
        }

        // Also add to internal queue
        {
            let mut queue = self.message_queue.write().await;
            for entry in &payload.entry {
                for change in &entry.changes {
                    if let Some(messages) = &change.value.messages {
                        queue.extend(messages.iter().cloned());
                    }
                }
            }
        }

        incoming
    }

    /// Poll for new messages from the queue
    pub async fn poll_messages(&self) -> Vec<IncomingMessage> {
        let mut queue = self.message_queue.write().await;
        let messages: Vec<_> = queue.drain(..).collect();
        
        messages.into_iter()
            .filter_map(|msg| {
                if msg.msg_type != "text" {
                    return None;
                }
                
                let text = msg.text?.body;
                
                if !self.is_authorized(&msg.from) {
                    return None;
                }

                let mut incoming = IncomingMessage::new(
                    text,
                    msg.from,
                    FrontendType::WhatsApp,
                );
                incoming.message_id = Some(msg.id);
                
                Some(incoming)
            })
            .collect()
    }

    /// Mark a message as read
    pub async fn mark_read(&self, message_id: &str) -> Result<()> {
        let url = format!("{}/{}/messages", WHATSAPP_API_BASE, self.phone_number_id);
        
        let body = serde_json::json!({
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id
        });

        let resp = self.http
            .post(&url)
            .bearer_auth(&self.access_token)
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("WhatsApp API error marking read {}: {}", status, body));
        }

        Ok(())
    }
}

#[async_trait]
impl MessageFrontend for WhatsAppClient {
    async fn receive(&mut self) -> Option<IncomingMessage> {
        // Poll the message queue
        let messages = self.poll_messages().await;
        messages.into_iter().next()
    }

    async fn send(&self, msg: OutgoingMessage) -> Result<()> {
        self.send_message(&msg.recipient_id, &msg.text).await?;
        Ok(())
    }

    fn frontend_type(&self) -> FrontendType {
        FrontendType::WhatsApp
    }

    async fn is_healthy(&self) -> bool {
        // Could ping the API here, but for now just return true
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phone_number_authorization() {
        let config = WhatsAppConfig {
            phone_number_id: "123".to_string(),
            access_token: "token".to_string(),
            allowed_numbers: vec!["+1234567890".to_string(), "9876543210".to_string()],
            verify_token: "verify".to_string(),
        };
        
        let client = WhatsAppClient::new(config);
        
        assert!(client.is_authorized("+1234567890"));
        assert!(client.is_authorized("1234567890"));
        assert!(client.is_authorized("9876543210"));
        assert!(!client.is_authorized("+1111111111"));
    }

    #[test]
    fn test_empty_allowed_numbers_allows_all() {
        let config = WhatsAppConfig {
            phone_number_id: "123".to_string(),
            access_token: "token".to_string(),
            allowed_numbers: vec![],
            verify_token: "verify".to_string(),
        };
        
        let client = WhatsAppClient::new(config);
        
        assert!(client.is_authorized("+1234567890"));
        assert!(client.is_authorized("anything"));
    }
}
