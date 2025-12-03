//! Common types for messaging frontends

use chrono::{DateTime, Utc};
use std::collections::HashMap;
use async_trait::async_trait;
use anyhow::Result;

/// Type of messaging frontend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrontendType {
    Cli,
    WhatsApp,
    // Future:
    // Slack,
    // Discord,
    // Telegram,
}

impl std::fmt::Display for FrontendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrontendType::Cli => write!(f, "CLI"),
            FrontendType::WhatsApp => write!(f, "WhatsApp"),
        }
    }
}

/// An incoming message from any frontend
#[derive(Debug, Clone)]
pub struct IncomingMessage {
    /// The message text
    pub text: String,
    /// Sender identifier (phone number, user ID, etc.)
    pub sender_id: String,
    /// When the message was received
    pub timestamp: DateTime<Utc>,
    /// Which frontend this came from
    pub frontend: FrontendType,
    /// Platform-specific metadata
    pub metadata: HashMap<String, String>,
    /// Message ID from the platform (for replies, reactions, etc.)
    pub message_id: Option<String>,
}

impl IncomingMessage {
    pub fn new(text: impl Into<String>, sender_id: impl Into<String>, frontend: FrontendType) -> Self {
        Self {
            text: text.into(),
            sender_id: sender_id.into(),
            timestamp: Utc::now(),
            frontend,
            metadata: HashMap::new(),
            message_id: None,
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// An outgoing message to send via any frontend
#[derive(Debug, Clone)]
pub struct OutgoingMessage {
    /// The message text
    pub text: String,
    /// Recipient identifier
    pub recipient_id: String,
    /// Optional formatting hints
    pub formatting: MessageFormat,
    /// Reply to a specific message (if supported)
    pub reply_to: Option<String>,
}

impl OutgoingMessage {
    pub fn new(text: impl Into<String>, recipient_id: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            recipient_id: recipient_id.into(),
            formatting: MessageFormat::Plain,
            reply_to: None,
        }
    }

    pub fn with_format(mut self, format: MessageFormat) -> Self {
        self.formatting = format;
        self
    }

    pub fn replying_to(mut self, message_id: impl Into<String>) -> Self {
        self.reply_to = Some(message_id.into());
        self
    }
}

/// Message formatting options
#[derive(Debug, Clone, Copy, Default)]
pub enum MessageFormat {
    #[default]
    Plain,
    Markdown,
    /// WhatsApp-specific formatting (*bold*, _italic_, ```code```)
    WhatsApp,
}

/// Trait for message frontends (CLI, WhatsApp, etc.)
#[async_trait]
pub trait MessageFrontend: Send + Sync {
    /// Receive the next message (may block)
    async fn receive(&mut self) -> Option<IncomingMessage>;
    
    /// Send a message
    async fn send(&self, msg: OutgoingMessage) -> Result<()>;
    
    /// Get the frontend type
    fn frontend_type(&self) -> FrontendType;
    
    /// Check if the frontend is connected/healthy
    async fn is_healthy(&self) -> bool;
}

/// Convert DOSA's output format to WhatsApp-friendly format
pub fn format_for_whatsapp(text: &str) -> String {
    // WhatsApp supports: *bold*, _italic_, ~strikethrough~, ```code```
    // Convert some common patterns
    text
        // Convert emoji markers
        .replace("✓", "✓")
        .replace("✗", "✗")
        .replace("📅", "📅")
        .replace("📧", "📧")
        // Convert headers to bold
        .lines()
        .map(|line| {
            if line.starts_with("##") {
                format!("*{}*", line.trim_start_matches('#').trim())
            } else if line.starts_with('#') {
                format!("*{}*", line.trim_start_matches('#').trim())
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}
