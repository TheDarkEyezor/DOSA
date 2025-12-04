//! Messaging frontends for DOSA
//!
//! This module provides an abstraction layer for different messaging interfaces:
//! - CLI (default REPL interface)
//! - WhatsApp Business API
//! - Future: Slack, Discord, Telegram
//!
//! All frontends share the same conversation history and backend capabilities.

pub mod types;
pub mod whatsapp;
pub mod server;
pub mod samantha;
pub mod handler;

pub use types::{IncomingMessage, OutgoingMessage, FrontendType, MessageFrontend};
pub use whatsapp::WhatsAppClient;
pub use server::MessageServer;
pub use samantha::*;
pub use handler::SamanthaHandler;
