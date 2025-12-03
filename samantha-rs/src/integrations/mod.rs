//! External service integrations (Google Calendar, Gmail, Web Search)
//! 
//! This module provides OAuth2 authentication and API clients for:
//! - Google Calendar (read & write)
//! - Gmail (read-only)
//! - Web Search (DuckDuckGo, Weather)
//! 
//! All tokens are stored securely in the macOS Keychain.

pub mod oauth;
pub mod google;
pub mod importance;
pub mod calendar_ai;
pub mod email_ai;
pub mod email_reply;
pub mod intent;
pub mod web_search;

pub use oauth::{OAuthManager, Provider, AuthStatus};
pub use google::calendar::{GoogleCalendarClient, NewCalendarEvent, ParsedEvent, EventUpdate};
pub use google::gmail::{GmailClient, EmailDraft};
pub use importance::ImportanceScorer;
pub use calendar_ai::{CalendarAI, EventCreationResult, EventUpdateResult, EventDeleteResult, ResolvedAttendee};
pub use email_ai::{EmailAI, EmailComposeResult};
pub use email_reply::{SmartReply, SmartReplyResult, ReplyIntent, parse_reply_intent};
pub use intent::{is_calendar_intent, is_calendar_query, detect_calendar_intent, CalendarIntent, detect_email_intent, EmailIntent};
pub use web_search::{WebSearch, SearchResult, WeatherInfo};

