//! Phase 6 Commands: External Integrations
//!
//! Commands for Google Calendar and Gmail integration.

use std::sync::{Arc, Mutex};
use anyhow::Result;
use chrono::Timelike;
use tokio::runtime::Handle;
use once_cell::sync::Lazy;

use crate::router::{Command, CommandContext, CommandResult, Router};
use crate::integrations::{OAuthManager, Provider, GoogleCalendarClient, GmailClient, ImportanceScorer, CalendarAI, NewCalendarEvent, EventUpdate, EmailDraft, EmailAI, EmailComposeResult};

/// Pending event waiting for confirmation
static PENDING_EVENT: Lazy<Mutex<Option<NewCalendarEvent>>> = Lazy::new(|| Mutex::new(None));

/// Pending update waiting for confirmation (event_id, update)
static PENDING_UPDATE: Lazy<Mutex<Option<(String, EventUpdate)>>> = Lazy::new(|| Mutex::new(None));

/// Pending delete waiting for confirmation (event_id)
static PENDING_DELETE: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));

/// Pending email waiting for confirmation
static PENDING_EMAIL: Lazy<Mutex<Option<EmailComposeResult>>> = Lazy::new(|| Mutex::new(None));

/// Enum for pending action type
#[derive(Debug, Clone)]
enum PendingAction {
    Create,
    Update,
    Delete,
    Email,
    None,
}

fn get_pending_action() -> PendingAction {
    if PENDING_EVENT.lock().unwrap().is_some() {
        PendingAction::Create
    } else if PENDING_UPDATE.lock().unwrap().is_some() {
        PendingAction::Update
    } else if PENDING_DELETE.lock().unwrap().is_some() {
        PendingAction::Delete
    } else if PENDING_EMAIL.lock().unwrap().is_some() {
        PendingAction::Email
    } else {
        PendingAction::None
    }
}

/// Store a pending event (called from main.rs for natural language calendar creation)
pub fn store_pending_event(event: NewCalendarEvent) {
    // Clear any other pending actions
    PENDING_UPDATE.lock().unwrap().take();
    PENDING_DELETE.lock().unwrap().take();
    PENDING_EMAIL.lock().unwrap().take();
    PENDING_EVENT.lock().unwrap().replace(event);
}

/// Store a pending update
pub fn store_pending_update(event_id: String, update: EventUpdate) {
    // Clear any other pending actions
    PENDING_EVENT.lock().unwrap().take();
    PENDING_DELETE.lock().unwrap().take();
    PENDING_EMAIL.lock().unwrap().take();
    PENDING_UPDATE.lock().unwrap().replace((event_id, update));
}

/// Store a pending delete
pub fn store_pending_delete(event_id: String) {
    // Clear any other pending actions
    PENDING_EVENT.lock().unwrap().take();
    PENDING_UPDATE.lock().unwrap().take();
    PENDING_EMAIL.lock().unwrap().take();
    PENDING_DELETE.lock().unwrap().replace(event_id);
}

/// Store a pending email
pub fn store_pending_email(draft: EmailComposeResult) {
    // Clear any other pending actions
    PENDING_EVENT.lock().unwrap().take();
    PENDING_UPDATE.lock().unwrap().take();
    PENDING_DELETE.lock().unwrap().take();
    PENDING_EMAIL.lock().unwrap().replace(draft);
}

/// Helper to run async code from sync context when already inside a tokio runtime
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    // We're inside a tokio runtime (from #[tokio::main])
    // Use task::block_in_place to run sync code that blocks on async
    tokio::task::block_in_place(|| {
        Handle::current().block_on(f)
    })
}

// ============================================================================
// Auth Command
// ============================================================================

pub struct AuthCommand;

impl Command for AuthCommand {
    fn name(&self) -> &str { "/auth" }
    fn description(&self) -> &str { "Authenticate with external services (Google Calendar, Gmail)" }
    fn usage(&self) -> &str { "/auth <google|status|revoke>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/auth")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let args: Vec<&str> = input.trim()
            .strip_prefix("/auth")
            .unwrap_or("")
            .trim()
            .split_whitespace()
            .collect();

        if args.is_empty() {
            return Ok(CommandResult::Success(format!(
                "🔐 Authentication Help\n\n\
                 Before authenticating, you need to:\n\
                 1. Create a project at https://console.cloud.google.com\n\
                 2. Enable Calendar API and Gmail API\n\
                 3. Create OAuth 2.0 credentials (Desktop app)\n\
                 4. Set environment variables:\n\
                    export GOOGLE_CLIENT_ID=\"your-client-id\"\n\
                    export GOOGLE_CLIENT_SECRET=\"your-secret\"\n\n\
                 Commands:\n\
                 • /auth google  - Authenticate with Google (Calendar + Gmail)\n\
                 • /auth status  - Show authentication status\n\
                 • /auth revoke  - Remove stored credentials\n\n\
                 Usage: {}", self.usage()
            )));
        }

        let oauth = OAuthManager::new();

        match args[0].to_lowercase().as_str() {
            "google" => {
                if !oauth.has_credentials(Provider::Google) {
                    return Ok(CommandResult::Error(
                        "Google credentials not configured.\n\
                         Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables.\n\n\
                         Get credentials from: https://console.cloud.google.com/apis/credentials".to_string()
                    ));
                }

                println!("🔐 Starting Google OAuth flow...");
                block_on(oauth.authenticate(Provider::Google))?;

                Ok(CommandResult::Success(
                    "✓ Successfully authenticated with Google!\n\
                     You can now use /gcal and /email commands.".to_string()
                ))
            }
            "status" => {
                let statuses = block_on(oauth.get_status());
                
                let mut output = String::from("🔐 Authentication Status\n\n");
                
                for status in statuses {
                    let icon = if status.authenticated { "✓" } else { "✗" };
                    output.push_str(&format!("{} {}: ", icon, 
                        match status.provider {
                            Provider::Google => "Google (Calendar + Gmail)",
                        }
                    ));
                    
                    if status.authenticated {
                        if let Some(email) = status.email {
                            output.push_str(&format!("{}", email));
                        } else {
                            output.push_str("Authenticated");
                        }
                        if let Some(expires) = status.expires_at {
                            output.push_str(&format!(" (expires: {})", 
                                expires.format("%b %d, %H:%M")));
                        }
                    } else {
                        output.push_str("Not authenticated");
                    }
                    output.push('\n');
                }

                if !oauth.has_credentials(Provider::Google) {
                    output.push_str("\n⚠️  Google credentials not configured.\n");
                    output.push_str("   Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET");
                }

                Ok(CommandResult::Success(output))
            }
            "revoke" => {
                block_on(oauth.revoke(Provider::Google))?;
                Ok(CommandResult::Success(
                    "✓ Google credentials revoked. You'll need to re-authenticate to use integrations.".to_string()
                ))
            }
            _ => {
                Ok(CommandResult::Error(format!("Unknown auth command: {}\n{}", args[0], self.usage())))
            }
        }
    }
}

// ============================================================================
// Google Calendar Command
// ============================================================================

pub struct GCalCommand;

impl Command for GCalCommand {
    fn name(&self) -> &str { "/gcal" }
    fn description(&self) -> &str { "View and manage Google Calendar events" }
    fn usage(&self) -> &str { "/gcal [today|tomorrow|week|summary|create <description>]" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/gcal")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let args = input.trim()
            .strip_prefix("/gcal")
            .unwrap_or("")
            .trim();

        let oauth = Arc::new(OAuthManager::new());
        
        // Load stored tokens
        block_on(oauth.load_stored_tokens());

        if !block_on(oauth.is_authenticated(Provider::Google)) {
            return Ok(CommandResult::Error(
                "Not authenticated with Google. Use /auth google first.".to_string()
            ));
        }

        let client = GoogleCalendarClient::new(oauth.clone());
        let lower = args.to_lowercase();

        // Handle "create" subcommand
        if lower.starts_with("create ") {
            let description = args.strip_prefix("create ").or(args.strip_prefix("Create "))
                .unwrap_or("").trim();
            
            if description.is_empty() {
                return Ok(CommandResult::Error(
                    "Please describe the event.\nExample: /gcal create meeting with John tomorrow at 2pm".to_string()
                ));
            }

            let calendar_ai = CalendarAI::new(ctx.llm, ctx.graph, oauth);
            
            match block_on(calendar_ai.create_from_natural_language(description)) {
                Ok(result) => {
                    let mut output = result.display_preview();
                    
                    if result.has_unresolved() {
                        output.push_str("\n\n⚠️  Some attendees don't have emails. Add emails to your contacts or create without them.");
                    }
                    
                    output.push_str("\n\n💡 To confirm, use: /gcal confirm");
                    output.push_str("\n   To cancel: /gcal cancel");
                    
                    // Store pending event in a simple way (we'll use static for now)
                    // In a real app, you'd store this in the context
                    PENDING_EVENT.lock().unwrap().replace(result.event);
                    
                    return Ok(CommandResult::Success(output));
                }
                Err(e) => {
                    return Ok(CommandResult::Error(format!("Failed to parse event: {}", e)));
                }
            }
        }

        // Handle "confirm" - execute the pending action (create/update/delete)
        if lower == "confirm" || lower == "yes" {
            match get_pending_action() {
                PendingAction::Create => {
                    let pending = PENDING_EVENT.lock().unwrap().take();
                    if let Some(event) = pending {
                        let calendar_ai = CalendarAI::new(ctx.llm, ctx.graph, oauth);
                        let send_invites = !event.attendees.is_empty();
                        
                        match block_on(calendar_ai.commit_event(&event, send_invites)) {
                            Ok(msg) => return Ok(CommandResult::Success(msg)),
                            Err(e) => return Ok(CommandResult::Error(format!("Failed to create event: {}", e))),
                        }
                    }
                }
                PendingAction::Update => {
                    let pending = PENDING_UPDATE.lock().unwrap().take();
                    if let Some((event_id, update)) = pending {
                        let calendar_ai = CalendarAI::new(ctx.llm, ctx.graph, oauth);
                        
                        match block_on(calendar_ai.commit_update(&event_id, &update)) {
                            Ok(msg) => return Ok(CommandResult::Success(msg)),
                            Err(e) => return Ok(CommandResult::Error(format!("Failed to update event: {}", e))),
                        }
                    }
                }
                PendingAction::Delete => {
                    let pending = PENDING_DELETE.lock().unwrap().take();
                    if let Some(event_id) = pending {
                        let calendar_ai = CalendarAI::new(ctx.llm, ctx.graph, oauth);
                        
                        match block_on(calendar_ai.commit_delete(&event_id)) {
                            Ok(msg) => return Ok(CommandResult::Success(msg)),
                            Err(e) => return Ok(CommandResult::Error(format!("Failed to delete event: {}", e))),
                        }
                    }
                }
                PendingAction::Email => {
                    // Email confirmation should use /email confirm
                    return Ok(CommandResult::Error("Use /email confirm to send pending email.".to_string()));
                }
                PendingAction::None => {
                    return Ok(CommandResult::Error("No pending action to confirm.".to_string()));
                }
            }
        }

        // Handle "cancel" - discard pending action
        if lower == "cancel" || lower == "no" {
            let action = get_pending_action();
            PENDING_EVENT.lock().unwrap().take();
            PENDING_UPDATE.lock().unwrap().take();
            PENDING_DELETE.lock().unwrap().take();
            PENDING_EMAIL.lock().unwrap().take();
            
            let msg = match action {
                PendingAction::Create => "Event creation cancelled.",
                PendingAction::Update => "Event update cancelled.",
                PendingAction::Delete => "Event deletion cancelled.",
                PendingAction::Email => "Email discarded.",
                PendingAction::None => "Nothing to cancel.",
            };
            return Ok(CommandResult::Success(msg.to_string()));
        }

        // Handle "summary" for detailed week view
        if lower == "summary" || lower == "week summary" {
            match block_on(client.get_week_summary()) {
                Ok(summary) => return Ok(CommandResult::Success(summary)),
                Err(e) => return Ok(CommandResult::Error(format!("Failed to get calendar summary: {}", e))),
            }
        }

        // Regular event listing
        let result: Result<(Vec<_>, String), anyhow::Error> = block_on(async {
            match lower.as_str() {
                "tomorrow" => {
                    let events = client.get_tomorrow_events().await?;
                    Ok((events, "Tomorrow's Events".to_string()))
                }
                "week" => {
                    // Use the detailed week summary
                    let summary = client.get_week_summary().await?;
                    Ok((Vec::new(), summary))
                }
                "upcoming" => {
                    let events = client.get_upcoming_events(10).await?;
                    Ok((events, "Upcoming Events".to_string()))
                }
                _ => {
                    let events = client.get_today_events().await?;
                    Ok((events, "Today's Events".to_string()))
                }
            }
        });

        match result {
            Ok((events, title)) => {
                // If title is long (week summary), return it directly
                if title.len() > 50 {
                    return Ok(CommandResult::Success(title));
                }
                
                if events.is_empty() {
                    return Ok(CommandResult::Success(format!("📅 {}\n\nNo events scheduled.", title)));
                }

                let mut output = format!("📅 {} ({} events)\n\n", title, events.len());
                for event in events {
                    output.push_str(&event.display());
                    output.push_str("\n\n");
                }
                Ok(CommandResult::Success(output.trim().to_string()))
            }
            Err(e) => Ok(CommandResult::Error(format!("Failed to fetch calendar: {}", e)))
        }
    }
}

// ============================================================================
// Email Command
// ============================================================================

pub struct EmailCommand;

impl Command for EmailCommand {
    fn name(&self) -> &str { "/email" }
    fn description(&self) -> &str { "View and send Gmail emails" }
    fn usage(&self) -> &str { "/email [list|unread|read <id>|summary|from:<name>|compose <message>|confirm|cancel]" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/email")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let args = input.trim()
            .strip_prefix("/email")
            .unwrap_or("")
            .trim();

        let oauth = Arc::new(OAuthManager::new());
        
        // Load stored tokens
        block_on(oauth.load_stored_tokens());

        if !block_on(oauth.is_authenticated(Provider::Google)) {
            return Ok(CommandResult::Error(
                "Not authenticated with Google. Use /auth google first.".to_string()
            ));
        }

        let client = GmailClient::new(oauth.clone());

        // Parse subcommand
        let lower = args.to_lowercase();
        
        // Handle "compose" subcommand
        if lower.starts_with("compose ") {
            let description = args.strip_prefix("compose ").or(args.strip_prefix("Compose "))
                .unwrap_or("").trim();
            
            if description.is_empty() {
                return Ok(CommandResult::Error(
                    "Please describe the email you want to send.\nExample: /email compose send a note to John about the project deadline".to_string()
                ));
            }

            let email_ai = EmailAI::new(ctx.llm, ctx.graph, oauth);
            
            match block_on(email_ai.compose_email(description)) {
                Ok(result) => {
                    let mut output = result.display_preview();
                    
                    output.push_str("\n\n💡 To send, use: /email confirm");
                    output.push_str("\n   To cancel: /email cancel");
                    
                    PENDING_EMAIL.lock().unwrap().replace(result);
                    
                    return Ok(CommandResult::Success(output));
                }
                Err(e) => {
                    return Ok(CommandResult::Error(format!("Failed to compose email: {}", e)));
                }
            }
        }

        // Handle "confirm" - send the pending email
        if lower == "confirm" || lower == "yes" || lower == "send" {
            let pending = PENDING_EMAIL.lock().unwrap().take();
            if let Some(draft_result) = pending {
                match block_on(client.send_message(&draft_result.draft)) {
                    Ok(_) => {
                        return Ok(CommandResult::Success(format!(
                            "✓ Email sent to {}!\n  Subject: {}",
                            draft_result.draft.to.join(", "),
                            draft_result.draft.subject
                        )));
                    }
                    Err(e) => {
                        // Put it back in case they want to retry
                        PENDING_EMAIL.lock().unwrap().replace(draft_result);
                        return Ok(CommandResult::Error(format!("Failed to send email: {}", e)));
                    }
                }
            } else {
                return Ok(CommandResult::Error("No pending email to send.".to_string()));
            }
        }

        // Handle "cancel" - discard pending email
        if lower == "cancel" || lower == "no" {
            let pending = PENDING_EMAIL.lock().unwrap().take();
            if pending.is_some() {
                return Ok(CommandResult::Success("Email discarded.".to_string()));
            } else {
                return Ok(CommandResult::Error("No pending email to cancel.".to_string()));
            }
        }
        
        if lower.is_empty() || lower == "list" {
            // List recent emails
            let emails = block_on(client.list_messages(None, 10))?;
            
            if emails.is_empty() {
                return Ok(CommandResult::Success("📧 No emails found.".to_string()));
            }

            let mut output = format!("📧 Recent Emails ({} shown)\n\n", emails.len());
            for email in &emails {
                output.push_str(&format!("  {} {}\n", email.id, email.display_short()));
            }
            output.push_str(&format!("\nUse /email read <id> to view a message"));
            return Ok(CommandResult::Success(output));
        }
        
        if lower == "unread" {
            let emails = block_on(client.get_unread(10))?;
            
            if emails.is_empty() {
                return Ok(CommandResult::Success("📧 No unread emails!".to_string()));
            }

            let mut output = format!("📧 Unread Emails ({} shown)\n\n", emails.len());
            for email in &emails {
                output.push_str(&format!("  {} {}\n", email.id, email.display_short()));
            }
            return Ok(CommandResult::Success(output));
        }

        if lower == "summary" {
            let emails = block_on(client.get_unread(20))?;
            
            if emails.is_empty() {
                return Ok(CommandResult::Success("📧 No unread emails to summarize!".to_string()));
            }

            let scorer = ImportanceScorer::new(ctx.llm, ctx.graph);
            let summary = block_on(scorer.summarize_emails(&emails))?;

            return Ok(CommandResult::Success(format!(
                "📧 Email Summary ({} unread)\n\n{}", emails.len(), summary
            )));
        }

        if lower.starts_with("read ") {
            let id = args.strip_prefix("read ").or(args.strip_prefix("Read "))
                .unwrap_or("")
                .trim();
            
            if id.is_empty() {
                return Ok(CommandResult::Error("Please provide a message ID".to_string()));
            }

            let email = block_on(client.get_message(id))?;
            return Ok(CommandResult::Success(email.display()));
        }

        if lower.starts_with("from:") {
            let sender = args.strip_prefix("from:").or(args.strip_prefix("From:"))
                .unwrap_or("")
                .trim();
            
            let emails = block_on(client.get_from(sender, 10))?;
            
            if emails.is_empty() {
                return Ok(CommandResult::Success(format!("📧 No emails from '{}'", sender)));
            }

            let mut output = format!("📧 Emails from '{}' ({} found)\n\n", sender, emails.len());
            for email in &emails {
                output.push_str(&format!("  {} {}\n", email.id, email.display_short()));
            }
            return Ok(CommandResult::Success(output));
        }

        Ok(CommandResult::Error(format!("Unknown email command. Usage: {}", self.usage())))
    }
}

// ============================================================================
// Inbox Command - Unified View
// ============================================================================

pub struct InboxCommand;

impl Command for InboxCommand {
    fn name(&self) -> &str { "/inbox" }
    fn description(&self) -> &str { "View unified inbox summary across all channels" }
    fn usage(&self) -> &str { "/inbox [summary]" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/inbox")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let oauth = Arc::new(OAuthManager::new());
        
        // Load stored tokens
        block_on(oauth.load_stored_tokens());

        let is_authenticated = block_on(oauth.is_authenticated(Provider::Google));

        if !is_authenticated {
            return Ok(CommandResult::Error(
                "Not authenticated with Google. Use /auth google first.".to_string()
            ));
        }

        let calendar = GoogleCalendarClient::new(oauth.clone());
        let gmail = GmailClient::new(oauth);

        // Fetch data
        let events = block_on(calendar.get_today_events()).unwrap_or_default();
        let emails = block_on(gmail.get_unread(20)).unwrap_or_default();

        // Generate briefing
        let scorer = ImportanceScorer::new(ctx.llm, ctx.graph);
        let briefing = block_on(scorer.generate_briefing(&events, &emails))?;

        Ok(CommandResult::Success(briefing))
    }
}

// ============================================================================
// Enhanced Briefing Command
// ============================================================================

pub struct EnhancedBriefingCommand;

impl Command for EnhancedBriefingCommand {
    fn name(&self) -> &str { "/briefing" }
    fn description(&self) -> &str { "Get daily briefing with calendar, email, and tasks" }
    fn usage(&self) -> &str { "/briefing" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower == "/briefing" || lower.starts_with("/briefing ")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let oauth = Arc::new(OAuthManager::new());
        
        // Load stored tokens
        block_on(oauth.load_stored_tokens());

        let is_google_auth = block_on(oauth.is_authenticated(Provider::Google));

        // Get local data
        let local_calendar = crate::calendar::Calendar::new(ctx.graph.database());
        let local_events = local_calendar.get_today_events().unwrap_or_default();
        
        let reminders = crate::reminders::Reminders::new(ctx.graph.database());
        let tasks = reminders.get_pending_tasks().unwrap_or_default();
        let overdue = reminders.get_overdue_tasks().unwrap_or_default();

        let user_name = ctx.graph.get_user_identity_name()
            .ok()
            .flatten()
            .unwrap_or_else(|| "there".to_string());

        let now = chrono::Local::now();
        let greeting = if now.hour() < 12 {
            "Good morning"
        } else if now.hour() < 17 {
            "Good afternoon"
        } else {
            "Good evening"
        };

        let mut briefing = format!(
            "📅 {} {}! Here's your briefing for {}.\n\n",
            greeting, user_name, now.format("%A, %B %d")
        );

        // Google Calendar section (if authenticated)
        if is_google_auth {
            let calendar = GoogleCalendarClient::new(oauth.clone());
            let gmail = GmailClient::new(oauth);

            // Google Calendar
            let gcal_events = block_on(calendar.get_today_events()).unwrap_or_default();
            
            briefing.push_str("📆 GOOGLE CALENDAR\n");
            if gcal_events.is_empty() {
                briefing.push_str("   No events scheduled today.\n");
            } else {
                for event in gcal_events.iter().take(5) {
                    briefing.push_str(&format!("   • {}\n", event.display_short()));
                }
                if gcal_events.len() > 5 {
                    briefing.push_str(&format!("   ...and {} more\n", gcal_events.len() - 5));
                }
            }

            // Gmail
            let unread = block_on(gmail.get_unread(10)).unwrap_or_default();
            briefing.push_str("\n📧 GMAIL\n");
            if unread.is_empty() {
                briefing.push_str("   All caught up!\n");
            } else {
                briefing.push_str(&format!("   {} unread message(s)\n", unread.len()));
                for email in unread.iter().take(3) {
                    let from = if let Some(idx) = email.from.find('<') {
                        email.from[..idx].trim()
                    } else {
                        &email.from
                    };
                    let subject = if email.subject.len() > 35 {
                        format!("{}...", &email.subject[..32])
                    } else {
                        email.subject.clone()
                    };
                    briefing.push_str(&format!("   • {} - {}\n", from, subject));
                }
            }
            briefing.push('\n');
        } else {
            briefing.push_str("📆 CALENDAR (local)\n");
            if local_events.is_empty() {
                briefing.push_str("   No events scheduled today.\n");
            } else {
                for event in local_events.iter().take(5) {
                    briefing.push_str(&format!("   • {}\n", event.display_short()));
                }
            }
            briefing.push_str("\n💡 Tip: Use /auth google to sync with Google Calendar & Gmail\n\n");
        }

        // Tasks section
        briefing.push_str("✅ TASKS\n");
        if !overdue.is_empty() {
            briefing.push_str(&format!("   ⚠️ {} overdue!\n", overdue.len()));
            for task in overdue.iter().take(2) {
                briefing.push_str(&format!("   • {} [{:?}]\n", task.title, task.priority));
            }
        }
        
        let pending: Vec<_> = tasks.iter().filter(|t| !overdue.iter().any(|o| o.id == t.id)).collect();
        if pending.is_empty() && overdue.is_empty() {
            briefing.push_str("   All tasks complete!\n");
        } else if !pending.is_empty() {
            briefing.push_str(&format!("   {} pending task(s)\n", pending.len()));
            for task in pending.iter().take(3) {
                briefing.push_str(&format!("   • {} [{:?}]\n", task.title, task.priority));
            }
        }

        Ok(CommandResult::Success(briefing.trim().to_string()))
    }
}

// ============================================================================
// Register Phase 6 Commands
// ============================================================================

pub fn register_phase6_commands(router: &mut Router) {
    router.register(Arc::new(AuthCommand));
    router.register(Arc::new(GCalCommand));
    router.register(Arc::new(EmailCommand));
    router.register(Arc::new(InboxCommand));
    // Replace the old briefing command with enhanced version
    router.register(Arc::new(EnhancedBriefingCommand));
}
