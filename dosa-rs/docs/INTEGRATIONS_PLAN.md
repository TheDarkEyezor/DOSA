# DOSA External Integrations Plan

## Overview

This document outlines the integration architecture for connecting DOSA to:
1. **Google Calendar** - Event synchronization, creation, and invitations
2. **Gmail** - Email retrieval, importance-based summaries, and sending
3. **WhatsApp** - Message retrieval, summaries, and sending (via WhatsApp Business API)

All integrations share a common pattern:
- **OAuth2 authentication** for Google services
- **Importance-based AI summaries** using local LLM
- **Two-way sync** where applicable
- **Knowledge graph integration** (linking contacts, events, messages to entities)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                              DOSA Core                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ KnowledgeGraph│  │   Calendar   │  │      Contacts            │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
│         │                 │                      │                   │
│         └────────────────┼──────────────────────┘                   │
│                          │                                           │
│  ┌───────────────────────▼───────────────────────────────────────┐  │
│  │                    Integration Layer                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │  │
│  │  │ SyncManager │  │ OAuthManager│  │ ImportanceSummarizer│   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                          │                                           │
└──────────────────────────┼───────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│Google Calendar│  │    Gmail     │  │ WhatsApp Business│
│     API       │  │     API      │  │       API        │
└──────────────┘  └──────────────┘  └──────────────────┘
```

---

## Module Structure

```
src/integrations/
├── mod.rs                      # Module exports
├── oauth.rs                    # OAuth2 token management
├── sync_manager.rs             # Unified sync orchestrator
├── importance.rs               # AI-based importance scoring
├── google/
│   ├── mod.rs
│   ├── calendar.rs             # Google Calendar API client
│   ├── gmail.rs                # Gmail API client
│   └── auth.rs                 # Google OAuth specifics
└── whatsapp/
    ├── mod.rs
    ├── client.rs               # WhatsApp Business API client
    └── webhook.rs              # Webhook receiver for real-time messages
```

---

## Phase 6A: OAuth2 Foundation

### Dependencies (Cargo.toml)
```toml
[dependencies]
oauth2 = "4.4"                  # OAuth2 client
jsonwebtoken = "9"              # JWT for service accounts
keyring = "2"                   # Secure credential storage (macOS Keychain)
tokio = { version = "1", features = ["full"] }  # Async runtime (already have)
reqwest = { version = "0.11", features = ["json"] }  # HTTP client (already have)
```

### OAuth Manager (`src/integrations/oauth.rs`)
```rust
pub struct OAuthManager {
    google_client: Option<GoogleOAuthClient>,
    token_store: SecureTokenStore,  // Uses macOS Keychain
}

impl OAuthManager {
    /// Start OAuth flow for a provider
    pub async fn authenticate(&mut self, provider: Provider) -> Result<()>;
    
    /// Get valid access token (auto-refresh if expired)
    pub async fn get_token(&self, provider: Provider) -> Result<String>;
    
    /// Revoke access
    pub async fn revoke(&mut self, provider: Provider) -> Result<()>;
    
    /// Check if authenticated
    pub fn is_authenticated(&self, provider: Provider) -> bool;
}

pub enum Provider {
    Google,
    WhatsAppBusiness,
}
```

### Token Storage
- Tokens stored in **macOS Keychain** via `keyring` crate
- Refresh tokens automatically used when access tokens expire
- Service: `dosa-assistant`, Account: `google-oauth` / `whatsapp`

### New Commands
| Command | Description |
|---------|-------------|
| `/auth google` | Start Google OAuth flow (opens browser) |
| `/auth whatsapp` | Configure WhatsApp Business API token |
| `/auth status` | Show authentication status for all services |
| `/auth revoke <service>` | Revoke access for a service |

---

## Phase 6B: Google Calendar Integration

### Google Calendar Client (`src/integrations/google/calendar.rs`)
```rust
pub struct GoogleCalendarClient {
    oauth: Arc<OAuthManager>,
    http: reqwest::Client,
}

impl GoogleCalendarClient {
    // === Read Operations ===
    
    /// Fetch events for a time range
    pub async fn list_events(&self, start: DateTime, end: DateTime) -> Result<Vec<GCalEvent>>;
    
    /// Get a single event by ID
    pub async fn get_event(&self, event_id: &str) -> Result<GCalEvent>;
    
    /// List all calendars
    pub async fn list_calendars(&self) -> Result<Vec<GCalCalendar>>;
    
    // === Write Operations ===
    
    /// Create a new event
    pub async fn create_event(&self, event: CreateEventRequest) -> Result<GCalEvent>;
    
    /// Update an existing event
    pub async fn update_event(&self, event_id: &str, update: UpdateEventRequest) -> Result<GCalEvent>;
    
    /// Delete an event
    pub async fn delete_event(&self, event_id: &str) -> Result<()>;
    
    /// Add attendees to an event (sends invites)
    pub async fn invite_attendees(&self, event_id: &str, emails: Vec<String>) -> Result<()>;
    
    /// RSVP to an event
    pub async fn respond_to_invite(&self, event_id: &str, response: RsvpStatus) -> Result<()>;
    
    // === Sync ===
    
    /// Sync events to local calendar (with conflict resolution)
    pub async fn sync_to_local(&self, local_calendar: &Calendar) -> Result<SyncResult>;
    
    /// Push local events to Google Calendar
    pub async fn sync_from_local(&self, local_calendar: &Calendar) -> Result<SyncResult>;
}

#[derive(Debug)]
pub struct CreateEventRequest {
    pub summary: String,
    pub description: Option<String>,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub location: Option<String>,
    pub attendees: Vec<String>,          // Email addresses
    pub reminders: Vec<ReminderOverride>,
    pub recurrence: Option<RecurrenceRule>,
}

pub enum RsvpStatus { Accepted, Declined, Tentative }
```

### Two-Way Sync Strategy
1. **Local → Google**: Events created in DOSA get `google_event_id` after push
2. **Google → Local**: Events from Google get synced with `external_id` field
3. **Conflict Resolution**: Last-modified wins, or prompt user for manual conflicts
4. **Sync Trigger**: On startup, and via `/sync calendar` command

### Database Schema Update
```sql
ALTER TABLE events ADD COLUMN google_event_id TEXT;
ALTER TABLE events ADD COLUMN sync_status TEXT DEFAULT 'local';  -- 'local', 'synced', 'conflict'
ALTER TABLE events ADD COLUMN last_synced_at DATETIME;
```

### New Commands
| Command | Description |
|---------|-------------|
| `/sync calendar` | Sync with Google Calendar |
| `/gcal list [today\|week\|month]` | List Google Calendar events |
| `/gcal create <title> at <time> with <attendees>` | Create event and invite |
| `/invite <event> <person>` | Invite contact to event |

---

## Phase 6C: Gmail Integration

### Gmail Client (`src/integrations/google/gmail.rs`)
```rust
pub struct GmailClient {
    oauth: Arc<OAuthManager>,
    http: reqwest::Client,
}

impl GmailClient {
    // === Read Operations ===
    
    /// Fetch emails with filtering
    pub async fn list_messages(&self, query: EmailQuery) -> Result<Vec<EmailSummary>>;
    
    /// Get full email content
    pub async fn get_message(&self, message_id: &str) -> Result<Email>;
    
    /// Get unread count
    pub async fn unread_count(&self) -> Result<u32>;
    
    /// Get labels (folders)
    pub async fn list_labels(&self) -> Result<Vec<Label>>;
    
    // === Write Operations ===
    
    /// Send an email
    pub async fn send(&self, email: ComposeEmail) -> Result<SentEmail>;
    
    /// Reply to an email
    pub async fn reply(&self, message_id: &str, body: &str) -> Result<SentEmail>;
    
    /// Forward an email
    pub async fn forward(&self, message_id: &str, to: Vec<String>) -> Result<SentEmail>;
    
    /// Mark as read/unread
    pub async fn mark_read(&self, message_id: &str, read: bool) -> Result<()>;
    
    /// Archive email
    pub async fn archive(&self, message_id: &str) -> Result<()>;
    
    /// Apply label
    pub async fn apply_label(&self, message_id: &str, label: &str) -> Result<()>;
    
    // === Draft Operations ===
    
    /// Create draft
    pub async fn create_draft(&self, email: ComposeEmail) -> Result<Draft>;
    
    /// Send draft
    pub async fn send_draft(&self, draft_id: &str) -> Result<SentEmail>;
}

#[derive(Debug)]
pub struct EmailQuery {
    pub from: Option<String>,
    pub to: Option<String>,
    pub subject: Option<String>,
    pub after: Option<DateTime<Utc>>,
    pub before: Option<DateTime<Utc>>,
    pub is_unread: Option<bool>,
    pub label: Option<String>,
    pub limit: usize,
}

#[derive(Debug)]
pub struct ComposeEmail {
    pub to: Vec<String>,
    pub cc: Option<Vec<String>>,
    pub bcc: Option<Vec<String>>,
    pub subject: String,
    pub body: String,
    pub attachments: Vec<Attachment>,
}
```

### Knowledge Graph Integration
- **Auto-link emails to contacts**: Match sender/recipient to Person entities
- **Extract entities from emails**: Use document ingestion on email bodies
- **Track communication history**: Create `EMAILED` relationships

```sql
CREATE TABLE email_messages (
    id INTEGER PRIMARY KEY,
    gmail_id TEXT UNIQUE NOT NULL,
    thread_id TEXT,
    from_email TEXT,
    from_entity_id INTEGER,  -- Link to knowledge graph
    subject TEXT,
    snippet TEXT,
    received_at DATETIME,
    is_read BOOLEAN,
    importance_score REAL,  -- AI-computed 0.0-1.0
    summary TEXT,           -- AI-generated summary
    FOREIGN KEY (from_entity_id) REFERENCES entities(id)
);
```

### New Commands
| Command | Description |
|---------|-------------|
| `/email list [unread\|important\|from:<name>]` | List emails |
| `/email read <id>` | Read full email |
| `/email summary` | AI summary of unread/important emails |
| `/email compose <to> <subject>` | Start composing email |
| `/email reply <id> <message>` | Reply to email |
| `/email send <to> <subject> <body>` | Quick send |

---

## Phase 6D: WhatsApp Integration

### WhatsApp Business API Client (`src/integrations/whatsapp/client.rs`)

> **Note**: WhatsApp integration requires a WhatsApp Business API account (Meta Business).
> For personal use, consider alternative approaches like WhatsApp Web automation.

```rust
pub struct WhatsAppClient {
    api_token: String,          // From Meta Business
    phone_number_id: String,    // Your business phone number
    http: reqwest::Client,
}

impl WhatsAppClient {
    // === Read Operations (via Webhooks) ===
    
    /// Get recent messages (from local cache, filled by webhooks)
    pub fn get_messages(&self, contact: &str, limit: usize) -> Result<Vec<WhatsAppMessage>>;
    
    /// Get unread message count
    pub fn unread_count(&self) -> Result<u32>;
    
    // === Write Operations ===
    
    /// Send a text message
    pub async fn send_text(&self, to: &str, message: &str) -> Result<MessageId>;
    
    /// Send a template message (required for initiating conversations)
    pub async fn send_template(&self, to: &str, template: &str, params: Vec<String>) -> Result<MessageId>;
    
    /// Send media (image, document, etc.)
    pub async fn send_media(&self, to: &str, media: MediaMessage) -> Result<MessageId>;
    
    /// Mark message as read
    pub async fn mark_read(&self, message_id: &str) -> Result<()>;
}

#[derive(Debug)]
pub struct WhatsAppMessage {
    pub id: String,
    pub from: String,           // Phone number
    pub from_entity_id: Option<i64>,  // Linked contact
    pub text: Option<String>,
    pub media_type: Option<MediaType>,
    pub timestamp: DateTime<Utc>,
    pub is_read: bool,
}
```

### Webhook Receiver (`src/integrations/whatsapp/webhook.rs`)
```rust
/// Run a small HTTP server to receive WhatsApp webhooks
pub async fn start_webhook_server(
    port: u16,
    message_handler: impl Fn(WhatsAppMessage) -> Result<()>,
) -> Result<()>;
```

### Database Schema
```sql
CREATE TABLE whatsapp_messages (
    id INTEGER PRIMARY KEY,
    wa_message_id TEXT UNIQUE NOT NULL,
    from_phone TEXT NOT NULL,
    from_entity_id INTEGER,  -- Link to knowledge graph contact
    text TEXT,
    media_url TEXT,
    received_at DATETIME,
    is_read BOOLEAN DEFAULT FALSE,
    importance_score REAL,
    summary TEXT,
    FOREIGN KEY (from_entity_id) REFERENCES entities(id)
);
```

### New Commands
| Command | Description |
|---------|-------------|
| `/wa list [unread\|from:<name>]` | List WhatsApp messages |
| `/wa read <contact>` | Read conversation with contact |
| `/wa summary` | AI summary of unread messages |
| `/wa send <contact> <message>` | Send message to contact |
| `/wa reply <message>` | Reply to last message from contact |

---

## Phase 6E: Importance-Based AI Summaries

### Importance Scorer (`src/integrations/importance.rs`)
```rust
pub struct ImportanceScorer<'a> {
    llm: &'a OllamaClient,
    graph: &'a KnowledgeGraph,
}

impl<'a> ImportanceScorer<'a> {
    /// Score message/email importance (0.0 - 1.0)
    pub async fn score(&self, content: &MessageContent) -> Result<f32>;
    
    /// Generate summary of multiple items, prioritized by importance
    pub async fn summarize_batch(&self, items: Vec<&MessageContent>) -> Result<String>;
    
    /// Get personalized importance based on user's knowledge graph
    pub async fn personalized_score(&self, content: &MessageContent) -> Result<f32>;
}

impl ImportanceScorer {
    fn build_scoring_prompt(&self, content: &MessageContent, user_context: &str) -> String {
        format!(r#"
Score the importance of this message for the user on a scale of 0.0 to 1.0.
Consider:
- Is it from someone in their contacts/knowledge graph?
- Does it require action or response?
- Is it time-sensitive?
- Does it relate to their work, projects, or interests?

User context:
{}

Message:
From: {}
Subject/Preview: {}
Content: {}

Output only a number between 0.0 and 1.0.
"#, user_context, content.sender, content.subject_or_preview, content.body)
    }
}
```

### Scoring Factors
| Factor | Weight | Description |
|--------|--------|-------------|
| Known Contact | +0.2 | Sender is in knowledge graph |
| VIP Contact | +0.3 | Sender has "important" or "family" relationship |
| Action Required | +0.2 | Contains questions or requests |
| Time Sensitive | +0.2 | Contains dates, deadlines, urgency words |
| Project Related | +0.1 | Mentions known projects |
| Unread Time | +0.1 | Older unread messages get higher priority |

### New Commands
| Command | Description |
|---------|-------------|
| `/inbox` | Combined important summary across all channels |
| `/inbox summary` | AI-generated summary of all important items |
| `/inbox [email\|whatsapp\|calendar]` | Filter by channel |

---

## Phase 6F: Unified Inbox & Daily Briefing

### Enhanced Daily Briefing
```rust
pub struct UnifiedInbox<'a> {
    gmail: Option<&'a GmailClient>,
    whatsapp: Option<&'a WhatsAppClient>,
    calendar: &'a Calendar<'a>,
    scorer: ImportanceScorer<'a>,
}

impl<'a> UnifiedInbox<'a> {
    /// Get daily briefing with all channels
    pub async fn daily_briefing(&self) -> Result<Briefing>;
    
    /// Get important items needing attention
    pub async fn get_attention_items(&self) -> Result<Vec<AttentionItem>>;
    
    /// Generate natural language summary
    pub async fn generate_summary(&self) -> Result<String>;
}

#[derive(Debug)]
pub struct Briefing {
    pub date: NaiveDate,
    pub calendar_events: Vec<Event>,
    pub important_emails: Vec<EmailSummary>,
    pub important_messages: Vec<WhatsAppMessage>,
    pub pending_tasks: Vec<Task>,
    pub summary: String,  // AI-generated
}
```

### Enhanced `/briefing` Command
```
📅 Good morning, Adi! Here's your briefing for Monday, December 1st:

📆 TODAY'S SCHEDULE (3 events)
• 10:00 AM - Team standup (Google Meet)
• 2:00 PM - 1:1 with Sarah
• 6:00 PM - Gym with Mike

📧 IMPORTANT EMAILS (5 unread, 2 need action)
• [URGENT] From: HR Team - "Benefits enrollment deadline tomorrow"
• From: John Smith - "Re: Project proposal" (awaiting your reply)

💬 WHATSAPP (12 unread, 3 important)
• Mom (3 messages) - "Call me when you're free"
• VaNI Team (8 messages) - Discussion about deployment
• Sarah - "Running 10 min late"

✅ TASKS DUE TODAY (2)
• Complete quarterly report (HIGH)
• Review PR #123 (MEDIUM)

Would you like me to summarize any of these in detail?
```

---

## Implementation Order

### Phase 6A: OAuth Foundation (Week 1)
1. Add OAuth2 dependencies
2. Implement `OAuthManager` with macOS Keychain storage
3. Add `/auth` commands
4. Test Google OAuth flow

### Phase 6B: Google Calendar (Week 2)
1. Implement `GoogleCalendarClient`
2. Add sync logic with conflict resolution
3. Update database schema
4. Add calendar commands
5. Test two-way sync

### Phase 6C: Gmail (Week 3)
1. Implement `GmailClient`
2. Add email storage and linking
3. Implement importance scoring
4. Add email commands
5. Test send/receive flow

### Phase 6D: WhatsApp (Week 4)
1. Set up WhatsApp Business API account
2. Implement `WhatsAppClient`
3. Set up webhook receiver
4. Add message storage
5. Add WhatsApp commands

### Phase 6E: AI Summaries (Week 5)
1. Implement `ImportanceScorer`
2. Add scoring to all message types
3. Implement batch summarization
4. Update `/briefing` command

### Phase 6F: Unified Inbox (Week 6)
1. Implement `UnifiedInbox`
2. Create combined briefing
3. Add `/inbox` commands
4. Polish user experience

---

## Security Considerations

1. **Token Storage**: Use macOS Keychain (never store tokens in plain files)
2. **Scopes**: Request minimum necessary OAuth scopes
3. **Token Rotation**: Implement automatic refresh token rotation
4. **Audit Logging**: Log all API calls for debugging
5. **Rate Limiting**: Respect API rate limits (especially Gmail)

### Required OAuth Scopes

**Google**:
- `https://www.googleapis.com/auth/calendar` (read/write calendar)
- `https://www.googleapis.com/auth/gmail.modify` (read/send email)
- `https://www.googleapis.com/auth/gmail.compose` (compose only, optional)

**WhatsApp Business**:
- `whatsapp_business_messaging`
- `whatsapp_business_management`

---

## Configuration

### Environment Variables / Config File
```toml
# ~/.config/dosa/config.toml

[google]
client_id = "xxx.apps.googleusercontent.com"
client_secret = "xxx"  # Or use GCP service account

[whatsapp]
phone_number_id = "xxx"
business_id = "xxx"
# Token stored in Keychain

[sync]
calendar_interval_minutes = 15
email_check_interval_minutes = 5

[importance]
min_score_for_briefing = 0.3
vip_contacts = ["mom", "boss", "partner"]
```

---

## Testing Strategy

1. **Mock APIs**: Create mock implementations for offline testing
2. **Sandbox Accounts**: Use Google's sandbox for calendar/gmail testing
3. **Integration Tests**: Test full flows with real APIs (gated by feature flag)
4. **Load Testing**: Ensure sync doesn't overwhelm APIs

---

## Future Enhancements

1. **Voice Integration**: "Hey Dosa, what's on my calendar today?"
2. **Slack Integration**: Similar pattern to WhatsApp
3. **Microsoft 365**: Outlook calendar and email
4. **Telegram**: Alternative to WhatsApp
5. **Apple Native**: Direct Calendar.app and Mail.app integration (macOS only)
6. **Smart Scheduling**: Suggest meeting times based on availability
7. **Email Templates**: Quick responses for common patterns
8. **Auto-categorization**: ML-based email/message labeling
