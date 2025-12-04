//! Gmail API client (read-only)
//!
//! Provides access to Gmail messages for display and summarization.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::integrations::oauth::{OAuthManager, Provider};

const GMAIL_API_BASE: &str = "https://gmail.googleapis.com/gmail/v1";

/// Email message summary (from list)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailSummary {
    pub id: String,
    pub thread_id: String,
    pub from: String,
    pub to: Option<String>,
    pub subject: String,
    pub snippet: String,
    pub date: DateTime<Utc>,
    pub is_unread: bool,
    pub labels: Vec<String>,
}

impl EmailSummary {
    /// Format for display
    pub fn display(&self) -> String {
        let unread_marker = if self.is_unread { "📬" } else { "📭" };
        let mut output = format!(
            "{} {} - {}\n   From: {}\n   {}",
            unread_marker,
            self.date.format("%b %d, %I:%M %p"),
            self.subject,
            self.from,
            self.snippet
        );

        if self.snippet.len() > 100 {
            output.truncate(output.len() - (self.snippet.len() - 100));
            output.push_str("...");
        }

        output
    }

    /// Short display for lists
    pub fn display_short(&self) -> String {
        let unread = if self.is_unread { "●" } else { "○" };
        let subject = if self.subject.len() > 50 {
            format!("{}...", &self.subject[..47])
        } else {
            self.subject.clone()
        };
        format!("{} {} - {}", unread, self.from_short(), subject)
    }

    /// Get short sender name
    pub fn from_short(&self) -> String {
        // Extract name from "Name <email>" format
        if let Some(idx) = self.from.find('<') {
            self.from[..idx].trim().to_string()
        } else if let Some(idx) = self.from.find('@') {
            self.from[..idx].to_string()
        } else {
            self.from.clone()
        }
    }
}

/// Full email message
#[derive(Debug, Clone)]
pub struct Email {
    pub id: String,
    pub thread_id: String,
    pub from: String,
    pub to: Vec<String>,
    pub cc: Vec<String>,
    pub subject: String,
    pub body_text: Option<String>,
    pub body_html: Option<String>,
    pub date: DateTime<Utc>,
    pub is_unread: bool,
    pub labels: Vec<String>,
}

impl Email {
    /// Get plain text body, falling back to stripped HTML
    pub fn text_content(&self) -> String {
        if let Some(ref text) = self.body_text {
            text.clone()
        } else if let Some(ref html) = self.body_html {
            // Simple HTML stripping (for basic display)
            html.replace("<br>", "\n")
                .replace("<br/>", "\n")
                .replace("<br />", "\n")
                .replace("</p>", "\n")
                .replace("</div>", "\n")
                .split('<')
                .map(|s| s.split('>').last().unwrap_or(""))
                .collect::<Vec<_>>()
                .join("")
        } else {
            String::new()
        }
    }

    /// Format for full display
    pub fn display(&self) -> String {
        let mut output = format!(
            "📧 {}\n\nFrom: {}\nTo: {}\nDate: {}\n",
            self.subject,
            self.from,
            self.to.join(", "),
            self.date.format("%a, %b %d, %Y at %I:%M %p"),
        );

        if !self.cc.is_empty() {
            output.push_str(&format!("Cc: {}\n", self.cc.join(", ")));
        }

        output.push_str(&format!("\n{}", self.text_content()));

        output
    }
}

/// API response structures
#[derive(Debug, Deserialize)]
struct MessagesListResponse {
    messages: Option<Vec<MessageRef>>,
    #[serde(rename = "nextPageToken")]
    next_page_token: Option<String>,
    #[serde(rename = "resultSizeEstimate")]
    result_size_estimate: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct MessageRef {
    id: String,
    #[serde(rename = "threadId")]
    thread_id: String,
}

#[derive(Debug, Deserialize)]
struct MessageResponse {
    id: String,
    #[serde(rename = "threadId")]
    thread_id: String,
    #[serde(rename = "labelIds")]
    label_ids: Option<Vec<String>>,
    snippet: Option<String>,
    payload: Option<MessagePayload>,
    #[serde(rename = "internalDate")]
    internal_date: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessagePayload {
    headers: Option<Vec<Header>>,
    #[serde(rename = "mimeType")]
    mime_type: Option<String>,
    body: Option<MessageBody>,
    parts: Option<Vec<MessagePart>>,
}

#[derive(Debug, Deserialize)]
struct Header {
    name: String,
    value: String,
}

#[derive(Debug, Deserialize)]
struct MessageBody {
    size: Option<i32>,
    data: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessagePart {
    #[serde(rename = "mimeType")]
    mime_type: Option<String>,
    body: Option<MessageBody>,
    parts: Option<Vec<MessagePart>>,
}

/// Thread response from Gmail API
#[derive(Debug, Deserialize)]
struct ThreadResponse {
    id: String,
    messages: Option<Vec<MessageResponse>>,
}

/// Email thread (conversation) with multiple messages
#[derive(Debug, Clone)]
pub struct EmailThread {
    pub id: String,
    pub subject: String,
    pub participants: Vec<String>,
    pub messages: Vec<Email>,
}

impl EmailThread {
    /// Format thread for display
    pub fn display(&self) -> String {
        let mut output = format!(
            "📧 Thread: {}\n   Participants: {}\n   Messages: {}\n\n",
            self.subject,
            self.participants.join(", "),
            self.messages.len()
        );
        
        for (i, msg) in self.messages.iter().enumerate() {
            output.push_str(&format!(
                "--- Message {} ({}) ---\n",
                i + 1,
                msg.date.format("%b %d, %I:%M %p")
            ));
            output.push_str(&format!("From: {}\n", msg.from));
            if !msg.to.is_empty() {
                output.push_str(&format!("To: {}\n", msg.to.join(", ")));
            }
            if let Some(ref text) = msg.body_text {
                let preview = if text.len() > 500 {
                    format!("{}...", &text[..500])
                } else {
                    text.clone()
                };
                output.push_str(&format!("\n{}\n\n", preview));
            }
        }
        
        output
    }
    
    /// Get conversation as text for summarization
    pub fn as_conversation_text(&self) -> String {
        let mut text = format!("Subject: {}\n\n", self.subject);
        
        for msg in &self.messages {
            text.push_str(&format!(
                "[{} - {}]\n",
                msg.from,
                msg.date.format("%b %d %Y, %I:%M %p")
            ));
            if let Some(ref body) = msg.body_text {
                text.push_str(body);
                text.push_str("\n\n");
            }
        }
        
        text
    }
}

/// Gmail client
pub struct GmailClient {
    oauth: Arc<OAuthManager>,
    http: reqwest::Client,
}

impl GmailClient {
    pub fn new(oauth: Arc<OAuthManager>) -> Self {
        GmailClient {
            oauth,
            http: reqwest::Client::new(),
        }
    }

    /// List messages with optional query
    pub async fn list_messages(
        &self,
        query: Option<&str>,
        max_results: usize,
    ) -> Result<Vec<EmailSummary>> {
        let token = self.oauth.get_token(Provider::Google).await?;

        let mut url = format!(
            "{}/users/me/messages?maxResults={}",
            GMAIL_API_BASE,
            max_results,
        );

        if let Some(q) = query {
            url.push_str(&format!("&q={}", urlencoding::encode(q)));
        }

        let resp = self.http
            .get(&url)
            .bearer_auth(&token)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Gmail API error {}: {}", status, body));
        }

        let data: MessagesListResponse = resp.json().await?;
        let message_refs = data.messages.unwrap_or_default();

        // Fetch details for each message (in parallel would be better, but keeping simple)
        let mut emails = Vec::new();
        for msg_ref in message_refs.iter().take(max_results) {
            if let Ok(summary) = self.get_message_summary(&msg_ref.id).await {
                emails.push(summary);
            }
        }

        Ok(emails)
    }

    /// Get message summary (headers + snippet)
    async fn get_message_summary(&self, message_id: &str) -> Result<EmailSummary> {
        let token = self.oauth.get_token(Provider::Google).await?;

        let url = format!(
            "{}/users/me/messages/{}?format=metadata&metadataHeaders=From&metadataHeaders=To&metadataHeaders=Subject&metadataHeaders=Date",
            GMAIL_API_BASE,
            message_id,
        );

        let resp = self.http
            .get(&url)
            .bearer_auth(&token)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Gmail API error {}: {}", status, body));
        }

        let msg: MessageResponse = resp.json().await?;
        self.parse_email_summary(msg)
    }

    /// Parse API response into EmailSummary
    fn parse_email_summary(&self, msg: MessageResponse) -> Result<EmailSummary> {
        let headers = msg.payload
            .as_ref()
            .and_then(|p| p.headers.as_ref())
            .map(|h| h.as_slice())
            .unwrap_or(&[]);

        let get_header = |name: &str| -> String {
            headers.iter()
                .find(|h| h.name.eq_ignore_ascii_case(name))
                .map(|h| h.value.clone())
                .unwrap_or_default()
        };

        let labels = msg.label_ids.unwrap_or_default();
        let is_unread = labels.contains(&"UNREAD".to_string());

        // Parse date
        let date = msg.internal_date
            .as_ref()
            .and_then(|d| d.parse::<i64>().ok())
            .and_then(|ts| DateTime::from_timestamp(ts / 1000, 0))
            .unwrap_or_else(Utc::now);

        Ok(EmailSummary {
            id: msg.id,
            thread_id: msg.thread_id,
            from: get_header("From"),
            to: Some(get_header("To")),
            subject: get_header("Subject"),
            snippet: msg.snippet.unwrap_or_default(),
            date,
            is_unread,
            labels,
        })
    }

    /// Get full message content
    pub async fn get_message(&self, message_id: &str) -> Result<Email> {
        let token = self.oauth.get_token(Provider::Google).await?;

        let url = format!(
            "{}/users/me/messages/{}?format=full",
            GMAIL_API_BASE,
            message_id,
        );

        let resp = self.http
            .get(&url)
            .bearer_auth(&token)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Gmail API error {}: {}", status, body));
        }

        let msg: MessageResponse = resp.json().await?;
        self.parse_full_email(msg)
    }

    /// Parse API response into full Email
    fn parse_full_email(&self, msg: MessageResponse) -> Result<Email> {
        let headers = msg.payload
            .as_ref()
            .and_then(|p| p.headers.as_ref())
            .map(|h| h.as_slice())
            .unwrap_or(&[]);

        let get_header = |name: &str| -> String {
            headers.iter()
                .find(|h| h.name.eq_ignore_ascii_case(name))
                .map(|h| h.value.clone())
                .unwrap_or_default()
        };

        let labels = msg.label_ids.clone().unwrap_or_default();
        let is_unread = labels.contains(&"UNREAD".to_string());

        // Parse date
        let date = msg.internal_date
            .as_ref()
            .and_then(|d| d.parse::<i64>().ok())
            .and_then(|ts| DateTime::from_timestamp(ts / 1000, 0))
            .unwrap_or_else(Utc::now);

        // Parse To and Cc
        let to = get_header("To")
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        
        let cc = get_header("Cc")
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        // Extract body
        let (body_text, body_html) = self.extract_body(&msg.payload);

        Ok(Email {
            id: msg.id,
            thread_id: msg.thread_id,
            from: get_header("From"),
            to,
            cc,
            subject: get_header("Subject"),
            body_text,
            body_html,
            date,
            is_unread,
            labels,
        })
    }

    /// Extract body from message payload
    fn extract_body(&self, payload: &Option<MessagePayload>) -> (Option<String>, Option<String>) {
        let payload = match payload {
            Some(p) => p,
            None => return (None, None),
        };

        let mut text = None;
        let mut html = None;

        // Check direct body
        if let Some(ref body) = payload.body {
            if let Some(ref data) = body.data {
                let decoded = self.decode_base64(data);
                let mime = payload.mime_type.as_deref().unwrap_or("");
                if mime.contains("text/plain") {
                    text = Some(decoded);
                } else if mime.contains("text/html") {
                    html = Some(decoded);
                }
            }
        }

        // Check parts
        if let Some(ref parts) = payload.parts {
            self.extract_body_from_parts(parts, &mut text, &mut html);
        }

        (text, html)
    }

    fn extract_body_from_parts(
        &self,
        parts: &[MessagePart],
        text: &mut Option<String>,
        html: &mut Option<String>,
    ) {
        for part in parts {
            let mime = part.mime_type.as_deref().unwrap_or("");

            if let Some(ref body) = part.body {
                if let Some(ref data) = body.data {
                    let decoded = self.decode_base64(data);
                    if mime.contains("text/plain") && text.is_none() {
                        *text = Some(decoded);
                    } else if mime.contains("text/html") && html.is_none() {
                        *html = Some(decoded);
                    }
                }
            }

            // Recurse into nested parts
            if let Some(ref nested) = part.parts {
                self.extract_body_from_parts(nested, text, html);
            }
        }
    }

    /// Decode base64url encoded string
    fn decode_base64(&self, data: &str) -> String {
        use base64::Engine;
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        
        engine.decode(data)
            .ok()
            .and_then(|bytes| String::from_utf8(bytes).ok())
            .unwrap_or_default()
    }

    /// Get unread count
    pub async fn unread_count(&self) -> Result<u32> {
        let messages = self.list_messages(Some("is:unread"), 100).await?;
        Ok(messages.len() as u32)
    }

    /// Get unread messages
    pub async fn get_unread(&self, max: usize) -> Result<Vec<EmailSummary>> {
        self.list_messages(Some("is:unread"), max).await
    }

    /// Get important unread messages
    pub async fn get_important_unread(&self, max: usize) -> Result<Vec<EmailSummary>> {
        self.list_messages(Some("is:unread is:important"), max).await
    }

    /// Get messages from a specific sender
    pub async fn get_from(&self, sender: &str, max: usize) -> Result<Vec<EmailSummary>> {
        let query = format!("from:{}", sender);
        self.list_messages(Some(&query), max).await
    }

    /// Get all messages in a conversation thread
    pub async fn get_thread(&self, thread_id: &str) -> Result<Vec<Email>> {
        let token = self.oauth.get_token(Provider::Google).await?;
        
        let url = format!("{}/users/me/threads/{}?format=full", GMAIL_API_BASE, thread_id);
        
        let resp = self.http
            .get(&url)
            .bearer_auth(&token)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Gmail thread error {}: {}", status, body));
        }

        let thread: ThreadResponse = resp.json().await?;
        
        let mut emails = Vec::new();
        for msg in thread.messages.unwrap_or_default() {
            if let Ok(email) = self.parse_message_to_email(msg).await {
                emails.push(email);
            }
        }
        
        // Sort by date ascending (oldest first)
        emails.sort_by(|a, b| a.date.cmp(&b.date));
        
        Ok(emails)
    }
    
    /// Get conversation with a specific person (groups by thread)
    pub async fn get_conversation_with(&self, person: &str, max_threads: usize) -> Result<Vec<EmailThread>> {
        // Search for emails involving this person
        let query = format!("from:{} OR to:{}", person, person);
        let messages = self.list_messages(Some(&query), max_threads * 5).await?;
        
        // Group by thread_id
        let mut thread_ids: Vec<String> = messages.iter()
            .map(|m| m.thread_id.clone())
            .collect();
        thread_ids.dedup();
        thread_ids.truncate(max_threads);
        
        let mut threads = Vec::new();
        for thread_id in thread_ids {
            if let Ok(emails) = self.get_thread(&thread_id).await {
                if !emails.is_empty() {
                    threads.push(EmailThread {
                        id: thread_id,
                        subject: emails.first().map(|e| e.subject.clone()).unwrap_or_default(),
                        participants: Self::extract_participants(&emails),
                        messages: emails,
                    });
                }
            }
        }
        
        Ok(threads)
    }
    
    /// Extract unique participants from a list of emails
    fn extract_participants(emails: &[Email]) -> Vec<String> {
        let mut participants: Vec<String> = emails.iter()
            .flat_map(|e| {
                let mut p = vec![e.from.clone()];
                p.extend(e.to.iter().cloned());
                p.extend(e.cc.iter().cloned());
                p
            })
            .collect();
        participants.sort();
        participants.dedup();
        participants
    }
    
    /// Parse a raw message into an Email struct (internal helper)
    async fn parse_message_to_email(&self, msg: MessageResponse) -> Result<Email> {
        // Extract headers using closure
        let get_header = |name: &str| -> String {
            msg.payload.as_ref()
                .and_then(|p| p.headers.as_ref())
                .and_then(|headers| {
                    headers.iter()
                        .find(|h| h.name.eq_ignore_ascii_case(name))
                        .map(|h| h.value.clone())
                })
                .unwrap_or_default()
        };

        let labels = msg.label_ids.clone().unwrap_or_default();
        let is_unread = labels.contains(&"UNREAD".to_string());

        // Parse date from internal_date (milliseconds since epoch)
        let date = msg.internal_date
            .as_ref()
            .and_then(|d| d.parse::<i64>().ok())
            .and_then(|ts| DateTime::from_timestamp(ts / 1000, 0))
            .unwrap_or_else(Utc::now);

        // Parse To and Cc
        let to: Vec<String> = get_header("To")
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        
        let cc: Vec<String> = get_header("Cc")
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        // Extract body
        let (body_text, body_html) = self.extract_body(&msg.payload);

        Ok(Email {
            id: msg.id,
            thread_id: msg.thread_id,
            from: get_header("From"),
            to,
            cc,
            subject: get_header("Subject"),
            body_text,
            body_html,
            date,
            is_unread,
            labels,
        })
    }

    /// Get summary text for LLM context
    pub async fn get_summary(&self) -> Result<String> {
        let unread_count = self.unread_count().await.unwrap_or(0);
        let unread = self.get_unread(10).await.unwrap_or_default();

        let mut summary = format!("Gmail: {} unread message(s)\n", unread_count);

        if !unread.is_empty() {
            summary.push_str("Recent unread:\n");
            for email in unread.iter().take(5) {
                summary.push_str(&format!("- From {}: {}\n", 
                    email.from_short(), 
                    if email.subject.len() > 50 { 
                        format!("{}...", &email.subject[..47]) 
                    } else { 
                        email.subject.clone() 
                    }
                ));
            }
        }

        Ok(summary)
    }

    /// Send an email
    pub async fn send_message(&self, draft: &EmailDraft) -> Result<String> {
        let token = self.oauth.get_token(Provider::Google).await?;

        // Build RFC 2822 message
        let raw_message = draft.to_rfc2822();
        
        // Base64url encode the message
        use base64::Engine;
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let encoded = engine.encode(raw_message.as_bytes());

        let url = format!("{}/users/me/messages/send", GMAIL_API_BASE);

        let body = serde_json::json!({
            "raw": encoded
        });

        let resp = self.http
            .post(&url)
            .bearer_auth(&token)
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Gmail send error {}: {}", status, body));
        }

        let result: serde_json::Value = resp.json().await?;
        let message_id = result.get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        Ok(message_id.to_string())
    }

    /// Reply to an email
    pub async fn send_reply(&self, original: &Email, reply_body: &str) -> Result<String> {
        let draft = EmailDraft::reply(original, reply_body);
        self.send_message(&draft).await
    }
}

// ============================================================================
// Email Draft for Composing
// ============================================================================

/// Email draft for sending
#[derive(Debug, Clone)]
pub struct EmailDraft {
    pub to: Vec<String>,
    pub cc: Vec<String>,
    pub bcc: Vec<String>,
    pub subject: String,
    pub body: String,
    pub reply_to_message_id: Option<String>,
    pub thread_id: Option<String>,
}

impl EmailDraft {
    pub fn new(to: Vec<String>, subject: &str, body: &str) -> Self {
        EmailDraft {
            to,
            cc: Vec::new(),
            bcc: Vec::new(),
            subject: subject.to_string(),
            body: body.to_string(),
            reply_to_message_id: None,
            thread_id: None,
        }
    }

    pub fn with_cc(mut self, cc: Vec<String>) -> Self {
        self.cc = cc;
        self
    }

    pub fn with_bcc(mut self, bcc: Vec<String>) -> Self {
        self.bcc = bcc;
        self
    }

    /// Create a reply to an existing email
    pub fn reply(original: &Email, body: &str) -> Self {
        let subject = if original.subject.starts_with("Re:") {
            original.subject.clone()
        } else {
            format!("Re: {}", original.subject)
        };

        EmailDraft {
            to: vec![original.from.clone()],
            cc: Vec::new(),
            bcc: Vec::new(),
            subject,
            body: body.to_string(),
            reply_to_message_id: Some(original.id.clone()),
            thread_id: Some(original.thread_id.clone()),
        }
    }

    /// Convert to RFC 2822 format for Gmail API
    fn to_rfc2822(&self) -> String {
        let mut message = String::new();

        // Headers
        message.push_str(&format!("To: {}\r\n", self.to.join(", ")));
        
        if !self.cc.is_empty() {
            message.push_str(&format!("Cc: {}\r\n", self.cc.join(", ")));
        }
        
        if !self.bcc.is_empty() {
            message.push_str(&format!("Bcc: {}\r\n", self.bcc.join(", ")));
        }

        message.push_str(&format!("Subject: {}\r\n", self.subject));
        message.push_str("Content-Type: text/plain; charset=utf-8\r\n");
        
        // Threading headers for replies
        if let Some(ref msg_id) = self.reply_to_message_id {
            message.push_str(&format!("In-Reply-To: <{}>\r\n", msg_id));
            message.push_str(&format!("References: <{}>\r\n", msg_id));
        }

        // Empty line before body
        message.push_str("\r\n");
        
        // Body
        message.push_str(&self.body);

        message
    }

    /// Preview the email before sending
    pub fn preview(&self) -> String {
        let mut output = String::new();
        output.push_str("📧 Email Preview\n\n");
        output.push_str(&format!("To: {}\n", self.to.join(", ")));
        
        if !self.cc.is_empty() {
            output.push_str(&format!("Cc: {}\n", self.cc.join(", ")));
        }
        
        output.push_str(&format!("Subject: {}\n", self.subject));
        output.push_str(&format!("\n{}\n", "-".repeat(50)));
        output.push_str(&self.body);
        output.push_str(&format!("\n{}\n", "-".repeat(50)));
        
        output
    }
}
