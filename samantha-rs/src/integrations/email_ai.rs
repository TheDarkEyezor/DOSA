//! AI-powered email composition
//!
//! Uses LLM to compose emails based on natural language instructions,
//! and resolves recipient names to contacts from the knowledge graph.

use anyhow::{anyhow, Result};
use std::sync::Arc;

use crate::llm::OllamaClient;
use crate::knowledge::KnowledgeGraph;
use crate::integrations::google::gmail::{GmailClient, EmailDraft, Email, EmailThread};
use crate::integrations::google::calendar::{GoogleCalendarClient, GCalEvent};
use crate::integrations::oauth::{OAuthManager, Provider};

/// AI-powered email assistant
pub struct EmailAI<'a> {
    llm: &'a OllamaClient,
    graph: &'a KnowledgeGraph,
    oauth: Arc<OAuthManager>,
}

impl<'a> EmailAI<'a> {
    pub fn new(llm: &'a OllamaClient, graph: &'a KnowledgeGraph, oauth: Arc<OAuthManager>) -> Self {
        EmailAI { llm, graph, oauth }
    }

    /// Compose an email from natural language
    pub async fn compose_email(&self, instruction: &str) -> Result<EmailComposeResult> {
        // Step 1: Parse the instruction to extract recipients, subject, and compose body
        let parsed = self.parse_email_request(instruction).await?;
        
        // Step 2: Resolve recipients to email addresses
        let resolved = self.resolve_recipients(&parsed.recipients);
        
        let resolved_emails: Vec<String> = resolved.iter()
            .filter_map(|r| r.email.clone())
            .collect();
        
        let unresolved: Vec<String> = resolved.iter()
            .filter(|r| r.email.is_none())
            .map(|r| r.name.clone())
            .collect();

        if resolved_emails.is_empty() && !parsed.recipients.is_empty() {
            return Err(anyhow!("Could not find email addresses for: {}", parsed.recipients.join(", ")));
        }

        // Step 3: Build the draft
        let draft = EmailDraft::new(resolved_emails, &parsed.subject, &parsed.body);

        Ok(EmailComposeResult {
            draft,
            parsed,
            resolved_recipients: resolved,
            unresolved_names: unresolved,
        })
    }

    /// Parse a natural language email request
    async fn parse_email_request(&self, instruction: &str) -> Result<ParsedEmailRequest> {
        let prompt = format!(r#"Parse this email request and generate JSON.

User request: "{}"

Return JSON with:
- recipients: Array of recipient names
- subject: Subject line (max 50 chars)
- body: Email body with \n for newlines. Keep SHORT (2-3 sentences)
- tone: One of: formal, casual, friendly, urgent

Example:
{{"recipients":["Alice"],"subject":"Quick hello","body":"Hi Alice,\n\nHope you're well.\n\nBest","tone":"casual"}}"#,
            instruction
        );

        // Use JSON mode for guaranteed valid JSON
        let response = self.llm.query_json(&prompt).await?;
        let json_str = extract_json(&response)?;
        
        // Try to fix common JSON issues (still useful for edge cases)
        let fixed_json = fix_truncated_json(&json_str);
        
        let parsed: ParsedEmailRequest = serde_json::from_str(&fixed_json)
            .map_err(|e| anyhow!("Failed to parse email request: {}. Response: {}", e, fixed_json))?;

        Ok(parsed)
    }

    /// Compose a reply to an email
    pub async fn compose_reply(&self, original: &Email, instruction: &str) -> Result<EmailComposeResult> {
        let prompt = format!(r#"You are helping compose a reply to an email.

Original email:
From: {}
Subject: {}
Body:
{}

User instruction for reply: "{}"

Generate a JSON response with:
- body: The reply email body (professional, well-formatted, appropriate greeting and sign-off)
- tone: The tone (formal, casual, friendly)

The reply should be complete and ready to send. Reference the original email appropriately.

Respond with ONLY valid JSON:
{{"body": "...", "tone": "formal"}}"#,
            original.from,
            original.subject,
            original.text_content(),
            instruction
        );

        let response = self.llm.query(&prompt).await?;
        let json_str = extract_json(&response)?;
        
        let reply: ParsedReply = serde_json::from_str(&json_str)
            .map_err(|e| anyhow!("Failed to parse reply: {}. Response: {}", e, json_str))?;

        let draft = EmailDraft::reply(original, &reply.body);
        let subject = draft.subject.clone();

        Ok(EmailComposeResult {
            draft,
            parsed: ParsedEmailRequest {
                recipients: vec![original.from.clone()],
                subject,
                body: reply.body,
                tone: reply.tone,
            },
            resolved_recipients: vec![ResolvedRecipient {
                name: original.from.clone(),
                email: Some(original.from.clone()),
                from_contacts: false,
            }],
            unresolved_names: vec![],
        })
    }

    /// Compose a reply from natural language input
    /// 
    /// Parses the input to find who to reply to, fetches recent emails,
    /// matches the sender, and composes a reply.
    pub async fn compose_reply_from_input(&self, input: &str) -> Result<EmailComposeResult> {
        // Step 1: Parse the reply request
        let prompt = format!(r#"Extract the reply target and message from this request.

Request: "{}"

Return JSON with:
- sender_name: The name of the person to reply to (or null if not specified)
- reply_content: What the user wants to say in the reply
- original_subject: The subject being replied to (or null if not specified)

Examples:
- "reply to john's email saying I'll be there" -> {{"sender_name": "john", "reply_content": "I'll be there", "original_subject": null}}
- "respond to the meeting invite from Sarah" -> {{"sender_name": "sarah", "reply_content": "accept the meeting invite", "original_subject": null}}
- "get back to that email about the project" -> {{"sender_name": null, "reply_content": "about the project", "original_subject": "project"}}

Respond with ONLY valid JSON:"#,
            input
        );

        let response = self.llm.query(&prompt).await?;
        let json_str = extract_json(&response)?;
        
        let parsed: ParsedReplyRequest = serde_json::from_str(&json_str)
            .map_err(|e| anyhow!("Failed to parse reply request: {}. Response: {}", e, json_str))?;

        // Step 2: Fetch recent email summaries
        let client = GmailClient::new(self.oauth.clone());
        let summaries = client.list_messages(None, 20).await?;
        
        if summaries.is_empty() {
            return Err(anyhow!("No recent emails found to reply to"));
        }

        // Step 3: Find matching email summary
        let matched_summary = self.find_matching_summary(&summaries, &parsed)?;

        // Step 4: Fetch the full email
        let email = client.get_message(&matched_summary.id).await?;

        // Step 5: Compose the reply
        self.compose_reply(&email, &parsed.reply_content).await
    }

    /// Find the email summary that best matches the reply request
    fn find_matching_summary(&self, summaries: &[crate::integrations::google::gmail::EmailSummary], request: &ParsedReplyRequest) -> Result<crate::integrations::google::gmail::EmailSummary> {
        // First try to match by sender name
        if let Some(ref sender_name) = request.sender_name {
            let sender_lower = sender_name.to_lowercase();
            for summary in summaries {
                let from_lower = summary.from.to_lowercase();
                if from_lower.contains(&sender_lower) || sender_lower.contains(&from_lower.split('@').next().unwrap_or("").split('<').next().unwrap_or("").trim()) {
                    return Ok(summary.clone());
                }
            }
        }

        // Try to match by subject
        if let Some(ref subject) = request.original_subject {
            let subject_lower = subject.to_lowercase();
            for summary in summaries {
                if summary.subject.to_lowercase().contains(&subject_lower) {
                    return Ok(summary.clone());
                }
            }
        }

        // Fall back to most recent email
        summaries.first()
            .cloned()
            .ok_or_else(|| anyhow!("No emails found to reply to"))
    }

    /// Resolve recipient names to email addresses
    fn resolve_recipients(&self, names: &[String]) -> Vec<ResolvedRecipient> {
        names.iter().map(|name| {
            // Check if it's already an email address
            if name.contains('@') {
                return ResolvedRecipient {
                    name: name.clone(),
                    email: Some(name.clone()),
                    from_contacts: false,
                };
            }

            // Try to find person in knowledge graph
            if let Ok(Some(entity)) = self.graph.find_person(name) {
                if let Ok(Some(email)) = self.graph.database().get_entity_property(entity.id, "email") {
                    return ResolvedRecipient {
                        name: name.clone(),
                        email: Some(email),
                        from_contacts: true,
                    };
                }
            }
            
            // Try find_any_entity
            if let Ok(Some(entity)) = self.graph.find_any_entity(name) {
                if let Ok(Some(email)) = self.graph.database().get_entity_property(entity.id, "email") {
                    return ResolvedRecipient {
                        name: name.clone(),
                        email: Some(email),
                        from_contacts: true,
                    };
                }
            }
            
            ResolvedRecipient {
                name: name.clone(),
                email: None,
                from_contacts: false,
            }
        }).collect()
    }

    /// Send the composed email
    pub async fn send_email(&self, draft: &EmailDraft) -> Result<String> {
        let client = GmailClient::new(self.oauth.clone());
        let _message_id = client.send_message(draft).await?;
        
        let mut result = "✓ Email sent successfully!\n".to_string();
        result.push_str(&format!("   To: {}\n", draft.to.join(", ")));
        result.push_str(&format!("   Subject: {}", draft.subject));
        
        Ok(result)
    }

    /// Compose an email to attendees of a calendar event (multi-hop)
    pub async fn compose_for_attendees(&self, instruction: &str) -> Result<EmailComposeResult> {
        // Step 1: Parse the instruction to find the event reference and message intent
        let parsed_ref = self.parse_event_reference(instruction).await?;
        
        // Step 2: Find the matching event
        let client = GoogleCalendarClient::new(self.oauth.clone());
        let event = self.find_matching_event(&client, &parsed_ref.event_search).await?;
        
        // Step 3: Extract attendee emails from the event
        let event_title = event.summary.as_deref().unwrap_or("(untitled)");
        let attendees = event.attendees.as_ref()
            .ok_or_else(|| anyhow!("Event '{}' has no attendees", event_title))?;
        
        let attendee_emails: Vec<String> = attendees.iter()
            .filter_map(|a| {
                // Skip self/organizer if desired
                if a.is_self.unwrap_or(false) {
                    None
                } else {
                    a.email.clone()
                }
            })
            .collect();
        
        if attendee_emails.is_empty() {
            return Err(anyhow!("No attendees found for event '{}'", event_title));
        }
        
        // Step 4: Compose the email content based on the instruction
        let event_time = event.start.date_time.as_ref()
            .or(event.start.date.as_ref())
            .map(|s| s.as_str())
            .unwrap_or("TBD");
        
        let email_prompt = format!(
            "Compose an email to attendees about: {}. Event: '{}' on {}",
            parsed_ref.message_intent,
            event_title,
            event_time
        );
        
        let parsed = self.parse_email_request(&email_prompt).await?;
        
        // Step 5: Build the draft with attendee emails
        let draft = EmailDraft::new(attendee_emails.clone(), &parsed.subject, &parsed.body);
        
        // Build resolved recipients list
        let resolved: Vec<ResolvedRecipient> = attendee_emails.iter().map(|email| {
            ResolvedRecipient {
                name: email.clone(),
                email: Some(email.clone()),
                from_contacts: false,
            }
        }).collect();
        
        Ok(EmailComposeResult {
            draft,
            parsed,
            resolved_recipients: resolved,
            unresolved_names: vec![],
        })
    }

    /// Parse event reference from instruction
    async fn parse_event_reference(&self, instruction: &str) -> Result<ParsedEventReference> {
        let prompt = format!(r#"Parse this request to email event attendees.

Request: "{}"

Extract:
- event_search: Keywords to find the event (e.g., "dinner tomorrow", "meeting friday")
- message_intent: What the email should say (e.g., "time changed to 7pm", "meeting cancelled")

Example: {{"event_search":"dinner tomorrow","message_intent":"time changed to 7pm"}}"#,
            instruction
        );

        // Use JSON mode for guaranteed valid JSON
        let response = self.llm.query_json(&prompt).await?;
        let json_str = extract_json(&response)?;
        let fixed = fix_truncated_json(&json_str);
        
        serde_json::from_str(&fixed)
            .map_err(|e| anyhow!("Failed to parse event reference: {}. Response: {}", e, fixed))
    }

    /// Find a matching event from calendar
    async fn find_matching_event(&self, client: &GoogleCalendarClient, search: &str) -> Result<GCalEvent> {
        let lower = search.to_lowercase();
        
        // Determine date range based on search terms
        let events = if lower.contains("today") {
            client.get_today_events().await?
        } else if lower.contains("tomorrow") {
            client.get_tomorrow_events().await?
        } else {
            client.get_upcoming_events(20).await?
        };
        
        if events.is_empty() {
            return Err(anyhow!("No upcoming events found"));
        }
        
        // Try to match by event title/summary
        let search_words: Vec<&str> = lower.split_whitespace()
            .filter(|w| !["the", "a", "an", "tomorrow", "today", "meeting", "event", "attendees", "of"].contains(w))
            .collect();
        
        for event in &events {
            if let Some(ref summary) = event.summary {
                let event_lower = summary.to_lowercase();
                // Check if any search word matches the event title
                if search_words.iter().any(|w| event_lower.contains(w)) {
                    return Ok(event.clone());
                }
            }
        }
        
        // If no match, try using LLM to pick the best match
        let event_list: Vec<String> = events.iter().enumerate()
            .map(|(i, e)| format!("{}: {}", i, e.summary.as_deref().unwrap_or("(untitled)")))
            .collect();
        
        let prompt = format!(
            "Which event best matches '{}'? Events:\n{}\n\nRespond with just the number.",
            search,
            event_list.join("\n")
        );
        
        let response = self.llm.query(&prompt).await?;
        if let Ok(idx) = response.trim().parse::<usize>() {
            if idx < events.len() {
                return Ok(events[idx].clone());
            }
        }
        
        // Fall back to first event
        Ok(events[0].clone())
    }
    
    /// Get conversation history with a person and optionally summarize it
    pub async fn get_conversation(&self, person: &str, summarize: bool) -> Result<ConversationResult> {
        let client = GmailClient::new(self.oauth.clone());
        
        // Get email threads with this person
        let threads = client.get_conversation_with(person, 5).await?;
        
        if threads.is_empty() {
            return Err(anyhow!("No emails found with {}", person));
        }
        
        let total_messages: usize = threads.iter().map(|t| t.messages.len()).sum();
        
        let summary = if summarize {
            Some(self.summarize_threads(&threads).await?)
        } else {
            None
        };
        
        Ok(ConversationResult {
            person: person.to_string(),
            threads,
            total_messages,
            summary,
        })
    }
    
    /// Summarize email threads using LLM
    async fn summarize_threads(&self, threads: &[crate::integrations::google::gmail::EmailThread]) -> Result<String> {
        // Build conversation text for summarization
        let mut conversation_text = String::new();
        
        for thread in threads {
            conversation_text.push_str(&format!("\n=== Thread: {} ===\n", thread.subject));
            conversation_text.push_str(&thread.as_conversation_text());
        }
        
        // Truncate if too long
        if conversation_text.len() > 8000 {
            conversation_text.truncate(8000);
            conversation_text.push_str("\n... (truncated)");
        }
        
        let prompt = format!(
            r#"Summarize this email conversation. Extract:
1. Key topics discussed
2. Important decisions or agreements
3. Action items or follow-ups mentioned
4. Overall tone/status of the conversation

Conversation:
{}

Provide a concise summary in 3-5 bullet points."#,
            conversation_text
        );
        
        self.llm.query(&prompt).await
    }
}

/// Parsed email request from natural language
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParsedEmailRequest {
    pub recipients: Vec<String>,
    pub subject: String,
    pub body: String,
    pub tone: String,
}

/// Parsed event reference for attendee emails
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ParsedEventReference {
    event_search: String,
    message_intent: String,
}

/// Result of conversation retrieval/summarization
#[derive(Debug)]
pub struct ConversationResult {
    pub person: String,
    pub threads: Vec<crate::integrations::google::gmail::EmailThread>,
    pub total_messages: usize,
    pub summary: Option<String>,
}

impl ConversationResult {
    /// Display the conversation summary
    pub fn display(&self) -> String {
        let mut output = format!(
            "📧 Conversation with {}\n   {} thread(s), {} total message(s)\n\n",
            self.person,
            self.threads.len(),
            self.total_messages
        );
        
        if let Some(ref summary) = self.summary {
            output.push_str("📋 Summary:\n");
            output.push_str(summary);
            output.push_str("\n\n");
        }
        
        output.push_str("📂 Threads:\n");
        for thread in &self.threads {
            output.push_str(&format!(
                "   • {} ({} messages)\n",
                thread.subject,
                thread.messages.len()
            ));
        }
        
        output
    }
    
    /// Display full conversation with all messages
    pub fn display_full(&self) -> String {
        let mut output = format!(
            "📧 Full Conversation with {}\n   {} thread(s), {} total message(s)\n\n",
            self.person,
            self.threads.len(),
            self.total_messages
        );
        
        for thread in &self.threads {
            output.push_str(&thread.display());
            output.push_str("\n");
        }
        
        output
    }
}

/// Parsed reply
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ParsedReply {
    body: String,
    tone: String,
}

/// Parsed reply request from natural language
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ParsedReplyRequest {
    sender_name: Option<String>,
    reply_content: String,
    original_subject: Option<String>,
}

/// Resolved recipient
#[derive(Debug, Clone)]
pub struct ResolvedRecipient {
    pub name: String,
    pub email: Option<String>,
    pub from_contacts: bool,
}

impl ResolvedRecipient {
    pub fn display(&self) -> String {
        if let Some(ref email) = self.email {
            if self.from_contacts {
                format!("✓ {} <{}>", self.name, email)
            } else {
                format!("  {}", email)
            }
        } else {
            format!("? {} (no email found)", self.name)
        }
    }
}

/// Result of email composition
#[derive(Debug)]
pub struct EmailComposeResult {
    pub draft: EmailDraft,
    pub parsed: ParsedEmailRequest,
    pub resolved_recipients: Vec<ResolvedRecipient>,
    pub unresolved_names: Vec<String>,
}

impl EmailComposeResult {
    pub fn display_preview(&self) -> String {
        let mut output = self.draft.preview();
        
        if !self.resolved_recipients.is_empty() {
            output.push_str("\n👥 Recipients:\n");
            for recipient in &self.resolved_recipients {
                output.push_str(&format!("   {}\n", recipient.display()));
            }
        }

        if !self.unresolved_names.is_empty() {
            output.push_str("\n⚠️  Could not find email for:\n");
            for name in &self.unresolved_names {
                output.push_str(&format!("   - {} (add with /contact {} email <email>)\n", name, name));
            }
        }

        output
    }

    pub fn has_unresolved(&self) -> bool {
        !self.unresolved_names.is_empty()
    }

    pub fn can_send(&self) -> bool {
        !self.draft.to.is_empty()
    }
}

/// Extract JSON from LLM response
fn extract_json(response: &str) -> Result<String> {
    let trimmed = response.trim();
    
    // If wrapped in code blocks, extract
    if trimmed.starts_with("```") {
        let lines: Vec<&str> = trimmed.lines().collect();
        let start = if lines.first().map(|l| l.starts_with("```")).unwrap_or(false) { 1 } else { 0 };
        let end = if lines.last().map(|l| l.trim() == "```").unwrap_or(false) { lines.len() - 1 } else { lines.len() };
        return Ok(lines[start..end].join("\n"));
    }
    
    // Try to find JSON object
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            return Ok(trimmed[start..=end].to_string());
        }
        // No closing brace - return from start, we'll try to fix it
        return Ok(trimmed[start..].to_string());
    }
    
    Ok(trimmed.to_string())
}

/// Try to fix truncated or malformed JSON
fn fix_truncated_json(json: &str) -> String {
    let mut result = json.trim().to_string();
    
    // Handle common truncation patterns:
    // 1. Trailing incomplete key: "tone:" -> remove or complete
    // 2. Trailing incomplete string value
    // 3. Missing closing brackets/braces
    
    // Check for incomplete key at end (e.g., "tone:" with no value)
    // Pattern: ,"key":} or ,"key":] or ,"key": at end
    let trailing_key_pattern = regex::Regex::new(r#",\s*"[^"]+"\s*:\s*$"#).ok();
    if let Some(re) = trailing_key_pattern {
        if re.is_match(&result) {
            // Remove the trailing incomplete key-value pair
            result = re.replace(&result, "").to_string();
        }
    }
    
    // Also handle case where value is incomplete: "key":"incomplete
    let incomplete_string_pattern = regex::Regex::new(r#",\s*"[^"]+"\s*:\s*"[^"]*$"#).ok();
    if let Some(re) = incomplete_string_pattern {
        if re.is_match(&result) {
            // Close the string value
            result.push('"');
        }
    }
    
    // Count braces and brackets
    let open_braces = result.matches('{').count();
    let close_braces = result.matches('}').count();
    let open_brackets = result.matches('[').count();
    let close_brackets = result.matches(']').count();
    
    // Check if we're in the middle of a string (odd number of unescaped quotes)
    let quote_count = result.matches('"').count() - result.matches("\\\"").count();
    if quote_count % 2 == 1 {
        // Close the string
        result.push('"');
    }
    
    // Add missing brackets
    for _ in close_brackets..open_brackets {
        result.push(']');
    }
    
    // Add missing braces
    for _ in close_braces..open_braces {
        result.push('}');
    }
    
    result
}
