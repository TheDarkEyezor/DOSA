// Smart email reply feature
// Finds recent emails from a person and helps compose contextual replies

use anyhow::{Result, anyhow};
use std::sync::Arc;

use crate::integrations::google::gmail::{GmailClient, Email, EmailDraft, EmailSummary};
use crate::llm::OllamaClient;

/// Smart reply engine for context-aware email responses
pub struct SmartReply {
    gmail: Arc<GmailClient>,
    llm: Arc<OllamaClient>,
}

/// Result of finding emails to reply to
#[derive(Debug)]
pub struct ReplyContext {
    pub original_email: Email,
    pub thread_emails: Vec<EmailSummary>,
    pub suggested_reply: Option<String>,
}

impl SmartReply {
    pub fn new(gmail: Arc<GmailClient>, llm: Arc<OllamaClient>) -> Self {
        Self { gmail, llm }
    }

    /// Find the most recent email from a person to reply to
    pub async fn find_email_to_reply(&self, person: &str) -> Result<Option<Email>> {
        // Search for recent emails from this person
        let emails = self.gmail.get_from(person, 5).await?;
        
        if emails.is_empty() {
            return Ok(None);
        }

        // Get the most recent one
        let most_recent = &emails[0];
        let full_email = self.gmail.get_message(&most_recent.id).await?;
        
        Ok(Some(full_email))
    }

    /// Get thread context for an email
    pub async fn get_thread_context(&self, email: &Email) -> Result<Vec<EmailSummary>> {
        // Search for emails in this thread
        let query = format!("thread:{}", email.thread_id);
        self.gmail.list_messages(Some(&query), 10).await
    }

    /// Generate a suggested reply using LLM
    pub async fn generate_reply(
        &self,
        email: &Email,
        user_intent: &str,
        tone: Option<&str>,
    ) -> Result<String> {
        let tone_instruction = match tone {
            Some("formal") => "Use a formal, professional tone.",
            Some("casual") => "Use a friendly, casual tone.",
            Some("brief") => "Keep it very brief and to the point.",
            _ => "Use an appropriate professional tone.",
        };

        let prompt = format!(
            r#"Generate an email reply based on the following:

ORIGINAL EMAIL:
From: {}
Subject: {}
Body:
{}

USER'S INTENT: {}

INSTRUCTIONS:
- {}
- Be concise but complete
- Don't include "Subject:" line in the reply
- Start with an appropriate greeting
- End with an appropriate sign-off

Generate only the reply text, nothing else:"#,
            email.from,
            email.subject,
            email.text_content().chars().take(2000).collect::<String>(),
            user_intent,
            tone_instruction,
        );

        let response = self.llm.query(&prompt).await?;
        Ok(response.trim().to_string())
    }

    /// Full smart reply flow: find email, generate reply, and prepare draft
    pub async fn smart_reply(
        &self,
        person: &str,
        user_intent: &str,
        tone: Option<&str>,
    ) -> Result<SmartReplyResult> {
        // Step 1: Find recent email from person
        let email = self.find_email_to_reply(person).await?
            .ok_or_else(|| anyhow!("No recent emails found from {}", person))?;

        // Step 2: Get thread context (for display)
        let thread = self.get_thread_context(&email).await.unwrap_or_default();

        // Step 3: Generate suggested reply
        let suggested = self.generate_reply(&email, user_intent, tone).await?;

        // Step 4: Prepare draft
        let draft = EmailDraft::reply(&email, &suggested);

        Ok(SmartReplyResult {
            original_email: email,
            thread_context: thread,
            suggested_reply: suggested,
            draft,
        })
    }

    /// Quick reply to the last email from someone
    pub async fn quick_reply(
        &self,
        person: &str,
        reply_content: &str,
    ) -> Result<SmartReplyResult> {
        let email = self.find_email_to_reply(person).await?
            .ok_or_else(|| anyhow!("No recent emails found from {}", person))?;

        let draft = EmailDraft::reply(&email, reply_content);

        Ok(SmartReplyResult {
            original_email: email.clone(),
            thread_context: vec![],
            suggested_reply: reply_content.to_string(),
            draft,
        })
    }

    /// Summarize an email thread
    pub async fn summarize_thread(&self, person: &str) -> Result<String> {
        let emails = self.gmail.get_from(person, 10).await?;
        
        if emails.is_empty() {
            return Ok(format!("No emails found from {}", person));
        }

        // Get full content of recent emails
        let mut email_contents = Vec::new();
        for summary in emails.iter().take(5) {
            if let Ok(full) = self.gmail.get_message(&summary.id).await {
                email_contents.push(format!(
                    "---\nDate: {}\nSubject: {}\n\n{}\n",
                    full.date.format("%Y-%m-%d %H:%M").to_string(),
                    full.subject,
                    full.text_content().chars().take(500).collect::<String>()
                ));
            }
        }

        if email_contents.is_empty() {
            return Ok(format!("Could not retrieve email contents from {}", person));
        }

        let prompt = format!(
            r#"Summarize this email conversation:

{}

Provide a concise summary covering:
1. Main topics discussed
2. Any action items or requests
3. Current status/state of the conversation

Summary:"#,
            email_contents.join("\n")
        );

        let summary = self.llm.query(&prompt).await?;
        Ok(summary.trim().to_string())
    }
}

/// Result of smart reply operation
#[derive(Debug)]
pub struct SmartReplyResult {
    pub original_email: Email,
    pub thread_context: Vec<EmailSummary>,
    pub suggested_reply: String,
    pub draft: EmailDraft,
}

impl SmartReplyResult {
    /// Format for user preview
    pub fn display(&self) -> String {
        let mut output = String::new();
        
        output.push_str("📧 Replying to email:\n");
        output.push_str(&format!("   From: {}\n", self.original_email.from));
        output.push_str(&format!("   Subject: {}\n", self.original_email.subject));
        
        if !self.thread_context.is_empty() {
            output.push_str(&format!("   (Part of thread with {} messages)\n", 
                self.thread_context.len()));
        }
        
        output.push_str("\n📝 Suggested reply:\n");
        output.push_str("─".repeat(40).as_str());
        output.push('\n');
        output.push_str(&self.suggested_reply);
        output.push('\n');
        output.push_str("─".repeat(40).as_str());
        output.push_str("\n\nSend this reply? (yes/edit/cancel)");
        
        output
    }
}

/// Parse reply intent from user input
pub fn parse_reply_intent(input: &str) -> Option<ReplyIntent> {
    let lower = input.to_lowercase();
    
    // Pattern: "reply to [person]'s email saying [content]"
    if lower.contains("reply") {
        // Extract person name - use original input to preserve case
        let person = extract_person_from_reply(input)?;
        
        // Extract what to say
        let content = if lower.contains("saying") {
            extract_after_keyword(&lower, "saying")
        } else if lower.contains("that") {
            extract_after_keyword(&lower, "that")
        } else {
            None
        };

        return Some(ReplyIntent {
            person,
            content,
            tone: parse_tone(&lower),
        });
    }

    None
}

/// Extract person name from reply intent (case-insensitive matching, preserves original case)
fn extract_person_from_reply(input: &str) -> Option<String> {
    // Use case-insensitive regex patterns
    let re1 = regex::Regex::new(r"(?i)reply to (\w+)'s").ok()?;
    if let Some(caps) = re1.captures(input) {
        return caps.get(1).map(|m| m.as_str().to_string());
    }
    
    let re2 = regex::Regex::new(r"(?i)reply to (?:email|message) from (\w+)").ok()?;
    if let Some(caps) = re2.captures(input) {
        return caps.get(1).map(|m| m.as_str().to_string());
    }
    
    let re3 = regex::Regex::new(r"(?i)reply to (\w+)").ok()?;
    if let Some(caps) = re3.captures(input) {
        let name = caps.get(1).map(|m| m.as_str())?;
        // Filter out common words (case-insensitive)
        let name_lower = name.to_lowercase();
        if !["the", "that", "this", "my", "his", "her", "email"].contains(&name_lower.as_str()) {
            return Some(name.to_string());
        }
    }
    
    None
}

/// Extract content after a keyword
fn extract_after_keyword(input: &str, keyword: &str) -> Option<String> {
    if let Some(pos) = input.find(keyword) {
        let after = &input[pos + keyword.len()..];
        let content = after.trim();
        if !content.is_empty() {
            return Some(content.to_string());
        }
    }
    None
}

/// Parse tone from input
fn parse_tone(input: &str) -> Option<String> {
    if input.contains("formal") {
        Some("formal".to_string())
    } else if input.contains("casual") || input.contains("friendly") {
        Some("casual".to_string())
    } else if input.contains("brief") || input.contains("short") {
        Some("brief".to_string())
    } else {
        None
    }
}

/// Parsed reply intent
#[derive(Debug)]
pub struct ReplyIntent {
    pub person: String,
    pub content: Option<String>,
    pub tone: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_reply_intent() {
        let intent = parse_reply_intent("reply to John's email saying I'll be there at 3pm");
        assert!(intent.is_some());
        let intent = intent.unwrap();
        assert_eq!(intent.person, "John");
        assert!(intent.content.is_some());
        assert!(intent.content.unwrap().contains("3pm"));
    }

    #[test]
    fn test_extract_person() {
        assert_eq!(
            extract_person_from_reply("reply to sarah's last email"),
            Some("sarah".to_string())
        );
        
        assert_eq!(
            extract_person_from_reply("reply to email from Bob"),
            Some("Bob".to_string())
        );
    }
}
