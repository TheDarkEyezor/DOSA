//! AI-based importance scoring for emails and messages
//!
//! Uses the local LLM to score message importance based on:
//! - Sender relationship (from knowledge graph)
//! - Content urgency
//! - Action requirements
//! - Time sensitivity

use anyhow::Result;
use chrono::Timelike;

use crate::knowledge::KnowledgeGraph;
use crate::llm::OllamaClient;
use crate::integrations::google::gmail::EmailSummary;
use crate::integrations::google::calendar::GCalEvent;

/// Importance scorer using LLM
pub struct ImportanceScorer<'a> {
    llm: &'a OllamaClient,
    graph: &'a KnowledgeGraph,
}

impl<'a> ImportanceScorer<'a> {
    pub fn new(llm: &'a OllamaClient, graph: &'a KnowledgeGraph) -> Self {
        ImportanceScorer { llm, graph }
    }

    /// Score an email's importance (0.0 - 1.0)
    pub async fn score_email(&self, email: &EmailSummary) -> Result<f32> {
        let user_context = self.get_user_context()?;
        let sender_known = self.is_sender_known(&email.from)?;
        
        let prompt = format!(r#"Score the importance of this email on a scale of 0.0 to 1.0.

Scoring guidelines:
- 0.0-0.3: Low priority (newsletters, promotions, automated)
- 0.4-0.6: Medium priority (general communication, FYI)
- 0.7-0.8: High priority (needs attention, from important contact)
- 0.9-1.0: Urgent (immediate action required, critical)

Consider:
- Is the sender known to the user? {}
- Does it require action or a response?
- Is it time-sensitive?
- Is it relevant to user's work/interests?

User context:
{}

Email:
From: {}
Subject: {}
Preview: {}

Output ONLY a number between 0.0 and 1.0, nothing else."#,
            if sender_known { "YES" } else { "NO" },
            user_context,
            email.from,
            email.subject,
            email.snippet
        );

        let response = self.llm.query(&prompt).await?;
        
        // Parse the score
        let score: f32 = response.trim()
            .parse()
            .unwrap_or(0.5);
        
        Ok(score.clamp(0.0, 1.0))
    }

    /// Check if sender is in knowledge graph
    fn is_sender_known(&self, from: &str) -> Result<bool> {
        // Extract email address
        let email = if let Some(start) = from.find('<') {
            if let Some(end) = from.find('>') {
                &from[start + 1..end]
            } else {
                from
            }
        } else {
            from
        };

        // Check if we have this email in the graph
        if let Ok(Some(_)) = self.graph.find_by_property("email", email) {
            return Ok(true);
        }

        // Extract name and check
        let name = if let Some(idx) = from.find('<') {
            from[..idx].trim()
        } else {
            from.split('@').next().unwrap_or(from)
        };

        if !name.is_empty() {
            if let Ok(Some(_)) = self.graph.find_person(name) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get user context for scoring
    fn get_user_context(&self) -> Result<String> {
        let mut context = String::new();

        // Get user identity
        if let Ok(Some(name)) = self.graph.get_user_identity_name() {
            context.push_str(&format!("User: {}\n", name));
        }

        // Get recent relationships (simplified)
        context.push_str("Known contacts and organizations in their network.\n");

        Ok(context)
    }

    /// Generate summary of important emails
    pub async fn summarize_emails(&self, emails: &[EmailSummary]) -> Result<String> {
        if emails.is_empty() {
            return Ok("No emails to summarize.".to_string());
        }

        let email_list: Vec<String> = emails.iter()
            .map(|e| format!("- From: {} | Subject: {} | Preview: {}", 
                e.from, e.subject, 
                if e.snippet.len() > 100 { format!("{}...", &e.snippet[..100]) } else { e.snippet.clone() }
            ))
            .collect();

        let prompt = format!(r#"Summarize these emails concisely for a daily briefing.
Group by priority/action needed. Be brief but capture key points.

Emails:
{}

Provide a 2-3 sentence summary of what needs attention."#,
            email_list.join("\n")
        );

        self.llm.query(&prompt).await
    }

    /// Generate summary of calendar events
    pub async fn summarize_calendar(&self, events: &[GCalEvent]) -> Result<String> {
        if events.is_empty() {
            return Ok("No events today.".to_string());
        }

        let event_list: Vec<String> = events.iter()
            .map(|e| {
                let time = e.start_time()
                    .map(|t| t.format("%I:%M %p").to_string())
                    .unwrap_or_else(|| "All day".to_string());
                format!("- {} - {}", time, e.title())
            })
            .collect();

        let prompt = format!(r#"Summarize this day's schedule in 1-2 sentences.
Highlight any conflicts, busy periods, or important meetings.

Today's events:
{}

Be concise."#,
            event_list.join("\n")
        );

        self.llm.query(&prompt).await
    }

    /// Generate unified daily briefing
    pub async fn generate_briefing(
        &self,
        events: &[GCalEvent],
        emails: &[EmailSummary],
    ) -> Result<String> {
        let user_name = self.graph.get_user_identity_name()
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
            "📅 {} {}, {}!\n\n",
            greeting,
            user_name,
            now.format("%A, %B %d")
        );

        // Calendar section
        briefing.push_str("📆 TODAY'S SCHEDULE\n");
        if events.is_empty() {
            briefing.push_str("   No events scheduled.\n");
        } else {
            for event in events.iter().take(5) {
                briefing.push_str(&format!("   • {}\n", event.display_short()));
            }
            if events.len() > 5 {
                briefing.push_str(&format!("   ...and {} more\n", events.len() - 5));
            }
        }

        // Email section
        briefing.push_str("\n📧 EMAIL\n");
        let unread: Vec<_> = emails.iter().filter(|e| e.is_unread).collect();
        if unread.is_empty() {
            briefing.push_str("   All caught up! No unread emails.\n");
        } else {
            briefing.push_str(&format!("   {} unread message(s)\n", unread.len()));
            
            // Show top 3 unread
            for email in unread.iter().take(3) {
                let from = email.from_short();
                let subject = if email.subject.len() > 40 {
                    format!("{}...", &email.subject[..37])
                } else {
                    email.subject.clone()
                };
                briefing.push_str(&format!("   • {} - {}\n", from, subject));
            }
        }

        // AI summary
        if !events.is_empty() || !unread.is_empty() {
            briefing.push_str("\n💡 SUMMARY\n");
            
            // Generate AI summary
            let summary_prompt = format!(r#"In 1-2 short sentences, give a quick overview of this person's day:

Events today: {}
Unread emails: {}

Be conversational and helpful. Start with "Today you have..." or similar."#,
                if events.is_empty() { "None".to_string() } else { 
                    events.iter().map(|e| e.title()).collect::<Vec<_>>().join(", ")
                },
                if unread.is_empty() { "None".to_string() } else {
                    format!("{} unread from {}", unread.len(), 
                        unread.iter().take(3).map(|e| e.from_short()).collect::<Vec<_>>().join(", "))
                }
            );

            if let Ok(summary) = self.llm.query(&summary_prompt).await {
                briefing.push_str(&format!("   {}\n", summary.trim()));
            }
        }

        Ok(briefing)
    }
}
