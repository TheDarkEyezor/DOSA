// Multi-hop reasoning engine
// Enables complex queries that chain multiple operations like:
// "email everyone I'm meeting with this week"
// "reschedule all meetings with John to next week"

use anyhow::{Result, anyhow};
use chrono::{DateTime, Local, Duration, Utc};
use std::collections::HashSet;
use std::sync::Arc;

use crate::integrations::google::calendar::{GoogleCalendarClient, GCalEvent};
use crate::integrations::google::gmail::{GmailClient, EmailDraft};
use crate::intelligence::{ConversationContext, PersonReference, EventReference};
use crate::knowledge::KnowledgeGraph;

/// Multi-hop query processor
pub struct MultiHopEngine {
    calendar: Option<Arc<GoogleCalendarClient>>,
    gmail: Option<Arc<GmailClient>>,
    graph: Arc<KnowledgeGraph>,
}

/// Result of a multi-hop operation
#[derive(Debug)]
pub struct MultiHopResult {
    pub success: bool,
    pub message: String,
    pub intermediate_steps: Vec<String>,
    pub final_data: Option<MultiHopData>,
}

/// Data produced by multi-hop operations
#[derive(Debug, Clone)]
pub enum MultiHopData {
    People(Vec<PersonInfo>),
    Events(Vec<EventInfo>),
    EmailDraft(EmailDraft),
    BatchEmails(Vec<EmailDraft>),
}

#[derive(Debug, Clone)]
pub struct PersonInfo {
    pub name: String,
    pub email: Option<String>,
    pub source: String,  // How we found them (calendar, email, etc.)
}

#[derive(Debug, Clone)]
pub struct EventInfo {
    pub id: String,
    pub title: String,
    pub start: Option<DateTime<Local>>,
    pub attendees: Vec<String>,
}

impl MultiHopEngine {
    pub fn new(
        calendar: Option<Arc<GoogleCalendarClient>>,
        gmail: Option<Arc<GmailClient>>,
        graph: Arc<KnowledgeGraph>,
    ) -> Self {
        Self { calendar, gmail, graph }
    }

    /// Process a multi-hop query
    pub async fn process(&self, query: &str, context: &mut ConversationContext) -> Result<MultiHopResult> {
        let query_lower = query.to_lowercase();

        // Pattern: "email everyone who knows [person] about [topic]"
        if query_lower.contains("email") && query_lower.contains("who knows") {
            return self.email_people_who_know(query, context).await;
        }

        // Pattern: "email everyone I'm meeting with [time period]"
        if (query_lower.contains("email") && query_lower.contains("meeting")) 
            || (query_lower.contains("email") && query_lower.contains("attendees")) {
            return self.email_meeting_attendees(&query_lower, context).await;
        }

        // Pattern: "who am I meeting with [time period]"
        if query_lower.contains("who") && query_lower.contains("meeting") {
            return self.find_meeting_attendees(&query_lower, context).await;
        }

        // Pattern: "reschedule all meetings with [person]"
        if query_lower.contains("reschedule") && query_lower.contains("all") {
            return self.reschedule_person_meetings(&query_lower, context).await;
        }

        // Pattern: "cancel meetings [time period]"
        if query_lower.contains("cancel") && query_lower.contains("meeting") {
            return self.cancel_meetings(&query_lower, context).await;
        }

        // Pattern: "what's [person]'s schedule"
        if query_lower.contains("schedule") && !query_lower.contains("my") {
            return self.get_person_schedule(&query_lower, context).await;
        }

        Err(anyhow!("Could not identify a multi-hop pattern in the query"))
    }

    /// Email all attendees from meetings in a time period
    async fn email_meeting_attendees(
        &self,
        query: &str,
        context: &mut ConversationContext,
    ) -> Result<MultiHopResult> {
        let calendar = self.calendar.as_ref()
            .ok_or_else(|| anyhow!("Calendar not connected"))?;

        let mut steps = Vec::new();

        // Step 1: Determine time period
        let (start, end, period_desc) = self.parse_time_period(query)?;
        steps.push(format!("📅 Looking at calendar for {}", period_desc));

        // Step 2: Get events in that period
        let events = calendar.list_events(
            start.with_timezone(&Utc),
            end.with_timezone(&Utc),
            None,
        ).await?;

        if events.is_empty() {
            return Ok(MultiHopResult {
                success: false,
                message: format!("No meetings found for {}", period_desc),
                intermediate_steps: steps,
                final_data: None,
            });
        }

        steps.push(format!("📋 Found {} events", events.len()));

        // Step 3: Extract unique attendees
        let mut attendees: HashSet<String> = HashSet::new();
        let mut attendee_names: Vec<PersonInfo> = Vec::new();

        for event in &events {
            if let Some(ref atts) = event.attendees {
                for att in atts {
                    if att.is_self != Some(true) {
                        if let Some(ref email) = att.email {
                            if !attendees.contains(email) {
                                attendees.insert(email.clone());
                                attendee_names.push(PersonInfo {
                                    name: att.display_name.clone().unwrap_or_else(|| email.clone()),
                                    email: Some(email.clone()),
                                    source: event.title().to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }

        if attendees.is_empty() {
            return Ok(MultiHopResult {
                success: false,
                message: "No attendees found in the meetings (only solo events)".to_string(),
                intermediate_steps: steps,
                final_data: None,
            });
        }

        steps.push(format!("👥 Found {} unique attendees", attendees.len()));

        // Step 4: Extract subject from query or generate one
        let subject = self.extract_email_subject(query)
            .unwrap_or_else(|| format!("Regarding our meetings {}", period_desc));

        // Store in context for follow-up
        let people: Vec<PersonReference> = attendee_names.iter().map(|p| PersonReference {
            name: p.name.clone(),
            email: p.email.clone(),
            entity_id: None,
        }).collect();
        context.set_people(people);

        // Step 5: Create email draft (or drafts)
        let recipients: Vec<String> = attendees.into_iter().collect();
        
        let message = format!(
            "Found {} attendees from {} meetings {}:\n\n{}\n\n\
            What would you like to say to them? I can compose a batch email.",
            attendee_names.len(),
            events.len(),
            period_desc,
            attendee_names.iter()
                .map(|p| format!("  • {} ({})", p.name, p.source))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(MultiHopResult {
            success: true,
            message,
            intermediate_steps: steps,
            final_data: Some(MultiHopData::People(attendee_names)),
        })
    }

    /// Find who the user is meeting with
    async fn find_meeting_attendees(
        &self,
        query: &str,
        context: &mut ConversationContext,
    ) -> Result<MultiHopResult> {
        let calendar = self.calendar.as_ref()
            .ok_or_else(|| anyhow!("Calendar not connected"))?;

        let mut steps = Vec::new();

        // Parse time period
        let (start, end, period_desc) = self.parse_time_period(query)?;
        steps.push(format!("📅 Checking calendar for {}", period_desc));

        // Get events
        let events = calendar.list_events(
            start.with_timezone(&Utc),
            end.with_timezone(&Utc),
            None,
        ).await?;

        if events.is_empty() {
            return Ok(MultiHopResult {
                success: true,
                message: format!("📅 No meetings scheduled for {}", period_desc),
                intermediate_steps: steps,
                final_data: None,
            });
        }

        steps.push(format!("📋 Found {} events", events.len()));

        // Group by person
        let mut people_events: std::collections::HashMap<String, Vec<&GCalEvent>> = 
            std::collections::HashMap::new();

        for event in &events {
            if let Some(ref atts) = event.attendees {
                for att in atts {
                    if att.is_self != Some(true) {
                        let name = att.display_name.clone()
                            .or_else(|| att.email.clone())
                            .unwrap_or_else(|| "Unknown".to_string());
                        people_events.entry(name).or_default().push(event);
                    }
                }
            }
        }

        if people_events.is_empty() {
            return Ok(MultiHopResult {
                success: true,
                message: format!("📅 {} meetings {} but all are solo events (no other attendees)", 
                    events.len(), period_desc),
                intermediate_steps: steps,
                final_data: Some(MultiHopData::Events(
                    events.iter().map(|e| EventInfo {
                        id: e.id.clone(),
                        title: e.title().to_string(),
                        start: e.start_time(),
                        attendees: vec![],
                    }).collect()
                )),
            });
        }

        let mut message = format!("👥 You're meeting with {} people {}:\n\n", 
            people_events.len(), period_desc);

        for (person, evts) in &people_events {
            let meeting_count = evts.len();
            let first_meeting = evts.first().map(|e| {
                e.start_time()
                    .map(|t| t.format("%a %I:%M %p").to_string())
                    .unwrap_or_default()
            }).unwrap_or_default();
            
            message.push_str(&format!("  • {} - {} meeting{} (first: {})\n",
                person,
                meeting_count,
                if meeting_count > 1 { "s" } else { "" },
                first_meeting,
            ));
        }

        Ok(MultiHopResult {
            success: true,
            message,
            intermediate_steps: steps,
            final_data: Some(MultiHopData::People(
                people_events.keys().map(|name| PersonInfo {
                    name: name.clone(),
                    email: None,
                    source: "Calendar".to_string(),
                }).collect()
            )),
        })
    }

    /// Reschedule all meetings with a specific person
    async fn reschedule_person_meetings(
        &self,
        query: &str,
        _context: &mut ConversationContext,
    ) -> Result<MultiHopResult> {
        let calendar = self.calendar.as_ref()
            .ok_or_else(|| anyhow!("Calendar not connected"))?;

        let mut steps = Vec::new();

        // Extract person name from query
        let person_name = self.extract_person_name(query)
            .ok_or_else(|| anyhow!("Couldn't identify which person's meetings to reschedule"))?;

        steps.push(format!("🔍 Looking for meetings with {}", person_name));

        // Get upcoming events
        let events = calendar.get_week_events().await?;

        // Filter to meetings with this person
        let matching: Vec<_> = events.iter()
            .filter(|e| {
                if let Some(ref atts) = e.attendees {
                    atts.iter().any(|a| {
                        a.display_name.as_ref()
                            .map(|n| n.to_lowercase().contains(&person_name.to_lowercase()))
                            .unwrap_or(false)
                        || a.email.as_ref()
                            .map(|e| e.to_lowercase().contains(&person_name.to_lowercase()))
                            .unwrap_or(false)
                    })
                } else {
                    false
                }
            })
            .collect();

        if matching.is_empty() {
            return Ok(MultiHopResult {
                success: false,
                message: format!("No upcoming meetings found with {}", person_name),
                intermediate_steps: steps,
                final_data: None,
            });
        }

        steps.push(format!("📋 Found {} meetings with {}", matching.len(), person_name));

        let mut message = format!("Found {} meetings with {}:\n\n", matching.len(), person_name);
        for event in &matching {
            if let Some(start) = event.start_time() {
                message.push_str(&format!("  • {} - {}\n", 
                    start.format("%a %b %d at %I:%M %p"),
                    event.title()
                ));
            }
        }
        message.push_str("\nWhen would you like to reschedule these to?");

        Ok(MultiHopResult {
            success: true,
            message,
            intermediate_steps: steps,
            final_data: Some(MultiHopData::Events(
                matching.iter().map(|e| EventInfo {
                    id: e.id.clone(),
                    title: e.title().to_string(),
                    start: e.start_time(),
                    attendees: e.attendees.as_ref()
                        .map(|a| a.iter().filter_map(|att| att.email.clone()).collect())
                        .unwrap_or_default(),
                }).collect()
            )),
        })
    }

    /// Cancel meetings in a time period
    async fn cancel_meetings(
        &self,
        query: &str,
        _context: &mut ConversationContext,
    ) -> Result<MultiHopResult> {
        let calendar = self.calendar.as_ref()
            .ok_or_else(|| anyhow!("Calendar not connected"))?;

        let mut steps = Vec::new();

        // Parse time period
        let (start, end, period_desc) = self.parse_time_period(query)?;
        steps.push(format!("📅 Finding meetings {}", period_desc));

        // Get events
        let events = calendar.list_events(
            start.with_timezone(&Utc),
            end.with_timezone(&Utc),
            None,
        ).await?;

        if events.is_empty() {
            return Ok(MultiHopResult {
                success: false,
                message: format!("No meetings found {}", period_desc),
                intermediate_steps: steps,
                final_data: None,
            });
        }

        steps.push(format!("📋 Found {} events", events.len()));

        let mut message = format!("⚠️ Found {} meetings to cancel {}:\n\n", events.len(), period_desc);
        for event in &events {
            if let Some(start) = event.start_time() {
                message.push_str(&format!("  • {} - {}\n", 
                    start.format("%I:%M %p"),
                    event.title()
                ));
            }
        }
        message.push_str("\nAre you sure you want to cancel all of these? (yes/no)");

        Ok(MultiHopResult {
            success: true,
            message,
            intermediate_steps: steps,
            final_data: Some(MultiHopData::Events(
                events.iter().map(|e| EventInfo {
                    id: e.id.clone(),
                    title: e.title().to_string(),
                    start: e.start_time(),
                    attendees: vec![],
                }).collect()
            )),
        })
    }

    /// Get someone else's schedule (from shared calendar or meetings)
    async fn get_person_schedule(
        &self,
        query: &str,
        _context: &mut ConversationContext,
    ) -> Result<MultiHopResult> {
        let calendar = self.calendar.as_ref()
            .ok_or_else(|| anyhow!("Calendar not connected"))?;

        let mut steps = Vec::new();

        let person_name = self.extract_person_name(query)
            .ok_or_else(|| anyhow!("Couldn't identify whose schedule to check"))?;

        steps.push(format!("🔍 Looking for meetings with {}", person_name));

        // Get events where this person is an attendee
        let events = calendar.get_week_events().await?;

        let matching: Vec<_> = events.iter()
            .filter(|e| {
                if let Some(ref atts) = e.attendees {
                    atts.iter().any(|a| {
                        a.display_name.as_ref()
                            .map(|n| n.to_lowercase().contains(&person_name.to_lowercase()))
                            .unwrap_or(false)
                    })
                } else {
                    false
                }
            })
            .collect();

        if matching.is_empty() {
            return Ok(MultiHopResult {
                success: true,
                message: format!("No shared meetings with {} this week", person_name),
                intermediate_steps: steps,
                final_data: None,
            });
        }

        let mut message = format!("📅 Meetings with {} this week:\n\n", person_name);
        for event in &matching {
            if let Some(start) = event.start_time() {
                message.push_str(&format!("  • {} {} - {}\n", 
                    start.format("%a"),
                    start.format("%I:%M %p"),
                    event.title()
                ));
            }
        }

        Ok(MultiHopResult {
            success: true,
            message,
            intermediate_steps: steps,
            final_data: None,
        })
    }

    /// Email all people who know a specific person (from knowledge graph)
    /// Pattern: "email everyone who knows Alice about her surprise birthday party"
    async fn email_people_who_know(
        &self,
        query: &str,
        context: &mut ConversationContext,
    ) -> Result<MultiHopResult> {
        let query_lower = query.to_lowercase();
        let mut steps = Vec::new();

        // Step 1: Extract the person name from "who knows [person]"
        let person_name = self.extract_name_from_who_knows(&query_lower)
            .ok_or_else(|| anyhow!("Could not find person name in query"))?;
        
        steps.push(format!("🔍 Looking up people who know {}", person_name));

        // Step 2: Find the person in knowledge graph
        let person = self.graph.find_person(&person_name)?
            .ok_or_else(|| anyhow!("I don't know anyone named '{}'", person_name))?;

        // Step 3: Get all people connected to this person
        let connections = self.graph.get_connections(&person.name)?;
        
        if connections.is_empty() {
            return Ok(MultiHopResult {
                success: false,
                message: format!("I don't know anyone who knows {}.", person.name),
                intermediate_steps: steps,
                final_data: None,
            });
        }

        steps.push(format!("👥 Found {} people who know {}", connections.len(), person.name));

        // Step 4: Collect their info and emails
        let mut people_info: Vec<PersonInfo> = Vec::new();
        let mut missing_emails: Vec<String> = Vec::new();

        for conn in &connections {
            // Try to get email from knowledge graph via database
            let email = self.graph.database().get_entity_property(conn.id, "email").ok().flatten();
            
            if email.is_some() {
                people_info.push(PersonInfo {
                    name: conn.name.clone(),
                    email,
                    source: format!("knows {}", person.name),
                });
            } else {
                missing_emails.push(conn.name.clone());
            }
        }

        // Step 5: Extract email topic/subject from query
        let topic = self.extract_topic_from_query(&query_lower, &person.name)
            .unwrap_or_else(|| format!("about {}", person.name));

        steps.push(format!("📧 Topic: {}", topic));

        // Step 6: Set up context for email composition
        let recipient_refs: Vec<PersonReference> = people_info.iter()
            .map(|p| PersonReference {
                name: p.name.clone(),
                email: p.email.clone(),
                entity_id: None,
            })
            .collect();
        
        context.set_people(recipient_refs);

        // Build result message
        let mut message = format!(
            "Found {} people who know {}:\n",
            connections.len(),
            person.name
        );

        for p in &people_info {
            if let Some(ref email) = p.email {
                message.push_str(&format!("  • {} ({})\n", p.name, email));
            } else {
                message.push_str(&format!("  • {} (no email)\n", p.name));
            }
        }

        if !missing_emails.is_empty() {
            message.push_str(&format!(
                "\n⚠️ Missing emails for: {}",
                missing_emails.join(", ")
            ));
        }

        // Add topic info to message
        message.push_str(&format!("\n📧 Topic: {}", topic));

        if !people_info.is_empty() {
            let recipient_list: Vec<String> = people_info.iter()
                .filter_map(|p| p.email.clone())
                .collect();
            
            if !recipient_list.is_empty() {
                message.push_str(&format!(
                    "\n\n📧 Ready to compose email to {} recipient(s) about: {}\n\
                     Say 'yes' to compose, or provide more details.",
                    recipient_list.len(),
                    topic
                ));
            }
        }

        Ok(MultiHopResult {
            success: true,
            message,
            intermediate_steps: steps,
            final_data: Some(MultiHopData::People(people_info)),
        })
    }

    /// Extract person name from "who knows [person]" pattern
    fn extract_name_from_who_knows(&self, query: &str) -> Option<String> {
        // Pattern: "who knows alice" or "everyone who knows alice"
        if let Some(pos) = query.find("who knows") {
            let after = &query[pos + 9..]; // "who knows".len() = 9
            let after = after.trim();
            
            // Stop at common words that indicate topic/purpose
            let stop_words = ["about", "regarding", "saying", "telling", "for", "to"];
            
            let mut name = after.to_string();
            for stop_word in &stop_words {
                if let Some(stop_pos) = after.find(stop_word) {
                    name = after[..stop_pos].trim().to_string();
                    break;
                }
            }
            
            // Clean up the name
            let name = name.trim_matches(|c: char| !c.is_alphanumeric() && c != ' ');
            
            if !name.is_empty() {
                // Capitalize first letter
                return Some(capitalize_name(name));
            }
        }
        None
    }

    /// Extract topic from email query
    fn extract_topic_from_query(&self, query: &str, exclude_name: &str) -> Option<String> {
        // Pattern: "about [topic]"
        if let Some(pos) = query.find("about") {
            let after = &query[pos + 5..]; // "about".len() = 5
            let topic = after.trim()
                .trim_matches(|c: char| c == '?' || c == '!' || c == '.');
            
            if !topic.is_empty() && !topic.eq_ignore_ascii_case(exclude_name) {
                return Some(topic.to_string());
            }
        }
        
        // Pattern: "saying [topic]"
        if let Some(pos) = query.find("saying") {
            let after = &query[pos + 6..];
            let topic = after.trim()
                .trim_matches(|c: char| c == '?' || c == '!' || c == '.');
            
            if !topic.is_empty() {
                return Some(topic.to_string());
            }
        }
        
        None
    }

    // ========================================================================
    // Helper functions
    // ========================================================================

    /// Parse time period from query
    fn parse_time_period(&self, query: &str) -> Result<(DateTime<Local>, DateTime<Local>, String)> {
        let now = Local::now();

        if query.contains("today") {
            let start = now.date_naive().and_hms_opt(0, 0, 0).unwrap()
                .and_local_timezone(Local).unwrap();
            let end = now.date_naive().and_hms_opt(23, 59, 59).unwrap()
                .and_local_timezone(Local).unwrap();
            return Ok((start, end, "today".to_string()));
        }

        if query.contains("tomorrow") {
            let tomorrow = now + Duration::days(1);
            let start = tomorrow.date_naive().and_hms_opt(0, 0, 0).unwrap()
                .and_local_timezone(Local).unwrap();
            let end = tomorrow.date_naive().and_hms_opt(23, 59, 59).unwrap()
                .and_local_timezone(Local).unwrap();
            return Ok((start, end, "tomorrow".to_string()));
        }

        if query.contains("this week") || query.contains("week") {
            let end = now + Duration::days(7);
            return Ok((now, end, "this week".to_string()));
        }

        if query.contains("next week") {
            let start = now + Duration::days(7);
            let end = now + Duration::days(14);
            return Ok((start, end, "next week".to_string()));
        }

        if query.contains("this month") || query.contains("month") {
            let end = now + Duration::days(30);
            return Ok((now, end, "this month".to_string()));
        }

        // Default to this week
        let end = now + Duration::days(7);
        Ok((now, end, "this week".to_string()))
    }

    /// Extract person name from query
    fn extract_person_name(&self, query: &str) -> Option<String> {
        // Common patterns: "with John", "for Sarah", "'s schedule"
        let patterns = [
            (r"with\s+(\w+)", 1),
            (r"for\s+(\w+)", 1),
            (r"(\w+)'s\s+schedule", 1),
            (r"(\w+)'s\s+meetings", 1),
        ];

        for (pattern, group) in &patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                if let Some(caps) = re.captures(query) {
                    if let Some(name) = caps.get(*group) {
                        let n = name.as_str();
                        // Skip common words
                        if !["all", "the", "my", "your", "our", "their"].contains(&n.to_lowercase().as_str()) {
                            return Some(n.to_string());
                        }
                    }
                }
            }
        }

        None
    }

    /// Extract email subject from query
    fn extract_email_subject(&self, query: &str) -> Option<String> {
        // Pattern: "about [subject]", "regarding [subject]"
        let patterns = [
            r"about\s+(.+?)(?:\s*$|\s+to\s)",
            r"regarding\s+(.+?)(?:\s*$|\s+to\s)",
            r"saying\s+(.+?)(?:\s*$)",
        ];

        for pattern in &patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                if let Some(caps) = re.captures(query) {
                    if let Some(subject) = caps.get(1) {
                        return Some(subject.as_str().trim().to_string());
                    }
                }
            }
        }

        None
    }
}

/// Capitalize first letter of each word in a name
fn capitalize_name(name: &str) -> String {
    name.split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().chain(chars).collect(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Database;

    #[test]
    fn test_parse_time_period() {
        let db = Database::in_memory().unwrap();
        let graph = KnowledgeGraph::new(db);
        let engine = MultiHopEngine {
            calendar: None,
            gmail: None,
            graph: Arc::new(graph),
        };

        let (_, _, desc) = engine.parse_time_period("email everyone I'm meeting with today").unwrap();
        assert_eq!(desc, "today");

        let (_, _, desc) = engine.parse_time_period("who am I meeting this week").unwrap();
        assert_eq!(desc, "this week");
    }

    #[test]
    fn test_extract_person_name() {
        let db = Database::in_memory().unwrap();
        let graph = KnowledgeGraph::new(db);
        let engine = MultiHopEngine {
            calendar: None,
            gmail: None,
            graph: Arc::new(graph),
        };

        assert_eq!(
            engine.extract_person_name("reschedule meetings with John"),
            Some("John".to_string())
        );

        assert_eq!(
            engine.extract_person_name("what's Sarah's schedule"),
            Some("Sarah".to_string())
        );
    }

    #[test]
    fn test_extract_name_from_who_knows() {
        let db = Database::in_memory().unwrap();
        let graph = KnowledgeGraph::new(db);
        let engine = MultiHopEngine {
            calendar: None,
            gmail: None,
            graph: Arc::new(graph),
        };

        // Basic case
        assert_eq!(
            engine.extract_name_from_who_knows("email everyone who knows alice"),
            Some("Alice".to_string())
        );

        // With "about" topic
        assert_eq!(
            engine.extract_name_from_who_knows("email everyone who knows bob about the party"),
            Some("Bob".to_string())
        );

        // With "regarding" topic  
        assert_eq!(
            engine.extract_name_from_who_knows("email people who knows sarah regarding the meeting"),
            Some("Sarah".to_string())
        );
    }

    #[test]
    fn test_extract_topic_from_query() {
        let db = Database::in_memory().unwrap();
        let graph = KnowledgeGraph::new(db);
        let engine = MultiHopEngine {
            calendar: None,
            gmail: None,
            graph: Arc::new(graph),
        };

        // "about" pattern
        assert_eq!(
            engine.extract_topic_from_query("email who knows alice about the surprise birthday party", "alice"),
            Some("the surprise birthday party".to_string())
        );

        // "saying" pattern
        assert_eq!(
            engine.extract_topic_from_query("email everyone saying thanks for coming", "bob"),
            Some("thanks for coming".to_string())
        );
    }
}
