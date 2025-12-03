//! AI-powered calendar event creation
//!
//! Uses LLM to parse natural language into calendar events,
//! and resolves attendee names to contacts from the knowledge graph.

use anyhow::{anyhow, Result};
use std::sync::Arc;

use crate::llm::OllamaClient;
use crate::knowledge::KnowledgeGraph;
use crate::integrations::google::calendar::{GoogleCalendarClient, NewCalendarEvent, ParsedEvent, EventUpdate, GCalEvent};
use crate::integrations::oauth::{OAuthManager, Provider};
use crate::calendar::parsing::{parse_datetime, parse_recurrence};

/// AI-powered calendar assistant
pub struct CalendarAI<'a> {
    llm: &'a OllamaClient,
    graph: &'a KnowledgeGraph,
    oauth: Arc<OAuthManager>,
}

impl<'a> CalendarAI<'a> {
    pub fn new(llm: &'a OllamaClient, graph: &'a KnowledgeGraph, oauth: Arc<OAuthManager>) -> Self {
        CalendarAI { llm, graph, oauth }
    }

    /// Parse natural language into a calendar event
    pub async fn parse_event(&self, input: &str) -> Result<ParsedEvent> {
        let today = chrono::Local::now().format("%Y-%m-%d (%A)").to_string();
        
        let prompt = format!(r#"Parse this calendar event request into JSON. Today is {}.

Request: "{}"

Extract:
- title: Event name
- date: When ("today", "tomorrow", weekday, or YYYY-MM-DD)
- time: Start time (24h format like "14:00")
- duration_minutes: Duration (default 60)
- location: Where (null if not mentioned)
- attendees: List of people names (empty if none)
- description: Details (null if none)
- is_all_day: true if all-day event
- recurrence: Repetition pattern (null if one-time, or "daily", "weekly", "every Monday", "every 2 weeks", "monthly", "every month on first Monday", etc.)

Example 1: {{"title": "Lunch", "date": "tomorrow", "time": "12:00", "duration_minutes": 60, "location": null, "attendees": [], "description": null, "is_all_day": false, "recurrence": null}}

Example 2: {{"title": "Team Standup", "date": "monday", "time": "10:00", "duration_minutes": 30, "location": null, "attendees": [], "description": null, "is_all_day": false, "recurrence": "every Monday"}}

Example 3: {{"title": "Weekly Review", "date": "friday", "time": "14:00", "duration_minutes": 60, "location": null, "attendees": [], "description": null, "is_all_day": false, "recurrence": "weekly"}}"#,
            today, input
        );

        // Use JSON mode for guaranteed valid JSON
        let response = self.llm.query_json(&prompt).await?;
        
        // Extract JSON from response
        let json_str = extract_json(&response)?;
        
        let parsed: ParsedEvent = serde_json::from_str(&json_str)
            .map_err(|e| anyhow!("Failed to parse LLM response: {}. Response was: {}", e, json_str))?;

        Ok(parsed)
    }

    /// Resolve attendee names to email addresses using the knowledge graph
    pub fn resolve_attendees(&self, names: &[String]) -> Vec<ResolvedAttendee> {
        names.iter().map(|name| {
            // Try to find person in knowledge graph
            if let Ok(Some(entity)) = self.graph.find_person(name) {
                // Look for email property
                if let Ok(Some(email)) = self.graph.database().get_entity_property(entity.id, "email") {
                    return ResolvedAttendee {
                        name: name.clone(),
                        email: Some(email),
                        from_contacts: true,
                        entity_id: Some(entity.id),
                    };
                }
            }
            
            // Also try find_any_entity for flexibility
            if let Ok(Some(entity)) = self.graph.find_any_entity(name) {
                if let Ok(Some(email)) = self.graph.database().get_entity_property(entity.id, "email") {
                    return ResolvedAttendee {
                        name: name.clone(),
                        email: Some(email),
                        from_contacts: true,
                        entity_id: Some(entity.id),
                    };
                }
            }
            
            // Not found in contacts
            ResolvedAttendee {
                name: name.clone(),
                email: None,
                from_contacts: false,
                entity_id: None,
            }
        }).collect()
    }

    /// Create an event from natural language
    pub async fn create_from_natural_language(&self, input: &str) -> Result<EventCreationResult> {
        // Step 1: Parse the input
        let parsed = self.parse_event(input).await?;

        // Step 2: Resolve attendees
        let resolved = self.resolve_attendees(&parsed.attendees);
        
        // Collect unresolved names first (before moving resolved)
        let unresolved_names: Vec<String> = resolved.iter()
            .filter(|a| a.email.is_none())
            .map(|a| a.name.clone())
            .collect();
        
        let resolved_emails: Vec<String> = resolved.iter()
            .filter_map(|a| a.email.clone())
            .collect();

        // Step 3: Build the event
        let event = parsed.to_calendar_event(resolved_emails)?;
        
        // Step 4: Check for conflicts
        let client = GoogleCalendarClient::new(self.oauth.clone());
        let conflicts = if !event.all_day {
            client.check_conflicts(event.start, event.end).await.unwrap_or_default()
        } else {
            Vec::new()
        };

        Ok(EventCreationResult {
            event,
            parsed,
            resolved_attendees: resolved,
            unresolved_names,
            conflicts,
        })
    }

    /// Create the event in Google Calendar
    pub async fn commit_event(&self, event: &NewCalendarEvent, send_invites: bool) -> Result<String> {
        let client = GoogleCalendarClient::new(self.oauth.clone());
        
        let created = if send_invites && !event.attendees.is_empty() {
            client.create_event_with_invites(event).await?
        } else {
            client.create_event(event).await?
        };

        let mut result = format!("✓ Event created: {}\n", created.title());
        
        if let Some(link) = created.html_link {
            result.push_str(&format!("   🔗 {}\n", link));
        }
        
        if send_invites && !event.attendees.is_empty() {
            result.push_str(&format!("   📧 Invites sent to {} attendee(s)", event.attendees.len()));
        }

        Ok(result)
    }

    /// Parse an update request from natural language
    pub async fn parse_update_request(&self, input: &str) -> Result<ParsedUpdate> {
        let today = chrono::Local::now().format("%Y-%m-%d (%A)").to_string();
        
        let prompt = format!(r#"Parse this calendar UPDATE request. Today is {}.

Request: "{}"

Extract:
- event_search: Keywords to find the event
- new_time: New time (24h format or null)
- new_date: New date ("today", "tomorrow", weekday, or YYYY-MM-DD, or null)
- new_title: New title (or null)
- new_location: New location (or null)
- add_attendees: People to add (or empty array)

Example: {{"event_search": "lunch with bob", "new_time": "14:00", "new_date": null, "new_title": null, "new_location": null, "add_attendees": []}}"#,
            today, input
        );

        // Use JSON mode for guaranteed valid JSON
        let response = self.llm.query_json(&prompt).await?;
        let json_str = extract_json(&response)?;
        
        let parsed: ParsedUpdate = serde_json::from_str(&json_str)
            .map_err(|e| anyhow!("Failed to parse update request: {}. Response: {}", e, json_str))?;

        Ok(parsed)
    }

    /// Find and prepare an event update
    pub async fn prepare_update(&self, input: &str) -> Result<EventUpdateResult> {
        // Step 1: Parse the update request
        let parsed = self.parse_update_request(input).await?;
        
        // Step 2: Find matching events
        let client = GoogleCalendarClient::new(self.oauth.clone());
        let matches = client.find_event_by_title(&parsed.event_search, 30).await?;
        
        if matches.is_empty() {
            return Err(anyhow!("No events found matching '{}'", parsed.event_search));
        }
        
        // Step 3: Build the update
        let mut update = EventUpdate::new();
        
        if let Some(ref title) = parsed.new_title {
            update = update.with_title(title);
        }
        
        if let Some(ref loc) = parsed.new_location {
            update = update.with_location(loc);
        }
        
        // Handle time changes
        if parsed.new_time.is_some() || parsed.new_date.is_some() {
            let target_event = &matches[0];
            let original_start = target_event.start_time()
                .ok_or_else(|| anyhow!("Cannot determine original event time"))?;
            
            // Parse new date (or keep original)
            let new_date = if let Some(ref d) = parsed.new_date {
                crate::integrations::google::calendar::parse_relative_date(d, chrono::Local::now().date_naive())?
            } else {
                original_start.date_naive()
            };
            
            // Parse new time (or keep original)
            let new_time = if let Some(ref t) = parsed.new_time {
                crate::integrations::google::calendar::parse_time(t)?
            } else {
                original_start.time()
            };
            
            // Calculate duration from original event
            let original_end = target_event.end_time().unwrap_or(original_start + chrono::Duration::hours(1));
            let duration = original_end - original_start;
            
            let new_start = new_date.and_time(new_time)
                .and_local_timezone(chrono::Local)
                .single()
                .ok_or_else(|| anyhow!("Invalid datetime"))?;
            let new_end = new_start + duration;
            
            update = update.with_time(new_start, new_end);
        }
        
        // Resolve attendees
        if !parsed.add_attendees.is_empty() {
            let resolved = self.resolve_attendees(&parsed.add_attendees);
            for att in resolved.iter().filter_map(|a| a.email.as_ref()) {
                update = update.with_attendee(att);
            }
        }
        
        Ok(EventUpdateResult {
            target_event: matches[0].clone(),
            all_matches: matches,
            update,
            parsed,
        })
    }

    /// Execute an event update
    pub async fn commit_update(&self, event_id: &str, update: &EventUpdate) -> Result<String> {
        let client = GoogleCalendarClient::new(self.oauth.clone());
        let updated = client.update_event(event_id, update).await?;
        
        let mut result = format!("✓ Event updated: {}\n", updated.title());
        if let Some(start) = updated.start_time() {
            result.push_str(&format!("   🕐 {}\n", start.format("%a %b %d, %I:%M %p")));
        }
        if let Some(link) = updated.html_link {
            result.push_str(&format!("   🔗 {}", link));
        }
        
        Ok(result)
    }

    /// Parse a delete request
    pub async fn parse_delete_request(&self, input: &str) -> Result<String> {
        let prompt = format!(r#"Extract the event to delete from this request.

User request: "{}"

Return ONLY the event title/description to search for, nothing else.
For example, if user says "cancel dinner with Charlie", return "dinner Charlie".
If user says "delete my meeting tomorrow", return "meeting"."#,
            input
        );

        let response = self.llm.query(&prompt).await?;
        Ok(response.trim().to_string())
    }

    /// Find event to delete
    pub async fn prepare_delete(&self, input: &str) -> Result<EventDeleteResult> {
        let search_term = self.parse_delete_request(input).await?;
        
        let client = GoogleCalendarClient::new(self.oauth.clone());
        let matches = client.find_event_by_title(&search_term, 30).await?;
        
        if matches.is_empty() {
            return Err(anyhow!("No events found matching '{}'", search_term));
        }
        
        Ok(EventDeleteResult {
            target_event: matches[0].clone(),
            all_matches: matches,
            search_term,
        })
    }

    /// Delete an event
    pub async fn commit_delete(&self, event_id: &str) -> Result<String> {
        let client = GoogleCalendarClient::new(self.oauth.clone());
        client.delete_event(event_id).await?;
        Ok("✓ Event deleted successfully".to_string())
    }

    /// Suggest available time slots based on existing calendar
    pub async fn suggest_time(&self, duration_minutes: i64, preferred_date: Option<chrono::NaiveDate>) -> Result<Vec<TimeSlot>> {
        let client = GoogleCalendarClient::new(self.oauth.clone());
        
        let date = preferred_date.unwrap_or_else(|| chrono::Local::now().date_naive());
        let events = client.get_day_summary(date).await;
        
        // For now, suggest common meeting times
        // TODO: Actually analyze busy times and find gaps
        let slots = vec![
            TimeSlot { date, hour: 9, minute: 0, description: "Morning".to_string() },
            TimeSlot { date, hour: 10, minute: 0, description: "Mid-morning".to_string() },
            TimeSlot { date, hour: 14, minute: 0, description: "Early afternoon".to_string() },
            TimeSlot { date, hour: 15, minute: 0, description: "Afternoon".to_string() },
            TimeSlot { date, hour: 16, minute: 0, description: "Late afternoon".to_string() },
        ];

        Ok(slots)
    }
}

/// Result of attendee resolution
#[derive(Debug, Clone)]
pub struct ResolvedAttendee {
    pub name: String,
    pub email: Option<String>,
    pub from_contacts: bool,
    pub entity_id: Option<i64>,
}

impl ResolvedAttendee {
    pub fn display(&self) -> String {
        if let Some(ref email) = self.email {
            if self.from_contacts {
                format!("✓ {} <{}>", self.name, email)
            } else {
                format!("  {} <{}>", self.name, email)
            }
        } else {
            format!("? {} (no email found)", self.name)
        }
    }
}

/// Result of event creation preparation
#[derive(Debug)]
pub struct EventCreationResult {
    pub event: NewCalendarEvent,
    pub parsed: ParsedEvent,
    pub resolved_attendees: Vec<ResolvedAttendee>,
    pub unresolved_names: Vec<String>,
    pub conflicts: Vec<crate::integrations::google::calendar::ConflictInfo>,
}

impl EventCreationResult {
    pub fn display_preview(&self) -> String {
        let mut output = String::new();
        output.push_str("📅 Event Preview\n\n");
        output.push_str(&self.event.preview());

        if !self.resolved_attendees.is_empty() {
            output.push_str("\n👥 Attendees:\n");
            for att in &self.resolved_attendees {
                output.push_str(&format!("   {}\n", att.display()));
            }
        }

        if !self.unresolved_names.is_empty() {
            output.push_str("\n⚠️  Could not find email for:\n");
            for name in &self.unresolved_names {
                output.push_str(&format!("   - {} (add email with /contact {} email <email>)\n", name, name));
            }
        }

        // Show conflicts if any
        if !self.conflicts.is_empty() {
            output.push_str("\n⚠️  Scheduling Conflicts:\n");
            for conflict in &self.conflicts {
                output.push_str(&format!("   {}\n", conflict.display()));
            }
            output.push_str("\n   The event will still be created, but you may want to adjust the time.\n");
        }

        output
    }

    pub fn has_unresolved(&self) -> bool {
        !self.unresolved_names.is_empty()
    }
    
    pub fn has_conflicts(&self) -> bool {
        !self.conflicts.is_empty()
    }
}

/// Time slot suggestion
#[derive(Debug, Clone)]
pub struct TimeSlot {
    pub date: chrono::NaiveDate,
    pub hour: u32,
    pub minute: u32,
    pub description: String,
}

impl TimeSlot {
    pub fn display(&self) -> String {
        let time = chrono::NaiveTime::from_hms_opt(self.hour, self.minute, 0).unwrap();
        format!("{} {} - {}", 
            self.date.format("%a %b %d"),
            time.format("%I:%M %p"),
            self.description
        )
    }
}

/// Parsed update request from natural language
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParsedUpdate {
    pub event_search: String,
    pub new_time: Option<String>,
    pub new_date: Option<String>,
    pub new_title: Option<String>,
    pub new_location: Option<String>,
    pub add_attendees: Vec<String>,
}

/// Result of preparing an event update
#[derive(Debug)]
pub struct EventUpdateResult {
    pub target_event: GCalEvent,
    pub all_matches: Vec<GCalEvent>,
    pub update: EventUpdate,
    pub parsed: ParsedUpdate,
}

impl EventUpdateResult {
    pub fn display_preview(&self) -> String {
        let mut output = self.update.preview(self.target_event.title());
        
        if self.all_matches.len() > 1 {
            output.push_str(&format!("\n⚠️  Found {} matching events. Will update the first one.\n", self.all_matches.len()));
            output.push_str("   Other matches:\n");
            for event in self.all_matches.iter().skip(1).take(3) {
                if let Some(start) = event.start_time() {
                    output.push_str(&format!("   - {} at {}\n", event.title(), start.format("%a %b %d %I:%M %p")));
                }
            }
        }
        
        output
    }
}

/// Result of preparing an event deletion
#[derive(Debug)]
pub struct EventDeleteResult {
    pub target_event: GCalEvent,
    pub all_matches: Vec<GCalEvent>,
    pub search_term: String,
}

impl EventDeleteResult {
    pub fn display_preview(&self) -> String {
        let mut output = format!("🗑️  Delete Event\n\n");
        output.push_str(&format!("   📅 {}\n", self.target_event.title()));
        
        if let Some(start) = self.target_event.start_time() {
            output.push_str(&format!("   🕐 {}\n", start.format("%a %b %d, %I:%M %p")));
        }
        
        if let Some(ref loc) = self.target_event.location {
            output.push_str(&format!("   📍 {}\n", loc));
        }
        
        if self.all_matches.len() > 1 {
            output.push_str(&format!("\n⚠️  Found {} matching events. Will delete the first one.\n", self.all_matches.len()));
        }
        
        output
    }
}

/// Extract JSON from LLM response (handles markdown code blocks)
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
    }
    
    Ok(trimmed.to_string())
}
