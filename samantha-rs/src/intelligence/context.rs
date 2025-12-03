// Conversation context memory for multi-turn dialogue
// Tracks recent entities, pending actions, and enables pronoun resolution

use anyhow::Result;
use chrono::{DateTime, Local};
use std::collections::VecDeque;

/// Maximum number of turns to keep in history
const MAX_HISTORY: usize = 10;

/// Reference to a calendar event in conversation
#[derive(Debug, Clone)]
pub struct EventReference {
    pub id: Option<String>,
    pub title: String,
    pub start_time: Option<DateTime<Local>>,
    pub attendees: Vec<String>,
}

/// Reference to an email in conversation
#[derive(Debug, Clone)]
pub struct EmailReference {
    pub id: Option<String>,
    pub subject: String,
    pub sender: String,
    pub recipients: Vec<String>,
}

/// Reference to a person mentioned in conversation
#[derive(Debug, Clone)]
pub struct PersonReference {
    pub name: String,
    pub email: Option<String>,
    pub entity_id: Option<i64>,
}

/// A pending action that needs confirmation or additional info
#[derive(Debug, Clone)]
pub enum PendingAction {
    CreateEvent {
        title: String,
        start_time: Option<DateTime<Local>>,
        end_time: Option<DateTime<Local>>,
        attendees: Vec<String>,
        recurrence: Option<String>,
    },
    UpdateEvent {
        event_ref: EventReference,
        updates: EventUpdates,
    },
    SendEmail {
        to: Vec<String>,
        subject: String,
        body: Option<String>,
    },
    ReplyToEmail {
        email_ref: EmailReference,
        body: Option<String>,
    },
}

/// Updates to apply to an event
#[derive(Debug, Clone, Default)]
pub struct EventUpdates {
    pub new_title: Option<String>,
    pub new_start: Option<DateTime<Local>>,
    pub new_end: Option<DateTime<Local>>,
    pub add_attendees: Vec<String>,
    pub remove_attendees: Vec<String>,
}

/// A single conversation turn
#[derive(Debug, Clone)]
pub struct Turn {
    pub user_input: String,
    pub assistant_response: String,
    pub timestamp: DateTime<Local>,
    pub intent: Option<String>,
}

/// Main conversation context tracker
#[derive(Debug, Clone)]
pub struct ConversationContext {
    /// Most recently mentioned event
    pub last_event: Option<EventReference>,
    
    /// Most recently mentioned email
    pub last_email: Option<EmailReference>,
    
    /// Most recently mentioned person
    pub last_person: Option<PersonReference>,
    
    /// Multiple people mentioned (for "them", "everyone")
    pub mentioned_people: Vec<PersonReference>,
    
    /// Action waiting for confirmation or more info
    pub pending_action: Option<PendingAction>,
    
    /// Recent conversation history
    pub turn_history: VecDeque<Turn>,
    
    /// Last query results (for follow-ups)
    pub last_query_results: Option<QueryResults>,
}

/// Results from a query that can be referenced later
#[derive(Debug, Clone)]
pub enum QueryResults {
    Events(Vec<EventReference>),
    Emails(Vec<EmailReference>),
    People(Vec<PersonReference>),
}

impl Default for ConversationContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationContext {
    pub fn new() -> Self {
        Self {
            last_event: None,
            last_email: None,
            last_person: None,
            mentioned_people: Vec::new(),
            pending_action: None,
            turn_history: VecDeque::with_capacity(MAX_HISTORY),
            last_query_results: None,
        }
    }
    
    /// Record a conversation turn
    pub fn add_turn(&mut self, user_input: &str, assistant_response: &str, intent: Option<&str>) {
        if self.turn_history.len() >= MAX_HISTORY {
            self.turn_history.pop_front();
        }
        
        self.turn_history.push_back(Turn {
            user_input: user_input.to_string(),
            assistant_response: assistant_response.to_string(),
            timestamp: Local::now(),
            intent: intent.map(|s| s.to_string()),
        });
    }
    
    /// Update context with a mentioned event
    pub fn set_event(&mut self, event: EventReference) {
        self.last_event = Some(event);
    }
    
    /// Update context with a mentioned email
    pub fn set_email(&mut self, email: EmailReference) {
        self.last_email = Some(email);
    }
    
    /// Update context with a mentioned person
    pub fn set_person(&mut self, person: PersonReference) {
        // Add to mentioned people list
        if !self.mentioned_people.iter().any(|p| p.name == person.name) {
            self.mentioned_people.push(person.clone());
            // Keep only last 5 people
            if self.mentioned_people.len() > 5 {
                self.mentioned_people.remove(0);
            }
        }
        self.last_person = Some(person);
    }
    
    /// Set multiple people (for "email everyone")
    pub fn set_people(&mut self, people: Vec<PersonReference>) {
        self.mentioned_people = people;
        if let Some(first) = self.mentioned_people.first() {
            self.last_person = Some(first.clone());
        }
    }
    
    /// Set a pending action that needs more info
    pub fn set_pending(&mut self, action: PendingAction) {
        self.pending_action = Some(action);
    }
    
    /// Clear pending action
    pub fn clear_pending(&mut self) {
        self.pending_action = None;
    }
    
    /// Store query results for follow-up
    pub fn set_query_results(&mut self, results: QueryResults) {
        self.last_query_results = Some(results);
    }
    
    /// Resolve pronouns in user input based on context
    pub fn resolve_pronouns(&self, input: &str) -> String {
        let mut resolved = input.to_string();
        
        // Resolve "it" / "that" / "the meeting" / "the event"
        if let Some(ref event) = self.last_event {
            let event_pronouns = ["it", "that", "the meeting", "the event", "this meeting", "this event"];
            for pronoun in &event_pronouns {
                if resolved.to_lowercase().contains(pronoun) {
                    // Only replace if it seems to refer to an event
                    if self.seems_event_context(&resolved) {
                        resolved = resolved.replace(pronoun, &format!("the event '{}'", event.title));
                    }
                }
            }
        }
        
        // Resolve "him" / "her" / "them" / person names
        if let Some(ref person) = self.last_person {
            let person_pronouns = ["him", "her"];
            for pronoun in &person_pronouns {
                if resolved.to_lowercase().contains(pronoun) {
                    resolved = resolved.replace(pronoun, &person.name);
                }
            }
        }
        
        // Resolve "them" / "everyone" to multiple people
        if !self.mentioned_people.is_empty() {
            if resolved.to_lowercase().contains("them") || resolved.to_lowercase().contains("everyone") {
                let names: Vec<&str> = self.mentioned_people.iter()
                    .map(|p| p.name.as_str())
                    .collect();
                let names_str = names.join(", ");
                resolved = resolved.replace("them", &names_str);
                resolved = resolved.replace("everyone", &names_str);
            }
        }
        
        resolved
    }
    
    /// Check if context suggests we're talking about an event
    fn seems_event_context(&self, input: &str) -> bool {
        let event_keywords = [
            "move", "reschedule", "cancel", "delete", "change time",
            "add", "invite", "attendee", "when", "where", "duration"
        ];
        
        let input_lower = input.to_lowercase();
        event_keywords.iter().any(|k| input_lower.contains(k))
    }
    
    /// Get the last N turns for context
    pub fn get_recent_context(&self, n: usize) -> Vec<&Turn> {
        self.turn_history.iter().rev().take(n).collect()
    }
    
    /// Check if there's a pending action that can be completed with new info
    pub fn try_complete_pending(&mut self, new_info: &str) -> Option<PendingAction> {
        if let Some(ref mut action) = self.pending_action {
            match action {
                PendingAction::CreateEvent { start_time, .. } => {
                    // If we were missing time and user provides it
                    if start_time.is_none() {
                        if let Ok(parsed) = crate::calendar::parse_datetime(new_info) {
                            *start_time = Some(parsed.start);
                            return self.pending_action.take();
                        }
                    }
                }
                PendingAction::SendEmail { body, .. } => {
                    // If we were missing body and user provides it
                    if body.is_none() {
                        *body = Some(new_info.to_string());
                        return self.pending_action.take();
                    }
                }
                _ => {}
            }
        }
        None
    }
    
    /// Apply incremental modification to pending event
    pub fn apply_modification(&mut self, modification: &str) -> Result<bool> {
        let mod_lower = modification.to_lowercase();
        
        // Handle time modifications: "make it at noon", "change to 3pm"
        if mod_lower.contains("at ") || mod_lower.contains("to ") {
            if let Ok(parsed) = crate::calendar::parse_datetime(modification) {
                if let Some(PendingAction::CreateEvent { ref mut start_time, .. }) = self.pending_action {
                    *start_time = Some(parsed.start);
                    return Ok(true);
                }
                if let Some(PendingAction::UpdateEvent { ref mut updates, .. }) = self.pending_action {
                    updates.new_start = Some(parsed.start);
                    return Ok(true);
                }
            }
        }
        
        // Handle attendee additions: "add Sarah", "also invite John"
        if mod_lower.contains("add ") || mod_lower.contains("invite ") || mod_lower.contains("also ") {
            // Extract name after "add" or "invite"
            let name = extract_name_after_keyword(modification, &["add", "invite", "also"]);
            if let Some(name) = name {
                if let Some(PendingAction::CreateEvent { ref mut attendees, .. }) = self.pending_action {
                    attendees.push(name);
                    return Ok(true);
                }
                if let Some(PendingAction::UpdateEvent { ref mut updates, .. }) = self.pending_action {
                    updates.add_attendees.push(name);
                    return Ok(true);
                }
            }
        }
        
        // Handle title changes: "call it X", "name it X"
        if mod_lower.contains("call it ") || mod_lower.contains("name it ") {
            let title = extract_after_keyword(modification, &["call it", "name it"]);
            if let Some(title) = title {
                if let Some(PendingAction::CreateEvent { title: ref mut t, .. }) = self.pending_action {
                    *t = title;
                    return Ok(true);
                }
                if let Some(PendingAction::UpdateEvent { ref mut updates, .. }) = self.pending_action {
                    updates.new_title = Some(title);
                    return Ok(true);
                }
            }
        }
        
        Ok(false)
    }
    
    /// Clear all context (for new conversation)
    pub fn clear(&mut self) {
        self.last_event = None;
        self.last_email = None;
        self.last_person = None;
        self.mentioned_people.clear();
        self.pending_action = None;
        self.turn_history.clear();
        self.last_query_results = None;
    }
}

/// Extract a name after keywords like "add", "invite"
fn extract_name_after_keyword(input: &str, keywords: &[&str]) -> Option<String> {
    let input_lower = input.to_lowercase();
    
    for keyword in keywords {
        if let Some(pos) = input_lower.find(keyword) {
            let after = &input[pos + keyword.len()..].trim();
            // Get first word(s) that look like a name
            let name: String = after.split_whitespace()
                .take_while(|w| !["to", "at", "for", "and", "the"].contains(&w.to_lowercase().as_str()))
                .collect::<Vec<_>>()
                .join(" ");
            
            if !name.is_empty() {
                return Some(name);
            }
        }
    }
    
    None
}

/// Extract text after keywords
fn extract_after_keyword(input: &str, keywords: &[&str]) -> Option<String> {
    let input_lower = input.to_lowercase();
    
    for keyword in keywords {
        if let Some(pos) = input_lower.find(keyword) {
            let after = input[pos + keyword.len()..].trim().to_string();
            if !after.is_empty() {
                return Some(after);
            }
        }
    }
    
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pronoun_resolution() {
        let mut ctx = ConversationContext::new();
        ctx.set_event(EventReference {
            id: Some("123".to_string()),
            title: "Team Standup".to_string(),
            start_time: None,
            attendees: vec![],
        });
        
        let resolved = ctx.resolve_pronouns("move it to 3pm");
        assert!(resolved.contains("Team Standup"));
    }
    
    #[test]
    fn test_pending_action_completion() {
        let mut ctx = ConversationContext::new();
        ctx.set_pending(PendingAction::SendEmail {
            to: vec!["john@example.com".to_string()],
            subject: "Meeting".to_string(),
            body: None,
        });
        
        let completed = ctx.try_complete_pending("Here is the email body");
        assert!(completed.is_some());
    }
}
