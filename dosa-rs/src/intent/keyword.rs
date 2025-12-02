//! Fast keyword-based intent classification
//!
//! Uses pattern matching for quick classification without LLM.
//! Returns multiple candidates with confidence scores.

use super::types::*;
use super::{IntentClassifier, IntentContext};
use anyhow::Result;
use async_trait::async_trait;

/// Keyword-based classifier for fast intent detection
pub struct KeywordClassifier;

impl KeywordClassifier {
    pub fn new() -> Self {
        KeywordClassifier
    }
    
    /// Score how well input matches contact update patterns
    fn score_contact_update(&self, lower: &str) -> f32 {
        let mut score: f32 = 0.0;
        
        // Person/contact keywords
        let person_keywords = ["person", "contact", "people", "friend", "colleague", "someone"];
        if person_keywords.iter().any(|k| lower.contains(k)) {
            score += 0.5;
        }
        
        // "I know" pattern - strong signal for adding contacts
        if lower.contains("i know") || lower.contains("know them") || lower.contains("we know") {
            score += 0.4;
        }
        
        // Email/property setting patterns
        if lower.contains("'s email") || lower.contains("email is") || lower.contains("email address") {
            score += 0.7;
        }
        
        // Phone patterns
        if lower.contains("'s phone") || lower.contains("phone is") || lower.contains("number is") {
            score += 0.6;
        }
        
        // Works at/studies at patterns (adding relationships) - strong signal for person context
        if lower.contains("works at") || lower.contains("works for") || 
           lower.contains("studies at") || lower.contains("goes to") ||
           lower.contains("he works") || lower.contains("she works") ||
           lower.contains("they work") || lower.contains("he studies") ||
           lower.contains("she studies") || lower.contains("they study") {
            score += 0.5;  // Increased - clear person context
        }
        
        // Add + person combination
        if lower.contains("add") && person_keywords.iter().any(|k| lower.contains(k)) {
            score += 0.3;
        }
        
        // "Add <name>, he/she/they" pattern - clear person context
        if lower.contains("add") && (lower.contains(", he ") || lower.contains(", she ") || 
           lower.contains(", they ") || lower.contains(" who ")) {
            score += 0.5;
        }
        
        // Remember pattern
        if lower.contains("remember") && (lower.contains("name") || lower.contains("email") || lower.contains("phone")) {
            score += 0.5;
        }
        
        score.min(1.0)
    }
    
    /// Score how well input matches calendar create patterns
    fn score_calendar_create(&self, lower: &str) -> f32 {
        // Don't match if this looks like an update (reschedule, move, change)
        let update_keywords = ["reschedule", "move", "change", "update", "modify", "push", "delay"];
        if update_keywords.iter().any(|k| lower.contains(k)) {
            return 0.0;
        }
        
        // Don't match if this looks like adding a contact/person
        let contact_keywords = ["person", "contact", "people", "friend", "colleague", "i know"];
        if contact_keywords.iter().any(|k| lower.contains(k)) {
            return 0.0;
        }
        
        // Don't match "Add <name>, he/she/they" patterns (clearly about adding a person)
        if lower.contains("add") && (lower.contains(", he ") || lower.contains(", she ") || 
           lower.contains(", they ") || lower.contains(" who ")) {
            return 0.0;
        }
        
        // Don't match work/study relationship patterns
        if lower.contains("he works") || lower.contains("she works") || lower.contains("they work") ||
           lower.contains("he studies") || lower.contains("she studies") || lower.contains("they study") {
            return 0.0;
        }
        
        // Don't match email/phone setting patterns
        if lower.contains("'s email") || lower.contains("email is") || 
           lower.contains("'s phone") || lower.contains("phone is") {
            return 0.0;
        }
        
        let mut score: f32 = 0.0;
        
        // Strong signals
        let create_keywords = ["schedule", "create", "add", "set up", "book", "arrange"];
        if create_keywords.iter().any(|k| lower.contains(k)) {
            score += 0.5;
        }
        
        // Event type indicators
        let event_types = ["meeting", "lunch", "dinner", "call", "appointment", "event"];
        if event_types.iter().any(|k| lower.contains(k)) {
            score += 0.3;
        }
        
        // Time indicators
        let time_words = ["tomorrow", "today", "monday", "tuesday", "wednesday", 
                         "thursday", "friday", "saturday", "sunday", "at ", "pm", "am",
                         "next week", "this week"];
        if time_words.iter().any(|k| lower.contains(k)) {
            score += 0.2;
        }
        
        // "with" pattern for attendees
        if lower.contains(" with ") {
            score += 0.1;
        }
        
        score.min(1.0)
    }
    
    /// Score how well input matches calendar query patterns
    fn score_calendar_query(&self, lower: &str) -> (f32, CalendarQueryType) {
        let mut score: f32 = 0.0;
        let mut query_type = CalendarQueryType::Today;
        
        // Query keywords
        let query_words = ["what", "show", "list", "do i have", "am i", "my calendar",
                          "my schedule", "what's on", "what is on", "any events"];
        if query_words.iter().any(|k| lower.contains(k)) {
            score += 0.5;
        }
        
        // Time specifiers determine query type
        if lower.contains("tomorrow") {
            query_type = CalendarQueryType::Tomorrow;
            score += 0.3;
        } else if lower.contains("this week") || lower.contains("week") {
            query_type = CalendarQueryType::Week;
            score += 0.3;
        } else if lower.contains("upcoming") || lower.contains("next") {
            query_type = CalendarQueryType::Upcoming;
            score += 0.3;
        } else if lower.contains("today") {
            query_type = CalendarQueryType::Today;
            score += 0.3;
        }
        
        (score.min(1.0), query_type)
    }
    
    /// Score how well input matches calendar update patterns
    fn score_calendar_update(&self, lower: &str) -> f32 {
        // Skip if this looks like emailing about a change
        if self.is_email_about_change(lower) {
            return 0.0;
        }
        
        let mut score: f32 = 0.0;
        
        let update_keywords = ["move", "reschedule", "change", "update", "modify", "push", "delay"];
        if update_keywords.iter().any(|k| lower.contains(k)) {
            score += 0.5;
        }
        
        // Must reference an event
        let event_refs = ["meeting", "event", "call", "appointment", "the ", "my "];
        if event_refs.iter().any(|k| lower.contains(k)) {
            score += 0.3;
        }
        
        // New time reference
        let time_change = ["to ", "from ", "instead"];
        if time_change.iter().any(|k| lower.contains(k)) {
            score += 0.2;
        }
        
        score.min(1.0)
    }
    
    /// Score how well input matches calendar delete patterns
    fn score_calendar_delete(&self, lower: &str) -> f32 {
        let mut score: f32 = 0.0;
        
        let delete_keywords = ["cancel", "delete", "remove", "drop"];
        if delete_keywords.iter().any(|k| lower.contains(k)) {
            score += 0.6;
        }
        
        // Must reference an event
        let event_refs = ["meeting", "event", "call", "appointment", "the ", "my ", " with "];
        if event_refs.iter().any(|k| lower.contains(k)) {
            score += 0.3;
        }
        
        score.min(1.0)
    }
    
    /// Score how well input matches email compose patterns
    fn score_email_compose(&self, lower: &str) -> f32 {
        let mut score: f32 = 0.0;
        
        let compose_keywords = ["send an email", "send email", "write an email", "write email",
                               "compose email", "email to", "send a message", "send a note",
                               "draft an email"];
        if compose_keywords.iter().any(|k| lower.contains(k)) {
            score += 0.7;
        }
        
        // Pattern: "email <name> about/saying/telling"
        let action_patterns = ["email saying", "email about", "email telling", 
                              "message saying", "message about"];
        if action_patterns.iter().any(|k| lower.contains(k)) {
            score += 0.6;
        }
        
        // "send <name> an email" pattern
        if lower.contains("send") && lower.contains("email") {
            score += 0.4;
        }
        
        score.min(1.0)
    }
    
    /// Score how well input matches email attendees pattern
    fn score_email_attendees(&self, lower: &str) -> f32 {
        let mut score: f32 = 0.0;
        
        // Strong signal: attendees + email action
        let attendee_words = ["attendee", "participant", "invitee", "everyone in", "them"];
        if attendee_words.iter().any(|k| lower.contains(k)) {
            score += 0.5;
        }
        
        let email_actions = ["email", "notify", "message", "let them know", "tell them", "send"];
        if email_actions.iter().any(|k| lower.contains(k)) {
            score += 0.4;
        }
        
        // Event reference
        let event_refs = ["meeting", "event", "dinner", "lunch", "call", "'s"];
        if event_refs.iter().any(|k| lower.contains(k)) {
            score += 0.2;
        }
        
        // Boost for "let them know" or "tell them" patterns
        if lower.contains("let") && lower.contains("know") {
            score += 0.3;
        }
        
        score.min(1.0)
    }
    
    /// Score how well input matches email query patterns
    fn score_email_query(&self, lower: &str) -> (f32, EmailQueryType) {
        let mut score: f32 = 0.0;
        let mut query_type = EmailQueryType::List;
        
        let email_query_words = ["my emails", "my inbox", "my mail", "check email",
                                "show email", "any emails", "any mail", "unread emails",
                                "new emails", "show me my"];
        if email_query_words.iter().any(|k| lower.contains(k)) {
            score += 0.5;
        }
        
        // Email word by itself with query intent
        if lower.contains("email") && (lower.contains("show") || lower.contains("check") || lower.contains("my")) {
            score += 0.3;
        }
        
        // Determine subtype
        if lower.contains("unread") || lower.contains("new ") {
            query_type = EmailQueryType::Unread;
            score += 0.3;
        } else if lower.contains("from ") {
            // Extract sender
            if let Some(idx) = lower.find("from ") {
                let after = &lower[idx + 5..];
                let sender = after.split_whitespace().next().unwrap_or("");
                if !sender.is_empty() {
                    query_type = EmailQueryType::From(sender.to_string());
                    score += 0.3;
                }
            }
        } else if lower.contains("summary") || lower.contains("summarize") {
            query_type = EmailQueryType::Summary;
            score += 0.3;
        }
        
        (score.min(1.0), query_type)
    }
    
    /// Score how well input matches knowledge graph query patterns
    fn score_knowledge_query(&self, lower: &str) -> f32 {
        let mut score: f32 = 0.0;
        
        let kg_patterns = ["who is", "who works", "what is", "tell me about",
                          "where does", "what does", "how do i contact",
                          "who can help", "who knows"];
        if kg_patterns.iter().any(|k| lower.contains(k)) {
            score += 0.6;
        }
        
        // Questions about relationships
        let relationship_words = ["manager", "team", "works at", "works with", "colleague"];
        if relationship_words.iter().any(|k| lower.contains(k)) {
            score += 0.3;
        }
        
        score.min(1.0)
    }
    
    /// Check if this is emailing ABOUT a change (not changing calendar)
    fn is_email_about_change(&self, lower: &str) -> bool {
        let email_signals = ["email", "notify", "let them know", "letting them know",
                            "tell them", "message", "attendee"];
        email_signals.iter().any(|k| lower.contains(k))
    }
    
    /// Score how well input matches email reply patterns
    fn score_email_reply(&self, lower: &str) -> f32 {
        let mut score: f32 = 0.0;
        
        // Strong reply signals
        let reply_keywords = ["reply to", "respond to", "answer", "write back", "get back to"];
        if reply_keywords.iter().any(|k| lower.contains(k)) {
            score += 0.6;
        }
        
        // Email reference patterns
        if lower.contains("email") || lower.contains("message") {
            score += 0.2;
        }
        
        // Person reference patterns - "John's email", "email from Sarah"
        if lower.contains("'s email") || lower.contains("from ") {
            score += 0.2;
        }
        
        // "last email" pattern
        if lower.contains("last") && lower.contains("email") {
            score += 0.3;
        }
        
        // "their email" pattern
        if lower.contains("their") && lower.contains("email") {
            score += 0.2;
        }
        
        score.min(1.0)
    }
    
    /// Get all candidate intents with scores
    pub fn get_candidates(&self, input: &str) -> Vec<(Intent, f32)> {
        let lower = input.to_lowercase();
        let mut candidates = Vec::new();
        
        // Check each intent type
        // Contact update check FIRST (highest priority for contact patterns)
        let contact_update = self.score_contact_update(&lower);
        if contact_update > 0.4 {
            candidates.push((
                Intent::ContactUpdate { description: input.to_string() },
                contact_update
            ));
        }
        
        let cal_create = self.score_calendar_create(&lower);
        if cal_create > 0.3 {
            candidates.push((
                Intent::CalendarCreate { description: input.to_string() },
                cal_create
            ));
        }
        
        let (cal_query, query_type) = self.score_calendar_query(&lower);
        if cal_query > 0.3 {
            candidates.push((
                Intent::CalendarQuery { query_type },
                cal_query
            ));
        }
        
        let cal_update = self.score_calendar_update(&lower);
        if cal_update > 0.3 {
            candidates.push((
                Intent::CalendarUpdate { description: input.to_string() },
                cal_update
            ));
        }
        
        let cal_delete = self.score_calendar_delete(&lower);
        if cal_delete > 0.3 {
            candidates.push((
                Intent::CalendarDelete { description: input.to_string() },
                cal_delete
            ));
        }
        
        // Email attendees check BEFORE general compose
        let email_attendees = self.score_email_attendees(&lower);
        if email_attendees > 0.5 {
            candidates.push((
                Intent::EmailAttendees { description: input.to_string() },
                email_attendees
            ));
        }
        
        // Email reply check BEFORE general compose
        let email_reply = self.score_email_reply(&lower);
        if email_reply > 0.5 {
            candidates.push((
                Intent::EmailReply { description: input.to_string() },
                email_reply
            ));
        }
        
        let email_compose = self.score_email_compose(&lower);
        if email_compose > 0.3 {
            candidates.push((
                Intent::EmailCompose { description: input.to_string() },
                email_compose
            ));
        }
        
        let (email_query, eq_type) = self.score_email_query(&lower);
        if email_query > 0.3 {
            candidates.push((
                Intent::EmailQuery { query_type: eq_type },
                email_query
            ));
        }
        
        let kg_query = self.score_knowledge_query(&lower);
        if kg_query > 0.3 {
            candidates.push((
                Intent::KnowledgeQuery { query: input.to_string() },
                kg_query
            ));
        }
        
        // Sort by confidence descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        candidates
    }
}

#[async_trait]
impl IntentClassifier for KeywordClassifier {
    async fn classify(&self, input: &str, _ctx: &IntentContext) -> Result<IntentResult> {
        let lower = input.to_lowercase();
        
        // Check for explicit commands first
        if lower.starts_with('/') {
            return Ok(IntentResult::new(
                Intent::Command(input.to_string()),
                1.0,
                input
            ));
        }
        
        // Get all candidates
        let candidates = self.get_candidates(input);
        
        if candidates.is_empty() {
            // No strong matches - conversation fallback
            return Ok(IntentResult::new(
                Intent::Conversation { input: input.to_string() },
                0.5,
                input
            ));
        }
        
        // Take the best candidate
        let (best_intent, best_confidence) = candidates[0].clone();
        
        // Collect alternatives (other candidates with reasonable scores)
        let alternatives: Vec<(Intent, f32)> = candidates.into_iter()
            .skip(1)
            .filter(|(_, conf)| *conf > 0.3)
            .collect();
        
        Ok(IntentResult::new(best_intent, best_confidence, input)
            .with_alternatives(alternatives))
    }
    
    fn name(&self) -> &str {
        "keyword"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_email_attendees_detection() {
        let classifier = KeywordClassifier::new();
        let score = classifier.score_email_attendees(
            "email the attendees of tomorrow's dinner letting them know the time changed"
        );
        assert!(score > 0.7, "Should detect email attendees intent, got {}", score);
    }
    
    #[test]
    fn test_calendar_update_not_triggered_by_email() {
        let classifier = KeywordClassifier::new();
        let score = classifier.score_calendar_update(
            "email the attendees letting them know the time changed"
        );
        assert!(score < 0.3, "Should NOT detect calendar update when emailing about change, got {}", score);
    }
    
    #[test]
    fn test_email_compose_detection() {
        let classifier = KeywordClassifier::new();
        let score = classifier.score_email_compose("send charlie an email saying hi");
        assert!(score > 0.3, "Should detect email compose, got {}", score);
    }
}
