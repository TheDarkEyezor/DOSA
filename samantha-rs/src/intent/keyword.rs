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
                          "my schedule", "what's on", "what is on", "any events", "meetings"];
        if query_words.iter().any(|k| lower.contains(k)) {
            score += 0.5;
        }
        
        // Check for person-based queries ("meetings with John")
        let with_patterns = ["with ", "involving ", "including "];
        for pattern in with_patterns {
            if lower.contains(pattern) {
                score += 0.4;
                // Extract person name
                if let Some(pos) = lower.find(pattern) {
                    let after = &lower[pos + pattern.len()..];
                    let words: Vec<&str> = after.split_whitespace().take(3).collect();
                    if !words.is_empty() && !["the", "a", "my", "on", "for", "at"].contains(&words[0]) {
                        let person = words.join(" ");
                        query_type = CalendarQueryType::WithPerson(person);
                        return (score.min(1.0), query_type);
                    }
                }
            }
        }
        
        // Check for topic-based queries ("meetings about AI")
        let topic_patterns = [
            ("about ", 0.4),
            ("related to ", 0.4),
            ("regarding ", 0.4),
            ("concerning ", 0.4),
            ("for ", 0.3),
        ];
        for (pattern, boost) in topic_patterns {
            if lower.contains(pattern) {
                if let Some(pos) = lower.find(pattern) {
                    let after = &lower[pos + pattern.len()..];
                    let topic = after.trim()
                        .trim_start_matches("the ")
                        .trim_start_matches("a ");
                    // Skip if it's a time word
                    if !topic.is_empty() && !["today", "tomorrow", "week", "next", "this"].iter().any(|t| topic.starts_with(t)) {
                        score += boost;
                        query_type = CalendarQueryType::Topic(topic.to_string());
                        return (score.min(1.0), query_type);
                    }
                }
            }
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
        
        // Conversation/thread patterns - check first as they have specific structure
        let conversation_patterns = [
            "summarize my conversation with",
            "summarize conversation with", 
            "summarize the conversation with",
            "conversation with",
            "my emails with",
            "email thread with",
            "thread with",
            "discussion with",
            "correspondence with",
            "emails between me and",
        ];
        for pattern in conversation_patterns {
            if lower.contains(pattern) {
                // Extract person name
                if let Some(idx) = lower.find(pattern) {
                    let after = &lower[idx + pattern.len()..].trim();
                    // Take words until end or stop word
                    let stop_words = ["?", ".", "!", ",", " about ", " from ", " today", " this week"];
                    let mut end_idx = after.len();
                    for stop in stop_words {
                        if let Some(stop_idx) = after.find(stop) {
                            end_idx = end_idx.min(stop_idx);
                        }
                    }
                    let person = after[..end_idx].trim().to_string();
                    if !person.is_empty() {
                        query_type = EmailQueryType::ConversationWith(person);
                        score = 0.85;
                        return (score, query_type);
                    }
                }
            }
        }
        
        let email_query_words = ["my emails", "my inbox", "my mail", "check email",
                                "show email", "any emails", "any mail", "unread emails",
                                "new emails", "show me my", "emails i", "what emails"];
        if email_query_words.iter().any(|k| lower.contains(k)) {
            score += 0.5;
        }
        
        // Email word by itself with query intent
        if lower.contains("email") && (lower.contains("show") || lower.contains("check") || lower.contains("my") || lower.contains("what")) {
            score += 0.3;
        }
        
        // Topic/semantic query patterns - "emails about X", "emails related to X"
        let topic_patterns = [
            ("about ", "emails about"),
            ("related to ", "related to"),
            ("regarding ", "regarding"),
            ("concerning ", "concerning"),
            ("mentioning ", "mentioning"),
            ("on the topic of ", "topic of"),
        ];
        
        for (pattern, context) in topic_patterns {
            if lower.contains(context) || (lower.contains("email") && lower.contains(pattern)) {
                // Extract the topic
                if let Some(topic) = self.extract_topic_from_query(lower, pattern) {
                    query_type = EmailQueryType::Topic(topic);
                    score += 0.6;
                    return (score.min(1.0), query_type);
                }
            }
        }
        
        // Determine subtype
        if lower.contains("unread") || (lower.contains("new ") && lower.contains("email")) {
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
    
    /// Extract topic from email query
    fn extract_topic_from_query(&self, lower: &str, pattern: &str) -> Option<String> {
        if let Some(idx) = lower.find(pattern) {
            let after = &lower[idx + pattern.len()..];
            // Take words until end or common stop words
            let stop_words = ["?", ".", "!", ",", " and ", " or ", " that ", " which "];
            let mut end_idx = after.len();
            for stop in stop_words {
                if let Some(stop_idx) = after.find(stop) {
                    end_idx = end_idx.min(stop_idx);
                }
            }
            let topic = after[..end_idx].trim();
            if !topic.is_empty() {
                return Some(topic.to_string());
            }
        }
        None
    }
    
    /// Score how well input matches knowledge graph query patterns
    fn score_knowledge_query(&self, lower: &str) -> f32 {
        let mut score: f32 = 0.0;
        
        // Strong personal contact patterns - high score
        let strong_personal_patterns = [
            "do i know", "do we have", "contact for", "who do i know",
            "my contacts", "my network", "in my contacts", "anyone who",
            "someone who", "people who", "find someone", "find people",
            "contacts at", "who from", "person in",
        ];
        if strong_personal_patterns.iter().any(|k| lower.contains(k)) {
            score += 0.85;
        }
        
        // Direct person queries - moderate-high score
        let person_patterns = ["who is", "who works", "tell me about", 
                              "where does", "what does", "how do i contact",
                              "who can help", "who knows", "what's", "background on"];
        if person_patterns.iter().any(|k| lower.contains(k)) {
            score += 0.7;
        }
        
        // Questions about relationships - add score
        let relationship_words = ["manager", "team", "works at", "works with", 
                                 "colleague", "coworker", "reports to", "works for"];
        if relationship_words.iter().any(|k| lower.contains(k)) {
            score += 0.4;
        }
        
        // Skill-based queries
        let skill_patterns = ["knows", "experience with", "experienced in", "skilled in",
                             "expert in", "good at", "familiar with", "can help with",
                             "python", "rust", "javascript", "react", "machine learning"];
        if skill_patterns.iter().any(|k| lower.contains(k)) {
            score += 0.5;
        }
        
        // Organization-related queries
        let org_patterns = ["at google", "at meta", "at apple", "at microsoft", 
                           "at amazon", "at netflix", "at uber", "at airbnb",
                           "at stripe", "from stanford", "from mit", "from berkeley",
                           "engineers at", "developers at", "team at"];
        if org_patterns.iter().any(|k| lower.contains(k)) {
            score += 0.4;
        }
        
        // Negative signals - reduce score for web search indicators
        let web_indicators = ["weather", "stock", "price", "news", "latest", 
                             "ceo of", "founder of", "define", "definition"];
        if web_indicators.iter().any(|k| lower.contains(k)) {
            score -= 0.4;
        }
        
        score.max(0.0).min(1.0)
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
        
        // Web search scoring
        let web_search = self.score_web_search(&lower);
        if web_search > 0.3 {
            candidates.push((
                Intent::WebSearch { query: input.to_string() },
                web_search
            ));
        }
        
        // Sort by confidence descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        candidates
    }
    
    /// Score how well input matches web search patterns
    fn score_web_search(&self, lower: &str) -> f32 {
        let mut score: f32 = 0.0;
        
        // Weather queries - very strong signal
        if lower.contains("weather") || lower.contains("temperature") || lower.contains("forecast") {
            score += 0.9;
        }
        
        // Stock/finance queries
        let stock_patterns = ["stock price", "share price", "market cap", "trading at", "worth", "earnings"];
        if stock_patterns.iter().any(|p| lower.contains(p)) {
            score += 0.8;
        }
        
        // CEO/company leadership queries
        let ceo_patterns = ["ceo of", "who runs", "who founded", "who started", "who leads", "founder of"];
        if ceo_patterns.iter().any(|p| lower.contains(p)) {
            score += 0.85;
        }
        
        // "what is" for definitions - but not "who is" (could be personal contact)
        if lower.starts_with("what is") && !lower.contains("my contact") && !lower.contains("do i know") {
            score += 0.7;
        }
        
        // "who is" should NOT be scored high here - prefer knowledge_query for personal contacts
        // Only score if it has famous person indicators
        if lower.starts_with("who is") {
            // Check for famous person context
            let famous_indicators = ["ceo", "founder", "president", "actor", "singer", "politician", "famous"];
            if famous_indicators.iter().any(|p| lower.contains(p)) {
                score += 0.6;
            }
            // Otherwise don't score - let knowledge_query handle it
        }
        
        // General search patterns
        let search_patterns = ["search for", "look up", "google", "define", "definition of"];
        if search_patterns.iter().any(|p| lower.contains(p)) {
            score += 0.7;
        }
        
        // How to queries
        if lower.starts_with("how to") || lower.starts_with("how do") || lower.starts_with("how can") {
            score += 0.75;
        }
        
        // News queries
        if lower.contains("news about") || lower.contains("latest on") || lower.starts_with("news") {
            score += 0.7;
        }
        
        // Comparison queries
        if lower.contains(" vs ") || lower.contains(" versus ") || lower.contains("compare ") {
            score += 0.6;
        }
        
        // Recipe queries
        if lower.contains("recipe") || lower.contains("how to make") || lower.contains("how to cook") {
            score += 0.7;
        }
        
        // Filter out personal/local queries that should stay local
        if lower.contains("my calendar") || lower.contains("my contacts") || 
           lower.contains("my emails") || lower.contains("who do i know") {
            return 0.0;
        }
        
        score.min(1.0)
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
