//! Natural language query system for the knowledge graph
//! 
//! This module enables querying the knowledge graph using natural language:
//! - "Who works at X?"
//! - "What do I know about Y?"
//! - "Who knows Z?"
//! - "List all relationships for X"
//! 
//! Phase 4: Added Intent-based routing for natural language queries

use anyhow::Result;
use std::collections::HashMap;

use crate::knowledge::KnowledgeGraph;
use crate::llm::OllamaClient;

// ============================================================================
// Intent System (Phase 4B)
// ============================================================================

/// High-level user intent classification
#[derive(Debug, Clone, PartialEq)]
pub enum Intent {
    /// User wants contact information (email, phone, etc.)
    ContactInfo { person_name: String },
    /// User wants location information
    LocationQuery { entity_name: String },
    /// User wants project information or team members
    ProjectQuery { project_name: String },
    /// User wants to understand relationships/network
    RelationshipQuery { entity_name: String },
    /// User wants to perform an action (send email, schedule meeting, etc.)
    ActionRequest { action: ActionType, entity_name: String },
    /// User wants to know how to reach someone
    ReachabilityQuery { person_name: String },
    /// General information query
    InformationQuery { topic: String },
    /// Unknown intent
    Unknown { raw_query: String },
}

/// Types of actions the user might want to perform
#[derive(Debug, Clone, PartialEq)]
pub enum ActionType {
    SendEmail,
    SendMessage,
    ScheduleMeeting,
    MakeCall,
    ShareDocument,
    Collaborate,
    Unknown,
}

/// Intent classification result with confidence
#[derive(Debug, Clone)]
pub struct IntentResult {
    pub intent: Intent,
    pub confidence: f32,
    pub entities: Vec<String>,
    pub action_ready: bool,  // Whether we have enough info to execute
}

/// Intent Router - classifies natural language into intents
pub struct IntentRouter<'a> {
    graph: &'a KnowledgeGraph,
}

impl<'a> IntentRouter<'a> {
    pub fn new(graph: &'a KnowledgeGraph) -> Self {
        Self { graph }
    }

    /// Classify a natural language query into an intent
    pub fn classify(&self, input: &str) -> IntentResult {
        let lower = input.to_lowercase();
        
        // Check for contact info intent first (highest priority for info queries)
        if let Some(result) = self.detect_contact_intent(&lower) {
            return result;
        }
        
        // Check for reachability intent ("how can I reach/contact X")
        if let Some(result) = self.detect_reachability_intent(&lower) {
            return result;
        }
        
        // Check for action intents (send email, schedule meeting)
        if let Some(result) = self.detect_action_intent(&lower) {
            return result;
        }
        
        // Check for location intent
        if let Some(result) = self.detect_location_intent(&lower) {
            return result;
        }
        
        // Check for project intent
        if let Some(result) = self.detect_project_intent(&lower) {
            return result;
        }
        
        // Check for relationship intent
        if let Some(result) = self.detect_relationship_intent(&lower) {
            return result;
        }
        
        // Default to unknown
        IntentResult {
            intent: Intent::Unknown { raw_query: input.to_string() },
            confidence: 0.0,
            entities: vec![],
            action_ready: false,
        }
    }

    /// Detect action-related intents (send email, schedule meeting, etc.)
    fn detect_action_intent(&self, input: &str) -> Option<IntentResult> {
        let action_patterns = [
            (vec!["send", "email"], ActionType::SendEmail),
            (vec!["email"], ActionType::SendEmail),
            (vec!["message"], ActionType::SendMessage),
            (vec!["text"], ActionType::SendMessage),
            (vec!["schedule", "meeting"], ActionType::ScheduleMeeting),
            (vec!["set up", "meeting"], ActionType::ScheduleMeeting),
            (vec!["book", "meeting"], ActionType::ScheduleMeeting),
            (vec!["call"], ActionType::MakeCall),
            (vec!["phone"], ActionType::MakeCall),
            (vec!["share"], ActionType::ShareDocument),
            (vec!["collaborate"], ActionType::Collaborate),
            (vec!["work with"], ActionType::Collaborate),
        ];
        
        for (patterns, action) in &action_patterns {
            let all_match = patterns.iter().all(|p| input.contains(p));
            if all_match {
                // Extract entity name (the "to" recipient)
                let entity = self.extract_entity_for_action(input);
                return Some(IntentResult {
                    intent: Intent::ActionRequest { 
                        action: action.clone(), 
                        entity_name: entity.clone() 
                    },
                    confidence: 0.85,
                    entities: vec![entity],
                    action_ready: false,  // Need to check if we have contact info
                });
            }
        }
        None
    }

    /// Detect contact information intent
    fn detect_contact_intent(&self, input: &str) -> Option<IntentResult> {
        let contact_patterns = [
            "contact info", "contact details", "contact information",
            "email address", "phone number", "email for", "phone for",
            "how do i contact", "how to contact", "how can i contact",
            "'s email", "'s phone", "email of", "phone of",
            "s email", "s phone",  // handles what's -> whats
        ];
        
        if contact_patterns.iter().any(|p| input.contains(p)) {
            let entity = self.extract_person_name(input);
            return Some(IntentResult {
                intent: Intent::ContactInfo { person_name: entity.clone() },
                confidence: 0.9,
                entities: vec![entity],
                action_ready: true,
            });
        }
        None
    }

    /// Detect reachability intent (how to reach someone)
    fn detect_reachability_intent(&self, input: &str) -> Option<IntentResult> {
        let reach_patterns = [
            "how can i reach", "how do i reach", "reach out to",
            "get in touch", "get ahold of", "how to reach",
        ];
        
        if reach_patterns.iter().any(|p| input.contains(p)) {
            let entity = self.extract_person_name(input);
            return Some(IntentResult {
                intent: Intent::ReachabilityQuery { person_name: entity.clone() },
                confidence: 0.9,
                entities: vec![entity],
                action_ready: true,
            });
        }
        None
    }

    /// Detect location-related intent
    fn detect_location_intent(&self, input: &str) -> Option<IntentResult> {
        let location_patterns = [
            "where is", "where does", "located", "location of",
            "based in", "works from", "lives in", "staying",
        ];
        
        if location_patterns.iter().any(|p| input.contains(p)) {
            let entity = self.extract_entity_name(input);
            return Some(IntentResult {
                intent: Intent::LocationQuery { entity_name: entity.clone() },
                confidence: 0.85,
                entities: vec![entity],
                action_ready: true,
            });
        }
        None
    }

    /// Detect project-related intent
    fn detect_project_intent(&self, input: &str) -> Option<IntentResult> {
        let project_patterns = [
            "working on", "works on", "project", "involved in",
            "team for", "who is on", "members of",
        ];
        
        if project_patterns.iter().any(|p| input.contains(p)) {
            let entity = self.extract_project_name(input);
            return Some(IntentResult {
                intent: Intent::ProjectQuery { project_name: entity.clone() },
                confidence: 0.85,
                entities: vec![entity],
                action_ready: true,
            });
        }
        None
    }

    /// Detect relationship-related intent
    fn detect_relationship_intent(&self, input: &str) -> Option<IntentResult> {
        let rel_patterns = [
            "who knows", "knows who", "connected to", "relationship",
            "network", "colleagues", "reports to", "manages",
        ];
        
        if rel_patterns.iter().any(|p| input.contains(p)) {
            let entity = self.extract_entity_name(input);
            return Some(IntentResult {
                intent: Intent::RelationshipQuery { entity_name: entity.clone() },
                confidence: 0.8,
                entities: vec![entity],
                action_ready: true,
            });
        }
        None
    }

    /// Extract entity name for action patterns (after "to" or similar)
    fn extract_entity_for_action(&self, input: &str) -> String {
        let prepositions = ["to ", "for ", "with "];
        
        for prep in prepositions {
            if let Some(pos) = input.rfind(prep) {
                let after = &input[pos + prep.len()..];
                let name = after
                    .trim()
                    .trim_matches(|c: char| c == '?' || c == '.' || c == '!')
                    .trim();
                if !name.is_empty() {
                    return capitalize_words(name);
                }
            }
        }
        String::new()
    }

    /// Extract person name from query
    fn extract_person_name(&self, input: &str) -> String {
        // Look for possessive pattern first (X's email)
        if input.contains("'s ") {
            let parts: Vec<&str> = input.split("'s ").collect();
            if !parts.is_empty() {
                // Get the word(s) before 's
                let before = parts[0];
                let words: Vec<&str> = before.split_whitespace().collect();
                // Take last 1-2 words as the name
                let name = words.iter().rev().take(2).rev()
                    .map(|s| *s)
                    .collect::<Vec<_>>()
                    .join(" ");
                if !name.is_empty() {
                    return capitalize_words(&name);
                }
            }
        }
        
        // Look for common patterns
        let patterns = [
            "contact ", "reach ", "email ", "call ", "for ", "about ",
        ];
        
        for pattern in patterns {
            if let Some(pos) = input.rfind(pattern) {
                let after = &input[pos + pattern.len()..];
                let name = after
                    .trim()
                    .trim_matches(|c: char| c == '?' || c == '.' || c == '!')
                    .split_whitespace()
                    .take(2)  // Take first two words as name
                    .collect::<Vec<_>>()
                    .join(" ");
                if !name.is_empty() {
                    return capitalize_words(&name);
                }
            }
        }
        String::new()
    }

    /// Extract project name from query
    fn extract_project_name(&self, input: &str) -> String {
        let patterns = ["project ", "on ", "for "];
        
        for pattern in patterns {
            if let Some(pos) = input.find(pattern) {
                let after = &input[pos + pattern.len()..];
                let name = after
                    .trim()
                    .trim_matches(|c: char| c == '?' || c == '.' || c == '!')
                    .split(&['?', '.', '!', ','][..])
                    .next()
                    .unwrap_or("")
                    .trim();
                if !name.is_empty() {
                    return capitalize_words(name);
                }
            }
        }
        String::new()
    }

    /// Extract generic entity name
    fn extract_entity_name(&self, input: &str) -> String {
        let patterns = ["is ", "does ", "of ", "about ", "for "];
        
        for pattern in patterns {
            if let Some(pos) = input.rfind(pattern) {
                let after = &input[pos + pattern.len()..];
                let name = after
                    .trim()
                    .trim_matches(|c: char| c == '?' || c == '.' || c == '!' || c == '\'')
                    .split(&['?', '.', '!', ','][..])
                    .next()
                    .unwrap_or("")
                    .trim();
                if !name.is_empty() && !["work", "live", "stay"].contains(&name) {
                    return capitalize_words(name);
                }
            }
        }
        String::new()
    }

    /// Execute intent and return formatted response with action context
    pub fn execute_intent(&self, intent_result: &IntentResult) -> Result<IntentResponse> {
        match &intent_result.intent {
            Intent::ContactInfo { person_name } => self.get_contact_info(person_name),
            Intent::ReachabilityQuery { person_name } => self.get_reachability(person_name),
            Intent::LocationQuery { entity_name } => self.get_location_info(entity_name),
            Intent::ProjectQuery { project_name } => self.get_project_info(project_name),
            Intent::RelationshipQuery { entity_name } => self.get_relationship_info(entity_name),
            Intent::ActionRequest { action, entity_name } => {
                self.prepare_action(action, entity_name)
            }
            Intent::InformationQuery { topic } => self.get_general_info(topic),
            Intent::Unknown { raw_query } => Ok(IntentResponse {
                message: format!("I'm not sure what you're asking about '{}'. Try:\n\
                    • How can I contact [person]?\n\
                    • Where is [person] located?\n\
                    • Who works on [project]?\n\
                    • Send email to [person]", raw_query),
                action_context: None,
            }),
        }
    }

    /// Get contact information for a person
    fn get_contact_info(&self, person_name: &str) -> Result<IntentResponse> {
        if let Some(person) = self.graph.find_person(person_name)? {
            let props = self.graph.get_entity_properties(person.id)?;
            
            let mut contact_info = Vec::new();
            let mut action_context = ActionContext::new(person.name.clone());
            
            if let Some(email) = props.get("email") {
                contact_info.push(format!("📧 Email: {}", email));
                action_context.email = Some(email.clone());
            }
            if let Some(phone) = props.get("phone") {
                contact_info.push(format!("📱 Phone: {}", phone));
                action_context.phone = Some(phone.clone());
            }
            if let Some(role) = props.get("role") {
                contact_info.push(format!("💼 Role: {}", role));
            }
            
            // Get organization
            let rels = self.graph.get_entity_relationships(person.id)?;
            for (rel_type, target) in &rels {
                if rel_type == "WORKS_AT" {
                    contact_info.push(format!("🏢 Works at: {}", target));
                    action_context.organization = Some(target.clone());
                }
            }
            
            if contact_info.is_empty() {
                Ok(IntentResponse {
                    message: format!("I know {}, but I don't have their contact details yet.\n\
                        You can add them with: /set {} email <email>", person.name, person.name),
                    action_context: None,
                })
            } else {
                Ok(IntentResponse {
                    message: format!("📋 Contact info for {}:\n{}", 
                        person.name, contact_info.join("\n")),
                    action_context: Some(action_context),
                })
            }
        } else {
            Ok(IntentResponse {
                message: format!("I don't know anyone named '{}'. Add them with: /add person {}", 
                    person_name, person_name),
                action_context: None,
            })
        }
    }

    /// Get reachability info (similar to contact but more action-oriented)
    fn get_reachability(&self, person_name: &str) -> Result<IntentResponse> {
        self.get_contact_info(person_name)
    }

    /// Get location info for an entity
    fn get_location_info(&self, entity_name: &str) -> Result<IntentResponse> {
        if let Some(entity) = self.graph.find_by_name(entity_name)? {
            if let Some(location) = self.graph.get_location(&entity.name)? {
                let loc_props = self.graph.get_entity_properties(location.id)?;
                
                let mut location_info = vec![format!("📍 {}", location.name)];
                
                if let Some(country) = loc_props.get("country") {
                    location_info.push(format!("   Country: {}", country));
                }
                if let Some(tz) = loc_props.get("timezone") {
                    location_info.push(format!("   Timezone: {}", tz));
                }
                
                return Ok(IntentResponse {
                    message: format!("{} is located in:\n{}", entity.name, location_info.join("\n")),
                    action_context: None,
                });
            }
            
            Ok(IntentResponse {
                message: format!("I don't have location information for {}.", entity.name),
                action_context: None,
            })
        } else {
            Ok(IntentResponse {
                message: format!("I don't know about '{}'.", entity_name),
                action_context: None,
            })
        }
    }

    /// Get project info including team members
    fn get_project_info(&self, project_name: &str) -> Result<IntentResponse> {
        if let Some(project) = self.graph.find_project(project_name)? {
            let members = self.graph.get_project_members(&project.name)?;
            let props = self.graph.get_entity_properties(project.id)?;
            
            let mut info = vec![format!("📦 Project: {}", project.name)];
            
            if let Some(status) = props.get("status") {
                info.push(format!("   Status: {}", status));
            }
            if let Some(deadline) = props.get("deadline") {
                info.push(format!("   Deadline: {}", deadline));
            }
            if let Some(desc) = props.get("description") {
                info.push(format!("   Description: {}", desc));
            }
            
            if !members.is_empty() {
                info.push("\n👥 Team:".to_string());
                for member in members {
                    info.push(format!("   • {}", member.name));
                }
            }
            
            Ok(IntentResponse {
                message: info.join("\n"),
                action_context: None,
            })
        } else {
            Ok(IntentResponse {
                message: format!("I don't know about a project called '{}'. Add it with: /add project {}", 
                    project_name, project_name),
                action_context: None,
            })
        }
    }

    /// Get relationship/network info
    fn get_relationship_info(&self, entity_name: &str) -> Result<IntentResponse> {
        if let Some(entity) = self.graph.find_by_name(entity_name)? {
            let rels = self.graph.get_entity_relationships(entity.id)?;
            
            if rels.is_empty() {
                return Ok(IntentResponse {
                    message: format!("{} has no recorded relationships.", entity.name),
                    action_context: None,
                });
            }
            
            let mut info = vec![format!("🔗 Relationships for {}:", entity.name)];
            for (rel_type, target) in rels {
                let rel_display = rel_type.to_lowercase().replace('_', " ");
                info.push(format!("   • {} → {}", rel_display, target));
            }
            
            Ok(IntentResponse {
                message: info.join("\n"),
                action_context: None,
            })
        } else {
            Ok(IntentResponse {
                message: format!("I don't know about '{}'.", entity_name),
                action_context: None,
            })
        }
    }

    /// Prepare action with required context
    fn prepare_action(&self, action: &ActionType, entity_name: &str) -> Result<IntentResponse> {
        // First, get contact info
        let contact_result = self.get_contact_info(entity_name)?;
        
        match action {
            ActionType::SendEmail => {
                if let Some(ref ctx) = contact_result.action_context {
                    if let Some(ref email) = ctx.email {
                        return Ok(IntentResponse {
                            message: format!("Ready to send email to {}:\n📧 {}\n\n\
                                [Action: Open email client with address]", ctx.person_name, email),
                            action_context: contact_result.action_context,
                        });
                    }
                }
                Ok(IntentResponse {
                    message: format!("I don't have an email address for {}.\n\
                        Add it with: /set {} email <email>", entity_name, entity_name),
                    action_context: None,
                })
            }
            ActionType::MakeCall => {
                if let Some(ref ctx) = contact_result.action_context {
                    if let Some(ref phone) = ctx.phone {
                        return Ok(IntentResponse {
                            message: format!("Ready to call {}:\n📱 {}", ctx.person_name, phone),
                            action_context: contact_result.action_context,
                        });
                    }
                }
                Ok(IntentResponse {
                    message: format!("I don't have a phone number for {}.\n\
                        Add it with: /set {} phone <number>", entity_name, entity_name),
                    action_context: None,
                })
            }
            ActionType::ScheduleMeeting => {
                Ok(IntentResponse {
                    message: format!("To schedule a meeting with {}, I need:\n\
                        • Date and time\n\
                        • Meeting title\n\n\
                        Try: /event Meeting with {} at <date/time>", entity_name, entity_name),
                    action_context: contact_result.action_context,
                })
            }
            _ => {
                Ok(IntentResponse {
                    message: format!("I understand you want to {} with {}, but I need more information.", 
                        format!("{:?}", action).to_lowercase(), entity_name),
                    action_context: contact_result.action_context,
                })
            }
        }
    }

    /// Get general information about a topic
    fn get_general_info(&self, topic: &str) -> Result<IntentResponse> {
        // Try to find as any entity type
        if let Some(entity) = self.graph.find_by_name(topic)? {
            let props = self.graph.get_entity_properties(entity.id)?;
            let rels = self.graph.get_entity_relationships(entity.id)?;
            
            let mut info = vec![format!("ℹ️ {}", entity.name)];
            
            if !props.is_empty() {
                info.push("\nProperties:".to_string());
                for (key, value) in props {
                    info.push(format!("   • {}: {}", key, value));
                }
            }
            
            if !rels.is_empty() {
                info.push("\nRelationships:".to_string());
                for (rel_type, target) in rels {
                    info.push(format!("   • {} → {}", rel_type, target));
                }
            }
            
            return Ok(IntentResponse {
                message: info.join("\n"),
                action_context: None,
            });
        }
        
        Ok(IntentResponse {
            message: format!("I don't have information about '{}'.", topic),
            action_context: None,
        })
    }
}

/// Response from intent execution
#[derive(Debug, Clone)]
pub struct IntentResponse {
    pub message: String,
    pub action_context: Option<ActionContext>,
}

/// Context for executing actions
#[derive(Debug, Clone)]
pub struct ActionContext {
    pub person_name: String,
    pub email: Option<String>,
    pub phone: Option<String>,
    pub organization: Option<String>,
    pub location: Option<String>,
}

impl ActionContext {
    pub fn new(person_name: String) -> Self {
        Self {
            person_name,
            email: None,
            phone: None,
            organization: None,
            location: None,
        }
    }
}

/// Helper to capitalize words in a name
fn capitalize_words(s: &str) -> String {
    s.split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// ============================================================================
// Original Query Types (kept for backward compatibility)
// ============================================================================

/// Query types that can be detected from natural language
#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    /// Find people working at an organization
    WorksAt { org_name: String },
    /// Find what we know about an entity
    InfoAbout { entity_name: String },
    /// Find people who know someone
    KnowsPerson { person_name: String },
    /// Find all relationships for an entity
    Relationships { entity_name: String },
    /// Find members of an organization
    MembersOf { org_name: String },
    /// List all people
    ListPeople,
    /// List all organizations
    ListOrgs,
    /// General search
    Search { term: String },
    /// Unknown query type
    Unknown { query: String },
}

/// Natural language query processor
pub struct NaturalLanguageQuery<'a> {
    graph: &'a KnowledgeGraph,
}

impl<'a> NaturalLanguageQuery<'a> {
    pub fn new(graph: &'a KnowledgeGraph) -> Self {
        Self { graph }
    }

    /// Parse a natural language query into a QueryType
    pub fn parse_query(&self, input: &str) -> QueryType {
        let lower = input.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        
        // "Who works at X?"
        if (lower.contains("who") && lower.contains("works at")) ||
           (lower.contains("who") && lower.contains("work at")) {
            if let Some(org) = self.extract_after_phrase(&lower, "works at")
                .or_else(|| self.extract_after_phrase(&lower, "work at")) {
                return QueryType::WorksAt { org_name: org };
            }
        }
        
        // "Who knows X?"
        if lower.contains("who") && lower.contains("knows") {
            if let Some(person) = self.extract_after_phrase(&lower, "knows") {
                return QueryType::KnowsPerson { person_name: person };
            }
        }
        
        // "What do you/I know about X?"
        if (lower.contains("what") && lower.contains("know about")) ||
           lower.contains("tell me about") ||
           lower.contains("info about") ||
           lower.contains("information about") {
            let phrases = ["know about", "tell me about", "info about", "information about"];
            for phrase in phrases {
                if let Some(entity) = self.extract_after_phrase(&lower, phrase) {
                    return QueryType::InfoAbout { entity_name: entity };
                }
            }
        }
        
        // "Who is in X?" or "Members of X"
        if (lower.contains("who") && lower.contains("in ")) ||
           lower.contains("members of") ||
           lower.contains("people at") {
            let phrases = ["who is in", "who are in", "members of", "people at"];
            for phrase in phrases {
                if let Some(org) = self.extract_after_phrase(&lower, phrase) {
                    return QueryType::MembersOf { org_name: org };
                }
            }
        }
        
        // "Relationships for X" or "connections of X"
        if lower.contains("relationships") || lower.contains("connections") {
            let phrases = ["relationships for", "relationships of", "connections for", "connections of"];
            for phrase in phrases {
                if let Some(entity) = self.extract_after_phrase(&lower, phrase) {
                    return QueryType::Relationships { entity_name: entity };
                }
            }
        }
        
        // "List all people" or "show people"
        if (lower.contains("list") || lower.contains("show") || lower.contains("all")) &&
           (lower.contains("people") || lower.contains("persons") || lower.contains("contacts")) {
            return QueryType::ListPeople;
        }
        
        // "List all organizations" or "show orgs"
        if (lower.contains("list") || lower.contains("show") || lower.contains("all")) &&
           (lower.contains("orgs") || lower.contains("organizations") || lower.contains("companies")) {
            return QueryType::ListOrgs;
        }
        
        // Generic search - look for entity names
        if words.len() >= 2 {
            // Try last word(s) as search term
            let search_term = words.last().map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()));
            if let Some(term) = search_term {
                if !term.is_empty() && term.len() > 2 {
                    return QueryType::Search { term: term.to_string() };
                }
            }
        }
        
        QueryType::Unknown { query: input.to_string() }
    }

    /// Extract text after a phrase
    fn extract_after_phrase(&self, text: &str, phrase: &str) -> Option<String> {
        text.find(phrase).map(|pos| {
            text[pos + phrase.len()..]
                .trim()
                .trim_matches(|c: char| c == '?' || c == '.' || c == '!')
                .trim()
                .to_string()
        }).filter(|s| !s.is_empty())
    }

    /// Execute a parsed query against the knowledge graph
    pub fn execute(&self, query_type: &QueryType) -> Result<String> {
        match query_type {
            QueryType::WorksAt { org_name } => self.query_works_at(org_name),
            QueryType::InfoAbout { entity_name } => self.query_info(entity_name),
            QueryType::KnowsPerson { person_name } => self.query_knows(person_name),
            QueryType::Relationships { entity_name } => self.query_relationships(entity_name),
            QueryType::MembersOf { org_name } => self.query_members(org_name),
            QueryType::ListPeople => self.list_people(),
            QueryType::ListOrgs => self.list_orgs(),
            QueryType::Search { term } => self.search(term),
            QueryType::Unknown { query } => Ok(format!("I'm not sure how to answer '{}'. Try asking:\n\
                • Who works at [company]?\n\
                • What do you know about [person]?\n\
                • Who knows [person]?\n\
                • List all people/organizations", query)),
        }
    }

    /// Find people who work at an organization
    fn query_works_at(&self, org_name: &str) -> Result<String> {
        // Find the organization
        if let Some(org) = self.graph.find_organization(org_name)? {
            let people = self.graph.get_employees(&org.name)?;
            
            if people.is_empty() {
                Ok(format!("I don't know anyone who works at {}.", org.name))
            } else {
                let mut result = format!("People who work at {}:\n", org.name);
                for person in people {
                    result.push_str(&format!("  • {}\n", person.name));
                }
                Ok(result)
            }
        } else {
            Ok(format!("I don't know about an organization called '{}'.", org_name))
        }
    }

    /// Get information about an entity
    fn query_info(&self, entity_name: &str) -> Result<String> {
        // Try person first
        if let Some(person) = self.graph.find_person(entity_name)? {
            let properties = self.graph.get_entity_properties(person.id)?;
            let relationships = self.graph.get_entity_relationships(person.id)?;
            
            let mut result = format!("📋 Information about {}:\n\n", person.name);
            
            if !properties.is_empty() {
                result.push_str("Properties:\n");
                for (key, value) in &properties {
                    result.push_str(&format!("  • {}: {}\n", key, value));
                }
            }
            
            if !relationships.is_empty() {
                result.push_str("\nRelationships:\n");
                for rel in &relationships {
                    result.push_str(&format!("  • {} → {}\n", rel.0, rel.1));
                }
            }
            
            if properties.is_empty() && relationships.is_empty() {
                result.push_str("  (No additional details stored)\n");
            }
            
            return Ok(result);
        }
        
        // Try organization
        if let Some(org) = self.graph.find_organization(entity_name)? {
            let properties = self.graph.get_entity_properties(org.id)?;
            let employees = self.graph.get_employees(&org.name)?;
            
            let mut result = format!("🏢 Information about {}:\n\n", org.name);
            
            if !properties.is_empty() {
                result.push_str("Properties:\n");
                for (key, value) in &properties {
                    result.push_str(&format!("  • {}: {}\n", key, value));
                }
            }
            
            if !employees.is_empty() {
                result.push_str("\nEmployees:\n");
                for emp in &employees {
                    result.push_str(&format!("  • {}\n", emp.name));
                }
            }
            
            if properties.is_empty() && employees.is_empty() {
                result.push_str("  (No additional details stored)\n");
            }
            
            return Ok(result);
        }
        
        Ok(format!("I don't have any information about '{}'.", entity_name))
    }

    /// Find people who know someone
    fn query_knows(&self, person_name: &str) -> Result<String> {
        if let Some(person) = self.graph.find_person(person_name)? {
            let connections = self.graph.get_connections(&person.name)?;
            
            if connections.is_empty() {
                Ok(format!("I don't know who knows {}.", person.name))
            } else {
                let mut result = format!("People connected to {}:\n", person.name);
                for conn in connections {
                    result.push_str(&format!("  • {}\n", conn.name));
                }
                Ok(result)
            }
        } else {
            Ok(format!("I don't know anyone named '{}'.", person_name))
        }
    }

    /// Get all relationships for an entity
    fn query_relationships(&self, entity_name: &str) -> Result<String> {
        // Try to find as person
        if let Some(person) = self.graph.find_person(entity_name)? {
            let rels = self.graph.get_entity_relationships(person.id)?;
            
            if rels.is_empty() {
                return Ok(format!("{} has no recorded relationships.", person.name));
            }
            
            let mut result = format!("Relationships for {}:\n", person.name);
            for (rel_type, target) in rels {
                result.push_str(&format!("  • {} → {}\n", rel_type, target));
            }
            return Ok(result);
        }
        
        // Try organization
        if let Some(org) = self.graph.find_organization(entity_name)? {
            let rels = self.graph.get_entity_relationships(org.id)?;
            
            if rels.is_empty() {
                return Ok(format!("{} has no recorded relationships.", org.name));
            }
            
            let mut result = format!("Relationships for {}:\n", org.name);
            for (rel_type, target) in rels {
                result.push_str(&format!("  • {} → {}\n", rel_type, target));
            }
            return Ok(result);
        }
        
        Ok(format!("I don't know about '{}'.", entity_name))
    }

    /// Get members of an organization
    fn query_members(&self, org_name: &str) -> Result<String> {
        // Same as works_at for now
        self.query_works_at(org_name)
    }

    /// List all people
    fn list_people(&self) -> Result<String> {
        let people = self.graph.list_people()?;
        
        if people.is_empty() {
            Ok("No people in the knowledge graph yet.".to_string())
        } else {
            let mut result = format!("👥 {} people:\n", people.len());
            for person in people {
                result.push_str(&format!("  • {}\n", person.name));
            }
            Ok(result)
        }
    }

    /// List all organizations
    fn list_orgs(&self) -> Result<String> {
        let orgs = self.graph.list_organizations()?;
        
        if orgs.is_empty() {
            Ok("No organizations in the knowledge graph yet.".to_string())
        } else {
            let mut result = format!("🏢 {} organizations:\n", orgs.len());
            for org in orgs {
                result.push_str(&format!("  • {}\n", org.name));
            }
            Ok(result)
        }
    }

    /// Search for entities matching a term
    fn search(&self, term: &str) -> Result<String> {
        let mut results = Vec::new();
        
        // Search people
        for person in self.graph.list_people()? {
            if person.name.to_lowercase().contains(&term.to_lowercase()) {
                results.push(format!("👤 {}", person.name));
            }
        }
        
        // Search organizations
        for org in self.graph.list_organizations()? {
            if org.name.to_lowercase().contains(&term.to_lowercase()) {
                results.push(format!("🏢 {}", org.name));
            }
        }
        
        if results.is_empty() {
            Ok(format!("No results found for '{}'.", term))
        } else {
            let mut output = format!("Search results for '{}':\n", term);
            for r in results {
                output.push_str(&format!("  • {}\n", r));
            }
            Ok(output)
        }
    }

    /// Check if a query is a knowledge graph question
    pub fn is_kg_query(&self, input: &str) -> bool {
        let lower = input.to_lowercase();
        
        // Direct indicators of KG queries
        let kg_indicators = [
            "who works at", "who work at",
            "who knows", 
            "what do you know about", "what do i know about",
            "tell me about",
            "info about", "information about",
            "who is in", "who are in",
            "members of", "people at",
            "relationships for", "connections of",
            "list all people", "list people",
            "list all org", "show org",
            "list contacts", "show contacts",
        ];
        
        kg_indicators.iter().any(|indicator| lower.contains(indicator))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Database;

    fn test_graph() -> KnowledgeGraph {
        let db = Database::in_memory().unwrap();
        KnowledgeGraph::new(db)
    }

    #[test]
    fn test_parse_works_at() {
        // We can't test with actual graph, but we can test the parse logic
        let input = "who works at google?";
        let lower = input.to_lowercase();
        
        // Simulate the extraction
        let phrase = "works at";
        let pos = lower.find(phrase).unwrap();
        let extracted = lower[pos + phrase.len()..].trim().trim_matches('?').to_string();
        
        assert_eq!(extracted, "google");
    }

    #[test]
    fn test_parse_knows() {
        let input = "who knows alice?";
        let lower = input.to_lowercase();
        
        let phrase = "knows";
        let pos = lower.find(phrase).unwrap();
        let extracted = lower[pos + phrase.len()..].trim().trim_matches('?').to_string();
        
        assert_eq!(extracted, "alice");
    }

    #[test]
    fn test_is_kg_query_indicators() {
        let queries = vec![
            ("who works at google", true),
            ("what do you know about alice", true),
            ("tell me about bob", true),
            ("list all people", true),
            ("what's the weather", false),
            ("hello there", false),
        ];
        
        for (query, expected) in queries {
            let lower = query.to_lowercase();
            let indicators = ["who works at", "what do you know about", "tell me about", "list all people"];
            let is_kg = indicators.iter().any(|i| lower.contains(i));
            assert_eq!(is_kg, expected, "Failed for: {}", query);
        }
    }

    #[test]
    fn test_intent_router_contact_intent() {
        let graph = test_graph();
        let router = IntentRouter::new(&graph);
        
        let queries = [
            "how can I contact alice",
            "contact info for bob",
            "what is charlie's email address",
        ];
        
        for query in queries {
            let result = router.classify(query);
            match result.intent {
                Intent::ContactInfo { .. } | Intent::ReachabilityQuery { .. } => {},
                _ => panic!("Expected ContactInfo or ReachabilityQuery for: {} (got {:?})", query, result.intent),
            }
        }
    }

    #[test]
    fn test_intent_router_action_intent() {
        let graph = test_graph();
        let router = IntentRouter::new(&graph);
        
        let result = router.classify("send email to alice");
        match result.intent {
            Intent::ActionRequest { action: ActionType::SendEmail, .. } => {},
            _ => panic!("Expected SendEmail action"),
        }
        
        let result = router.classify("schedule a meeting with bob");
        match result.intent {
            Intent::ActionRequest { action: ActionType::ScheduleMeeting, .. } => {},
            _ => panic!("Expected ScheduleMeeting action"),
        }
    }

    #[test]
    fn test_intent_router_location_intent() {
        let graph = test_graph();
        let router = IntentRouter::new(&graph);
        
        let result = router.classify("where is alice located");
        match result.intent {
            Intent::LocationQuery { .. } => {},
            _ => panic!("Expected LocationQuery"),
        }
    }

    #[test]
    fn test_intent_router_project_intent() {
        let graph = test_graph();
        let router = IntentRouter::new(&graph);
        
        let result = router.classify("who is working on project alpha");
        match result.intent {
            Intent::ProjectQuery { .. } => {},
            _ => panic!("Expected ProjectQuery"),
        }
    }
}
