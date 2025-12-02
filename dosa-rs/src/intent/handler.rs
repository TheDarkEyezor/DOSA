//! Intent handler - routes classified intents to appropriate actions
//!
//! This module takes the classified intent and executes the corresponding action.

use super::types::*;
use super::hybrid::HybridClassifier;
use super::IntentClassifier;
use crate::integrations::{
    OAuthManager, Provider, CalendarAI, GoogleCalendarClient, 
    GmailClient, ImportanceScorer, EmailAI,
};
use crate::knowledge::{KnowledgeGraph, RelationshipType};
use crate::llm::OllamaClient;
use crate::router::phase6_commands;
use anyhow::Result;
use std::sync::Arc;
use std::path::Path;
use serde::Deserialize;

/// Actions for contact updates parsed from LLM response
#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum ContactAction {
    AddPerson { name: String },
    SetProperty { person: String, property: String, value: String },
    AddRelationship { person: String, relationship: String, target: String },
}

/// Result of handling an intent
pub enum HandleResult {
    /// Intent was handled successfully with output
    Handled(String),
    /// Intent requires authentication
    NeedsAuth(String),
    /// Intent was not handled (fallback to conversation)
    NotHandled,
    /// Error occurred
    Error(String),
}

/// Intent handler that routes intents to actions
pub struct IntentHandler<'a> {
    llm: &'a OllamaClient,
    graph: &'a KnowledgeGraph,
    classifier: HybridClassifier,
}

impl<'a> IntentHandler<'a> {
    pub fn new(llm: &'a OllamaClient, graph: &'a KnowledgeGraph) -> Self {
        IntentHandler {
            llm,
            graph,
            classifier: HybridClassifier::new(Arc::new(llm.clone())),
        }
    }

    /// Create intent handler with ONNX model support
    pub fn with_data_dir(llm: &'a OllamaClient, graph: &'a KnowledgeGraph, data_dir: &Path) -> Self {
        let classifier = HybridClassifier::try_with_onnx(Arc::new(llm.clone()), data_dir);
        IntentHandler {
            llm,
            graph,
            classifier,
        }
    }

    /// Classify and handle an input
    pub async fn handle(&self, input: &str) -> Result<HandleResult> {
        // Classify the intent
        let ctx = IntentContext::new(self.is_google_authenticated().await);
        let result = self.classifier.classify(input, &ctx).await?;
        
        // Handle based on intent type
        self.handle_intent(&result.intent, input).await
    }

    /// Handle a specific intent
    async fn handle_intent(&self, intent: &Intent, input: &str) -> Result<HandleResult> {
        match intent {
            Intent::Command(cmd) => {
                // Commands are handled by the router, not here
                Ok(HandleResult::NotHandled)
            }

            // Calendar intents
            Intent::CalendarCreate { .. } => {
                self.handle_calendar_create(input).await
            }
            Intent::CalendarQuery { query_type } => {
                self.handle_calendar_query(query_type).await
            }
            Intent::CalendarUpdate { .. } => {
                self.handle_calendar_update(input).await
            }
            Intent::CalendarDelete { .. } => {
                self.handle_calendar_delete(input).await
            }

            // Email intents
            Intent::EmailCompose { .. } => {
                self.handle_email_compose(input).await
            }
            Intent::EmailReply { .. } => {
                self.handle_email_reply(input).await
            }
            Intent::EmailAttendees { .. } => {
                self.handle_email_attendees(input).await
            }
            Intent::EmailQuery { query_type } => {
                self.handle_email_query(query_type).await
            }

            // Contact/Knowledge intents
            Intent::ContactUpdate { .. } => {
                self.handle_contact_update(input).await
            }

            // Knowledge graph query
            Intent::KnowledgeQuery { .. } => {
                // Handled separately by NaturalLanguageQuery
                Ok(HandleResult::NotHandled)
            }

            // Conversation fallback
            Intent::Conversation { .. } => {
                Ok(HandleResult::NotHandled)
            }

            // Ambiguous - could add disambiguation UI
            Intent::Ambiguous { .. } => {
                Ok(HandleResult::NotHandled)
            }
        }
    }

    /// Check if authenticated with Google
    async fn is_google_authenticated(&self) -> bool {
        let oauth = OAuthManager::new();
        oauth.load_stored_tokens().await;
        oauth.is_authenticated(Provider::Google).await
    }

    /// Get OAuth manager with loaded tokens
    async fn get_oauth(&self) -> Result<Arc<OAuthManager>> {
        let oauth = Arc::new(OAuthManager::new());
        oauth.load_stored_tokens().await;
        
        if !oauth.is_authenticated(Provider::Google).await {
            return Err(anyhow::anyhow!("Not authenticated"));
        }
        
        Ok(oauth)
    }

    // Calendar handlers

    async fn handle_calendar_create(&self, input: &str) -> Result<HandleResult> {
        let oauth = match self.get_oauth().await {
            Ok(o) => o,
            Err(_) => return Ok(HandleResult::NeedsAuth("calendar".to_string())),
        };

        let calendar_ai = CalendarAI::new(self.llm, self.graph, oauth);
        
        match calendar_ai.create_from_natural_language(input).await {
            Ok(result) => {
                let mut output = result.display_preview();
                
                if result.has_unresolved() {
                    output.push_str("\n\n⚠️  Some attendees don't have emails.");
                }
                
                output.push_str("\n\n💡 To confirm, use: /gcal confirm");
                output.push_str("\n   To cancel: /gcal cancel");
                
                phase6_commands::store_pending_event(result.event);
                
                Ok(HandleResult::Handled(output))
            }
            Err(e) => Ok(HandleResult::Error(format!("Could not parse calendar event: {}", e)))
        }
    }

    async fn handle_calendar_query(&self, query_type: &CalendarQueryType) -> Result<HandleResult> {
        let oauth = match self.get_oauth().await {
            Ok(o) => o,
            Err(_) => return Ok(HandleResult::NeedsAuth("calendar".to_string())),
        };

        let client = GoogleCalendarClient::new(oauth);

        match query_type {
            CalendarQueryType::Today => {
                match client.get_today_events().await {
                    Ok(events) => {
                        if events.is_empty() {
                            Ok(HandleResult::Handled("📅 No events scheduled for today.".to_string()))
                        } else {
                            let mut output = format!("📅 Today's Events ({}):\n", events.len());
                            for event in &events {
                                output.push_str(&format!("{}\n", event.display()));
                            }
                            Ok(HandleResult::Handled(output))
                        }
                    }
                    Err(e) => Ok(HandleResult::Error(format!("Could not fetch calendar: {}", e)))
                }
            }
            CalendarQueryType::Tomorrow => {
                match client.get_tomorrow_events().await {
                    Ok(events) => {
                        if events.is_empty() {
                            Ok(HandleResult::Handled("📅 No events scheduled for tomorrow.".to_string()))
                        } else {
                            let mut output = format!("📅 Tomorrow's Events ({}):\n", events.len());
                            for event in &events {
                                output.push_str(&format!("{}\n", event.display()));
                            }
                            Ok(HandleResult::Handled(output))
                        }
                    }
                    Err(e) => Ok(HandleResult::Error(format!("Could not fetch calendar: {}", e)))
                }
            }
            CalendarQueryType::Week | CalendarQueryType::Upcoming => {
                match client.get_week_summary().await {
                    Ok(summary) => Ok(HandleResult::Handled(summary)),
                    Err(e) => Ok(HandleResult::Error(format!("Could not fetch calendar: {}", e)))
                }
            }
            CalendarQueryType::Date(date) => {
                match client.get_day_summary(*date).await {
                    Ok(summary) => Ok(HandleResult::Handled(summary)),
                    Err(e) => Ok(HandleResult::Error(format!("Could not fetch calendar: {}", e)))
                }
            }
        }
    }

    async fn handle_calendar_update(&self, input: &str) -> Result<HandleResult> {
        let oauth = match self.get_oauth().await {
            Ok(o) => o,
            Err(_) => return Ok(HandleResult::NeedsAuth("calendar".to_string())),
        };

        let calendar_ai = CalendarAI::new(self.llm, self.graph, oauth);
        
        match calendar_ai.prepare_update(input).await {
            Ok(result) => {
                let mut output = result.display_preview();
                output.push_str("\n💡 To confirm, use: /gcal confirm");
                output.push_str("\n   To cancel: /gcal cancel");
                
                phase6_commands::store_pending_update(
                    result.target_event.id.clone(),
                    result.update
                );
                
                Ok(HandleResult::Handled(output))
            }
            Err(e) => Ok(HandleResult::Error(format!("Could not prepare update: {}", e)))
        }
    }

    async fn handle_calendar_delete(&self, input: &str) -> Result<HandleResult> {
        let oauth = match self.get_oauth().await {
            Ok(o) => o,
            Err(_) => return Ok(HandleResult::NeedsAuth("calendar".to_string())),
        };

        let calendar_ai = CalendarAI::new(self.llm, self.graph, oauth);
        
        match calendar_ai.prepare_delete(input).await {
            Ok(result) => {
                let mut output = result.display_preview();
                output.push_str("\n💡 To confirm deletion, use: /gcal confirm");
                output.push_str("\n   To cancel: /gcal cancel");
                
                phase6_commands::store_pending_delete(result.target_event.id.clone());
                
                Ok(HandleResult::Handled(output))
            }
            Err(e) => Ok(HandleResult::Error(format!("Could not find event: {}", e)))
        }
    }

    // Email handlers

    async fn handle_email_compose(&self, input: &str) -> Result<HandleResult> {
        let oauth = match self.get_oauth().await {
            Ok(o) => o,
            Err(_) => return Ok(HandleResult::NeedsAuth("email".to_string())),
        };

        let email_ai = EmailAI::new(self.llm, self.graph, oauth);
        
        match email_ai.compose_email(input).await {
            Ok(result) => {
                let mut output = result.display_preview();
                output.push_str("\n💡 To send, use: /email confirm");
                output.push_str("\n   To cancel: /email cancel");
                
                phase6_commands::store_pending_email(result);
                
                Ok(HandleResult::Handled(output))
            }
            Err(e) => Ok(HandleResult::Error(format!("Could not compose email: {}", e)))
        }
    }

    async fn handle_email_reply(&self, input: &str) -> Result<HandleResult> {
        let oauth = match self.get_oauth().await {
            Ok(o) => o,
            Err(_) => return Ok(HandleResult::NeedsAuth("email".to_string())),
        };

        let email_ai = EmailAI::new(self.llm, self.graph, oauth);
        
        match email_ai.compose_reply_from_input(input).await {
            Ok(result) => {
                let mut output = result.display_preview();
                output.push_str("\n💡 To send reply, use: /email confirm");
                output.push_str("\n   To cancel: /email cancel");
                
                phase6_commands::store_pending_email(result);
                
                Ok(HandleResult::Handled(output))
            }
            Err(e) => Ok(HandleResult::Error(format!("Could not compose reply: {}", e)))
        }
    }

    async fn handle_email_attendees(&self, input: &str) -> Result<HandleResult> {
        let oauth = match self.get_oauth().await {
            Ok(o) => o,
            Err(_) => return Ok(HandleResult::NeedsAuth("email".to_string())),
        };

        let email_ai = EmailAI::new(self.llm, self.graph, oauth);
        
        match email_ai.compose_for_attendees(input).await {
            Ok(result) => {
                let mut output = result.display_preview();
                output.push_str("\n💡 To send, use: /email confirm");
                output.push_str("\n   To cancel: /email cancel");
                
                phase6_commands::store_pending_email(result);
                
                Ok(HandleResult::Handled(output))
            }
            Err(e) => Ok(HandleResult::Error(format!("Could not compose email for attendees: {}", e)))
        }
    }

    async fn handle_email_query(&self, query_type: &EmailQueryType) -> Result<HandleResult> {
        let oauth = match self.get_oauth().await {
            Ok(o) => o,
            Err(_) => return Ok(HandleResult::NeedsAuth("email".to_string())),
        };

        let client = GmailClient::new(oauth.clone());

        match query_type {
            EmailQueryType::Unread => {
                match client.get_unread(10).await {
                    Ok(emails) => {
                        if emails.is_empty() {
                            Ok(HandleResult::Handled("📧 No unread emails!".to_string()))
                        } else {
                            let mut output = format!("📧 Unread Emails ({}):\n", emails.len());
                            for email in &emails {
                                output.push_str(&format!("  {}\n", email.display_short()));
                            }
                            Ok(HandleResult::Handled(output))
                        }
                    }
                    Err(e) => Ok(HandleResult::Error(format!("Could not fetch emails: {}", e)))
                }
            }
            EmailQueryType::List => {
                match client.list_messages(None, 10).await {
                    Ok(emails) => {
                        if emails.is_empty() {
                            Ok(HandleResult::Handled("📧 No emails found.".to_string()))
                        } else {
                            let mut output = format!("📧 Recent Emails ({}):\n", emails.len());
                            for email in &emails {
                                output.push_str(&format!("  {}\n", email.display_short()));
                            }
                            Ok(HandleResult::Handled(output))
                        }
                    }
                    Err(e) => Ok(HandleResult::Error(format!("Could not fetch emails: {}", e)))
                }
            }
            EmailQueryType::From(sender) => {
                match client.get_from(sender, 10).await {
                    Ok(emails) => {
                        if emails.is_empty() {
                            Ok(HandleResult::Handled(format!("📧 No emails from {}.", sender)))
                        } else {
                            let mut output = format!("📧 Emails from {} ({}):\n", sender, emails.len());
                            for email in &emails {
                                output.push_str(&format!("  {}\n", email.display_short()));
                            }
                            Ok(HandleResult::Handled(output))
                        }
                    }
                    Err(e) => Ok(HandleResult::Error(format!("Could not fetch emails: {}", e)))
                }
            }
            EmailQueryType::Summary => {
                match client.get_unread(20).await {
                    Ok(emails) => {
                        if emails.is_empty() {
                            Ok(HandleResult::Handled("📧 No unread emails to summarize!".to_string()))
                        } else {
                            let scorer = ImportanceScorer::new(self.llm, self.graph);
                            match scorer.summarize_emails(&emails).await {
                                Ok(summary) => {
                                    Ok(HandleResult::Handled(format!(
                                        "📧 Email Summary ({} unread):\n\n{}",
                                        emails.len(), summary
                                    )))
                                }
                                Err(e) => Ok(HandleResult::Error(format!("Could not summarize: {}", e)))
                            }
                        }
                    }
                    Err(e) => Ok(HandleResult::Error(format!("Could not fetch emails: {}", e)))
                }
            }
        }
    }

    // Contact/Knowledge handlers

    async fn handle_contact_update(&self, input: &str) -> Result<HandleResult> {
        // Use LLM to parse the contact update request
        let prompt = format!(r#"Parse this contact management request and extract the actions to perform.

Request: "{}"

Return JSON with an array of actions. Each action can be:
1. Add a person: {{"action": "add_person", "name": "Full Name"}}
2. Set a property: {{"action": "set_property", "person": "Name", "property": "email", "value": "email@example.com"}}
3. Add relationship: {{"action": "add_relationship", "person": "Name", "relationship": "works_at", "target": "Company Name"}}

Relationship types: works_at, studies_at, manages, knows, works_on

Examples:
- "Add John Smith as a person I know" -> [{{"action": "add_person", "name": "John Smith"}}]
- "John's email is john@example.com" -> [{{"action": "set_property", "person": "John", "property": "email", "value": "john@example.com"}}]
- "Add Sarah and Mike, they work at Google" -> [
    {{"action": "add_person", "name": "Sarah"}},
    {{"action": "add_person", "name": "Mike"}},
    {{"action": "add_relationship", "person": "Sarah", "relationship": "works_at", "target": "Google"}},
    {{"action": "add_relationship", "person": "Mike", "relationship": "works_at", "target": "Google"}}
  ]

Respond with ONLY valid JSON array:"#,
            input
        );

        let response = self.llm.query(&prompt).await?;
        
        // Extract JSON from response
        let json_str = self.extract_json_array(&response);
        
        let actions: Vec<ContactAction> = match serde_json::from_str(&json_str) {
            Ok(a) => a,
            Err(e) => {
                // Try direct parsing as fallback
                if let Some(actions) = self.try_parse_contact_directly(input) {
                    actions
                } else {
                    return Ok(HandleResult::Error(format!(
                        "Could not parse contact update: {}. Try using commands like /add person <name> or /set <name> email <email>",
                        e
                    )));
                }
            }
        };

        if actions.is_empty() {
            // Try direct parsing before giving up
            if let Some(actions) = self.try_parse_contact_directly(input) {
                if !actions.is_empty() {
                    return self.execute_contact_actions(&actions).await;
                }
            }
            return Ok(HandleResult::Error(
                "Could not understand the contact update. Try: /add person <name> or /set <name> email <email>".to_string()
            ));
        }

        self.execute_contact_actions(&actions).await
    }

    /// Try to parse contact updates directly without LLM
    fn try_parse_contact_directly(&self, input: &str) -> Option<Vec<ContactAction>> {
        let lower = input.to_lowercase();
        let mut actions = Vec::new();
        
        // Pattern: "<Name>'s email is <email>"
        if let Some(captures) = regex::Regex::new(r"(?i)(\w+)(?:'s|s)?\s+email\s+(?:is\s+)?([^\s]+@[^\s]+)")
            .ok()
            .and_then(|re| re.captures(input))
        {
            let name = captures.get(1)?.as_str().to_string();
            let email = captures.get(2)?.as_str().to_string();
            actions.push(ContactAction::SetProperty {
                person: name,
                property: "email".to_string(),
                value: email,
            });
            return Some(actions);
        }
        
        // Pattern: "add <name> as someone I know" or "<name> is someone I know"
        if lower.contains("someone i know") || lower.contains("person i know") {
            // Extract name - try to find it before "is" or "as"
            let name_pattern = regex::Regex::new(r"(?i)(?:add\s+)?(\w+(?:\s+\w+)?)\s+(?:is|as)\s+(?:a\s+)?(?:someone|person)")
                .ok()?;
            if let Some(cap) = name_pattern.captures(input) {
                let name = cap.get(1)?.as_str().to_string();
                actions.push(ContactAction::AddPerson { name });
                return Some(actions);
            }
        }
        
        // Pattern: "add <name>" at the start
        if lower.starts_with("add ") {
            let rest = &input[4..].trim();
            // Skip if it looks like a command
            if !rest.starts_with("person") && !rest.contains("email") && !rest.is_empty() {
                let name = rest.split(|c: char| !c.is_alphanumeric() && c != ' ')
                    .next()
                    .unwrap_or(rest)
                    .trim()
                    .to_string();
                if !name.is_empty() {
                    actions.push(ContactAction::AddPerson { name });
                    return Some(actions);
                }
            }
        }
        
        None
    }

    /// Execute contact actions (separated out for reuse)
    async fn execute_contact_actions(&self, actions: &[ContactAction]) -> Result<HandleResult> {
        let mut results = Vec::new();
        
        for action in actions {
            match action {
                ContactAction::AddPerson { name } => {
                    match self.graph.add_person(name.as_str()) {
                        Ok(entity) => {
                            results.push(format!("✓ Added person: {} (id: {})", entity.name, entity.id));
                        }
                        Err(e) => {
                            results.push(format!("✗ Could not add {}: {}", name, e));
                        }
                    }
                }
                ContactAction::SetProperty { person, property, value } => {
                    // Find the person first
                    match self.graph.find_person(person.as_str()) {
                        Ok(Some(entity)) => {
                            match self.graph.database().set_entity_property(entity.id, property.as_str(), value.as_str()) {
                                Ok(_) => {
                                    results.push(format!("✓ Set {}'s {} to {}", person, property, value));
                                }
                                Err(e) => {
                                    results.push(format!("✗ Could not set property: {}", e));
                                }
                            }
                        }
                        Ok(None) => {
                            results.push(format!("✗ Person '{}' not found. Add them first.", person));
                        }
                        Err(e) => {
                            results.push(format!("✗ Error finding person: {}", e));
                        }
                    }
                }
                ContactAction::AddRelationship { person, relationship, target } => {
                    // Find the person
                    let person_entity = match self.graph.find_person(person.as_str()) {
                        Ok(Some(e)) => e,
                        Ok(None) => {
                            results.push(format!("✗ Person '{}' not found", person));
                            continue;
                        }
                        Err(e) => {
                            results.push(format!("✗ Error: {}", e));
                            continue;
                        }
                    };

                    // Find or create the target entity
                    let target_entity = match relationship.as_str() {
                        "works_at" | "studies_at" => {
                            match self.graph.find_organization(target.as_str()) {
                                Ok(Some(e)) => e,
                                Ok(None) => {
                                    // Create the organization
                                    match self.graph.add_organization(target.as_str()) {
                                        Ok(e) => {
                                            results.push(format!("  (Created organization: {})", target));
                                            e
                                        }
                                        Err(e) => {
                                            results.push(format!("✗ Could not create org: {}", e));
                                            continue;
                                        }
                                    }
                                }
                                Err(e) => {
                                    results.push(format!("✗ Error: {}", e));
                                    continue;
                                }
                            }
                        }
                        "works_on" => {
                            match self.graph.find_project(target.as_str()) {
                                Ok(Some(e)) => e,
                                Ok(None) => {
                                    match self.graph.add_project(target.as_str()) {
                                        Ok(e) => {
                                            results.push(format!("  (Created project: {})", target));
                                            e
                                        }
                                        Err(e) => {
                                            results.push(format!("✗ Could not create project: {}", e));
                                            continue;
                                        }
                                    }
                                }
                                Err(e) => {
                                    results.push(format!("✗ Error: {}", e));
                                    continue;
                                }
                            }
                        }
                        "knows" | "manages" => {
                            match self.graph.find_person(target.as_str()) {
                                Ok(Some(e)) => e,
                                Ok(None) => {
                                    results.push(format!("✗ Person '{}' not found", target));
                                    continue;
                                }
                                Err(e) => {
                                    results.push(format!("✗ Error: {}", e));
                                    continue;
                                }
                            }
                        }
                        _ => {
                            results.push(format!("✗ Unknown relationship type: {}", relationship));
                            continue;
                        }
                    };

                    // Convert string to RelationshipType
                    let rel_type = match relationship.as_str() {
                        "works_at" => RelationshipType::WorksAt,
                        "studies_at" => RelationshipType::MemberOf, // Use MemberOf for students
                        "works_on" => RelationshipType::WorksOn,
                        "knows" => RelationshipType::Knows,
                        "manages" => RelationshipType::Manages,
                        _ => RelationshipType::RelatedTo,
                    };

                    // Add the relationship
                    match self.graph.create_relationship(person_entity.id, target_entity.id, rel_type) {
                        Ok(_) => {
                            results.push(format!("✓ {} {} {}", person, relationship.replace("_", " "), target));
                        }
                        Err(e) => {
                            results.push(format!("✗ Could not add relationship: {}", e));
                        }
                    }
                }
            }
        }

        let output = format!("👤 Contact Update:\n{}", results.join("\n"));
        Ok(HandleResult::Handled(output))
    }

    /// Extract JSON array from LLM response
    fn extract_json_array(&self, response: &str) -> String {
        // Try to find JSON array in response
        if let Some(start) = response.find('[') {
            if let Some(end) = response.rfind(']') {
                let json_str = &response[start..=end];
                // Clean up common LLM formatting issues
                let cleaned = json_str
                    .replace('\n', " ")
                    .replace('\r', "")
                    .replace("  ", " ");
                
                // Validate it parses as JSON before returning
                if serde_json::from_str::<serde_json::Value>(&cleaned).is_ok() {
                    return cleaned;
                }
                // Return the original if cleaning didn't help
                return json_str.to_string();
            }
        }
        
        // Check for markdown code blocks
        if let Some(code_start) = response.find("```json") {
            if let Some(code_end) = response[code_start..].find("```\n") {
                let code_block = &response[code_start + 7..code_start + code_end];
                if let Some(arr_start) = code_block.find('[') {
                    if let Some(arr_end) = code_block.rfind(']') {
                        return code_block[arr_start..=arr_end].to_string();
                    }
                }
            }
        }
        
        // Return empty array if no JSON found
        "[]".to_string()
    }
}
