//! Intent handler - routes classified intents to appropriate actions
//!
//! This module takes the classified intent and executes the corresponding action.

use super::types::*;
use super::hybrid::HybridClassifier;
use super::IntentClassifier;
use crate::integrations::{
    OAuthManager, Provider, CalendarAI, GoogleCalendarClient, 
    GmailClient, ImportanceScorer, EmailAI, SmartReply, WebSearch,
};
use crate::knowledge::{KnowledgeGraph, RelationshipType};
use crate::llm::OllamaClient;
use crate::router::phase6_commands;
use crate::intelligence::{ConversationContext, EventReference, EmailReference, PersonReference};
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
    shared_classifier: Option<Arc<HybridClassifier>>,
    context: Option<&'a mut ConversationContext>,
}

impl<'a> IntentHandler<'a> {
    pub fn new(llm: &'a OllamaClient, graph: &'a KnowledgeGraph) -> Self {
        IntentHandler {
            llm,
            graph,
            classifier: HybridClassifier::new(Arc::new(llm.clone())),
            shared_classifier: None,
            context: None,
        }
    }

    /// Create intent handler with ONNX model support
    pub fn with_data_dir(llm: &'a OllamaClient, graph: &'a KnowledgeGraph, data_dir: &Path) -> Self {
        let classifier = HybridClassifier::try_with_onnx(Arc::new(llm.clone()), data_dir);
        IntentHandler {
            llm,
            graph,
            classifier,
            shared_classifier: None,
            context: None,
        }
    }

    /// Create intent handler with conversation context
    pub fn with_context(
        llm: &'a OllamaClient,
        graph: &'a KnowledgeGraph,
        data_dir: &Path,
        context: &'a mut ConversationContext,
    ) -> Self {
        let classifier = HybridClassifier::try_with_onnx(Arc::new(llm.clone()), data_dir);
        IntentHandler {
            llm,
            graph,
            classifier,
            shared_classifier: None,
            context: Some(context),
        }
    }

    /// Create intent handler with a pre-loaded (shared) classifier - avoids reloading the model
    pub fn with_shared_classifier(
        llm: &'a OllamaClient,
        graph: &'a KnowledgeGraph,
        classifier: Arc<HybridClassifier>,
        context: &'a mut ConversationContext,
    ) -> Self {
        IntentHandler {
            llm,
            graph,
            classifier: HybridClassifier::new(Arc::new(llm.clone())), // Placeholder, won't be used
            shared_classifier: Some(classifier),
            context: Some(context),
        }
    }

    /// Get the classifier to use (prefer shared if available)
    fn get_classifier(&self) -> &HybridClassifier {
        if let Some(ref shared) = self.shared_classifier {
            shared.as_ref()
        } else {
            &self.classifier
        }
    }

    /// Classify and handle an input
    pub async fn handle(&mut self, input: &str) -> Result<HandleResult> {
        // Apply pronoun resolution if we have context
        let resolved_input = if let Some(ref ctx) = self.context {
            ctx.resolve_pronouns(input)
        } else {
            input.to_string()
        };
        
        // Check if user is providing follow-up info for a pending action
        if let Some(ref mut ctx) = self.context {
            if let Some(completed) = ctx.try_complete_pending(&resolved_input) {
                return self.handle_completed_action(completed).await;
            }
            
            // Check for incremental modifications
            if ctx.pending_action.is_some() {
                if ctx.apply_modification(&resolved_input)? {
                    return Ok(HandleResult::Handled("✓ Updated.".to_string()));
                }
            }
        }
        
        // Classify the intent using the (shared or owned) classifier
        let ctx = IntentContext::new(self.is_google_authenticated().await);
        let result = self.get_classifier().classify(&resolved_input, &ctx).await?;
        
        // Handle based on intent type
        let handle_result = self.handle_intent(&result.intent, &resolved_input).await?;
        
        // Record turn if we have context and it was handled
        if let HandleResult::Handled(ref output) = handle_result {
            if let Some(ref mut conv_ctx) = self.context {
                conv_ctx.add_turn(input, output, Some(&format!("{:?}", result.intent)));
            }
        }
        
        Ok(handle_result)
    }

    /// Handle a completed pending action
    async fn handle_completed_action(&mut self, action: crate::intelligence::PendingAction) -> Result<HandleResult> {
        use crate::intelligence::PendingAction;
        
        match action {
            PendingAction::CreateEvent { title, start_time, end_time, attendees, recurrence } => {
                // Now that we have the info, create the event
                let oauth = match self.get_oauth().await {
                    Ok(o) => o,
                    Err(_) => return Ok(HandleResult::NeedsAuth("calendar".to_string())),
                };
                
                let description = format!("Create event '{}' with {}", 
                    title, 
                    attendees.join(", ")
                );
                
                // Use CalendarAI with the complete information
                let calendar_ai = CalendarAI::new(self.llm, self.graph, oauth);
                let result = calendar_ai.create_from_natural_language(&description).await?;
                
                let mut output = result.display_preview();
                output.push_str("\n\n💡 To confirm, use: /gcal confirm");
                phase6_commands::store_pending_event(result.event);
                
                Ok(HandleResult::Handled(output))
            }
            PendingAction::SendEmail { to, subject, body } => {
                let oauth = match self.get_oauth().await {
                    Ok(o) => o,
                    Err(_) => return Ok(HandleResult::NeedsAuth("email".to_string())),
                };
                
                let email_ai = EmailAI::new(self.llm, self.graph, oauth);
                let input = format!(
                    "Email {} about '{}' saying: {}",
                    to.join(", "),
                    subject,
                    body.unwrap_or_default()
                );
                let result = email_ai.compose_email(&input).await?;
                
                let mut output = result.display_preview();
                output.push_str("\n💡 To send, use: /email confirm");
                phase6_commands::store_pending_email(result);
                
                Ok(HandleResult::Handled(output))
            }
            _ => Ok(HandleResult::NotHandled),
        }
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

            // Web search - weather, facts, general internet queries
            Intent::WebSearch { query } => {
                self.handle_web_search(query).await
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

    // Web search handler
    
    async fn handle_web_search(&self, query: &str) -> Result<HandleResult> {
        let web_search = WebSearch::new();
        
        match web_search.quick_answer(query).await {
            Ok(answer) => Ok(HandleResult::Handled(answer)),
            Err(e) => {
                // If web search fails, return NotHandled to fall through to LLM
                // which can still provide a reasonable response
                Ok(HandleResult::NotHandled)
            }
        }
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
            CalendarQueryType::WithPerson(person) => {
                self.handle_calendar_with_person(&client, person).await
            }
            CalendarQueryType::Topic(topic) => {
                self.handle_calendar_topic(&client, topic).await
            }
            CalendarQueryType::WithOrg(org) => {
                self.handle_calendar_with_org(&client, org).await
            }
            CalendarQueryType::Filter(filter) => {
                self.handle_calendar_filter(&client, filter).await
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
            EmailQueryType::Topic(topic) => {
                self.handle_email_topic_query(&client, topic).await
            }
            EmailQueryType::AboutPerson(person) => {
                self.handle_email_about_person(&client, person).await
            }
            EmailQueryType::AboutOrg(org) => {
                self.handle_email_about_org(&client, org).await
            }
            EmailQueryType::DateRange { from, to } => {
                self.handle_email_date_range(&client, from.as_ref(), to.as_ref()).await
            }
            EmailQueryType::Filter(filter) => {
                self.handle_email_complex_filter(&client, filter).await
            }
        }
    }

    /// Handle topic-based email search
    async fn handle_email_topic_query(&self, client: &GmailClient, topic: &str) -> Result<HandleResult> {
        // Build Gmail search query for topic
        // Gmail supports subject: and body content search
        let query = format!("subject:{} OR {}", topic, topic);
        
        match client.list_messages(Some(&query), 20).await {
            Ok(emails) => {
                if emails.is_empty() {
                    Ok(HandleResult::Handled(format!("📧 No emails found about '{}'.", topic)))
                } else {
                    // Use LLM to filter and rank by relevance
                    let filtered = self.filter_emails_by_topic(&emails, topic).await;
                    
                    if filtered.is_empty() {
                        Ok(HandleResult::Handled(format!("📧 No emails closely related to '{}'.", topic)))
                    } else {
                        let mut output = format!("📧 Emails about '{}' ({}):\n", topic, filtered.len());
                        for email in &filtered {
                            output.push_str(&format!("  {}\n", email.display_short()));
                        }
                        Ok(HandleResult::Handled(output))
                    }
                }
            }
            Err(e) => Ok(HandleResult::Error(format!("Could not search emails: {}", e)))
        }
    }

    /// Filter emails by topic using LLM
    async fn filter_emails_by_topic(&self, emails: &[crate::integrations::google::gmail::EmailSummary], topic: &str) -> Vec<crate::integrations::google::gmail::EmailSummary> {
        // For efficiency, use a single LLM call to score all emails
        let email_list: Vec<String> = emails.iter().enumerate()
            .map(|(i, e)| format!("{}. From: {} | Subject: {} | Preview: {}", 
                i + 1, e.from, e.subject, e.snippet.chars().take(100).collect::<String>()))
            .collect();
        
        let prompt = format!(
            r#"Given these emails, identify which ones are related to the topic "{}".
Return ONLY a JSON array of the email numbers (1-indexed) that are relevant.

Emails:
{}

Respond with just the JSON array, e.g., [1, 3, 5] or [] if none are relevant:"#,
            topic,
            email_list.join("\n")
        );
        
        match self.llm.query(&prompt).await {
            Ok(response) => {
                // Parse the response to get relevant indices
                if let Some(start) = response.find('[') {
                    if let Some(end) = response.rfind(']') {
                        let json_str = &response[start..=end];
                        if let Ok(indices) = serde_json::from_str::<Vec<usize>>(json_str) {
                            return indices.iter()
                                .filter_map(|&i| emails.get(i.saturating_sub(1)).cloned())
                                .collect();
                        }
                    }
                }
                // If parsing fails, return all emails
                emails.to_vec()
            }
            Err(_) => emails.to_vec()
        }
    }

    /// Handle emails about a specific person (using KG)
    async fn handle_email_about_person(&self, client: &GmailClient, person: &str) -> Result<HandleResult> {
        // Look up person in KG to get email address
        let person_email = self.graph.find_person(person)
            .ok()
            .flatten()
            .and_then(|e| e.properties.get("email").cloned());
        
        // Build search query
        let query = if let Some(email) = &person_email {
            format!("from:{} OR to:{} OR {}", email, email, person)
        } else {
            person.to_string()
        };
        
        match client.list_messages(Some(&query), 15).await {
            Ok(emails) => {
                if emails.is_empty() {
                    Ok(HandleResult::Handled(format!("📧 No emails found involving {}.", person)))
                } else {
                    let mut output = format!("📧 Emails involving {} ({}):\n", person, emails.len());
                    for email in &emails {
                        output.push_str(&format!("  {}\n", email.display_short()));
                    }
                    Ok(HandleResult::Handled(output))
                }
            }
            Err(e) => Ok(HandleResult::Error(format!("Could not search emails: {}", e)))
        }
    }

    /// Handle emails about a specific organization (using KG)
    async fn handle_email_about_org(&self, client: &GmailClient, org: &str) -> Result<HandleResult> {
        // Look up org in KG to find related people/domains
        let org_domain = self.infer_org_domain(org);
        
        // Build search query
        let query = if let Some(domain) = &org_domain {
            format!("from:{} OR to:{} OR {}", domain, domain, org)
        } else {
            org.to_string()
        };
        
        match client.list_messages(Some(&query), 15).await {
            Ok(emails) => {
                if emails.is_empty() {
                    Ok(HandleResult::Handled(format!("📧 No emails found related to {}.", org)))
                } else {
                    let mut output = format!("📧 Emails related to {} ({}):\n", org, emails.len());
                    for email in &emails {
                        output.push_str(&format!("  {}\n", email.display_short()));
                    }
                    Ok(HandleResult::Handled(output))
                }
            }
            Err(e) => Ok(HandleResult::Error(format!("Could not search emails: {}", e)))
        }
    }

    /// Infer organization email domain from name
    fn infer_org_domain(&self, org: &str) -> Option<String> {
        let lower = org.to_lowercase();
        let common_domains = [
            ("google", "@google.com"),
            ("microsoft", "@microsoft.com"),
            ("apple", "@apple.com"),
            ("amazon", "@amazon.com"),
            ("meta", "@meta.com"),
            ("facebook", "@fb.com"),
            ("netflix", "@netflix.com"),
            ("uber", "@uber.com"),
            ("stripe", "@stripe.com"),
            ("airbnb", "@airbnb.com"),
        ];
        
        for (name, domain) in common_domains {
            if lower.contains(name) {
                return Some(domain.to_string());
            }
        }
        
        // Try to construct domain from org name
        let sanitized: String = lower.chars()
            .filter(|c| c.is_alphanumeric())
            .collect();
        if !sanitized.is_empty() {
            Some(format!("@{}.com", sanitized))
        } else {
            None
        }
    }

    /// Handle date range email query
    async fn handle_email_date_range(
        &self, 
        client: &GmailClient, 
        from: Option<&chrono::DateTime<chrono::Utc>>, 
        to: Option<&chrono::DateTime<chrono::Utc>>
    ) -> Result<HandleResult> {
        let mut query_parts = Vec::new();
        
        if let Some(from_date) = from {
            query_parts.push(format!("after:{}", from_date.format("%Y/%m/%d")));
        }
        if let Some(to_date) = to {
            query_parts.push(format!("before:{}", to_date.format("%Y/%m/%d")));
        }
        
        let query = if query_parts.is_empty() {
            None
        } else {
            Some(query_parts.join(" "))
        };
        
        match client.list_messages(query.as_deref(), 20).await {
            Ok(emails) => {
                if emails.is_empty() {
                    Ok(HandleResult::Handled("📧 No emails found in the specified date range.".to_string()))
                } else {
                    let mut output = format!("📧 Emails ({}):\n", emails.len());
                    for email in &emails {
                        output.push_str(&format!("  {}\n", email.display_short()));
                    }
                    Ok(HandleResult::Handled(output))
                }
            }
            Err(e) => Ok(HandleResult::Error(format!("Could not search emails: {}", e)))
        }
    }

    /// Handle complex email filter
    async fn handle_email_complex_filter(&self, client: &GmailClient, filter: &crate::intent::types::EmailFilter) -> Result<HandleResult> {
        // Build Gmail search query from filter
        let mut query_parts = Vec::new();
        
        // Topics/keywords
        for topic in &filter.topics {
            query_parts.push(format!("({})", topic));
        }
        
        // From filters
        for from in &filter.from {
            query_parts.push(format!("from:{}", from));
        }
        
        // To filters
        for to in &filter.to {
            query_parts.push(format!("to:{}", to));
        }
        
        // Unread only
        if filter.unread_only {
            query_parts.push("is:unread".to_string());
        }
        
        // Important only
        if filter.important_only {
            query_parts.push("is:important OR is:starred".to_string());
        }
        
        // Date filters
        if let Some(after) = &filter.after {
            query_parts.push(format!("after:{}", after.format("%Y/%m/%d")));
        }
        if let Some(before) = &filter.before {
            query_parts.push(format!("before:{}", before.format("%Y/%m/%d")));
        }
        
        let query = if query_parts.is_empty() {
            None
        } else {
            Some(query_parts.join(" "))
        };
        
        let limit = if filter.limit > 0 { filter.limit } else { 20 };
        
        match client.list_messages(query.as_deref(), limit).await {
            Ok(mut emails) => {
                // Post-filter by people/orgs if specified
                if !filter.mentions_people.is_empty() || !filter.mentions_orgs.is_empty() {
                    emails = self.post_filter_emails(&emails, &filter.mentions_people, &filter.mentions_orgs).await;
                }
                
                if emails.is_empty() {
                    Ok(HandleResult::Handled("📧 No emails match your filter criteria.".to_string()))
                } else {
                    let mut output = format!("📧 Filtered Emails ({}):\n", emails.len());
                    for email in &emails {
                        output.push_str(&format!("  {}\n", email.display_short()));
                    }
                    Ok(HandleResult::Handled(output))
                }
            }
            Err(e) => Ok(HandleResult::Error(format!("Could not search emails: {}", e)))
        }
    }

    /// Post-filter emails by people/org mentions
    async fn post_filter_emails(
        &self, 
        emails: &[crate::integrations::google::gmail::EmailSummary],
        people: &[String],
        orgs: &[String]
    ) -> Vec<crate::integrations::google::gmail::EmailSummary> {
        emails.iter()
            .filter(|email| {
                let content = format!("{} {} {}", email.from, email.subject, email.snippet).to_lowercase();
                
                // Check people mentions
                let people_match = people.is_empty() || 
                    people.iter().any(|p| content.contains(&p.to_lowercase()));
                
                // Check org mentions
                let org_match = orgs.is_empty() || 
                    orgs.iter().any(|o| content.contains(&o.to_lowercase()));
                
                people_match && org_match
            })
            .cloned()
            .collect()
    }

    // Calendar semantic query handlers

    /// Handle calendar query for meetings with a specific person
    async fn handle_calendar_with_person(&self, client: &GoogleCalendarClient, person: &str) -> Result<HandleResult> {
        // Look up person in KG to get email address for attendee matching
        let person_email = self.graph.find_person(person)
            .ok()
            .flatten()
            .and_then(|e| e.properties.get("email").cloned());
        
        // Get upcoming events
        match client.get_upcoming_events(30).await {
            Ok(events) => {
                let lower_person = person.to_lowercase();
                let filtered: Vec<_> = events.iter()
                    .filter(|e| {
                        let attendees = e.attendees.as_ref();
                        // Check if person is in attendees
                        let in_attendees = attendees.map(|atts| atts.iter().any(|a| {
                            a.email.as_ref().map(|em| em.to_lowercase().contains(&lower_person)).unwrap_or(false) ||
                            a.display_name.as_ref().map(|n| n.to_lowercase().contains(&lower_person)).unwrap_or(false)
                        })).unwrap_or(false);
                        // Check email match if we have it
                        let email_match = person_email.as_ref().map(|pe| {
                            attendees.map(|atts| atts.iter().any(|a| 
                                a.email.as_ref().map(|em| em.eq_ignore_ascii_case(pe)).unwrap_or(false)
                            )).unwrap_or(false)
                        }).unwrap_or(false);
                        // Check if mentioned in title/description
                        let title = e.summary.as_deref().unwrap_or("");
                        let in_title = title.to_lowercase().contains(&lower_person);
                        let in_desc = e.description.as_ref()
                            .map(|d| d.to_lowercase().contains(&lower_person))
                            .unwrap_or(false);
                        
                        in_attendees || email_match || in_title || in_desc
                    })
                    .collect();
                
                if filtered.is_empty() {
                    Ok(HandleResult::Handled(format!("📅 No upcoming meetings found with {}.", person)))
                } else {
                    let mut output = format!("📅 Meetings with {} ({}):\n", person, filtered.len());
                    for event in filtered {
                        output.push_str(&format!("{}\n", event.display()));
                    }
                    Ok(HandleResult::Handled(output))
                }
            }
            Err(e) => Ok(HandleResult::Error(format!("Could not fetch calendar: {}", e)))
        }
    }

    /// Handle calendar query for events about a topic
    async fn handle_calendar_topic(&self, client: &GoogleCalendarClient, topic: &str) -> Result<HandleResult> {
        match client.get_upcoming_events(30).await {
            Ok(events) => {
                // Use LLM to identify relevant events
                let filtered = self.filter_events_by_topic(&events, topic).await;
                
                if filtered.is_empty() {
                    Ok(HandleResult::Handled(format!("📅 No upcoming events found related to '{}'.", topic)))
                } else {
                    let mut output = format!("📅 Events related to '{}' ({}):\n", topic, filtered.len());
                    for event in filtered {
                        output.push_str(&format!("{}\n", event.display()));
                    }
                    Ok(HandleResult::Handled(output))
                }
            }
            Err(e) => Ok(HandleResult::Error(format!("Could not fetch calendar: {}", e)))
        }
    }

    /// Handle calendar query for events with an organization
    async fn handle_calendar_with_org(&self, client: &GoogleCalendarClient, org: &str) -> Result<HandleResult> {
        let org_domain = self.infer_org_domain(org);
        
        match client.get_upcoming_events(30).await {
            Ok(events) => {
                let lower_org = org.to_lowercase();
                let filtered: Vec<_> = events.iter()
                    .filter(|e| {
                        let attendees = e.attendees.as_ref();
                        // Check attendee emails for org domain
                        let domain_match = org_domain.as_ref().map(|d| {
                            attendees.map(|atts| atts.iter().any(|a| 
                                a.email.as_ref().map(|em| em.to_lowercase().contains(d)).unwrap_or(false)
                            )).unwrap_or(false)
                        }).unwrap_or(false);
                        // Check org name in attendees
                        let in_attendees = attendees.map(|atts| atts.iter().any(|a| {
                            a.display_name.as_ref().map(|n| n.to_lowercase().contains(&lower_org)).unwrap_or(false)
                        })).unwrap_or(false);
                        // Check if mentioned in title/description
                        let title = e.summary.as_deref().unwrap_or("");
                        let in_title = title.to_lowercase().contains(&lower_org);
                        let in_desc = e.description.as_ref()
                            .map(|d| d.to_lowercase().contains(&lower_org))
                            .unwrap_or(false);
                        
                        domain_match || in_attendees || in_title || in_desc
                    })
                    .collect();
                
                if filtered.is_empty() {
                    Ok(HandleResult::Handled(format!("📅 No upcoming meetings found with {}.", org)))
                } else {
                    let mut output = format!("📅 Meetings with {} ({}):\n", org, filtered.len());
                    for event in filtered {
                        output.push_str(&format!("{}\n", event.display()));
                    }
                    Ok(HandleResult::Handled(output))
                }
            }
            Err(e) => Ok(HandleResult::Error(format!("Could not fetch calendar: {}", e)))
        }
    }

    /// Handle complex calendar filter
    async fn handle_calendar_filter(&self, client: &GoogleCalendarClient, filter: &CalendarFilter) -> Result<HandleResult> {
        match client.get_upcoming_events(50).await {
            Ok(events) => {
                let mut filtered: Vec<_> = events.iter().collect();
                
                // Filter by people
                if !filter.with_people.is_empty() {
                    filtered.retain(|e| {
                        let attendees = e.attendees.as_ref();
                        filter.with_people.iter().any(|person| {
                            let lower = person.to_lowercase();
                            attendees.map(|atts| atts.iter().any(|a| {
                                a.email.as_ref().map(|em| em.to_lowercase().contains(&lower)).unwrap_or(false) ||
                                a.display_name.as_ref().map(|n| n.to_lowercase().contains(&lower)).unwrap_or(false)
                            })).unwrap_or(false) ||
                            e.summary.as_deref().unwrap_or("").to_lowercase().contains(&lower)
                        })
                    });
                }
                
                // Filter by organizations
                if !filter.with_orgs.is_empty() {
                    filtered.retain(|e| {
                        let attendees = e.attendees.as_ref();
                        filter.with_orgs.iter().any(|org| {
                            let lower = org.to_lowercase();
                            let domain = self.infer_org_domain(org);
                            
                            let domain_match = domain.as_ref().map(|d| {
                                attendees.map(|atts| atts.iter().any(|a| 
                                    a.email.as_ref().map(|em| em.to_lowercase().contains(d)).unwrap_or(false)
                                )).unwrap_or(false)
                            }).unwrap_or(false);
                            
                            domain_match ||
                            e.summary.as_deref().unwrap_or("").to_lowercase().contains(&lower) ||
                            e.description.as_ref().map(|d| d.to_lowercase().contains(&lower)).unwrap_or(false)
                        })
                    });
                }
                
                // Filter by topics using keyword matching
                if !filter.topics.is_empty() {
                    filtered.retain(|e| {
                        let content = format!("{} {}", 
                            e.summary.as_deref().unwrap_or(""),
                            e.description.as_deref().unwrap_or("")
                        ).to_lowercase();
                        filter.topics.iter().any(|t| content.contains(&t.to_lowercase()))
                    });
                }
                
                // Filter by date range - compare start times
                if let Some(after) = &filter.after {
                    filtered.retain(|e| {
                        e.start_time().map(|t| t.with_timezone(&chrono::Utc) >= *after).unwrap_or(true)
                    });
                }
                if let Some(before) = &filter.before {
                    filtered.retain(|e| {
                        e.start_time().map(|t| t.with_timezone(&chrono::Utc) <= *before).unwrap_or(true)
                    });
                }
                
                // Apply limit
                if filter.limit > 0 && filtered.len() > filter.limit {
                    filtered.truncate(filter.limit);
                }
                
                if filtered.is_empty() {
                    Ok(HandleResult::Handled("📅 No events found matching your criteria.".to_string()))
                } else {
                    let mut output = format!("📅 Filtered Events ({}):\n", filtered.len());
                    for event in filtered {
                        output.push_str(&format!("{}\n", event.display()));
                    }
                    Ok(HandleResult::Handled(output))
                }
            }
            Err(e) => Ok(HandleResult::Error(format!("Could not fetch calendar: {}", e)))
        }
    }

    /// Filter events by topic using LLM
    async fn filter_events_by_topic(&self, events: &[crate::integrations::google::calendar::GCalEvent], topic: &str) -> Vec<crate::integrations::google::calendar::GCalEvent> {
        let event_list: Vec<String> = events.iter().enumerate()
            .map(|(i, e)| format!("{}. {} | {}", 
                i + 1, 
                e.summary.as_deref().unwrap_or("(No title)"),
                e.description.as_deref().unwrap_or("").chars().take(100).collect::<String>()))
            .collect();
        
        let prompt = format!(
            r#"Given these calendar events, identify which ones are related to the topic "{}".
Return ONLY a JSON array of the event numbers (1-indexed) that are relevant.

Events:
{}

Respond with just the JSON array, e.g., [1, 3, 5] or [] if none are relevant:"#,
            topic,
            event_list.join("\n")
        );
        
        match self.llm.query(&prompt).await {
            Ok(response) => {
                if let Some(start) = response.find('[') {
                    if let Some(end) = response.rfind(']') {
                        let json_str = &response[start..=end];
                        if let Ok(indices) = serde_json::from_str::<Vec<usize>>(json_str) {
                            return indices.iter()
                                .filter_map(|&i| events.get(i.saturating_sub(1)).cloned())
                                .collect();
                        }
                    }
                }
                events.to_vec()
            }
            Err(_) => events.to_vec()
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
        
        // Pattern: "<Name> works at <Organization>" or "<Name> studies at <Organization>"
        let relationship_patterns = [
            (r"(?i)^(.+?)\s+works?\s+(?:at|for)\s+(.+)$", "works_at"),
            (r"(?i)^(.+?)\s+(?:studies|studied|is studying)\s+at\s+(.+)$", "studies_at"),
            (r"(?i)^(.+?)\s+manages?\s+(.+)$", "manages"),
            (r"(?i)^(.+?)\s+knows?\s+(.+)$", "knows"),
            (r"(?i)^(.+?)\s+works?\s+on\s+(.+)$", "works_on"),
            (r"(?i)^(.+?)\s+(?:is|are)\s+(?:at|from)\s+(.+)$", "works_at"),
        ];
        
        for (pattern, rel_type) in relationship_patterns {
            if let Some(captures) = regex::Regex::new(pattern)
                .ok()
                .and_then(|re| re.captures(input))
            {
                let person_name = captures.get(1)?.as_str().trim().to_string();
                let target = captures.get(2)?.as_str().trim().to_string();
                
                // Skip if names are too long (probably not a name)
                if person_name.split_whitespace().count() <= 4 && target.split_whitespace().count() <= 6 {
                    actions.push(ContactAction::AddRelationship {
                        person: person_name,
                        relationship: rel_type.to_string(),
                        target,
                    });
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
                        "studies_at" => {
                            // For studies_at, first try to find by exact name, then fuzzy match
                            let found = self.graph.find_by_name(target.as_str()).ok().flatten()
                                .or_else(|| self.find_entity_fuzzy(target.as_str(), &["university", "organization"]))
                                .or_else(|| self.graph.find_organization(target.as_str()).ok().flatten());
                            
                            match found {
                                Some(e) => e,
                                None => {
                                    // Create as organization (educational institution)
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
                            }
                        }
                        "works_at" => {
                            // For works_at, find or create organization
                            let found = self.graph.find_organization(target.as_str()).ok().flatten()
                                .or_else(|| self.find_entity_fuzzy(target.as_str(), &["organization"]));
                            
                            match found {
                                Some(e) => e,
                                None => {
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

    /// Find an entity by fuzzy name matching
    /// Searches through entities of the given types and returns the best match
    fn find_entity_fuzzy(&self, name: &str, entity_types: &[&str]) -> Option<crate::knowledge::entities::Entity> {
        use strsim::jaro_winkler;
        
        let name_lower = name.to_lowercase();
        let mut best_match: Option<(crate::knowledge::entities::Entity, f64)> = None;
        const THRESHOLD: f64 = 0.75;
        
        // Get all entities and filter by type
        if let Ok(all_entities) = self.graph.database().list_all_entities() {
            for (id, entity_name, entity_type) in all_entities {
                // Check if this entity type is in our search list
                if !entity_types.iter().any(|t| *t == entity_type.to_lowercase()) {
                    continue;
                }
                
                let entity_name_lower = entity_name.to_lowercase();
                
                // Calculate similarity
                let similarity = jaro_winkler(&name_lower, &entity_name_lower);
                
                // Also check if one contains the other (partial match)
                let partial_match = entity_name_lower.contains(&name_lower) 
                    || name_lower.contains(&entity_name_lower);
                
                let effective_score = if partial_match {
                    similarity.max(0.8) // Boost partial matches
                } else {
                    similarity
                };
                
                if effective_score >= THRESHOLD {
                    if best_match.is_none() || effective_score > best_match.as_ref().unwrap().1 {
                        let entity = crate::knowledge::entities::Entity::new(
                            id,
                            crate::knowledge::entities::EntityType::from_str(&entity_type),
                            entity_name
                        );
                        best_match = Some((entity, effective_score));
                    }
                }
            }
        }
        
        best_match.map(|(entity, _)| entity)
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
