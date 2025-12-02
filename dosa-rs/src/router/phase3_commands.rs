//! Phase 3 commands: Intelligence features
//! 
//! Adds commands for:
//! - Daily briefing
//! - Knowledge graph queries
//! - Alert management
//! - Phase 4: GraphRAG multi-hop queries and intent-based routing

use anyhow::Result;
use std::sync::Arc;

use crate::router::commands::{Command, CommandContext, CommandResult, Router};
use crate::intelligence::{AlertSystem, NaturalLanguageQuery};
use crate::intelligence::query::IntentRouter;
use crate::intelligence::graphrag::GraphRAG;
use crate::intelligence::kag::LogicalSolver;

// ============================================================================
// Briefing Command
// ============================================================================

/// Daily briefing command
pub struct BriefingCommand;

impl Command for BriefingCommand {
    fn name(&self) -> &str { "/briefing" }
    fn description(&self) -> &str { "Get your daily briefing" }
    fn usage(&self) -> &str { "/briefing" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower == "/briefing" || lower == "/brief" || lower == "/morning"
    }
    
    fn execute(&self, _input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let alert_system = AlertSystem::new(ctx.graph.database());
        let briefing = alert_system.get_daily_briefing()?;
        Ok(CommandResult::Success(briefing))
    }
}

// ============================================================================
// Alert/Notifications Command
// ============================================================================

/// Show current alerts/notifications
pub struct AlertsCommand;

impl Command for AlertsCommand {
    fn name(&self) -> &str { "/alerts" }
    fn description(&self) -> &str { "Show current alerts and notifications" }
    fn usage(&self) -> &str { "/alerts" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower == "/alerts" || lower == "/notifications"
    }
    
    fn execute(&self, _input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let alert_system = AlertSystem::new(ctx.graph.database());
        
        match alert_system.display_alerts()? {
            Some(alerts) => Ok(CommandResult::Success(alerts)),
            None => Ok(CommandResult::Success("✨ No alerts or notifications at this time.".to_string())),
        }
    }
}

// ============================================================================
// Knowledge Graph Query Command
// ============================================================================

/// Query command for knowledge graph
pub struct QueryCommand;

impl Command for QueryCommand {
    fn name(&self) -> &str { "/query" }
    fn description(&self) -> &str { "Query the knowledge graph" }
    fn usage(&self) -> &str { "/query <question> (e.g., 'who works at Google?')" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/query ") || lower.starts_with("/q ")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let query = input.trim()
            .strip_prefix("/query")
            .or_else(|| input.trim().strip_prefix("/q"))
            .map(|s| s.trim())
            .unwrap_or("");

        if query.is_empty() {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        let nl_query = NaturalLanguageQuery::new(ctx.graph);
        let query_type = nl_query.parse_query(query);
        let result = nl_query.execute(&query_type)?;
        
        Ok(CommandResult::Success(result))
    }
}

// ============================================================================
// Who Command (shortcut for relationship queries)
// ============================================================================

/// Who command for quick people queries
pub struct WhoCommand;

impl Command for WhoCommand {
    fn name(&self) -> &str { "/who" }
    fn description(&self) -> &str { "Find people (e.g., '/who works at Google')" }
    fn usage(&self) -> &str { "/who <query>" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase().starts_with("/who ")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let query = input.trim()
            .strip_prefix("/who")
            .map(|s| format!("who {}", s.trim()))
            .unwrap_or_default();

        if query.len() <= 4 {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        let nl_query = NaturalLanguageQuery::new(ctx.graph);
        let query_type = nl_query.parse_query(&query);
        let result = nl_query.execute(&query_type)?;
        
        Ok(CommandResult::Success(result))
    }
}

// ============================================================================
// Search Command
// ============================================================================

/// Search across knowledge graph
pub struct SearchCommand;

impl Command for SearchCommand {
    fn name(&self) -> &str { "/search" }
    fn description(&self) -> &str { "Search the knowledge graph" }
    fn usage(&self) -> &str { "/search <term>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/search ") || lower.starts_with("/find ")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let term = input.trim()
            .strip_prefix("/search")
            .or_else(|| input.trim().strip_prefix("/find"))
            .map(|s| s.trim())
            .unwrap_or("");

        if term.is_empty() {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        let nl_query = NaturalLanguageQuery::new(ctx.graph);
        let query_type = crate::intelligence::query::QueryType::Search { term: term.to_string() };
        let result = nl_query.execute(&query_type)?;
        
        Ok(CommandResult::Success(result))
    }
}

// ============================================================================
// Phase 4: GraphRAG Multi-hop Command
// ============================================================================

/// GraphRAG command for complex multi-hop queries
pub struct GraphRAGCommand;

impl Command for GraphRAGCommand {
    fn name(&self) -> &str { "/graph" }
    fn description(&self) -> &str { "Multi-hop knowledge graph query" }
    fn usage(&self) -> &str { "/graph <query> (e.g., 'who can help with Project X?')" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/graph ") || lower.starts_with("/rag ")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let query = input.trim()
            .strip_prefix("/graph")
            .or_else(|| input.trim().strip_prefix("/rag"))
            .map(|s| s.trim())
            .unwrap_or("");

        if query.is_empty() {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        let graphrag = GraphRAG::new(ctx.graph);
        
        // Try to parse as multi-hop query
        if let Some(mh_query) = graphrag.parse_query(query) {
            let result = graphrag.execute(&mh_query)?;
            return Ok(CommandResult::Success(result.explanation));
        }
        
        // Fall back to regular NL query
        let nl_query = NaturalLanguageQuery::new(ctx.graph);
        let query_type = nl_query.parse_query(query);
        let result = nl_query.execute(&query_type)?;
        
        Ok(CommandResult::Success(result))
    }
}

// ============================================================================
// Phase 4: Contact/Reach Command
// ============================================================================

/// Contact command for getting contact information with action context
pub struct ContactCommand;

impl Command for ContactCommand {
    fn name(&self) -> &str { "/contact" }
    fn description(&self) -> &str { "Get contact information for a person" }
    fn usage(&self) -> &str { "/contact <name>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/contact ") || lower.starts_with("/reach ")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let name = input.trim()
            .strip_prefix("/contact")
            .or_else(|| input.trim().strip_prefix("/reach"))
            .map(|s| s.trim())
            .unwrap_or("");

        if name.is_empty() || name == "add" {
            // Handle /contact add as separate case (existing functionality)
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        let intent_router = IntentRouter::new(ctx.graph);
        let query = format!("how can I contact {}", name);
        let intent_result = intent_router.classify(&query);
        let response = intent_router.execute_intent(&intent_result)?;
        
        Ok(CommandResult::Success(response.message))
    }
}

// ============================================================================
// Phase 4: Ask Command (Unified Natural Language)
// ============================================================================

/// Ask command - unified natural language interface
pub struct AskCommand;

impl Command for AskCommand {
    fn name(&self) -> &str { "/ask" }
    fn description(&self) -> &str { "Ask anything in natural language" }
    fn usage(&self) -> &str { "/ask <question>" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase().starts_with("/ask ")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let raw_query = input.trim()
            .strip_prefix("/ask")
            .map(|s| s.trim())
            .unwrap_or("");

        if raw_query.is_empty() {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        // Resolve first-person pronouns ("I", "me", "my") to user's identity
        let query = ctx.graph.resolve_pronouns(raw_query)?;
        let query = query.as_str();

        // First, try KAG LogicalSolver for reasoning queries (aggregation, inference, comparisons)
        let logical_solver = LogicalSolver::new(ctx.graph);
        if let Some(logical_form) = logical_solver.parse_to_logical_form(query) {
            let result = logical_solver.execute(&logical_form)?;
            // Format reasoning steps for transparency
            let mut output = result.answer.clone();
            if !result.reasoning_steps.is_empty() && result.reasoning_steps.len() > 1 {
                output.push_str("\n\n📊 Reasoning:");
                for step in &result.reasoning_steps {
                    output.push_str(&format!("\n  • {:?}: {}", step.step_type, step.description));
                }
            }
            if !result.rules_applied.is_empty() {
                output.push_str(&format!("\n\n🧠 Rules applied: {}", result.rules_applied.join(", ")));
            }
            return Ok(CommandResult::Success(output));
        }

        // Second, try GraphRAG for multi-hop queries
        let graphrag = GraphRAG::new(ctx.graph);
        if let Some(mh_query) = graphrag.parse_query(query) {
            let result = graphrag.execute(&mh_query)?;
            return Ok(CommandResult::Success(result.explanation));
        }
        
        // Third, try Intent Router for contact/action queries
        let intent_router = IntentRouter::new(ctx.graph);
        let intent_result = intent_router.classify(query);
        
        // Check if intent was recognized (not Unknown)
        match &intent_result.intent {
            crate::intelligence::query::Intent::Unknown { .. } => {
                // Fall back to regular NL query
                let nl_query = NaturalLanguageQuery::new(ctx.graph);
                let query_type = nl_query.parse_query(query);
                let result = nl_query.execute(&query_type)?;
                Ok(CommandResult::Success(result))
            }
            _ => {
                let response = intent_router.execute_intent(&intent_result)?;
                Ok(CommandResult::Success(response.message))
            }
        }
    }
}

// ============================================================================
// Phase 5: Document Ingestion Command
// ============================================================================

/// Ingest command - add documents to populate knowledge graph
pub struct IngestCommand;

impl Command for IngestCommand {
    fn name(&self) -> &str { "/ingest" }
    fn description(&self) -> &str { "Add a document to extract entities and relationships (supports PDF, TXT, MD)" }
    fn usage(&self) -> &str { "/ingest <text or file path>" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase().starts_with("/ingest ")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let content = input.trim()
            .strip_prefix("/ingest")
            .map(|s| s.trim())
            .unwrap_or("");

        if content.is_empty() {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        let ingester = crate::intelligence::DocumentIngester::new(ctx.graph);

        // Strip surrounding quotes if present
        let content = content.trim_matches('"').trim_matches('\'');

        // Check if it's a file path
        let is_file = content.starts_with('/') || content.starts_with("~/") 
            || content.ends_with(".txt") || content.ends_with(".md") || content.ends_with(".pdf");
        
        let result = if is_file {
            // Try to process as file
            let path = shellexpand::tilde(content).to_string();
            let path = std::path::Path::new(&path);
            
            if !path.exists() {
                // Not a file, use as text
                ingester.ingest_parallel(content)?
            } else {
                // Use the file ingestion (handles PDF, TXT, MD)
                ingester.ingest_file(path)?
            }
        } else {
            // Use parallel ingestion for text (splits by paragraphs)
            ingester.ingest_parallel(content)?
        };

        if result.entities_added.is_empty() && result.relationships_added.is_empty() && result.properties_set.is_empty() {
            Ok(CommandResult::Success("No entities or relationships found in the text. Try providing more structured information like:\n  \"I am John Smith. I work at Google. My email is john@google.com.\"".to_string()))
        } else {
            Ok(CommandResult::Success(format!(
                "📥 Document ingested!\n\n{}{}",
                result.summary(),
                result.details()
            )))
        }
    }
}

/// Ingest with LLM command - use AI to extract more comprehensively
pub struct IngestAICommand;

impl Command for IngestAICommand {
    fn name(&self) -> &str { "/ingest-ai" }
    fn description(&self) -> &str { "Use AI to extract entities from a document (slower but more thorough)" }
    fn usage(&self) -> &str { "/ingest-ai <text>" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase().starts_with("/ingest-ai ")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let content = input.trim()
            .strip_prefix("/ingest-ai")
            .map(|s| s.trim())
            .unwrap_or("");

        if content.is_empty() {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        // Use async runtime for LLM call
        let rt = tokio::runtime::Runtime::new()?;
        let ingester = crate::intelligence::DocumentIngester::new(ctx.graph);
        
        let document = content.to_string();
        let result = rt.block_on(async {
            ingester.ingest(&document).await
        })?;

        if result.entities_added.is_empty() && result.relationships_added.is_empty() {
            Ok(CommandResult::Success("AI couldn't extract structured information. The text might be too short or not contain clear entity information.".to_string()))
        } else {
            Ok(CommandResult::Success(format!(
                "🤖 AI ingestion complete!\n\n{}{}",
                result.summary(),
                result.details()
            )))
        }
    }
}

/// Learn about me command - shortcut for personal info ingestion
pub struct LearnCommand;

impl Command for LearnCommand {
    fn name(&self) -> &str { "/learn" }
    fn description(&self) -> &str { "Teach the assistant about yourself or others" }
    fn usage(&self) -> &str { "/learn <info about you or someone>" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase().starts_with("/learn ")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let content = input.trim()
            .strip_prefix("/learn")
            .map(|s| s.trim())
            .unwrap_or("");

        if content.is_empty() {
            return Ok(CommandResult::Error(
                "Tell me something! Examples:\n  /learn I am John, I work at Google\n  /learn Alice knows Bob, they work together at Microsoft\n  /learn My email is john@example.com".to_string()
            ));
        }

        let ingester = crate::intelligence::DocumentIngester::new(ctx.graph);
        let result = ingester.ingest_fast(content)?;

        if result.entities_added.is_empty() && result.relationships_added.is_empty() && result.properties_set.is_empty() {
            Ok(CommandResult::Success(format!(
                "🤔 I couldn't extract structured info from that. Try phrases like:\n  \"I am [name]\"\n  \"[name] works at [company]\"\n  \"[name]'s email is [email]\"\n\nYou said: \"{}\"",
                content
            )))
        } else {
            Ok(CommandResult::Success(format!(
                "📝 Got it!\n\n{}{}",
                result.summary(),
                result.details()
            )))
        }
    }
}

// ============================================================================
// Register Phase 3 Commands
// ============================================================================

pub fn register_phase3_commands(router: &mut Router) {
    // Note: BriefingCommand is now in phase6_commands.rs (EnhancedBriefingCommand)
    router.register(Arc::new(AlertsCommand));
    router.register(Arc::new(QueryCommand));
    router.register(Arc::new(WhoCommand));
    router.register(Arc::new(SearchCommand));
    // Phase 4 commands
    router.register(Arc::new(GraphRAGCommand));
    router.register(Arc::new(ContactCommand));
    router.register(Arc::new(AskCommand));
    // Phase 5 commands - Document Ingestion
    router.register(Arc::new(IngestCommand));
    router.register(Arc::new(IngestAICommand));
    router.register(Arc::new(LearnCommand));
}
