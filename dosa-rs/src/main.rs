mod llm;
mod knowledge;
mod router;
mod storage;
mod calendar;
mod reminders;
mod contacts;
mod intelligence;
mod completer;
mod integrations;
mod intent;

use anyhow::Result;
use rustyline::error::ReadlineError;
use rustyline::{Config, Editor};
use std::path::PathBuf;

use llm::{OllamaClient, ollama::Message};
use knowledge::KnowledgeGraph;
use router::{CommandContext, CommandResult, create_default_router, register_phase2_commands, register_phase3_commands, register_phase6_commands};
use storage::Database;
use calendar::Calendar;
use reminders::Reminders;
use intelligence::{AlertSystem, NaturalLanguageQuery, EntityExtractor};
use completer::DosaCompleter;
use intent::{IntentHandler, HandleResult};

const SYSTEM_PROMPT: &str = r#"You are DOSA (Digitally Optimized Smart Assistant), a helpful personal assistant with access to:
1. A knowledge graph about the user's contacts and organizations
2. A calendar with events
3. A task/reminder system

You help the user by:
- Answering questions using information from the knowledge graph, calendar, and tasks
- Helping them remember information about people and organizations they know
- Keeping track of their schedule and reminders
- Providing helpful, concise responses

When the user asks about people, events, or tasks, use the context provided to give accurate answers.
If you don't have information about something, say so and suggest they add it.

Be conversational but concise. Don't be overly verbose."#;

fn get_data_dir() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    
    // Try to use a data directory next to the executable, or fall back to current dir
    let data_dir = exe_dir.join("data");
    if data_dir.exists() || std::fs::create_dir_all(&data_dir).is_ok() {
        data_dir
    } else {
        PathBuf::from("./data")
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  DOSA - Digitally Optimized Smart Assistant                ║");
    println!("║  Type /help for available commands, or just chat!          ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize components
    let data_dir = get_data_dir();
    std::fs::create_dir_all(&data_dir)?;
    
    let db_path = data_dir.join("dosa.db");
    println!("📁 Database: {}", db_path.display());
    
    let db = Database::open(&db_path)?;
    
    // Initialize Phase 2 tables (calendar, reminders)
    Calendar::init_tables(&db)?;
    Reminders::init_tables(&db)?;
    
    let graph = KnowledgeGraph::new(db);
    
    // Initialize Ollama client
    let llm = OllamaClient::new("llama3.2:3b");
    
    // Check Ollama connection
    print!("🔌 Connecting to Ollama... ");
    if llm.health_check().await? {
        println!("✓ Connected (model: {})", llm.model());
    } else {
        println!("✗ Failed!");
        println!("   Make sure Ollama is running: ollama serve");
        return Ok(());
    }
    println!();

    // Create command router with all commands
    let mut router = create_default_router();
    register_phase2_commands(&mut router);
    register_phase3_commands(&mut router);
    register_phase6_commands(&mut router);
    
    // Show proactive alerts on startup
    let alert_system = AlertSystem::new(graph.database());
    if let Some(alerts) = alert_system.display_alerts()? {
        println!("{}", alerts);
    }
    
    // Create completer with entity names
    let mut completer = DosaCompleter::new();
    completer.refresh_entities(&graph);
    
    // Create readline editor with completion support
    let config = Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .auto_add_history(true)
        .build();
    let mut rl = Editor::with_config(config)?;
    rl.set_helper(Some(completer));
    
    // Try to load history
    let history_path = data_dir.join("history.txt");
    let _ = rl.load_history(&history_path);
    
    // Conversation history for context
    let mut conversation: Vec<Message> = vec![Message::system(SYSTEM_PROMPT)];
    
    // Commands that modify entities (require completer refresh)
    let entity_modifying_commands = [
        "/add ", "/delete ", "/rename ", "/learn ", "/ingest ", "/link "
    ];
    
    // Main loop
    loop {
        let readline = rl.readline("you> ");
        
        match readline {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                
                // Create command context
                let ctx = CommandContext {
                    graph: &graph,
                    llm: &llm,
                };
                
                // Try to route to a command first
                match router.route(input, &ctx)? {
                    CommandResult::Success(msg) => {
                        println!("\n{}\n", msg);
                        
                        // Refresh completer if entities were modified
                        let lower = input.to_lowercase();
                        if entity_modifying_commands.iter().any(|cmd| lower.starts_with(cmd)) {
                            if let Some(helper) = rl.helper_mut() {
                                helper.refresh_entities(&graph);
                            }
                        }
                    }
                    CommandResult::Error(msg) => {
                        println!("\n❌ {}\n", msg);
                    }
                    CommandResult::Exit => {
                        println!("\nGoodbye! 👋\n");
                        break;
                    }
                    CommandResult::NotHandled => {
                        // Check if this is a knowledge graph query first
                        let nl_query = NaturalLanguageQuery::new(&graph);
                        if nl_query.is_kg_query(input) {
                            let query_type = nl_query.parse_query(input);
                            match nl_query.execute(&query_type) {
                                Ok(result) => {
                                    println!("\n{}\n", result);
                                    continue;
                                }
                                Err(e) => {
                                    println!("\n❌ Query error: {}\n", e);
                                    continue;
                                }
                            }
                        }
                        
                        // Use the new intent handler for calendar/email intents
                        let intent_handler = IntentHandler::with_data_dir(&llm, &graph, &data_dir);
                        let handle_result = tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(
                                intent_handler.handle(input)
                            )
                        });
                        
                        match handle_result {
                            Ok(HandleResult::Handled(output)) => {
                                println!("\n{}\n", output);
                                continue;
                            }
                            Ok(HandleResult::NeedsAuth(service)) => {
                                let icon = if service == "email" { "📧" } else { "📅" };
                                println!("\n{} I can help with your {}!", icon, service);
                                println!("   To enable this, please authenticate with: /auth google\n");
                                continue;
                            }
                            Ok(HandleResult::Error(msg)) => {
                                println!("\n❌ {}\n", msg);
                                continue;
                            }
                            Ok(HandleResult::NotHandled) | Err(_) => {
                                // Fall through to LLM conversation
                            }
                        }
                        
                        // Not handled by intent system - send to LLM
                        print!("\ndosa> ");
                        
                        // Build context from all sources
                        let kg_summary = graph.get_summary()?;
                        let calendar = Calendar::new(graph.database());
                        let cal_summary = calendar.get_summary()?;
                        let reminders = Reminders::new(graph.database());
                        let task_summary = reminders.get_summary()?;
                        
                        // Build context string
                        let mut context_parts = Vec::new();
                        if !kg_summary.contains("No entities") {
                            context_parts.push(format!("Contacts:\n{}", kg_summary));
                        }
                        if !cal_summary.contains("No upcoming") {
                            context_parts.push(format!("Calendar:\n{}", cal_summary));
                        }
                        if !task_summary.contains("No pending") {
                            context_parts.push(format!("Tasks:\n{}", task_summary));
                        }
                        
                        let user_msg = if context_parts.is_empty() {
                            input.to_string()
                        } else {
                            format!(
                                "Context:\n{}\n\nUser: {}",
                                context_parts.join("\n\n"),
                                input
                            )
                        };
                        
                        // Add user message to conversation
                        conversation.push(Message::user(&user_msg));
                        
                        // Get response from LLM
                        match llm.chat(conversation.clone()).await {
                            Ok(response) => {
                                println!("{}\n", response);
                                
                                // Add assistant response to conversation
                                conversation.push(Message::assistant(&response));
                                
                                // Keep conversation history manageable (last 10 exchanges)
                                if conversation.len() > 21 {
                                    // Keep system prompt + last 10 exchanges (20 messages)
                                    let system = conversation[0].clone();
                                    let skip_count = conversation.len() - 20;
                                    conversation = std::iter::once(system)
                                        .chain(conversation.into_iter().skip(skip_count))
                                        .collect();
                                }
                                
                                // Try to extract entities from user input
                                let extractor = EntityExtractor::new(&graph);
                                if extractor.might_contain_entities(input) {
                                    let extraction = extractor.extract_patterns(input);
                                    if extraction.was_informative {
                                        if let Ok(actions) = extractor.apply_extraction(&extraction) {
                                            if !actions.is_empty() {
                                                println!("{}", extractor.display_extraction(&extraction));
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                println!("Error communicating with LLM: {}\n", e);
                            }
                        }
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("\nUse /exit to quit");
            }
            Err(ReadlineError::Eof) => {
                println!("\nGoodbye! 👋");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }
    
    // Save history
    let _ = rl.save_history(&history_path);
    
    Ok(())
}
