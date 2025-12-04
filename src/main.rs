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
mod proactive;
mod messaging;

#[cfg(feature = "voice")]
mod voice;

use anyhow::Result;
use rustyline::error::ReadlineError;
use rustyline::{Config, Editor};
use std::path::PathBuf;
use std::sync::Arc;

use llm::{OllamaClient, ollama::Message};
use knowledge::KnowledgeGraph;
use router::{CommandContext, CommandResult, create_default_router, register_phase2_commands, register_phase3_commands, register_phase6_commands};
use storage::Database;
use calendar::Calendar;
use reminders::Reminders;
use intelligence::{AlertSystem, NaturalLanguageQuery, EntityExtractor, ConversationContext, MultiHopEngine};
use integrations::WebSearch;
use completer::DosaCompleter;
use intent::{IntentHandler, HandleResult, HybridClassifier};
use messaging::{WhatsAppClient, MessageServer, FrontendType, IncomingMessage, OutgoingMessage};

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
    // Initialize logging
    env_logger::init();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let whatsapp_mode = args.iter().any(|a| a == "--whatsapp" || a == "-w");
    let help_mode = args.iter().any(|a| a == "--help" || a == "-h");

    if help_mode {
        println!("DOSA - Digitally Optimized Smart Assistant");
        println!();
        println!("Usage: dosa [OPTIONS]");
        println!();
        println!("Options:");
        println!("  --whatsapp, -w   Run in WhatsApp server mode");
        println!("  --help, -h       Show this help message");
        println!();
        println!("Environment variables (for WhatsApp mode):");
        println!("  WHATSAPP_PHONE_ID        Your WhatsApp Business Phone Number ID");
        println!("  WHATSAPP_TOKEN           Your WhatsApp Business API token");
        println!("  WHATSAPP_VERIFY_TOKEN    Webhook verification token");
        println!("  WHATSAPP_ALLOWED_NUMBERS Comma-separated allowed phone numbers");
        println!("  WEBHOOK_PORT             Port for webhook server (default: 8080)");
        return Ok(());
    }

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  DOSA - Digitally Optimized Smart Assistant                ║");
    if whatsapp_mode {
        println!("║  Running in WhatsApp mode                                  ║");
    } else {
        println!("║  Type /help for available commands, or just chat!          ║");
    }
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
    
    let graph = Arc::new(KnowledgeGraph::new(db));
    
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
    
    // Load NLU model ONCE at startup (not on every query)
    let nlu_classifier = Arc::new(HybridClassifier::try_with_onnx(
        Arc::new(llm.clone()),
        &data_dir
    ));
    
    // Show proactive alerts on startup
    let alert_system = AlertSystem::new(graph.database());
    if let Some(alerts) = alert_system.display_alerts()? {
        println!("{}", alerts);
    }
    
    // Run WhatsApp server mode if enabled
    if whatsapp_mode {
        return run_whatsapp_mode(
            graph,
            llm,
            router,
            nlu_classifier,
            data_dir,
        ).await;
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
    
    // Conversation history for context - includes recent turns for continuity
    let mut conversation: Vec<Message> = vec![Message::system(SYSTEM_PROMPT)];
    
    // Multi-turn context memory for calendar/email intents
    let mut intent_context = ConversationContext::new();
    
    // Track recent topics for follow-up questions
    let mut last_topic: Option<String> = None;
    let mut awaiting_followup = false;
    
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
                        // Check for multi-hop queries first (e.g., "email everyone who knows Alice")
                        let input_lower = input.to_lowercase();
                        if input_lower.contains("email") && input_lower.contains("who knows") {
                            let multihop = MultiHopEngine::new(None, None, Arc::clone(&graph));
                            let mut mh_context = ConversationContext::new();
                            
                            let result = tokio::task::block_in_place(|| {
                                tokio::runtime::Handle::current().block_on(
                                    multihop.process(input, &mut mh_context)
                                )
                            });
                            
                            match result {
                                Ok(mh_result) => {
                                    println!("\n{}\n", mh_result.message);
                                    continue;
                                }
                                Err(_) => {
                                    // Fall through to other handlers
                                }
                            }
                        }
                        
                        // Check for web search queries (weather, search, etc.)
                        if WebSearch::is_web_query(input) {
                            let web_search = WebSearch::new();
                            let result = tokio::task::block_in_place(|| {
                                tokio::runtime::Handle::current().block_on(
                                    web_search.quick_answer(input)
                                )
                            });
                            
                            match result {
                                Ok(answer) => {
                                    println!("\n{}\n", answer);
                                    continue;
                                }
                                Err(_) => {
                                    // Fall through to other handlers
                                }
                            }
                        }
                        
                        // Check if this is a knowledge graph query
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
                        
                        // Use the intent handler with cached NLU classifier (no reload)
                        let mut intent_handler = IntentHandler::with_shared_classifier(
                            &llm, &graph, Arc::clone(&nlu_classifier), &mut intent_context
                        );
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
                            Ok(HandleResult::NeedsInfo { question, .. }) => {
                                println!("\n❓ {}\n", question);
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

/// Run DOSA in WhatsApp server mode
/// This starts a webhook server to receive WhatsApp messages and responds via the API
async fn run_whatsapp_mode(
    graph: Arc<KnowledgeGraph>,
    llm: OllamaClient,
    router: router::Router,
    nlu_classifier: Arc<HybridClassifier>,
    data_dir: PathBuf,
) -> Result<()> {
    use messaging::whatsapp::WhatsAppConfig;
    use messaging::SamanthaHandler;
    use tokio::sync::mpsc;
    
    // Load WhatsApp configuration
    let wa_config = match WhatsAppConfig::from_env() {
        Ok(config) => config,
        Err(e) => {
            // Try loading from file
            let config_path = data_dir.join("whatsapp_config.json");
            if config_path.exists() {
                WhatsAppConfig::from_file(&config_path)?
            } else {
                eprintln!("❌ WhatsApp configuration error: {}", e);
                eprintln!();
                eprintln!("Please set environment variables or create config file:");
                eprintln!("  WHATSAPP_PHONE_ID=your_phone_number_id");
                eprintln!("  WHATSAPP_TOKEN=your_access_token");
                eprintln!("  WHATSAPP_VERIFY_TOKEN=your_verify_token");
                eprintln!("  WHATSAPP_ALLOWED_NUMBERS=+1234567890,+0987654321");
                eprintln!();
                eprintln!("Or create: {}", config_path.display());
                return Err(e);
            }
        }
    };

    let whatsapp = Arc::new(WhatsAppClient::new(wa_config));
    let webhook_port: u16 = std::env::var("WEBHOOK_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    println!("📱 WhatsApp integration enabled (Samantha mode)");
    println!("🔗 Webhook endpoint: http://your-server:{}/webhook", webhook_port);
    println!();
    println!("To configure in Meta Business Portal:");
    println!("  1. Go to your WhatsApp Business app settings");
    println!("  2. Set webhook URL to: https://your-domain:{}/webhook", webhook_port);
    println!("  3. Set verify token to match WHATSAPP_VERIFY_TOKEN");
    println!("  4. Subscribe to 'messages' webhook field");
    println!();
    
    // Create database connection for Samantha features
    let db_path = data_dir.join("dosa.db");
    let db = Arc::new(tokio::sync::Mutex::new(Database::open(&db_path)?));
    
    // Create Samantha handler
    let llm_arc = Arc::new(llm.clone());
    let samantha = SamanthaHandler::new(
        Arc::clone(&db),
        Arc::clone(&llm_arc),
        Arc::clone(&whatsapp),
    );
    
    // Create message channel for webhook -> processor communication
    let (msg_tx, mut msg_rx) = mpsc::channel::<IncomingMessage>(100);
    
    // Spawn webhook server
    let whatsapp_clone = Arc::clone(&whatsapp);
    let msg_tx_clone = msg_tx.clone();
    
    tokio::spawn(async move {
        if let Err(e) = run_webhook_server(whatsapp_clone, msg_tx_clone, webhook_port).await {
            eprintln!("Webhook server error: {}", e);
        }
    });

    // Track last check times
    let mut last_reminder_check = std::time::Instant::now();
    let mut last_onboard_check = std::time::Instant::now();
    let reminder_interval = std::time::Duration::from_secs(60);
    let onboard_interval = std::time::Duration::from_secs(3600);

    // Also keep CLI input available for local testing
    let (cli_tx, mut cli_rx) = mpsc::channel::<String>(10);
    
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        loop {
            let mut input = String::new();
            if stdin.read_line(&mut input).is_ok() {
                let input = input.trim().to_string();
                if !input.is_empty() {
                    if cli_tx.blocking_send(input).is_err() {
                        break;
                    }
                }
            }
        }
    });

    println!("Ready! Waiting for messages (CLI input also available)...");
    println!("Type /exit to quit");
    println!();

    // Conversation state per user (for non-Samantha flows)
    let mut conversations: std::collections::HashMap<String, Vec<Message>> = std::collections::HashMap::new();
    let mut intent_contexts: std::collections::HashMap<String, ConversationContext> = std::collections::HashMap::new();

    loop {
        tokio::select! {
            // Handle WhatsApp messages
            Some(msg) = msg_rx.recv() => {
                let sender_name = msg.sender_id.clone(); // WhatsApp gives us the phone number
                println!("📨 [WhatsApp] {}: {}", msg.sender_id, msg.text);
                
                // First try Samantha handler (reminders, greetings, memory extraction)
                let handled = match samantha.handle_message(&msg.sender_id, &msg.text, &sender_name).await {
                    Ok(handled) => handled,
                    Err(e) => {
                        log::error!("Samantha handler error: {}", e);
                        false
                    }
                };

                // If Samantha handled it, we're done
                if handled {
                    // Mark as read
                    if let Some(msg_id) = &msg.message_id {
                        let _ = whatsapp.mark_read(msg_id).await;
                    }
                    continue;
                }

                // Otherwise, get Samantha-style AI response
                let response = match samantha.get_ai_response(&msg.sender_id, &msg.text, &sender_name).await {
                    Ok(resp) => resp,
                    Err(e) => {
                        log::error!("AI response error: {}", e);
                        "I'm having a moment... give me a sec and try again? 💭".to_string()
                    }
                };

                // Send response via WhatsApp
                if let Err(e) = whatsapp.send_message(&msg.sender_id, &response).await {
                    eprintln!("Failed to send WhatsApp response: {}", e);
                } else {
                    println!("📤 [WhatsApp] -> {}: {}", msg.sender_id, truncate_for_log(&response, 100));
                }

                // Mark as read
                if let Some(msg_id) = &msg.message_id {
                    let _ = whatsapp.mark_read(msg_id).await;
                }
            }

            // Handle CLI input (uses standard DOSA processing)
            Some(input) = cli_rx.recv() => {
                if input == "/exit" {
                    println!("Goodbye! 👋");
                    break;
                }

                let response = process_message(
                    &input,
                    "cli_user",
                    &graph,
                    &llm,
                    &router,
                    &nlu_classifier,
                    &mut conversations,
                    &mut intent_contexts,
                ).await;

                println!("\ndosa> {}\n", response);
            }

            // Periodic reminder and greeting checks (every minute)
            _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {
                // Check reminders every minute
                if last_reminder_check.elapsed() >= reminder_interval {
                    if let Err(e) = samantha.check_reminders().await {
                        log::error!("Reminder check error: {}", e);
                    }
                    if let Err(e) = samantha.check_greetings().await {
                        log::error!("Greeting check error: {}", e);
                    }
                    last_reminder_check = std::time::Instant::now();
                }

                // Check onboarding every hour
                if last_onboard_check.elapsed() >= onboard_interval {
                    if let Err(e) = samantha.check_timezone_onboarding().await {
                        log::error!("Timezone onboarding check error: {}", e);
                    }
                    if let Err(e) = samantha.check_greeting_onboarding().await {
                        log::error!("Greeting onboarding check error: {}", e);
                    }
                    last_onboard_check = std::time::Instant::now();
                }
            }
        }
    }

    Ok(())
}

/// Process a message from any frontend
async fn process_message(
    input: &str,
    user_id: &str,
    graph: &Arc<KnowledgeGraph>,
    llm: &OllamaClient,
    router: &router::Router,
    nlu_classifier: &Arc<HybridClassifier>,
    conversations: &mut std::collections::HashMap<String, Vec<Message>>,
    intent_contexts: &mut std::collections::HashMap<String, ConversationContext>,
) -> String {
    // Get or create conversation for this user
    let conversation = conversations.entry(user_id.to_string())
        .or_insert_with(|| vec![Message::system(SYSTEM_PROMPT)]);
    let intent_context = intent_contexts.entry(user_id.to_string())
        .or_insert_with(ConversationContext::new);

    let ctx = CommandContext {
        graph,
        llm,
    };

    // Try command routing first
    match router.route(input, &ctx) {
        Ok(CommandResult::Success(msg)) => return msg,
        Ok(CommandResult::Error(msg)) => return format!("❌ {}", msg),
        Ok(CommandResult::Exit) => return "Goodbye! 👋".to_string(),
        Ok(CommandResult::NotHandled) => {}
        Err(e) => return format!("Error: {}", e),
    }

    // Try intent handler
    let mut intent_handler = IntentHandler::with_shared_classifier(
        llm, graph, Arc::clone(nlu_classifier), intent_context
    );

    match intent_handler.handle(input).await {
        Ok(HandleResult::Handled(response)) => {
            conversation.push(Message::user(input));
            conversation.push(Message::assistant(&response));
            trim_conversation(conversation);
            return response;
        }
        Ok(HandleResult::NeedsAuth(service)) => {
            return format!("🔐 Please authenticate with {} first. Use /auth {} in CLI mode.", service, service);
        }
        Ok(HandleResult::NeedsInfo { question, .. }) => {
            return format!("❓ {}", question);
        }
        Ok(HandleResult::NotHandled) => {}
        Ok(HandleResult::Error(e)) => return format!("Error: {}", e),
        Err(e) => return format!("Error: {}", e),
    }

    // Fall back to LLM conversation
    let user_msg = build_context_message(input, graph);
    conversation.push(Message::user(&user_msg));

    match llm.chat(conversation.clone()).await {
        Ok(response) => {
            conversation.push(Message::assistant(&response));
            trim_conversation(conversation);
            response
        }
        Err(e) => format!("Error: {}", e),
    }
}

/// Build context message with KG info
fn build_context_message(input: &str, _graph: &Arc<KnowledgeGraph>) -> String {
    // For now, just return the input as-is
    // The intent handler and NL query system handle context enrichment
    input.to_string()
}

/// Keep conversation history manageable
fn trim_conversation(conversation: &mut Vec<Message>) {
    if conversation.len() > 21 {
        let system = conversation[0].clone();
        let skip_count = conversation.len() - 20;
        *conversation = std::iter::once(system)
            .chain(conversation.iter().skip(skip_count).cloned())
            .collect();
    }
}

/// Truncate string for logging
fn truncate_for_log(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

/// Simple webhook server using tiny_http (no axum dependency needed)
async fn run_webhook_server(
    whatsapp: Arc<WhatsAppClient>,
    msg_tx: tokio::sync::mpsc::Sender<IncomingMessage>,
    port: u16,
) -> Result<()> {
    use std::io::Read;
    
    let addr = format!("0.0.0.0:{}", port);
    let server = tiny_http::Server::http(&addr)
        .map_err(|e| anyhow::anyhow!("Failed to start server: {}", e))?;
    
    println!("🌐 Webhook server listening on {}", addr);

    loop {
        let mut request = match server.recv() {
            Ok(req) => req,
            Err(e) => {
                eprintln!("Server error: {}", e);
                continue;
            }
        };

        let method = request.method().to_string();
        let url = request.url().to_string();

        // Health check
        if url == "/health" {
            let response = tiny_http::Response::from_string("OK");
            let _ = request.respond(response);
            continue;
        }

        // Webhook verification (GET)
        if method == "GET" && url.starts_with("/webhook") {
            let query = url.split('?').nth(1).unwrap_or("");
            let params: std::collections::HashMap<_, _> = query
                .split('&')
                .filter_map(|p| {
                    let mut parts = p.splitn(2, '=');
                    Some((parts.next()?, parts.next()?))
                })
                .collect();

            let mode = params.get("hub.mode").map(|s| urlencoding::decode(s).unwrap_or_default());
            let token = params.get("hub.verify_token").map(|s| urlencoding::decode(s).unwrap_or_default());
            let challenge = params.get("hub.challenge").map(|s| urlencoding::decode(s).unwrap_or_default());

            if mode.as_deref() == Some("subscribe") 
                && token.as_deref() == Some(whatsapp.verify_token()) 
            {
                if let Some(challenge) = challenge {
                    println!("✓ Webhook verified");
                    let response = tiny_http::Response::from_string(challenge.to_string());
                    let _ = request.respond(response);
                    continue;
                }
            }

            let response = tiny_http::Response::from_string("Forbidden")
                .with_status_code(403);
            let _ = request.respond(response);
            continue;
        }

        // Webhook messages (POST)
        if method == "POST" && url.starts_with("/webhook") {
            let mut body = String::new();
            let mut reader = request.as_reader();
            let _ = reader.read_to_string(&mut body);

            // Parse webhook payload
            if let Ok(payload) = serde_json::from_str::<messaging::whatsapp::WebhookPayload>(&body) {
                let messages = whatsapp.process_webhook(payload).await;
                for msg in messages {
                    if msg_tx.send(msg).await.is_err() {
                        eprintln!("Failed to queue message");
                    }
                }
            }

            let response = tiny_http::Response::from_string("OK");
            let _ = request.respond(response);
            continue;
        }

        // 404 for other routes
        let response = tiny_http::Response::from_string("Not Found")
            .with_status_code(404);
        let _ = request.respond(response);
    }
}