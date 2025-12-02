use anyhow::Result;
use std::sync::Arc;
use crate::knowledge::KnowledgeGraph;
use crate::llm::OllamaClient;

/// Result of executing a command
#[derive(Debug)]
pub enum CommandResult {
    /// Command executed successfully with a message
    Success(String),
    /// Command failed with an error message
    Error(String),
    /// Request to exit the application
    Exit,
    /// Command not recognized, should be passed to LLM
    NotHandled,
}

/// Context available to commands
pub struct CommandContext<'a> {
    pub graph: &'a KnowledgeGraph,
    pub llm: &'a OllamaClient,
}

/// Trait for implementing commands
pub trait Command: Send + Sync {
    /// Get the command name/pattern (e.g., "/add", "/list")
    fn name(&self) -> &str;
    
    /// Get a brief description of the command
    fn description(&self) -> &str;
    
    /// Get usage information
    fn usage(&self) -> &str;
    
    /// Check if this command matches the input
    fn matches(&self, input: &str) -> bool;
    
    /// Execute the command
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult>;
}

/// Command router that dispatches to registered commands
pub struct Router {
    commands: Vec<Arc<dyn Command>>,
}

impl Router {
    pub fn new() -> Self {
        Router {
            commands: Vec::new(),
        }
    }

    /// Register a new command
    pub fn register(&mut self, command: Arc<dyn Command>) {
        self.commands.push(command);
    }

    /// Route input to the appropriate command
    pub fn route(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let input = input.trim();
        
        // Check for exact command matches first
        for cmd in &self.commands {
            if cmd.matches(input) {
                return cmd.execute(input, ctx);
            }
        }

        // No command matched - return NotHandled so LLM can process it
        Ok(CommandResult::NotHandled)
    }

    /// Get help text for all commands
    pub fn help(&self) -> String {
        let mut help = String::from("Available commands:\n\n");
        
        for cmd in &self.commands {
            help.push_str(&format!("  {} - {}\n", cmd.name(), cmd.description()));
            help.push_str(&format!("    Usage: {}\n\n", cmd.usage()));
        }

        help.push_str("Any other input will be processed by the AI assistant.\n");
        help
    }

    /// List of registered commands
    pub fn list_commands(&self) -> Vec<&str> {
        self.commands.iter().map(|c| c.name()).collect()
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Built-in Commands
// ============================================================================

/// Exit command
pub struct ExitCommand;

impl Command for ExitCommand {
    fn name(&self) -> &str { "/exit" }
    fn description(&self) -> &str { "Exit the assistant" }
    fn usage(&self) -> &str { "/exit" }
    
    fn matches(&self, input: &str) -> bool {
        let input = input.trim().to_lowercase();
        input == "/exit" || input == "/quit" || input == "/q"
    }
    
    fn execute(&self, _input: &str, _ctx: &CommandContext) -> Result<CommandResult> {
        Ok(CommandResult::Exit)
    }
}

/// Help command - now static
pub struct HelpCommand;

impl Command for HelpCommand {
    fn name(&self) -> &str { "/help" }
    fn description(&self) -> &str { "Show available commands" }
    fn usage(&self) -> &str { "/help" }
    
    fn matches(&self, input: &str) -> bool {
        let input = input.trim().to_lowercase();
        input == "/help" || input == "/h" || input == "/?"
    }
    
    fn execute(&self, _input: &str, _ctx: &CommandContext) -> Result<CommandResult> {
        let help = r#"Available commands:

📋 KNOWLEDGE GRAPH - Entities
  /add person <name>     - Add a person
  /add org <name>        - Add an organization
  /add project <name>    - Add a project
  /add location <name>   - Add a location
  /add skill <name>      - Add a skill or technology
  /add industry <name>   - Add an industry or field
  /add product <name>    - Add a product or service
  /add goal <name>       - Add a goal or objective
  /add problem <name>    - Add a problem or challenge
  /add idea <name>       - Add an idea or concept
  
📋 KNOWLEDGE GRAPH - Relationships
  /link <A> works_at <B>   - Current employment
  /link <A> worked_at <B>  - Past employment
  /link <A> works_on <B>   - Current project
  /link <A> worked_on <B>  - Past project
  /link <A> knows <B>      - Link people
  /link <A> founded <B>    - Founder relationship
  /link <A> has_skill <B>  - Person has skill
  /link <A> in_industry <B>- In an industry/field
  /link <A> solves <B>     - Project solves problem
  /link <A> targets <B>    - Project targets goal
  /link <A> mentors <B>    - Mentorship
  /link <A> studied_at <B> - Education
  
📋 KNOWLEDGE GRAPH - Management
  /set <name> <key> <val>- Set a property on an entity
  /rename <old> to <new> - Rename an entity
  /delete <name>         - Delete an entity
  /unlink <A> <rel> <B>  - Remove a relationship
  /unset <name> <key>    - Remove a property
  /info <name>           - Get entity information
  /list [type]           - List entities (people|orgs|skills|...)
  /summary               - Get knowledge graph summary

📅 CALENDAR
  /event <title> at <time>      - Add a calendar event
  /events [today|tomorrow|week] - List events
  /today                        - Show today's schedule

✅ TASKS & REMINDERS
  /remind <task> [by <time>]    - Add a quick reminder
  /task <title> [priority:high] [by <time>]
  /tasks [all|overdue]          - List tasks
  /done <id or title>           - Mark task as done

👥 CONTACTS
  /contact <name>        - View contact details
  /contacts              - List all contacts

🧠 INTELLIGENCE (GraphRAG)
  /briefing              - Get your daily briefing
  /ask <question>        - Natural language query
  /query <question>      - Query knowledge graph
  /graph <query>         - Multi-hop graph traversal
  /who <query>           - Find people
  /search <term>         - Search knowledge graph

📚 DOCUMENT INGESTION
  /learn <text>          - Learn from text
  /ingest <file>         - Ingest a document file

👤 USER IDENTITY
  /iam <name>            - Set yourself as an entity
  /whoami                - Show current identity
  /forget me             - Clear identity

💭 MEMORY
  /remember <key> <val>  - Store a memory
  /memories              - List memories
  /forget <key>          - Forget a memory

🔧 SYSTEM
  /help                  - Show this help
  /exit                  - Exit

💡 Examples: "Who works at Google?" | "Who do I know?" | "What skills does Alice have?"
Any other input will be processed by the AI assistant."#;
        Ok(CommandResult::Success(help.to_string()))
    }
}

/// Add person command
pub struct AddPersonCommand;

impl Command for AddPersonCommand {
    fn name(&self) -> &str { "/add person" }
    fn description(&self) -> &str { "Add a person to the knowledge graph" }
    fn usage(&self) -> &str { "/add person <name>" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase().starts_with("/add person")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let name = input.trim()
            .strip_prefix("/add person")
            .or_else(|| input.trim().strip_prefix("/add Person"))
            .map(|s| s.trim())
            .unwrap_or("");
        
        if name.is_empty() {
            return Ok(CommandResult::Error("Please provide a name: /add person <name>".to_string()));
        }

        // Check if person already exists
        if ctx.graph.find_person(name)?.is_some() {
            return Ok(CommandResult::Error(format!("Person '{}' already exists", name)));
        }

        let entity = ctx.graph.add_person(name)?;
        Ok(CommandResult::Success(format!("Added person: {} (id: {})", entity.name, entity.id)))
    }
}

/// Add organization command
pub struct AddOrgCommand;

impl Command for AddOrgCommand {
    fn name(&self) -> &str { "/add org" }
    fn description(&self) -> &str { "Add an organization to the knowledge graph" }
    fn usage(&self) -> &str { "/add org <name>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/add org") || lower.starts_with("/add organization")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let input_lower = input.to_lowercase();
        let name = if input_lower.contains("organization") {
            input.trim()
                .get(input_lower.find("organization").unwrap() + "organization".len()..)
                .map(|s| s.trim())
                .unwrap_or("")
        } else {
            input.trim()
                .get(input_lower.find("org").unwrap() + "org".len()..)
                .map(|s| s.trim())
                .unwrap_or("")
        };
        
        if name.is_empty() {
            return Ok(CommandResult::Error("Please provide a name: /add org <name>".to_string()));
        }

        if ctx.graph.find_organization(name)?.is_some() {
            return Ok(CommandResult::Error(format!("Organization '{}' already exists", name)));
        }

        let entity = ctx.graph.add_organization(name)?;
        Ok(CommandResult::Success(format!("Added organization: {} (id: {})", entity.name, entity.id)))
    }
}

/// Add project command
pub struct AddProjectCommand;

impl Command for AddProjectCommand {
    fn name(&self) -> &str { "/add project" }
    fn description(&self) -> &str { "Add a project to the knowledge graph" }
    fn usage(&self) -> &str { "/add project <name>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/add project")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let name = input.trim()
            .strip_prefix("/add project")
            .or_else(|| input.trim().strip_prefix("/add Project"))
            .map(|s| s.trim())
            .unwrap_or("");
        
        if name.is_empty() {
            return Ok(CommandResult::Error("Please provide a name: /add project <name>".to_string()));
        }

        if ctx.graph.find_project(name)?.is_some() {
            return Ok(CommandResult::Error(format!("Project '{}' already exists", name)));
        }

        let entity = ctx.graph.add_project(name)?;
        Ok(CommandResult::Success(format!("Added project: {} (id: {})", entity.name, entity.id)))
    }
}

/// Add location command
pub struct AddLocationCommand;

impl Command for AddLocationCommand {
    fn name(&self) -> &str { "/add location" }
    fn description(&self) -> &str { "Add a location to the knowledge graph" }
    fn usage(&self) -> &str { "/add location <name>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/add location") || lower.starts_with("/add place")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let input_lower = input.to_lowercase();
        let name = if input_lower.contains("location") {
            input.trim()
                .get(input_lower.find("location").unwrap() + "location".len()..)
                .map(|s| s.trim())
                .unwrap_or("")
        } else {
            input.trim()
                .get(input_lower.find("place").unwrap() + "place".len()..)
                .map(|s| s.trim())
                .unwrap_or("")
        };
        
        if name.is_empty() {
            return Ok(CommandResult::Error("Please provide a name: /add location <name>".to_string()));
        }

        if ctx.graph.find_location(name)?.is_some() {
            return Ok(CommandResult::Error(format!("Location '{}' already exists", name)));
        }

        let entity = ctx.graph.add_location(name)?;
        Ok(CommandResult::Success(format!("Added location: {} (id: {})", entity.name, entity.id)))
    }
}

/// Add a skill/technology to the knowledge graph
pub struct AddSkillCommand;

impl Command for AddSkillCommand {
    fn name(&self) -> &str { "/add skill" }
    fn description(&self) -> &str { "Add a skill or technology to the knowledge graph" }
    fn usage(&self) -> &str { "/add skill <name>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/add skill") || lower.starts_with("/add tech")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let input_lower = input.to_lowercase();
        let name = if input_lower.contains("skill") {
            input.trim()
                .get(input_lower.find("skill").unwrap() + "skill".len()..)
                .map(|s| s.trim())
                .unwrap_or("")
        } else {
            input.trim()
                .get(input_lower.find("tech").unwrap() + "tech".len()..)
                .map(|s| s.trim())
                .unwrap_or("")
        };
        
        if name.is_empty() {
            return Ok(CommandResult::Error("Please provide a name: /add skill <name>".to_string()));
        }

        use crate::knowledge::entities::EntityType;
        let entity = ctx.graph.add_entity(EntityType::Skill, name)?;
        Ok(CommandResult::Success(format!("Added skill: {} (id: {})", entity.name, entity.id)))
    }
}

/// Add an industry/field to the knowledge graph
pub struct AddIndustryCommand;

impl Command for AddIndustryCommand {
    fn name(&self) -> &str { "/add industry" }
    fn description(&self) -> &str { "Add an industry or field to the knowledge graph" }
    fn usage(&self) -> &str { "/add industry <name>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/add industry") || lower.starts_with("/add field")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let input_lower = input.to_lowercase();
        let name = if input_lower.contains("industry") {
            input.trim()
                .get(input_lower.find("industry").unwrap() + "industry".len()..)
                .map(|s| s.trim())
                .unwrap_or("")
        } else {
            input.trim()
                .get(input_lower.find("field").unwrap() + "field".len()..)
                .map(|s| s.trim())
                .unwrap_or("")
        };
        
        if name.is_empty() {
            return Ok(CommandResult::Error("Please provide a name: /add industry <name>".to_string()));
        }

        use crate::knowledge::entities::EntityType;
        let entity = ctx.graph.add_entity(EntityType::Industry, name)?;
        Ok(CommandResult::Success(format!("Added industry: {} (id: {})", entity.name, entity.id)))
    }
}

/// Add a product to the knowledge graph
pub struct AddProductCommand;

impl Command for AddProductCommand {
    fn name(&self) -> &str { "/add product" }
    fn description(&self) -> &str { "Add a product or service to the knowledge graph" }
    fn usage(&self) -> &str { "/add product <name>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/add product") || lower.starts_with("/add service")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let input_lower = input.to_lowercase();
        let name = if input_lower.contains("product") {
            input.trim()
                .get(input_lower.find("product").unwrap() + "product".len()..)
                .map(|s| s.trim())
                .unwrap_or("")
        } else {
            input.trim()
                .get(input_lower.find("service").unwrap() + "service".len()..)
                .map(|s| s.trim())
                .unwrap_or("")
        };
        
        if name.is_empty() {
            return Ok(CommandResult::Error("Please provide a name: /add product <name>".to_string()));
        }

        use crate::knowledge::entities::EntityType;
        let entity = ctx.graph.add_entity(EntityType::Product, name)?;
        Ok(CommandResult::Success(format!("Added product: {} (id: {})", entity.name, entity.id)))
    }
}

/// Add a goal to the knowledge graph
pub struct AddGoalCommand;

impl Command for AddGoalCommand {
    fn name(&self) -> &str { "/add goal" }
    fn description(&self) -> &str { "Add a goal or objective to the knowledge graph" }
    fn usage(&self) -> &str { "/add goal <name>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/add goal") || lower.starts_with("/add objective")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let input_lower = input.to_lowercase();
        let name = if input_lower.contains("goal") {
            input.trim()
                .get(input_lower.find("goal").unwrap() + "goal".len()..)
                .map(|s| s.trim())
                .unwrap_or("")
        } else {
            input.trim()
                .get(input_lower.find("objective").unwrap() + "objective".len()..)
                .map(|s| s.trim())
                .unwrap_or("")
        };
        
        if name.is_empty() {
            return Ok(CommandResult::Error("Please provide a name: /add goal <name>".to_string()));
        }

        use crate::knowledge::entities::EntityType;
        let entity = ctx.graph.add_entity(EntityType::Goal, name)?;
        Ok(CommandResult::Success(format!("Added goal: {} (id: {})", entity.name, entity.id)))
    }
}

/// Add a problem to the knowledge graph
pub struct AddProblemCommand;

impl Command for AddProblemCommand {
    fn name(&self) -> &str { "/add problem" }
    fn description(&self) -> &str { "Add a problem or challenge to the knowledge graph" }
    fn usage(&self) -> &str { "/add problem <name>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/add problem") || lower.starts_with("/add challenge")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let input_lower = input.to_lowercase();
        let name = if input_lower.contains("problem") {
            input.trim()
                .get(input_lower.find("problem").unwrap() + "problem".len()..)
                .map(|s| s.trim())
                .unwrap_or("")
        } else {
            input.trim()
                .get(input_lower.find("challenge").unwrap() + "challenge".len()..)
                .map(|s| s.trim())
                .unwrap_or("")
        };
        
        if name.is_empty() {
            return Ok(CommandResult::Error("Please provide a name: /add problem <name>".to_string()));
        }

        use crate::knowledge::entities::EntityType;
        let entity = ctx.graph.add_entity(EntityType::Problem, name)?;
        Ok(CommandResult::Success(format!("Added problem: {} (id: {})", entity.name, entity.id)))
    }
}

/// Add an idea to the knowledge graph
pub struct AddIdeaCommand;

impl Command for AddIdeaCommand {
    fn name(&self) -> &str { "/add idea" }
    fn description(&self) -> &str { "Add an idea or concept to the knowledge graph" }
    fn usage(&self) -> &str { "/add idea <name>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/add idea") || lower.starts_with("/add concept")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let input_lower = input.to_lowercase();
        let name = if input_lower.contains("idea") {
            input.trim()
                .get(input_lower.find("idea").unwrap() + "idea".len()..)
                .map(|s| s.trim())
                .unwrap_or("")
        } else {
            input.trim()
                .get(input_lower.find("concept").unwrap() + "concept".len()..)
                .map(|s| s.trim())
                .unwrap_or("")
        };
        
        if name.is_empty() {
            return Ok(CommandResult::Error("Please provide a name: /add idea <name>".to_string()));
        }

        use crate::knowledge::entities::EntityType;
        let entity = ctx.graph.add_entity(EntityType::Idea, name)?;
        Ok(CommandResult::Success(format!("Added idea: {} (id: {})", entity.name, entity.id)))
    }
}

/// List entities command
pub struct ListCommand;

impl Command for ListCommand {
    fn name(&self) -> &str { "/list" }
    fn description(&self) -> &str { "List entities in the knowledge graph" }
    fn usage(&self) -> &str { "/list [people|orgs|projects|locations|skills|industries|products|goals|problems|ideas|all]" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase().starts_with("/list")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let arg = input.trim()
            .strip_prefix("/list")
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_default();

        let mut output = String::new();

        let show_people = arg.is_empty() || arg == "all" || arg == "people" || arg == "person";
        let show_orgs = arg.is_empty() || arg == "all" || arg == "orgs" || arg == "organizations";
        let show_projects = arg.is_empty() || arg == "all" || arg == "projects" || arg == "project";
        let show_locations = arg.is_empty() || arg == "all" || arg == "locations" || arg == "location" || arg == "places";
        let show_skills = arg == "all" || arg == "skills" || arg == "skill" || arg == "tech";
        let show_industries = arg == "all" || arg == "industries" || arg == "industry" || arg == "fields";
        let show_products = arg == "all" || arg == "products" || arg == "product" || arg == "services";
        let show_goals = arg == "all" || arg == "goals" || arg == "goal" || arg == "objectives";
        let show_problems = arg == "all" || arg == "problems" || arg == "problem" || arg == "challenges";
        let show_ideas = arg == "all" || arg == "ideas" || arg == "idea" || arg == "concepts";

        if show_people {
            let people = ctx.graph.list_people()?;
            if people.is_empty() {
                output.push_str("No people in knowledge graph.\n");
            } else {
                output.push_str("People:\n");
                for person in people {
                    output.push_str(&format!("  - {} (id: {})\n", person.name, person.id));
                }
            }
        }

        if show_orgs {
            let orgs = ctx.graph.list_organizations()?;
            if orgs.is_empty() {
                output.push_str("No organizations in knowledge graph.\n");
            } else {
                output.push_str("Organizations:\n");
                for org in orgs {
                    output.push_str(&format!("  - {} (id: {})\n", org.name, org.id));
                }
            }
        }

        if show_projects {
            let projects = ctx.graph.list_projects()?;
            if projects.is_empty() {
                output.push_str("No projects in knowledge graph.\n");
            } else {
                output.push_str("Projects:\n");
                for project in projects {
                    output.push_str(&format!("  - {} (id: {})\n", project.name, project.id));
                }
            }
        }

        if show_locations {
            let locations = ctx.graph.list_locations()?;
            if locations.is_empty() {
                output.push_str("No locations in knowledge graph.\n");
            } else {
                output.push_str("Locations:\n");
                for location in locations {
                    output.push_str(&format!("  - {} (id: {})\n", location.name, location.id));
                }
            }
        }

        if show_skills {
            let skills = ctx.graph.list_entities_by_type("skill")?;
            if !skills.is_empty() {
                output.push_str("Skills:\n");
                for skill in skills {
                    output.push_str(&format!("  - {} (id: {})\n", skill.name, skill.id));
                }
            }
        }

        if show_industries {
            let industries = ctx.graph.list_entities_by_type("industry")?;
            if !industries.is_empty() {
                output.push_str("Industries:\n");
                for industry in industries {
                    output.push_str(&format!("  - {} (id: {})\n", industry.name, industry.id));
                }
            }
        }

        if show_products {
            let products = ctx.graph.list_entities_by_type("product")?;
            if !products.is_empty() {
                output.push_str("Products:\n");
                for product in products {
                    output.push_str(&format!("  - {} (id: {})\n", product.name, product.id));
                }
            }
        }

        if show_goals {
            let goals = ctx.graph.list_entities_by_type("goal")?;
            if !goals.is_empty() {
                output.push_str("Goals:\n");
                for goal in goals {
                    output.push_str(&format!("  - {} (id: {})\n", goal.name, goal.id));
                }
            }
        }

        if show_problems {
            let problems = ctx.graph.list_entities_by_type("problem")?;
            if !problems.is_empty() {
                output.push_str("Problems:\n");
                for problem in problems {
                    output.push_str(&format!("  - {} (id: {})\n", problem.name, problem.id));
                }
            }
        }

        if show_ideas {
            let ideas = ctx.graph.list_entities_by_type("idea")?;
            if !ideas.is_empty() {
                output.push_str("Ideas:\n");
                for idea in ideas {
                    output.push_str(&format!("  - {} (id: {})\n", idea.name, idea.id));
                }
            }
        }

        Ok(CommandResult::Success(output))
    }
}

/// Link entities command
pub struct LinkCommand;

impl Command for LinkCommand {
    fn name(&self) -> &str { "/link" }
    fn description(&self) -> &str { "Create a relationship between entities" }
    fn usage(&self) -> &str { "/link <entity> works_at|worked_at|works_on|worked_on|knows|lives_in|located_in|manages|reports_to <entity>" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase().starts_with("/link")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let args = input.trim()
            .strip_prefix("/link")
            .map(|s| s.trim())
            .unwrap_or("");

        // Try to parse "X works_at Y" pattern (person -> organization)
        if let Some((person_name, org_name)) = parse_relationship(args, &["works_at", "works at"]) {
            let person = ctx.graph.find_person(&person_name)?
                .ok_or_else(|| anyhow::anyhow!("Person '{}' not found. Add them first with /add person {}", person_name, person_name))?;
            let org = ctx.graph.find_organization(&org_name)?
                .ok_or_else(|| anyhow::anyhow!("Organization '{}' not found. Add it first with /add org {}", org_name, org_name))?;
            
            ctx.graph.link_works_at(person.id, org.id)?;
            return Ok(CommandResult::Success(format!("Linked: {} works at {}", person.name, org.name)));
        }

        // Try to parse "X worked_at Y" pattern (person -> organization, past)
        if let Some((person_name, org_name)) = parse_relationship(args, &["worked_at", "worked at"]) {
            let person = ctx.graph.find_person(&person_name)?
                .ok_or_else(|| anyhow::anyhow!("Person '{}' not found. Add them first with /add person {}", person_name, person_name))?;
            let org = ctx.graph.find_organization(&org_name)?
                .ok_or_else(|| anyhow::anyhow!("Organization '{}' not found. Add it first with /add org {}", org_name, org_name))?;
            
            use crate::knowledge::entities::RelationshipType;
            ctx.graph.create_relationship(person.id, org.id, RelationshipType::WorkedAt)?;
            return Ok(CommandResult::Success(format!("Linked: {} worked at {} (past)", person.name, org.name)));
        }

        // Try to parse "X works_on Y" pattern (person -> project, current)
        if let Some((person_name, project_name)) = parse_relationship(args, &["works_on", "works on"]) {
            let person = ctx.graph.find_person(&person_name)?
                .ok_or_else(|| anyhow::anyhow!("Person '{}' not found. Add them first with /add person {}", person_name, person_name))?;
            let project = ctx.graph.find_project(&project_name)?
                .ok_or_else(|| anyhow::anyhow!("Project '{}' not found. Add it first with /add project {}", project_name, project_name))?;
            
            ctx.graph.link_works_on(person.id, project.id)?;
            return Ok(CommandResult::Success(format!("Linked: {} works on {}", person.name, project.name)));
        }

        // Try to parse "X worked_on Y" pattern (person -> project, past)
        if let Some((person_name, project_name)) = parse_relationship(args, &["worked_on", "worked on"]) {
            let person = ctx.graph.find_person(&person_name)?
                .ok_or_else(|| anyhow::anyhow!("Person '{}' not found. Add them first with /add person {}", person_name, person_name))?;
            let project = ctx.graph.find_project(&project_name)?
                .ok_or_else(|| anyhow::anyhow!("Project '{}' not found. Add it first with /add project {}", project_name, project_name))?;
            
            use crate::knowledge::entities::RelationshipType;
            ctx.graph.create_relationship(person.id, project.id, RelationshipType::WorkedOn)?;
            return Ok(CommandResult::Success(format!("Linked: {} worked on {} (past)", person.name, project.name)));
        }

        // Try to parse "X knows Y" pattern (person -> person)
        if let Some((person1_name, person2_name)) = parse_relationship(args, &["knows"]) {
            let person1 = ctx.graph.find_person(&person1_name)?
                .ok_or_else(|| anyhow::anyhow!("Person '{}' not found", person1_name))?;
            let person2 = ctx.graph.find_person(&person2_name)?
                .ok_or_else(|| anyhow::anyhow!("Person '{}' not found", person2_name))?;
            
            ctx.graph.link_knows(person1.id, person2.id)?;
            return Ok(CommandResult::Success(format!("Linked: {} knows {}", person1.name, person2.name)));
        }

        // Try to parse "X lives_in Y" pattern (person -> location)
        if let Some((person_name, location_name)) = parse_relationship(args, &["lives_in", "lives in"]) {
            let person = ctx.graph.find_person(&person_name)?
                .ok_or_else(|| anyhow::anyhow!("Person '{}' not found. Add them first with /add person {}", person_name, person_name))?;
            let location = ctx.graph.find_location(&location_name)?
                .ok_or_else(|| anyhow::anyhow!("Location '{}' not found. Add it first with /add location {}", location_name, location_name))?;
            
            use crate::knowledge::entities::RelationshipType;
            ctx.graph.create_relationship(person.id, location.id, RelationshipType::LivesIn)?;
            return Ok(CommandResult::Success(format!("Linked: {} lives in {}", person.name, location.name)));
        }

        // Try to parse "X located_in Y" pattern (person/org -> location)
        if let Some((entity_name, location_name)) = parse_relationship(args, &["located_in", "located in", "in"]) {
            let entity = ctx.graph.find_by_name(&entity_name)?
                .ok_or_else(|| anyhow::anyhow!("Entity '{}' not found", entity_name))?;
            let location = ctx.graph.find_location(&location_name)?
                .ok_or_else(|| anyhow::anyhow!("Location '{}' not found. Add it first with /add location {}", location_name, location_name))?;
            
            ctx.graph.link_located_in(entity.id, location.id)?;
            return Ok(CommandResult::Success(format!("Linked: {} is located in {}", entity.name, location.name)));
        }

        // Try to parse "X manages Y" pattern (person -> person/project)
        if let Some((manager_name, managed_name)) = parse_relationship(args, &["manages"]) {
            let manager = ctx.graph.find_person(&manager_name)?
                .ok_or_else(|| anyhow::anyhow!("Person '{}' not found", manager_name))?;
            let managed = ctx.graph.find_by_name(&managed_name)?
                .ok_or_else(|| anyhow::anyhow!("Entity '{}' not found", managed_name))?;
            
            ctx.graph.link_manages(manager.id, managed.id)?;
            return Ok(CommandResult::Success(format!("Linked: {} manages {}", manager.name, managed.name)));
        }

        // Try to parse "X reports_to Y" pattern (person -> person)
        if let Some((person_name, manager_name)) = parse_relationship(args, &["reports_to", "reports to"]) {
            let person = ctx.graph.find_person(&person_name)?
                .ok_or_else(|| anyhow::anyhow!("Person '{}' not found", person_name))?;
            let manager = ctx.graph.find_person(&manager_name)?
                .ok_or_else(|| anyhow::anyhow!("Person '{}' not found", manager_name))?;
            
            ctx.graph.link_reports_to(person.id, manager.id)?;
            return Ok(CommandResult::Success(format!("Linked: {} reports to {}", person.name, manager.name)));
        }

        Ok(CommandResult::Error(format!(
            "Could not parse relationship. Usage: {}", 
            self.usage()
        )))
    }
}

/// Info command - get information about an entity
pub struct InfoCommand;

impl Command for InfoCommand {
    fn name(&self) -> &str { "/info" }
    fn description(&self) -> &str { "Get information about an entity" }
    fn usage(&self) -> &str { "/info <name>" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase().starts_with("/info")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let name = input.trim()
            .strip_prefix("/info")
            .map(|s| s.trim())
            .unwrap_or("");
        
        if name.is_empty() {
            return Ok(CommandResult::Error("Please provide a name: /info <name>".to_string()));
        }

        if let Some(entity) = ctx.graph.find_by_name(name)? {
            let info = ctx.graph.get_entity_info(entity.id)?;
            Ok(CommandResult::Success(info))
        } else {
            Ok(CommandResult::Error(format!("Entity '{}' not found", name)))
        }
    }
}

/// Set property command
pub struct SetCommand;

impl Command for SetCommand {
    fn name(&self) -> &str { "/set" }
    fn description(&self) -> &str { "Set a property on an entity" }
    fn usage(&self) -> &str { "/set <name> <property> <value>" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase().starts_with("/set")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let args = input.trim()
            .strip_prefix("/set")
            .map(|s| s.trim())
            .unwrap_or("");
        
        // Parse: name property value (property and value are last two space-separated items)
        // This allows names with spaces like "Adipratama Prabaswara"
        let parts: Vec<&str> = args.rsplitn(3, ' ').collect();
        if parts.len() < 3 {
            return Ok(CommandResult::Error(format!("Usage: {}\nExample: /set John email john@example.com", self.usage())));
        }

        let value = parts[0];
        let property = parts[1];
        let name = parts[2];

        if let Some(entity) = ctx.graph.find_by_name(name)? {
            ctx.graph.set_property(entity.id, property, value)?;
            Ok(CommandResult::Success(format!("✓ Set {}.{} = {}", name, property, value)))
        } else {
            Ok(CommandResult::Error(format!("Entity '{}' not found", name)))
        }
    }
}

/// Summary command - get knowledge graph summary
pub struct SummaryCommand;

impl Command for SummaryCommand {
    fn name(&self) -> &str { "/summary" }
    fn description(&self) -> &str { "Get a summary of the knowledge graph" }
    fn usage(&self) -> &str { "/summary" }
    
    fn matches(&self, input: &str) -> bool {
        let input = input.trim().to_lowercase();
        input == "/summary" || input == "/kg" || input == "/graph"
    }
    
    fn execute(&self, _input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let summary = ctx.graph.get_summary()?;
        Ok(CommandResult::Success(summary))
    }
}

/// Rename an entity
pub struct RenameCommand;

impl Command for RenameCommand {
    fn name(&self) -> &str { "/rename" }
    fn description(&self) -> &str { "Rename an entity in the knowledge graph" }
    fn usage(&self) -> &str { "/rename <old name> to <new name>" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase().starts_with("/rename")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let args = input.trim()
            .strip_prefix("/rename")
            .map(|s| s.trim())
            .unwrap_or("");
        
        // Parse "old name to new name" pattern
        let parts: Vec<&str> = args.split(" to ").collect();
        if parts.len() != 2 {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        let old_name = parts[0].trim();
        let new_name = parts[1].trim();

        if old_name.is_empty() || new_name.is_empty() {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        match ctx.graph.rename_by_name(old_name, new_name)? {
            Some(_id) => Ok(CommandResult::Success(format!("✓ Renamed '{}' to '{}'", old_name, new_name))),
            None => Ok(CommandResult::Error(format!("Entity '{}' not found", old_name))),
        }
    }
}

/// Delete an entity
pub struct DeleteCommand;

impl Command for DeleteCommand {
    fn name(&self) -> &str { "/delete" }
    fn description(&self) -> &str { "Delete an entity from the knowledge graph" }
    fn usage(&self) -> &str { "/delete <name>" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase().starts_with("/delete")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let name = input.trim()
            .strip_prefix("/delete")
            .map(|s| s.trim())
            .unwrap_or("");

        if name.is_empty() {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        match ctx.graph.delete_by_name(name)? {
            Some(_id) => Ok(CommandResult::Success(format!("✓ Deleted entity '{}' and all its relationships", name))),
            None => Ok(CommandResult::Error(format!("Entity '{}' not found", name))),
        }
    }
}

/// Remove a relationship between entities
pub struct UnlinkCommand;

impl Command for UnlinkCommand {
    fn name(&self) -> &str { "/unlink" }
    fn description(&self) -> &str { "Remove a relationship between two entities" }
    fn usage(&self) -> &str { "/unlink <entity> <relationship> <entity>" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase().starts_with("/unlink")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let args = input.trim()
            .strip_prefix("/unlink")
            .map(|s| s.trim())
            .unwrap_or("");
        
        // Try to parse relationship patterns
        let rel_patterns = [
            ("works_at", "WORKS_AT"), ("works at", "WORKS_AT"),
            ("works_on", "WORKS_ON"), ("works on", "WORKS_ON"),
            ("knows", "KNOWS"),
            ("located_in", "LOCATED_IN"), ("located in", "LOCATED_IN"),
            ("lives_in", "LIVES_IN"), ("lives in", "LIVES_IN"),
            ("manages", "MANAGES"),
            ("reports_to", "REPORTS_TO"), ("reports to", "REPORTS_TO"),
            ("member_of", "MEMBER_OF"), ("member of", "MEMBER_OF"),
        ];

        for (pattern, rel_type) in rel_patterns {
            if let Some((from, to)) = parse_relationship(args, &[pattern]) {
                let deleted = ctx.graph.delete_relationship(&from, &to, rel_type)?;
                if deleted {
                    return Ok(CommandResult::Success(format!("✓ Removed: {} {} {}", from, pattern.to_uppercase(), to)));
                } else {
                    return Ok(CommandResult::Error(format!("Relationship not found: {} {} {}", from, pattern, to)));
                }
            }
        }
        
        Ok(CommandResult::Error(format!("Usage: {}\nExamples:\n  /unlink Alice works_at Google\n  /unlink Bob knows Charlie", self.usage())))
    }
}

/// Remove a property from an entity
pub struct UnsetCommand;

impl Command for UnsetCommand {
    fn name(&self) -> &str { "/unset" }
    fn description(&self) -> &str { "Remove a property from an entity" }
    fn usage(&self) -> &str { "/unset <name> <property>" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase().starts_with("/unset")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let args = input.trim()
            .strip_prefix("/unset")
            .map(|s| s.trim())
            .unwrap_or("");
        
        // Parse: name property (property is the last word, name is everything before)
        let parts: Vec<&str> = args.rsplitn(2, ' ').collect();
        if parts.len() < 2 {
            return Ok(CommandResult::Error(format!("Usage: {}\nExample: /unset John email", self.usage())));
        }

        let property = parts[0];
        let name = parts[1];

        if let Some(entity) = ctx.graph.find_by_name(name)? {
            let deleted = ctx.graph.delete_property(entity.id, property)?;
            if deleted {
                Ok(CommandResult::Success(format!("✓ Removed property '{}' from '{}'", property, name)))
            } else {
                Ok(CommandResult::Error(format!("Property '{}' not found on entity '{}'", property, name)))
            }
        } else {
            Ok(CommandResult::Error(format!("Entity '{}' not found", name)))
        }
    }
}

/// Set user identity - "I am X"
pub struct IAmCommand;

impl Command for IAmCommand {
    fn name(&self) -> &str { "/iam" }
    fn description(&self) -> &str { "Set your identity for first-person queries" }
    fn usage(&self) -> &str { "/iam <your name>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/iam ") || lower.starts_with("/i am ")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let name = input.trim()
            .strip_prefix("/iam")
            .or_else(|| input.trim().strip_prefix("/i am"))
            .map(|s| s.trim())
            .unwrap_or("");

        if name.is_empty() {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        match ctx.graph.set_user_identity(name)? {
            Some(entity) => {
                let mut response = format!("👤 Identity set: You are now '{}'\n", entity.name);
                response.push_str("\nYou can now use first-person queries like:\n");
                response.push_str("  • \"Who do I know?\"\n");
                response.push_str("  • \"Where do I work?\"\n");
                response.push_str("  • \"What are my tasks?\"\n");
                response.push_str("  • \"Show my connections\"");
                Ok(CommandResult::Success(response))
            }
            None => {
                // Offer to create the person
                Ok(CommandResult::Error(format!(
                    "Person '{}' not found in the knowledge graph.\n\
                    First add yourself with: /add person {}\n\
                    Then set your identity with: /iam {}", 
                    name, name, name
                )))
            }
        }
    }
}

/// Show current user identity
pub struct WhoAmICommand;

impl Command for WhoAmICommand {
    fn name(&self) -> &str { "/whoami" }
    fn description(&self) -> &str { "Show your current identity" }
    fn usage(&self) -> &str { "/whoami" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower == "/whoami" || lower == "/who am i"
    }
    
    fn execute(&self, _input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        match ctx.graph.get_user_identity()? {
            Some(entity) => {
                let info = ctx.graph.get_entity_info(entity.id)?;
                Ok(CommandResult::Success(format!("👤 You are: {}\n\n{}", entity.name, info)))
            }
            None => {
                Ok(CommandResult::Success(
                    "No identity set. Use /iam <name> to set yourself as an entity.".to_string()
                ))
            }
        }
    }
}

/// Clear user identity
pub struct ForgetMeCommand;

impl Command for ForgetMeCommand {
    fn name(&self) -> &str { "/forget me" }
    fn description(&self) -> &str { "Clear your identity" }
    fn usage(&self) -> &str { "/forget me" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower == "/forget me" || lower == "/forgetme"
    }
    
    fn execute(&self, _input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        if ctx.graph.clear_user_identity()? {
            Ok(CommandResult::Success("👤 Identity cleared. First-person pronouns will no longer be resolved.".to_string()))
        } else {
            Ok(CommandResult::Success("No identity was set.".to_string()))
        }
    }
}

// ============================================================================
// Memory Commands - Agent personality and snippets
// ============================================================================

/// Store a memory/personality snippet for the agent
pub struct RememberCommand;

impl Command for RememberCommand {
    fn name(&self) -> &str { "/remember" }
    fn description(&self) -> &str { "Store a memory or personality note for the assistant" }
    fn usage(&self) -> &str { "/remember <key> <value>" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase().starts_with("/remember ")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let rest = input.trim().strip_prefix("/remember").unwrap().trim();
        
        // Parse as "key value" or "key = value"
        let (key, value) = if rest.contains('=') {
            let parts: Vec<&str> = rest.splitn(2, '=').collect();
            if parts.len() != 2 {
                return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
            }
            (parts[0].trim().to_string(), parts[1].trim().to_string())
        } else {
            let parts: Vec<&str> = rest.splitn(2, char::is_whitespace).collect();
            if parts.len() != 2 {
                return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
            }
            (parts[0].trim().to_string(), parts[1].trim().to_string())
        };
        
        if key.is_empty() || value.is_empty() {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }
        
        // Store with memory: prefix
        let mem_key = format!("memory:{}", key.to_lowercase().replace(' ', "_"));
        ctx.graph.database().set_setting(&mem_key, &value)?;
        
        Ok(CommandResult::Success(format!("🧠 Remembered: {} = {}", key, value)))
    }
}

/// List all stored memories
pub struct MemoriesCommand;

impl Command for MemoriesCommand {
    fn name(&self) -> &str { "/memories" }
    fn description(&self) -> &str { "List all stored memories and personality notes" }
    fn usage(&self) -> &str { "/memories" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower == "/memories" || lower == "/memory"
    }
    
    fn execute(&self, _input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let all_settings = ctx.graph.database().list_settings_by_prefix("memory:")?;
        
        if all_settings.is_empty() {
            return Ok(CommandResult::Success(
                "🧠 No memories stored yet.\n\nUse /remember <key> <value> to store memories.".to_string()
            ));
        }
        
        let mut output = String::from("🧠 Stored Memories:\n\n");
        for (key, value) in all_settings {
            let display_key = key.strip_prefix("memory:").unwrap_or(&key);
            output.push_str(&format!("  • {}: {}\n", display_key, value));
        }
        
        Ok(CommandResult::Success(output))
    }
}

/// Forget a specific memory
pub struct ForgetMemoryCommand;

impl Command for ForgetMemoryCommand {
    fn name(&self) -> &str { "/forget" }
    fn description(&self) -> &str { "Forget a stored memory (use '/forget me' for identity)" }
    fn usage(&self) -> &str { "/forget <key>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/forget ") && lower != "/forget me"
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let key = input.trim().strip_prefix("/forget").unwrap().trim();
        
        if key.is_empty() || key.to_lowercase() == "me" {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }
        
        let mem_key = format!("memory:{}", key.to_lowercase().replace(' ', "_"));
        
        if ctx.graph.database().delete_setting(&mem_key)? {
            Ok(CommandResult::Success(format!("🧠 Forgot: {}", key)))
        } else {
            Ok(CommandResult::Error(format!("Memory '{}' not found.", key)))
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse a relationship pattern like "X relationship Y"
fn parse_relationship(input: &str, patterns: &[&str]) -> Option<(String, String)> {
    let input_lower = input.to_lowercase();
    
    for pattern in patterns {
        if let Some(pos) = input_lower.find(pattern) {
            let before = input[..pos].trim().to_string();
            let after = input[pos + pattern.len()..].trim().to_string();
            
            if !before.is_empty() && !after.is_empty() {
                return Some((before, after));
            }
        }
    }
    None
}

/// Create a router with all built-in commands
pub fn create_default_router() -> Router {
    let mut router = Router::new();
    
    // Register commands (order matters - more specific patterns first)
    router.register(Arc::new(ExitCommand));
    router.register(Arc::new(HelpCommand));
    router.register(Arc::new(AddPersonCommand));
    router.register(Arc::new(AddOrgCommand));
    router.register(Arc::new(AddProjectCommand));
    router.register(Arc::new(AddLocationCommand));
    router.register(Arc::new(AddSkillCommand));
    router.register(Arc::new(AddIndustryCommand));
    router.register(Arc::new(AddProductCommand));
    router.register(Arc::new(AddGoalCommand));
    router.register(Arc::new(AddProblemCommand));
    router.register(Arc::new(AddIdeaCommand));
    router.register(Arc::new(ListCommand));
    router.register(Arc::new(LinkCommand));
    router.register(Arc::new(InfoCommand));
    router.register(Arc::new(SetCommand));
    router.register(Arc::new(RenameCommand));
    router.register(Arc::new(DeleteCommand));
    router.register(Arc::new(UnlinkCommand));
    router.register(Arc::new(UnsetCommand));
    router.register(Arc::new(IAmCommand));
    router.register(Arc::new(WhoAmICommand));
    router.register(Arc::new(ForgetMeCommand));
    router.register(Arc::new(RememberCommand));
    router.register(Arc::new(MemoriesCommand));
    router.register(Arc::new(ForgetMemoryCommand));
    router.register(Arc::new(SummaryCommand));
    
    router
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_relationship() {
        let result = parse_relationship("John works_at Google", &["works_at"]);
        assert_eq!(result, Some(("John".to_string(), "Google".to_string())));

        let result = parse_relationship("Alice knows Bob", &["knows"]);
        assert_eq!(result, Some(("Alice".to_string(), "Bob".to_string())));
    }

    #[test]
    fn test_exit_command_matches() {
        let cmd = ExitCommand;
        assert!(cmd.matches("/exit"));
        assert!(cmd.matches("/quit"));
        assert!(cmd.matches("/q"));
        assert!(!cmd.matches("exit"));
    }
}
