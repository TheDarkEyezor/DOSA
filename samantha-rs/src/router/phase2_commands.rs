use anyhow::Result;
use std::sync::Arc;
use chrono::Local;

use crate::router::commands::{Command, CommandContext, CommandResult, Router};
use crate::calendar::{Calendar, events::parse_datetime};
use crate::reminders::{Reminders, TaskPriority};
use crate::contacts::ContactManager;

// ============================================================================
// Calendar Commands
// ============================================================================

/// Add event command
pub struct EventAddCommand;

impl Command for EventAddCommand {
    fn name(&self) -> &str { "/event" }
    fn description(&self) -> &str { "Add a calendar event" }
    fn usage(&self) -> &str { "/event <title> at <datetime>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        // Match /event but not /events
        (lower.starts_with("/event ") && !lower.starts_with("/events")) ||
        lower.starts_with("/event add") || lower.starts_with("/event new")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let args = input.trim()
            .to_lowercase()
            .replace("/event add", "")
            .replace("/event new", "")
            .replace("/event", "");
        let args = args.trim();

        if args.is_empty() {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        // Parse "title at datetime [at location]"
        let (title, datetime_str, location) = parse_event_args(args);
        
        let start_time = match parse_datetime(&datetime_str) {
            Some(dt) => dt,
            None => {
                return Ok(CommandResult::Error(format!(
                    "Could not parse datetime '{}'. Try formats like 'tomorrow 2pm' or 'dec 5 3:30pm'",
                    datetime_str
                )));
            }
        };

        let calendar = Calendar::new(ctx.graph.database());
        let event = calendar.add_event(
            &title,
            start_time,
            None,
            None,
            location.as_deref(),
        )?;

        Ok(CommandResult::Success(format!(
            "📅 Event added:\n{}",
            event.display()
        )))
    }
}

/// Parse event arguments
fn parse_event_args(input: &str) -> (String, String, Option<String>) {
    // Look for "at" to separate title from datetime
    // Format: "title at datetime [at location]"
    
    let parts: Vec<&str> = input.splitn(2, " at ").collect();
    
    if parts.len() < 2 {
        // No "at" found, treat entire input as title with default time
        return (input.to_string(), "today".to_string(), None);
    }

    let title = parts[0].trim().to_string();
    let rest = parts[1].trim();

    // Check if there's a location (another "at" in the rest)
    if let Some(loc_pos) = rest.rfind(" at ") {
        let datetime_str = rest[..loc_pos].trim().to_string();
        let location = rest[loc_pos + 4..].trim().to_string();
        (title, datetime_str, Some(location))
    } else {
        (title, rest.to_string(), None)
    }
}

/// List events command
pub struct EventListCommand;

impl Command for EventListCommand {
    fn name(&self) -> &str { "/event list" }
    fn description(&self) -> &str { "List calendar events" }
    fn usage(&self) -> &str { "/event list [today|tomorrow|week]" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/event list") || lower.starts_with("/events")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let calendar = Calendar::new(ctx.graph.database());
        
        let arg = input.trim()
            .to_lowercase()
            .replace("/event list", "")
            .replace("/events", "")
            .trim()
            .to_string();

        let (events, label) = match arg.as_str() {
            "today" | "" => (calendar.get_today_events()?, "Today's events"),
            "tomorrow" => (calendar.get_tomorrow_events()?, "Tomorrow's events"),
            "week" => (calendar.get_week_events()?, "This week's events"),
            _ => (calendar.get_upcoming_events(10)?, "Upcoming events"),
        };

        if events.is_empty() {
            return Ok(CommandResult::Success(format!("No events found for '{}'.", arg)));
        }

        let mut output = format!("{}:\n\n", label);
        for event in events {
            output.push_str(&event.display());
            output.push_str("\n\n");
        }

        Ok(CommandResult::Success(output))
    }
}

/// Today's events shortcut
pub struct EventTodayCommand;

impl Command for EventTodayCommand {
    fn name(&self) -> &str { "/today" }
    fn description(&self) -> &str { "Show today's events and tasks" }
    fn usage(&self) -> &str { "/today" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase() == "/today"
    }
    
    fn execute(&self, _input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let calendar = Calendar::new(ctx.graph.database());
        let reminders = Reminders::new(ctx.graph.database());
        
        let events = calendar.get_today_events()?;
        let tasks = reminders.get_tasks_due_soon()?;
        let overdue = reminders.get_overdue_tasks()?;

        let mut output = format!("📅 Today - {}\n\n", Local::now().format("%A, %B %d, %Y"));

        if !overdue.is_empty() {
            output.push_str("⚠️ OVERDUE:\n");
            for task in &overdue {
                output.push_str(&format!("  {}\n", task.display_short()));
            }
            output.push('\n');
        }

        if events.is_empty() {
            output.push_str("No events scheduled for today.\n");
        } else {
            output.push_str("Events:\n");
            for event in &events {
                output.push_str(&format!("  {}\n", event.display_short()));
            }
        }

        if !tasks.is_empty() {
            output.push_str("\nTasks due soon:\n");
            for task in &tasks {
                output.push_str(&format!("  {}\n", task.display_short()));
            }
        }

        Ok(CommandResult::Success(output))
    }
}

// ============================================================================
// Reminder Commands
// ============================================================================

/// Quick remind command
pub struct RemindCommand;

impl Command for RemindCommand {
    fn name(&self) -> &str { "/remind" }
    fn description(&self) -> &str { "Add a quick reminder" }
    fn usage(&self) -> &str { "/remind <task> [by <deadline>]" }
    
    fn matches(&self, input: &str) -> bool {
        input.trim().to_lowercase().starts_with("/remind")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let args = input.trim()
            .strip_prefix("/remind")
            .map(|s| s.trim())
            .unwrap_or("");

        if args.is_empty() {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        // Parse "task by deadline" or just "task"
        let (title, deadline) = if let Some(pos) = args.to_lowercase().find(" by ") {
            let title = args[..pos].trim();
            let deadline_str = args[pos + 4..].trim();
            (title, parse_datetime(deadline_str))
        } else {
            (args, None)
        };

        let reminders = Reminders::new(ctx.graph.database());
        let task = reminders.remind(title, deadline)?;

        let deadline_msg = deadline
            .map(|d| format!(" (due {})", d.format("%b %d, %I:%M %p")))
            .unwrap_or_default();

        Ok(CommandResult::Success(format!(
            "✅ Reminder added: {}{}",
            task.title,
            deadline_msg
        )))
    }
}

/// Add task with priority command
pub struct TaskAddCommand;

impl Command for TaskAddCommand {
    fn name(&self) -> &str { "/task" }
    fn description(&self) -> &str { "Add a task with priority" }
    fn usage(&self) -> &str { "/task <title> [priority:high] [by <deadline>]" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        // Match /task but not /tasks
        (lower.starts_with("/task ") && !lower.starts_with("/tasks")) ||
        lower.starts_with("/task add") || lower.starts_with("/task new")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let args = input.trim()
            .to_lowercase()
            .replace("/task add", "")
            .replace("/task new", "")
            .replace("/task", "");
        let args = args.trim();

        if args.is_empty() {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        // Extract priority if specified
        let (args, priority) = if args.contains("priority:") {
            let mut priority = TaskPriority::Medium;
            let mut cleaned = args.to_string();
            
            for p in ["priority:urgent", "priority:high", "priority:medium", "priority:low"] {
                if args.contains(p) {
                    priority = TaskPriority::from_str(p.strip_prefix("priority:").unwrap());
                    cleaned = cleaned.replace(p, "").trim().to_string();
                    break;
                }
            }
            (cleaned, priority)
        } else {
            (args.to_string(), TaskPriority::Medium)
        };

        // Parse deadline
        let (title, deadline) = if let Some(pos) = args.to_lowercase().find(" by ") {
            let title = args[..pos].trim();
            let deadline_str = args[pos + 4..].trim();
            (title.to_string(), parse_datetime(deadline_str))
        } else {
            (args, None)
        };

        let reminders = Reminders::new(ctx.graph.database());
        let task = reminders.add_task(&title, None, deadline, priority)?;

        Ok(CommandResult::Success(format!(
            "✅ Task added:\n{}",
            task.display()
        )))
    }
}

/// List tasks command
pub struct TaskListCommand;

impl Command for TaskListCommand {
    fn name(&self) -> &str { "/tasks" }
    fn description(&self) -> &str { "List tasks" }
    fn usage(&self) -> &str { "/tasks [all|overdue]" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower == "/tasks" || lower.starts_with("/tasks ")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let reminders = Reminders::new(ctx.graph.database());
        
        let arg = input.trim()
            .strip_prefix("/tasks")
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_default();

        let (tasks, label) = match arg.as_str() {
            "all" => (reminders.get_active_tasks()?, "All active tasks"),
            "overdue" => (reminders.get_overdue_tasks()?, "Overdue tasks"),
            _ => (reminders.get_pending_tasks()?, "Pending tasks"),
        };

        if tasks.is_empty() {
            return Ok(CommandResult::Success("No tasks found.".to_string()));
        }

        let mut output = format!("{}:\n\n", label);
        for task in tasks {
            output.push_str(&format!("[{}] {}\n", task.id, task.display()));
            output.push('\n');
        }

        Ok(CommandResult::Success(output))
    }
}

/// Complete task command
pub struct TaskDoneCommand;

impl Command for TaskDoneCommand {
    fn name(&self) -> &str { "/done" }
    fn description(&self) -> &str { "Mark a task as done" }
    fn usage(&self) -> &str { "/done <task_id or title>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/done") || lower.starts_with("/task done")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let args = input.trim()
            .to_lowercase()
            .replace("/task done", "")
            .replace("/done", "");
        let args = args.trim();

        if args.is_empty() {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        let reminders = Reminders::new(ctx.graph.database());

        // Try to parse as ID first
        if let Ok(id) = args.parse::<i64>() {
            if reminders.complete_task(id)? {
                return Ok(CommandResult::Success(format!("✅ Task {} marked as done!", id)));
            } else {
                return Ok(CommandResult::Error(format!("Task {} not found", id)));
            }
        }

        // Try to find by title
        if let Some(task) = reminders.find_task_by_title(args)? {
            reminders.complete_task(task.id)?;
            return Ok(CommandResult::Success(format!("✅ Task '{}' marked as done!", task.title)));
        }

        Ok(CommandResult::Error(format!("Task '{}' not found", args)))
    }
}

// ============================================================================
// Contact Commands
// ============================================================================

/// Add contact command
pub struct ContactAddCommand;

impl Command for ContactAddCommand {
    fn name(&self) -> &str { "/contact add" }
    fn description(&self) -> &str { "Add a new contact" }
    fn usage(&self) -> &str { "/contact add <name>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/contact add") || lower.starts_with("/contact new")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let name = input.trim()
            .to_lowercase()
            .replace("/contact add", "")
            .replace("/contact new", "");
        let name = name.trim();

        if name.is_empty() {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        // Capitalize name properly
        let name: String = name.split_whitespace()
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    Some(first) => first.to_uppercase().chain(chars).collect(),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ");

        let contacts = ContactManager::new(ctx.graph);
        let contact = contacts.add_contact(&name)?;

        Ok(CommandResult::Success(format!(
            "👤 Contact added: {}\n\nSet details with:\n  /set {} email <email>\n  /set {} phone <phone>",
            contact.name, contact.name, contact.name
        )))
    }
}

/// List contacts command
pub struct ContactListCommand;

impl Command for ContactListCommand {
    fn name(&self) -> &str { "/contacts" }
    fn description(&self) -> &str { "List all contacts" }
    fn usage(&self) -> &str { "/contacts [at <organization>]" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower == "/contacts" || lower.starts_with("/contacts ")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let contacts_mgr = ContactManager::new(ctx.graph);
        
        let arg = input.trim()
            .strip_prefix("/contacts")
            .map(|s| s.trim())
            .unwrap_or("");

        let contacts = if arg.starts_with("at ") {
            let org = arg.strip_prefix("at ").unwrap().trim();
            contacts_mgr.get_contacts_at_org(org)?
        } else {
            contacts_mgr.list_contacts()?
        };

        if contacts.is_empty() {
            return Ok(CommandResult::Success("No contacts found.".to_string()));
        }

        let mut output = format!("{} contact(s):\n\n", contacts.len());
        for contact in contacts {
            output.push_str(&contact.display());
            output.push_str("\n\n");
        }

        Ok(CommandResult::Success(output))
    }
}

/// Contact info command (alias for /info for people)
pub struct ContactInfoCommand;

impl Command for ContactInfoCommand {
    fn name(&self) -> &str { "/contact" }
    fn description(&self) -> &str { "Get contact details" }
    fn usage(&self) -> &str { "/contact <name>" }
    
    fn matches(&self, input: &str) -> bool {
        let lower = input.trim().to_lowercase();
        lower.starts_with("/contact ") && 
            !lower.starts_with("/contact add") && 
            !lower.starts_with("/contact new") &&
            !lower.starts_with("/contacts")
    }
    
    fn execute(&self, input: &str, ctx: &CommandContext) -> Result<CommandResult> {
        let name = input.trim()
            .strip_prefix("/contact")
            .map(|s| s.trim())
            .unwrap_or("");

        if name.is_empty() {
            return Ok(CommandResult::Error(format!("Usage: {}", self.usage())));
        }

        let contacts = ContactManager::new(ctx.graph);
        
        if let Some(contact) = contacts.find_contact(name)? {
            Ok(CommandResult::Success(contact.display()))
        } else {
            Ok(CommandResult::Error(format!("Contact '{}' not found", name)))
        }
    }
}

// ============================================================================
// Register All Phase 2 Commands
// ============================================================================

pub fn register_phase2_commands(router: &mut Router) {
    // Calendar commands
    router.register(Arc::new(EventAddCommand));
    router.register(Arc::new(EventListCommand));
    router.register(Arc::new(EventTodayCommand));
    
    // Reminder commands
    router.register(Arc::new(RemindCommand));
    router.register(Arc::new(TaskAddCommand));
    router.register(Arc::new(TaskListCommand));
    router.register(Arc::new(TaskDoneCommand));
    
    // Contact commands
    router.register(Arc::new(ContactAddCommand));
    router.register(Arc::new(ContactListCommand));
    router.register(Arc::new(ContactInfoCommand));
}
