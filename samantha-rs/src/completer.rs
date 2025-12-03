//! Tab completion for DOSA commands, entities, and relationships

use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::Helper;
use rustyline::Context;
use std::borrow::Cow;

use crate::knowledge::KnowledgeGraph;

/// All available commands
const COMMANDS: &[&str] = &[
    "/help", "/exit", "/quit",
    // Knowledge graph - entities
    "/add person ", "/add org ", "/add project ", "/add location ",
    "/add skill ", "/add industry ", "/add product ", "/add goal ", "/add problem ", "/add idea ",
    "/link ", "/unlink ",
    "/set ", "/unset ",
    "/rename ", "/delete ",
    "/info ", "/list ", "/summary ", "/search ",
    // Calendar
    "/event ", "/events ", "/today ",
    // Tasks
    "/remind ", "/task ", "/tasks ", "/done ",
    // Contacts
    "/contact ", "/contacts ",
    // Intelligence
    "/briefing ", "/alerts ", "/ask ", "/query ", "/graph ", "/who ",
    // Document ingestion
    "/learn ", "/ingest ",
    // User identity
    "/iam ", "/whoami ", "/forget me",
    // Memory
    "/remember ", "/memories ", "/forget ",
    // External integrations
    "/auth ", "/auth google", "/auth status", "/auth revoke",
    "/gcal ", "/gcal today", "/gcal tomorrow", "/gcal week", "/gcal summary",
    "/gcal create ", "/gcal confirm", "/gcal cancel",
    "/email ", "/email list", "/email unread", "/email summary", "/email read ",
    "/inbox ",
];

/// Relationship types for /link and /unlink
const RELATIONSHIPS: &[&str] = &[
    // Employment & Projects
    "works_at ", "worked_at ", "works_on ", "worked_on ", 
    // Social
    "knows ", "manages ", "reports_to ", "collaborates_with ",
    // Location
    "located_in ", "lives_in ", "based_in ",
    // Organization
    "member_of ", "founded ", "co_founded ",
    // Investment
    "invested ", "funded_by ",
    // Skills & Industry
    "has_skill ", "used_in ", "requires ", "in_industry ",
    // Goals & Problems
    "solves ", "targets ",
    // Mentorship & Partnerships
    "mentors ", "mentored_by ", "partners_with ", "competes_with ",
    // Education
    "studied_at ", "studied ", "has_degree ", "graduated_from ",
    // Languages & Achievements
    "speaks ", "won ", "achieved ",
    // Employment types
    "contractor_at ", "interned_at ",
    // Creation
    "built ", "developed ", "uses_tech ",
];

/// List types for /list command
const LIST_TYPES: &[&str] = &[
    "people", "orgs", "organizations", "projects", "locations", 
    "skills", "industries", "products", "goals", "problems", "ideas",
    "degrees", "universities", "languages", "achievements", "technologies",
    "all",
];

/// Event/task time keywords
const TIME_KEYWORDS: &[&str] = &[
    "today", "tomorrow", "week", "month", "all", "overdue",
];

/// DOSA command completer
pub struct DosaCompleter {
    /// Cached entity names (refreshed periodically)
    entity_names: Vec<String>,
}

impl DosaCompleter {
    pub fn new() -> Self {
        Self {
            entity_names: Vec::new(),
        }
    }

    /// Update the cached entity names from the graph
    pub fn refresh_entities(&mut self, graph: &KnowledgeGraph) {
        self.entity_names.clear();
        
        // Collect all entity names
        if let Ok(people) = graph.list_people() {
            for p in people {
                self.entity_names.push(p.name);
            }
        }
        if let Ok(orgs) = graph.list_organizations() {
            for o in orgs {
                self.entity_names.push(o.name);
            }
        }
        if let Ok(projects) = graph.list_projects() {
            for p in projects {
                self.entity_names.push(p.name);
            }
        }
        if let Ok(locations) = graph.list_locations() {
            for l in locations {
                self.entity_names.push(l.name);
            }
        }
    }

    fn complete_command(&self, line: &str, pos: usize) -> Vec<Pair> {
        let mut matches = Vec::new();
        let input = &line[..pos].to_lowercase();

        // Command completion (starts with /)
        if input.starts_with('/') || input.is_empty() {
            for cmd in COMMANDS {
                if cmd.to_lowercase().starts_with(input) {
                    matches.push(Pair {
                        display: cmd.trim().to_string(),
                        replacement: cmd.to_string(),
                    });
                }
            }
        }

        matches
    }

    fn complete_after_command(&self, line: &str, pos: usize) -> Vec<Pair> {
        let mut matches = Vec::new();
        let lower = line.to_lowercase();

        // After /link or /unlink, suggest entity names then relationships
        if lower.starts_with("/link ") || lower.starts_with("/unlink ") {
            let after_cmd = if lower.starts_with("/link ") {
                &line[6..pos]
            } else {
                &line[8..pos]
            };
            
            // Check if we already have an entity name
            let has_relationship = RELATIONSHIPS.iter().any(|r| after_cmd.to_lowercase().contains(r.trim()));
            
            if has_relationship {
                // Complete second entity name
                let word_start = line.rfind(' ').map(|i| i + 1).unwrap_or(pos);
                let partial = &line[word_start..pos].to_lowercase();
                
                for name in &self.entity_names {
                    if name.to_lowercase().starts_with(partial) {
                        matches.push(Pair {
                            display: name.clone(),
                            replacement: name.clone(),
                        });
                    }
                }
            } else {
                // Check if we're typing a relationship or entity
                let word_start = line.rfind(' ').map(|i| i + 1).unwrap_or(pos);
                let partial = &line[word_start..pos].to_lowercase();
                
                // Try relationship completions
                for rel in RELATIONSHIPS {
                    if rel.starts_with(partial) {
                        matches.push(Pair {
                            display: rel.trim().to_string(),
                            replacement: rel.to_string(),
                        });
                    }
                }
                
                // Also try entity names
                for name in &self.entity_names {
                    if name.to_lowercase().starts_with(partial) {
                        matches.push(Pair {
                            display: name.clone(),
                            replacement: format!("{} ", name),
                        });
                    }
                }
            }
        }
        
        // After /list, suggest list types
        else if lower.starts_with("/list ") {
            let partial = line[6..pos].to_lowercase();
            for list_type in LIST_TYPES {
                if list_type.starts_with(&partial) {
                    matches.push(Pair {
                        display: list_type.to_string(),
                        replacement: list_type.to_string(),
                    });
                }
            }
        }
        
        // After /events or /tasks, suggest time keywords
        else if lower.starts_with("/events ") || lower.starts_with("/tasks ") {
            let offset = if lower.starts_with("/events ") { 8 } else { 7 };
            let partial = line[offset..pos].to_lowercase();
            for kw in TIME_KEYWORDS {
                if kw.starts_with(&partial) {
                    matches.push(Pair {
                        display: kw.to_string(),
                        replacement: kw.to_string(),
                    });
                }
            }
        }
        
        // After /info, /delete, /rename, /set, /unset, /iam, /contact - suggest entity names
        else if lower.starts_with("/info ") || lower.starts_with("/delete ") ||
                lower.starts_with("/rename ") || lower.starts_with("/set ") ||
                lower.starts_with("/unset ") || lower.starts_with("/iam ") ||
                lower.starts_with("/contact ") {
            let word_start = line.rfind(' ').map(|i| i + 1).unwrap_or(pos);
            let partial = &line[word_start..pos].to_lowercase();
            
            for name in &self.entity_names {
                if name.to_lowercase().starts_with(partial) {
                    matches.push(Pair {
                        display: name.clone(),
                        replacement: name.clone(),
                    });
                }
            }
        }

        matches
    }
}

impl Completer for DosaCompleter {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> Result<(usize, Vec<Pair>), ReadlineError> {
        let line_to_cursor = &line[..pos];
        
        // Determine what to complete
        if line_to_cursor.is_empty() || !line_to_cursor.contains(' ') {
            // Command completion
            let matches = self.complete_command(line, pos);
            Ok((0, matches))
        } else {
            // Argument completion
            let matches = self.complete_after_command(line, pos);
            // Find start of current word
            let word_start = line.rfind(' ').map(|i| i + 1).unwrap_or(0);
            Ok((word_start, matches))
        }
    }
}

impl Hinter for DosaCompleter {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> Option<String> {
        if pos < line.len() {
            return None;
        }
        
        let lower = line.to_lowercase();
        
        // Show hint for incomplete commands
        if lower.starts_with('/') && !lower.contains(' ') {
            for cmd in COMMANDS {
                if cmd.to_lowercase().starts_with(&lower) && cmd.len() > line.len() {
                    // Return the rest of the command as a hint
                    return Some(cmd[line.len()..].to_string());
                }
            }
        }
        
        None
    }
}

impl Highlighter for DosaCompleter {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        // Could add syntax highlighting here
        Cow::Borrowed(line)
    }

    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        _default: bool,
    ) -> Cow<'b, str> {
        Cow::Borrowed(prompt)
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        // Gray out hints
        Cow::Owned(format!("\x1b[90m{}\x1b[0m", hint))
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _forced: bool) -> bool {
        false
    }
}

impl Validator for DosaCompleter {}

impl Helper for DosaCompleter {}
