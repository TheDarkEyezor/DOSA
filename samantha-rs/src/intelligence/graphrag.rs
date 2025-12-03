//! GraphRAG - Graph-based Retrieval Augmented Generation
//! 
//! This module provides multi-hop graph traversal capabilities for answering
//! complex questions that require traversing multiple relationships:
//! - "Who can help with Project X?" → Traverse WORKS_ON to find team members
//! - "Who at Google knows about AI?" → Traverse WORKS_AT then check skills
//! - "Find people in New York who work on Project Alpha" → Multi-filter traversal

use anyhow::Result;
use std::collections::{HashMap, HashSet};

use crate::knowledge::{KnowledgeGraph, entities::Entity};

/// A path in the knowledge graph
#[derive(Debug, Clone)]
pub struct GraphPath {
    /// The entities in this path, in traversal order
    pub entities: Vec<Entity>,
    /// The relationship types connecting them
    pub relationships: Vec<String>,
}

impl GraphPath {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            relationships: Vec::new(),
        }
    }

    pub fn add_step(&mut self, entity: Entity, relationship: String) {
        if !self.entities.is_empty() {
            self.relationships.push(relationship);
        }
        self.entities.push(entity);
    }

    /// Get a human-readable description of this path
    pub fn describe(&self) -> String {
        if self.entities.is_empty() {
            return String::new();
        }
        
        let mut parts = Vec::new();
        for (i, entity) in self.entities.iter().enumerate() {
            parts.push(entity.name.clone());
            if i < self.relationships.len() {
                parts.push(format!("→[{}]→", self.relationships[i].to_lowercase().replace('_', " ")));
            }
        }
        parts.join(" ")
    }
}

/// Result of a multi-hop query
#[derive(Debug, Clone)]
pub struct TraversalResult {
    /// Entities found at the end of traversal
    pub entities: Vec<Entity>,
    /// The paths taken to reach each entity
    pub paths: Vec<GraphPath>,
    /// Human-readable explanation
    pub explanation: String,
}

/// Multi-hop query types
#[derive(Debug, Clone)]
pub enum MultiHopQuery {
    /// Find people who can help with a project
    WhoCanHelp { project_name: String },
    /// Find people at a location working on a project
    PeopleAtLocationOnProject { location_name: String, project_name: String },
    /// Find people at a company who know someone
    AtCompanyWhoKnows { org_name: String, person_name: String },
    /// Find who a person knows that works at a company
    PersonKnowsAtCompany { person_name: String, org_name: String },
    /// Find manager of someone working on project
    ManagerOfProjectMember { project_name: String },
    /// Find colleagues of someone (same company or project)
    ColleaguesOf { person_name: String },
    /// Find experts on a project (via connections)
    ExpertsForProject { project_name: String },
    /// Generic path finding between two entities
    PathBetween { from_name: String, to_name: String },
}

/// GraphRAG engine for multi-hop traversal
pub struct GraphRAG<'a> {
    graph: &'a KnowledgeGraph,
}

impl<'a> GraphRAG<'a> {
    pub fn new(graph: &'a KnowledgeGraph) -> Self {
        Self { graph }
    }

    /// Parse a natural language query into a multi-hop query
    pub fn parse_query(&self, input: &str) -> Option<MultiHopQuery> {
        let lower = input.to_lowercase();
        
        // "Who can help with [project]?"
        if lower.contains("who can help") || lower.contains("who could help") || 
            lower.contains("help with") || lower.contains("help on") {
            if let Some(project) = self.extract_after(&lower, &["with ", "on "]) {
                return Some(MultiHopQuery::WhoCanHelp { project_name: project });
            }
        }
        
        // "Find people in [location] working on [project]"
        if (lower.contains("in ") && lower.contains("working on")) ||
           (lower.contains("from ") && lower.contains("on project")) {
            let location = self.extract_after(&lower, &["in ", "from "]);
            let project = self.extract_after(&lower, &["working on ", "on project "]);
            if let (Some(loc), Some(proj)) = (location, project) {
                return Some(MultiHopQuery::PeopleAtLocationOnProject {
                    location_name: loc,
                    project_name: proj,
                });
            }
        }
        
        // "Who at [company] knows [person]?"
        if lower.contains(" at ") && lower.contains(" knows ") && !lower.contains("does ") && !lower.contains("know anyone") {
            // For pattern "who at google knows charlie", extract google and charlie
            let org = self.extract_between(&lower, &[" at "], " knows");
            let person = self.extract_after(&lower, &[" knows "]);
            if let (Some(o), Some(p)) = (org, person) {
                return Some(MultiHopQuery::AtCompanyWhoKnows {
                    org_name: o,
                    person_name: p,
                });
            }
        }
        
        // "Does [person] know anyone who works at [company]?" / "[person]'s connections at [company]"
        if (lower.contains("know anyone") && lower.contains("works at")) || 
           (lower.contains("knows anyone") && lower.contains("work at")) ||
           (lower.contains("know someone") && lower.contains("at ")) ||
           (lower.contains("connections at")) {
            // Extract person name from start: "does charlie know..." or "charlie's connections..."
            let person = self.extract_person_from_start(&lower);
            let org = self.extract_after(&lower, &["works at ", "work at ", "at "]);
            if let (Some(p), Some(o)) = (person, org) {
                return Some(MultiHopQuery::PersonKnowsAtCompany {
                    person_name: p,
                    org_name: o,
                });
            }
        }
        
        // "Who manages people on [project]?"
        if lower.contains("manages") && (lower.contains("project") || lower.contains("on ")) {
            if let Some(project) = self.extract_after(&lower, &["on ", "project "]) {
                return Some(MultiHopQuery::ManagerOfProjectMember { project_name: project });
            }
        }
        
        // "Colleagues of [person]"
        if lower.contains("colleagues of") || lower.contains("coworkers of") {
            if let Some(person) = self.extract_after(&lower, &["of "]) {
                return Some(MultiHopQuery::ColleaguesOf { person_name: person });
            }
        }
        
        // "Experts for [project]" or "Who knows about [project]"
        if lower.contains("expert") || (lower.contains("who knows about") && lower.contains("project")) {
            if let Some(project) = self.extract_after(&lower, &["for ", "about ", "on "]) {
                return Some(MultiHopQuery::ExpertsForProject { project_name: project });
            }
        }
        
        // "Connection between [A] and [B]"
        if lower.contains("connection between") || lower.contains("path between") ||
           lower.contains("how is") && lower.contains("connected to") {
            let parts: Vec<&str> = lower.split(" and ").collect();
            if parts.len() == 2 {
                let from = parts[0].split_whitespace().last().unwrap_or("");
                let to = parts[1].trim_matches(|c: char| c == '?' || c == '.');
                if !from.is_empty() && !to.is_empty() {
                    return Some(MultiHopQuery::PathBetween {
                        from_name: capitalize_word(from),
                        to_name: capitalize_word(to),
                    });
                }
            }
        }
        
        None
    }

    /// Execute a multi-hop query
    pub fn execute(&self, query: &MultiHopQuery) -> Result<TraversalResult> {
        match query {
            MultiHopQuery::WhoCanHelp { project_name } => {
                self.find_project_helpers(project_name)
            }
            MultiHopQuery::PeopleAtLocationOnProject { location_name, project_name } => {
                self.find_at_location_on_project(location_name, project_name)
            }
            MultiHopQuery::AtCompanyWhoKnows { org_name, person_name } => {
                self.find_at_company_who_knows(org_name, person_name)
            }
            MultiHopQuery::PersonKnowsAtCompany { person_name, org_name } => {
                self.find_person_knows_at_company(person_name, org_name)
            }
            MultiHopQuery::ManagerOfProjectMember { project_name } => {
                self.find_project_managers(project_name)
            }
            MultiHopQuery::ColleaguesOf { person_name } => {
                self.find_colleagues(person_name)
            }
            MultiHopQuery::ExpertsForProject { project_name } => {
                self.find_project_experts(project_name)
            }
            MultiHopQuery::PathBetween { from_name, to_name } => {
                self.find_path(from_name, to_name)
            }
        }
    }

    /// Find people who can help with a project
    fn find_project_helpers(&self, project_name: &str) -> Result<TraversalResult> {
        let members = self.graph.get_project_members(project_name)?;
        
        if members.is_empty() {
            return Ok(TraversalResult {
                entities: vec![],
                paths: vec![],
                explanation: format!("No one is currently assigned to project '{}'.", project_name),
            });
        }
        
        let mut paths = Vec::new();
        for member in &members {
            let mut path = GraphPath::new();
            // Add project as starting point
            if let Some(proj) = self.graph.find_project(project_name)? {
                path.add_step(proj, String::new());
                path.add_step(member.clone(), "WORKS_ON".to_string());
                paths.push(path);
            }
        }
        
        let names: Vec<String> = members.iter().map(|m| m.name.clone()).collect();
        Ok(TraversalResult {
            entities: members,
            paths,
            explanation: format!("People working on '{}': {}", project_name, names.join(", ")),
        })
    }

    /// Find people at a location working on a project
    fn find_at_location_on_project(&self, location_name: &str, project_name: &str) -> Result<TraversalResult> {
        // Get people at location
        let at_location = self.graph.get_entities_at_location(location_name)?;
        
        // Get project members
        let project_members = self.graph.get_project_members(project_name)?;
        let member_ids: HashSet<i64> = project_members.iter().map(|m| m.id).collect();
        
        // Find intersection
        let matches: Vec<Entity> = at_location
            .into_iter()
            .filter(|e| member_ids.contains(&e.id))
            .collect();
        
        if matches.is_empty() {
            return Ok(TraversalResult {
                entities: vec![],
                paths: vec![],
                explanation: format!(
                    "No one in '{}' is working on '{}'.",
                    location_name, project_name
                ),
            });
        }
        
        let names: Vec<String> = matches.iter().map(|m| m.name.clone()).collect();
        Ok(TraversalResult {
            entities: matches,
            paths: vec![],
            explanation: format!(
                "People in '{}' working on '{}': {}", 
                location_name, project_name, names.join(", ")
            ),
        })
    }

    /// Find people at a company who know someone
    fn find_at_company_who_knows(&self, org_name: &str, person_name: &str) -> Result<TraversalResult> {
        // Get employees
        let employees = self.graph.get_employees(org_name)?;
        let employee_ids: HashSet<i64> = employees.iter().map(|e| e.id).collect();
        
        // Get connections of the person
        let connections = self.graph.get_connections(person_name)?;
        
        // Find intersection
        let matches: Vec<Entity> = connections
            .into_iter()
            .filter(|c| employee_ids.contains(&c.id))
            .collect();
        
        if matches.is_empty() {
            return Ok(TraversalResult {
                entities: vec![],
                paths: vec![],
                explanation: format!(
                    "No one at '{}' is connected to '{}'.",
                    org_name, person_name
                ),
            });
        }
        
        let names: Vec<String> = matches.iter().map(|m| m.name.clone()).collect();
        Ok(TraversalResult {
            entities: matches,
            paths: vec![],
            explanation: format!(
                "People at '{}' who know '{}': {}", 
                org_name, person_name, names.join(", ")
            ),
        })
    }

    /// Find who a person knows that works at a company
    /// e.g., "Does Charlie know anyone who works at Google?"
    fn find_person_knows_at_company(&self, person_name: &str, org_name: &str) -> Result<TraversalResult> {
        // Get the person's connections
        let connections = self.graph.get_connections(person_name)?;
        
        // Get employees at the org
        let employees = self.graph.get_employees(org_name)?;
        let employee_ids: HashSet<i64> = employees.iter().map(|e| e.id).collect();
        
        // Find intersection: connections that work at org
        let matches: Vec<Entity> = connections
            .into_iter()
            .filter(|c| employee_ids.contains(&c.id))
            .collect();
        
        if matches.is_empty() {
            // Check if org exists
            if employees.is_empty() {
                // Try case-insensitive lookup
                if let Ok(Some(_)) = self.graph.find_organization(org_name) {
                    return Ok(TraversalResult {
                        entities: vec![],
                        paths: vec![],
                        explanation: format!(
                            "{} doesn't know anyone who works at {}.",
                            person_name, org_name
                        ),
                    });
                }
                return Ok(TraversalResult {
                    entities: vec![],
                    paths: vec![],
                    explanation: format!(
                        "I don't have any information about an organization called '{}'.",
                        org_name
                    ),
                });
            }
            return Ok(TraversalResult {
                entities: vec![],
                paths: vec![],
                explanation: format!(
                    "{} doesn't know anyone who works at {}.",
                    person_name, org_name
                ),
            });
        }
        
        let names: Vec<String> = matches.iter().map(|m| m.name.clone()).collect();
        Ok(TraversalResult {
            entities: matches,
            paths: vec![],
            explanation: format!(
                "Yes! {} knows {} who work(s) at {}: {}",
                person_name,
                if names.len() == 1 { "someone" } else { "people" },
                org_name, 
                names.join(", ")
            ),
        })
    }

    /// Find managers of people on a project
    fn find_project_managers(&self, project_name: &str) -> Result<TraversalResult> {
        let members = self.graph.get_project_members(project_name)?;
        
        let mut managers = Vec::new();
        let mut seen_ids = HashSet::new();
        
        for member in &members {
            if let Some(manager) = self.graph.get_manager(&member.name)? {
                if !seen_ids.contains(&manager.id) {
                    seen_ids.insert(manager.id);
                    managers.push(manager);
                }
            }
        }
        
        if managers.is_empty() {
            return Ok(TraversalResult {
                entities: vec![],
                paths: vec![],
                explanation: format!("No managers found for people on '{}'.", project_name),
            });
        }
        
        let names: Vec<String> = managers.iter().map(|m| m.name.clone()).collect();
        Ok(TraversalResult {
            entities: managers,
            paths: vec![],
            explanation: format!("Managers of people on '{}': {}", project_name, names.join(", ")),
        })
    }

    /// Find colleagues of a person (same org or project)
    fn find_colleagues(&self, person_name: &str) -> Result<TraversalResult> {
        let person = match self.graph.find_person(person_name)? {
            Some(p) => p,
            None => return Ok(TraversalResult {
                entities: vec![],
                paths: vec![],
                explanation: format!("Person '{}' not found.", person_name),
            }),
        };
        
        let mut colleagues = Vec::new();
        let mut seen_ids = HashSet::new();
        seen_ids.insert(person.id);
        
        // Get relationships
        let rels = self.graph.get_entity_relationships(person.id)?;
        
        for (rel_type, target_name) in rels {
            if rel_type == "WORKS_AT" {
                // Find all employees at same org
                let employees = self.graph.get_employees(&target_name)?;
                for emp in employees {
                    if !seen_ids.contains(&emp.id) {
                        seen_ids.insert(emp.id);
                        colleagues.push(emp);
                    }
                }
            } else if rel_type == "WORKS_ON" {
                // Find all members of same project
                let members = self.graph.get_project_members(&target_name)?;
                for member in members {
                    if !seen_ids.contains(&member.id) {
                        seen_ids.insert(member.id);
                        colleagues.push(member);
                    }
                }
            }
        }
        
        if colleagues.is_empty() {
            return Ok(TraversalResult {
                entities: vec![],
                paths: vec![],
                explanation: format!("No colleagues found for '{}'.", person_name),
            });
        }
        
        let names: Vec<String> = colleagues.iter().map(|c| c.name.clone()).collect();
        Ok(TraversalResult {
            entities: colleagues,
            paths: vec![],
            explanation: format!("Colleagues of '{}': {}", person_name, names.join(", ")),
        })
    }

    /// Find experts for a project (people connected to project members)
    fn find_project_experts(&self, project_name: &str) -> Result<TraversalResult> {
        let members = self.graph.get_project_members(project_name)?;
        let member_ids: HashSet<i64> = members.iter().map(|m| m.id).collect();
        
        let mut experts = Vec::new();
        let mut seen_ids = HashSet::new();
        
        for member in &members {
            // Get connections of each member
            let connections = self.graph.get_connections(&member.name)?;
            for conn in connections {
                // Skip if already a project member or already seen
                if !member_ids.contains(&conn.id) && !seen_ids.contains(&conn.id) {
                    seen_ids.insert(conn.id);
                    experts.push(conn);
                }
            }
        }
        
        if experts.is_empty() {
            let member_names = members.iter().map(|m| m.name.clone()).collect::<Vec<_>>().join(", ");
            return Ok(TraversalResult {
                entities: members,  // Return members themselves
                paths: vec![],
                explanation: format!("The project members are the experts: {}", member_names),
            });
        }
        
        let names: Vec<String> = experts.iter().map(|e| e.name.clone()).collect();
        Ok(TraversalResult {
            entities: experts,
            paths: vec![],
            explanation: format!("Extended network for '{}': {}", project_name, names.join(", ")),
        })
    }

    /// Find path between two entities (BFS)
    fn find_path(&self, from_name: &str, to_name: &str) -> Result<TraversalResult> {
        let from = match self.graph.find_by_name(from_name)? {
            Some(e) => e,
            None => return Ok(TraversalResult {
                entities: vec![],
                paths: vec![],
                explanation: format!("Entity '{}' not found.", from_name),
            }),
        };
        
        let to = match self.graph.find_by_name(to_name)? {
            Some(e) => e,
            None => return Ok(TraversalResult {
                entities: vec![],
                paths: vec![],
                explanation: format!("Entity '{}' not found.", to_name),
            }),
        };
        
        // BFS to find shortest path
        let mut visited = HashSet::new();
        let mut queue: Vec<(Entity, GraphPath)> = Vec::new();
        
        let mut initial_path = GraphPath::new();
        initial_path.add_step(from.clone(), String::new());
        queue.push((from.clone(), initial_path));
        visited.insert(from.id);
        
        while let Some((current, path)) = queue.pop() {
            // Check if we reached destination
            if current.id == to.id {
                return Ok(TraversalResult {
                    entities: vec![to],
                    paths: vec![path.clone()],
                    explanation: format!("Path found: {}", path.describe()),
                });
            }
            
            // Get all relationships
            let rels = self.graph.get_entity_relationships(current.id)?;
            for (rel_type, target_name) in rels {
                if let Some(target) = self.graph.find_by_name(&target_name)? {
                    if !visited.contains(&target.id) {
                        visited.insert(target.id);
                        let mut new_path = path.clone();
                        new_path.add_step(target.clone(), rel_type);
                        queue.push((target, new_path));
                    }
                }
            }
            
            // Limit search depth
            if path.entities.len() > 5 {
                continue;
            }
        }
        
        Ok(TraversalResult {
            entities: vec![],
            paths: vec![],
            explanation: format!("No path found between '{}' and '{}'.", from_name, to_name),
        })
    }

    /// Extract person name from the start of a query like "does charlie know..."
    fn extract_person_from_start(&self, text: &str) -> Option<String> {
        // "does charlie know anyone..." -> extract "charlie"
        if text.starts_with("does ") {
            let after_does = &text[5..]; // skip "does "
            // Find the next word before "know"
            if let Some(know_pos) = after_does.find(" know") {
                let name = after_does[..know_pos].trim();
                if !name.is_empty() {
                    return Some(capitalize_word(name));
                }
            }
        }
        
        // "charlie knows someone..." or "charlie's connections..."
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() > 1 {
            let first = words[0].trim_matches(|c: char| !c.is_alphabetic());
            let stop_words = ["who", "what", "does", "find", "show", "get", "list", "are", "is"];
            if !stop_words.contains(&first) && !first.is_empty() {
                return Some(capitalize_word(first));
            }
        }
        
        None
    }

    /// Extract text after one of the given phrases
    fn extract_after(&self, text: &str, phrases: &[&str]) -> Option<String> {
        for phrase in phrases {
            if let Some(pos) = text.find(phrase) {
                let after = &text[pos + phrase.len()..];
                let result = after
                    .trim()
                    .trim_matches(|c: char| c == '?' || c == '.' || c == '!' || c == ',')
                    .split(&['?', '.', '!', ','][..])
                    .next()
                    .unwrap_or("")
                    .trim();
                
                // Filter out common words that aren't entity names
                let stop_words = ["the", "a", "an", "who", "what", "where", "when", "how"];
                if !result.is_empty() && !stop_words.contains(&result) {
                    return Some(capitalize_word(result));
                }
            }
        }
        None
    }

    /// Extract text between start phrases and end phrase
    fn extract_between(&self, text: &str, start_phrases: &[&str], end_phrase: &str) -> Option<String> {
        for start in start_phrases {
            if let Some(start_pos) = text.find(start) {
                let after_start = &text[start_pos + start.len()..];
                if let Some(end_pos) = after_start.find(end_phrase) {
                    let result = after_start[..end_pos]
                        .trim()
                        .trim_matches(|c: char| c == '?' || c == '.' || c == '!' || c == ',');
                    
                    if !result.is_empty() {
                        return Some(capitalize_word(result));
                    }
                }
            }
        }
        None
    }

    /// Check if this is a multi-hop query
    pub fn is_multihop_query(&self, input: &str) -> bool {
        self.parse_query(input).is_some()
    }
}

/// Capitalize first letter of each word
fn capitalize_word(s: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Database;

    fn test_graph() -> KnowledgeGraph {
        let db = Database::in_memory().unwrap();
        KnowledgeGraph::new(db)
    }

    #[test]
    fn test_parse_who_can_help() {
        let graph = test_graph();
        let rag = GraphRAG::new(&graph);
        
        let query = rag.parse_query("who can help with project alpha");
        assert!(query.is_some());
        
        match query.unwrap() {
            MultiHopQuery::WhoCanHelp { project_name } => {
                assert!(project_name.to_lowercase().contains("alpha"));
            }
            _ => panic!("Expected WhoCanHelp query"),
        }
    }

    #[test]
    fn test_parse_colleagues_of() {
        let graph = test_graph();
        let rag = GraphRAG::new(&graph);
        
        let query = rag.parse_query("colleagues of alice");
        assert!(query.is_some());
        
        match query.unwrap() {
            MultiHopQuery::ColleaguesOf { person_name } => {
                assert_eq!(person_name.to_lowercase(), "alice");
            }
            _ => panic!("Expected ColleaguesOf query"),
        }
    }

    #[test]
    fn test_find_project_helpers() {
        let graph = test_graph();
        
        // Set up data
        let alice = graph.add_person("Alice").unwrap();
        let bob = graph.add_person("Bob").unwrap();
        let project = graph.add_project("Project Alpha").unwrap();
        
        graph.link_works_on(alice.id, project.id).unwrap();
        graph.link_works_on(bob.id, project.id).unwrap();
        
        let rag = GraphRAG::new(&graph);
        let result = rag.find_project_helpers("Project Alpha").unwrap();
        
        assert_eq!(result.entities.len(), 2);
        assert!(result.explanation.contains("Alice"));
        assert!(result.explanation.contains("Bob"));
    }

    #[test]
    fn test_find_colleagues() {
        let graph = test_graph();
        
        // Set up: Alice and Bob work at Google
        let alice = graph.add_person("Alice").unwrap();
        let bob = graph.add_person("Bob").unwrap();
        let google = graph.add_organization("Google").unwrap();
        
        graph.link_works_at(alice.id, google.id).unwrap();
        graph.link_works_at(bob.id, google.id).unwrap();
        
        let rag = GraphRAG::new(&graph);
        let result = rag.find_colleagues("Alice").unwrap();
        
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0].name, "Bob");
    }

    #[test]
    fn test_parse_person_knows_at_company() {
        let graph = test_graph();
        let rag = GraphRAG::new(&graph);
        
        // Test "does charlie know anyone who works at google"
        let query = rag.parse_query("does charlie know anyone who works at google");
        assert!(query.is_some(), "Should parse 'does X know anyone who works at Y'");
        
        match query.unwrap() {
            MultiHopQuery::PersonKnowsAtCompany { person_name, org_name } => {
                assert_eq!(person_name.to_lowercase(), "charlie");
                assert_eq!(org_name.to_lowercase(), "google");
            }
            other => panic!("Expected PersonKnowsAtCompany, got {:?}", other),
        }
    }

    #[test]
    fn test_find_person_knows_at_company() {
        let graph = test_graph();
        
        // Set up: Charlie knows Bob, and Bob works at Google
        let charlie = graph.add_person("Charlie").unwrap();
        let bob = graph.add_person("Bob").unwrap();
        let google = graph.add_organization("Google").unwrap();
        
        graph.link_knows(charlie.id, bob.id).unwrap();
        graph.link_works_at(bob.id, google.id).unwrap();
        
        let rag = GraphRAG::new(&graph);
        let result = rag.find_person_knows_at_company("Charlie", "Google").unwrap();
        
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0].name, "Bob");
        assert!(result.explanation.contains("Charlie"));
        assert!(result.explanation.contains("Bob"));
        assert!(result.explanation.contains("Google"));
    }
}
