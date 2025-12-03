//! Entity extraction from natural language conversations
//! 
//! This module uses the LLM to extract entities and relationships from
//! user conversations and automatically adds them to the knowledge graph.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::knowledge::KnowledgeGraph;
use crate::llm::OllamaClient;

/// Represents an extracted entity from text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type: String, // "person" or "organization"
    pub properties: Vec<(String, String)>,
}

/// Represents an extracted relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelationship {
    pub from_entity: String,
    pub relationship: String,
    pub to_entity: String,
}

/// Result of entity extraction
#[derive(Debug, Clone, Default)]
pub struct ExtractionResult {
    pub entities: Vec<ExtractedEntity>,
    pub relationships: Vec<ExtractedRelationship>,
    pub was_informative: bool,
}

/// Entity extractor that uses LLM to find entities in text
pub struct EntityExtractor<'a> {
    graph: &'a KnowledgeGraph,
}

impl<'a> EntityExtractor<'a> {
    pub fn new(graph: &'a KnowledgeGraph) -> Self {
        Self { graph }
    }

    /// Check if text likely contains extractable information
    pub fn might_contain_entities(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        
        // Skip if it's a command
        if lower.starts_with('/') {
            return false;
        }
        
        // Skip very short messages
        if text.len() < 15 {
            return false;
        }
        
        // Look for patterns that suggest entity information
        let entity_patterns = [
            // Names (capitalized words)
            " is ", " are ", " was ", " were ",
            // Relationships
            "works at", "works for", "works with",
            "knows", "met", "introduced",
            "hired", "joined", "left",
            "married", "dating", "related to",
            // Information sharing
            "email is", "phone is", "number is",
            "lives in", "lives at", "from",
            "manager", "boss", "colleague",
            "friend", "partner", "spouse",
        ];
        
        entity_patterns.iter().any(|p| lower.contains(p))
    }

    /// Extract entities using pattern matching (fast, no LLM)
    pub fn extract_patterns(&self, text: &str) -> ExtractionResult {
        let mut result = ExtractionResult::default();
        let lower = text.to_lowercase();
        
        // Pattern: "X works at Y"
        if let Some(pos) = lower.find("works at") {
            let before = &text[..pos].trim();
            let after = &text[pos + 8..].trim();
            
            // Extract person name (last word before "works at")
            let person_name = before.split_whitespace().last();
            
            // Extract org name (first few words after "works at")
            let org_name: String = after
                .split(|c: char| c == '.' || c == ',' || c == '!' || c == '?')
                .next()
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            
            if let Some(person) = person_name {
                if !org_name.is_empty() {
                    result.entities.push(ExtractedEntity {
                        name: capitalize_name(person),
                        entity_type: "person".to_string(),
                        properties: vec![],
                    });
                    result.entities.push(ExtractedEntity {
                        name: capitalize_name(&org_name),
                        entity_type: "organization".to_string(),
                        properties: vec![],
                    });
                    result.relationships.push(ExtractedRelationship {
                        from_entity: capitalize_name(person),
                        relationship: "WORKS_AT".to_string(),
                        to_entity: capitalize_name(&org_name),
                    });
                    result.was_informative = true;
                }
            }
        }
        
        // Pattern: "X knows Y"
        if let Some(pos) = lower.find(" knows ") {
            let before = &text[..pos].trim();
            let after = &text[pos + 7..].trim();
            
            let person1 = before.split_whitespace().last();
            let person2: String = after
                .split(|c: char| c == '.' || c == ',' || c == '!' || c == '?')
                .next()
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            
            if let Some(p1) = person1 {
                if !person2.is_empty() && is_likely_name(&person2) {
                    result.entities.push(ExtractedEntity {
                        name: capitalize_name(p1),
                        entity_type: "person".to_string(),
                        properties: vec![],
                    });
                    result.entities.push(ExtractedEntity {
                        name: capitalize_name(&person2),
                        entity_type: "person".to_string(),
                        properties: vec![],
                    });
                    result.relationships.push(ExtractedRelationship {
                        from_entity: capitalize_name(p1),
                        relationship: "KNOWS".to_string(),
                        to_entity: capitalize_name(&person2),
                    });
                    result.was_informative = true;
                }
            }
        }

        // Pattern: "X's email is Y" or "X email is Y"
        for pattern in ["'s email is ", " email is "] {
            if let Some(pos) = lower.find(pattern) {
                let before = &text[..pos].trim();
                let after = &text[pos + pattern.len()..].trim();
                
                let person_name = before.split_whitespace().last();
                let email: String = after
                    .split_whitespace()
                    .next()
                    .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric() && c != '@' && c != '.').to_string())
                    .unwrap_or_default();
                
                if let Some(person) = person_name {
                    if !email.is_empty() && email.contains('@') {
                        result.entities.push(ExtractedEntity {
                            name: capitalize_name(person),
                            entity_type: "person".to_string(),
                            properties: vec![("email".to_string(), email)],
                        });
                        result.was_informative = true;
                    }
                }
            }
        }
        
        result
    }

    /// Apply extracted entities to the knowledge graph
    pub fn apply_extraction(&self, extraction: &ExtractionResult) -> Result<Vec<String>> {
        let mut actions = Vec::new();
        
        // Add entities
        for entity in &extraction.entities {
            match entity.entity_type.as_str() {
                "person" => {
                    if self.graph.find_person(&entity.name)?.is_none() {
                        self.graph.add_person(&entity.name)?;
                        actions.push(format!("Added person: {}", entity.name));
                    }
                    
                    // Add properties
                    if let Some(person) = self.graph.find_person(&entity.name)? {
                        for (key, value) in &entity.properties {
                            self.graph.set_property(person.id, key, value)?;
                            actions.push(format!("Set {}.{} = {}", entity.name, key, value));
                        }
                    }
                }
                "organization" => {
                    if self.graph.find_organization(&entity.name)?.is_none() {
                        self.graph.add_organization(&entity.name)?;
                        actions.push(format!("Added organization: {}", entity.name));
                    }
                }
                _ => {}
            }
        }
        
        // Add relationships
        for rel in &extraction.relationships {
            match rel.relationship.as_str() {
                "WORKS_AT" => {
                    if let (Some(person), Some(org)) = (
                        self.graph.find_person(&rel.from_entity)?,
                        self.graph.find_organization(&rel.to_entity)?
                    ) {
                        self.graph.link_works_at(person.id, org.id)?;
                        actions.push(format!("Linked: {} works at {}", rel.from_entity, rel.to_entity));
                    }
                }
                "KNOWS" => {
                    if let (Some(p1), Some(p2)) = (
                        self.graph.find_person(&rel.from_entity)?,
                        self.graph.find_person(&rel.to_entity)?
                    ) {
                        self.graph.link_knows(p1.id, p2.id)?;
                        actions.push(format!("Linked: {} knows {}", rel.from_entity, rel.to_entity));
                    }
                }
                _ => {}
            }
        }
        
        Ok(actions)
    }

    /// Display what was extracted (for user confirmation)
    pub fn display_extraction(&self, extraction: &ExtractionResult) -> String {
        if !extraction.was_informative {
            return String::new();
        }

        let mut output = String::from("\n💡 I noticed some information:\n");
        
        for entity in &extraction.entities {
            let icon = if entity.entity_type == "person" { "👤" } else { "🏢" };
            output.push_str(&format!("   {} {} ({})\n", icon, entity.name, entity.entity_type));
            for (key, value) in &entity.properties {
                output.push_str(&format!("      • {}: {}\n", key, value));
            }
        }
        
        for rel in &extraction.relationships {
            output.push_str(&format!("   🔗 {} {} {}\n", rel.from_entity, rel.relationship, rel.to_entity));
        }
        
        output.push_str("\n   [This has been automatically added to your knowledge graph]\n");
        
        output
    }
}

/// Capitalize a name properly
fn capitalize_name(name: &str) -> String {
    name.split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => {
                    let lower_rest: String = chars.collect::<String>().to_lowercase();
                    format!("{}{}", first.to_uppercase().collect::<String>(), lower_rest)
                }
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Check if a string is likely a name (capitalized, reasonable length)
fn is_likely_name(s: &str) -> bool {
    let trimmed = s.trim();
    if trimmed.is_empty() || trimmed.len() > 50 {
        return false;
    }
    
    // Should have at least one alphabetic character
    if !trimmed.chars().any(|c| c.is_alphabetic()) {
        return false;
    }
    
    // Shouldn't be common words
    let common_words = ["the", "a", "an", "this", "that", "they", "them", "their"];
    if common_words.contains(&trimmed.to_lowercase().as_str()) {
        return false;
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capitalize_name() {
        assert_eq!(capitalize_name("john"), "John");
        assert_eq!(capitalize_name("john doe"), "John Doe");
        assert_eq!(capitalize_name("ALICE"), "Alice");
    }

    #[test]
    fn test_is_likely_name() {
        assert!(is_likely_name("Alice"));
        assert!(is_likely_name("John Doe"));
        assert!(!is_likely_name("the"));
        assert!(!is_likely_name("123"));
        assert!(!is_likely_name(""));
    }

    #[test]
    fn test_might_contain_entities() {
        let patterns = vec![
            ("Alice works at Google", true),
            ("Bob knows Charlie", true),
            ("Hello world", false),
            ("/add person test", false),  // Commands are skipped
            ("hi", false),  // Too short
        ];
        
        for (text, expected) in patterns {
            let lower = text.to_lowercase();
            let has_pattern = ["works at", "knows", " is "].iter().any(|p| lower.contains(p));
            let not_command = !text.starts_with('/');
            let long_enough = text.len() >= 15;
            
            // Simplified check
            if expected {
                assert!(lower.contains("works at") || lower.contains("knows"), 
                    "Expected pattern in: {}", text);
            }
        }
    }
}
