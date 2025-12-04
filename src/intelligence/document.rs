//! Document Ingestion Module
//!
//! Allows users to add documents (text files, notes, bios) and have the LLM
//! automatically extract entities, relationships, and properties to populate
//! the knowledge graph.
//!
//! Supports:
//! - Personal bios ("I am John, I work at Google...")
//! - Meeting notes ("Met with Sarah from Microsoft...")
//! - Contact info ("Bob's phone is 555-1234, email bob@example.com")
//! - Relationship descriptions ("Alice manages Bob and Charlie")

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use strsim::jaro_winkler;

use crate::knowledge::KnowledgeGraph;
use crate::knowledge::entities::{EntityType, RelationshipType};
use crate::llm::{OllamaClient, Message};

/// Extracted entity from a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentEntity {
    pub name: String,
    pub entity_type: String,
    pub properties: std::collections::HashMap<String, String>,
}

/// Extracted relationship from a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRelationship {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// Result of document ingestion
#[derive(Debug, Clone)]
pub struct IngestionResult {
    pub entities_added: Vec<String>,
    pub relationships_added: Vec<String>,
    pub properties_set: Vec<String>,
    pub errors: Vec<String>,
}

impl Default for IngestionResult {
    fn default() -> Self {
        Self {
            entities_added: Vec::new(),
            relationships_added: Vec::new(),
            properties_set: Vec::new(),
            errors: Vec::new(),
        }
    }
}

impl IngestionResult {
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        
        if !self.entities_added.is_empty() {
            parts.push(format!("📊 Added {} entities", self.entities_added.len()));
        }
        if !self.relationships_added.is_empty() {
            parts.push(format!("🔗 Added {} relationships", self.relationships_added.len()));
        }
        if !self.properties_set.is_empty() {
            parts.push(format!("📝 Set {} properties", self.properties_set.len()));
        }
        if !self.errors.is_empty() {
            parts.push(format!("⚠️ {} errors", self.errors.len()));
        }
        
        if parts.is_empty() {
            "No information extracted".to_string()
        } else {
            parts.join("\n")
        }
    }

    pub fn details(&self) -> String {
        let mut output = String::new();
        
        if !self.entities_added.is_empty() {
            output.push_str("\n📊 Entities added:\n");
            for e in &self.entities_added {
                output.push_str(&format!("  • {}\n", e));
            }
        }
        
        if !self.relationships_added.is_empty() {
            output.push_str("\n🔗 Relationships:\n");
            for r in &self.relationships_added {
                output.push_str(&format!("  • {}\n", r));
            }
        }
        
        if !self.properties_set.is_empty() {
            output.push_str("\n📝 Properties:\n");
            for p in &self.properties_set {
                output.push_str(&format!("  • {}\n", p));
            }
        }
        
        if !self.errors.is_empty() {
            output.push_str("\n⚠️ Errors:\n");
            for e in &self.errors {
                output.push_str(&format!("  • {}\n", e));
            }
        }
        
        output
    }
}

/// Document ingestion engine
pub struct DocumentIngester<'a> {
    graph: &'a KnowledgeGraph,
    llm: OllamaClient,
}

impl<'a> DocumentIngester<'a> {
    pub fn new(graph: &'a KnowledgeGraph) -> Self {
        Self {
            graph,
            llm: OllamaClient::new("llama3.2:3b"),
        }
    }

    pub fn with_model(graph: &'a KnowledgeGraph, model: &str) -> Self {
        Self {
            graph,
            llm: OllamaClient::new(model),
        }
    }

    /// Ingest a document and extract entities/relationships
    pub async fn ingest(&self, document: &str) -> Result<IngestionResult> {
        let mut result = IngestionResult::default();

        // Step 1: Use LLM to extract structured data
        let extraction = self.extract_with_llm(document).await?;

        // Step 2: Parse the LLM response
        let (entities, relationships) = self.parse_llm_response(&extraction)?;

        // Step 3: Apply to knowledge graph
        self.apply_entities(&entities, &mut result)?;
        self.apply_relationships(&relationships, &mut result)?;

        Ok(result)
    }

    /// Quick ingest using pattern matching only (no LLM call)
    pub fn ingest_fast(&self, document: &str) -> Result<IngestionResult> {
        self.ingest_fast_for(document, None)
    }
    
    /// Quick ingest with explicit subject specification
    pub fn ingest_fast_for(&self, document: &str, subject: Option<String>) -> Result<IngestionResult> {
        let mut result = IngestionResult::default();

        // Determine the document subject
        let user_identity = self.graph.get_user_identity_name().ok().flatten();
        let subject_name = subject.or_else(|| Self::detect_document_subject(document, &user_identity));
        
        // If we detected a subject that isn't the user, create an entity for them
        if let Some(ref name) = subject_name {
            if user_identity.as_ref() != Some(name) {
                let _ = self.graph.add_entity(EntityType::Person, name);
            }
        }

        // Extract using patterns with the detected subject
        let (entities, relationships) = Self::extract_patterns_static(document, &subject_name);

        // Apply to graph
        self.apply_entities(&entities, &mut result)?;
        self.apply_relationships(&relationships, &mut result)?;

        Ok(result)
    }

    /// Parallel document ingestion - splits document into paragraphs and processes in parallel
    /// Uses multi-threading for faster extraction on large documents
    pub fn ingest_parallel(&self, document: &str) -> Result<IngestionResult> {
        self.ingest_parallel_for(document, None)
    }
    
    /// Parallel document ingestion with explicit subject specification
    /// If `subject` is provided, use it; otherwise, auto-detect from document content
    pub fn ingest_parallel_for(&self, document: &str, subject: Option<String>) -> Result<IngestionResult> {
        // Split document into paragraphs (preserving context)
        let paragraphs = Self::split_paragraphs(document);
        
        if paragraphs.is_empty() {
            return self.ingest_fast(document); // Fall back to single-threaded for small docs
        }

        // Determine the document subject:
        // 1. If explicitly provided, use that
        // 2. Otherwise, try to detect from document content (e.g., resume header)
        // 3. Only fall back to user identity if document is self-referential
        let user_identity = self.graph.get_user_identity_name().ok().flatten();
        let subject_name = subject.or_else(|| Self::detect_document_subject(document, &user_identity));
        
        // If we detected a subject that isn't the user, create an entity for them
        if let Some(ref name) = subject_name {
            if user_identity.as_ref() != Some(name) {
                // Create a person entity for the document subject
                let _ = self.graph.add_entity(EntityType::Person, name);
            }
        }
        
        // Process paragraphs in parallel using rayon
        // Note: We use static methods to avoid capturing self (which has non-Send fields)
        let extracted: Vec<(Vec<DocumentEntity>, Vec<DocumentRelationship>)> = paragraphs
            .par_iter()
            .map(|para| Self::extract_patterns_static(para, &subject_name))
            .collect();

        // Merge all extractions
        let mut all_entities: Vec<DocumentEntity> = Vec::new();
        let mut all_relationships: Vec<DocumentRelationship> = Vec::new();

        for (entities, relationships) in extracted {
            all_entities.extend(entities);
            all_relationships.extend(relationships);
        }

        // Deduplicate entities using fuzzy matching
        let unique_entities = Self::dedupe_entities_fuzzy(&all_entities);

        // Apply to graph (sequentially, since SQLite isn't thread-safe)
        let mut result = IngestionResult::default();
        self.apply_entities(&unique_entities, &mut result)?;
        self.apply_relationships(&all_relationships, &mut result)?;

        Ok(result)
    }

    /// Split document into paragraphs (static method)
    fn split_paragraphs(document: &str) -> Vec<String> {
        document
            .split("\n\n")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty() && s.len() > 20)
            .map(String::from)
            .collect()
    }

    /// Detect the document subject (e.g., person name from resume header)
    /// Returns the detected name if found, or user_identity if document is clearly self-referential
    pub fn detect_document_subject(document: &str, user_identity: &Option<String>) -> Option<String> {
        let lower = document.to_lowercase();
        
        // FIRST: Try to detect a person name from the header (most reliable)
        // This takes priority over self-referential detection
        
        // Get first few lines for header analysis
        let first_lines: String = document.lines()
            .take(5)
            .filter(|l| !l.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        
        let first_line = document.lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("");
        
        // Pattern 1: First line is just a name (2-4 capitalized words, no special chars)
        let trimmed_first = first_line.trim();
        let words: Vec<&str> = trimmed_first.split_whitespace()
            .filter(|w| w.chars().all(|c| c.is_alphabetic() || c == '-' || c == '\''))
            .collect();
        
        if words.len() >= 2 && words.len() <= 4 {
            let capitalized_count = words.iter()
                .filter(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
                .count();
            
            if capitalized_count >= 2 {
                let name = words.join(" ");
                let non_names = ["resume", "curriculum vitae", "cv", "cover letter", "application"];
                if !non_names.iter().any(|n| name.to_lowercase().contains(n)) {
                    return Some(name);
                }
            }
        }
        
        // Pattern 2: "<Name> - <Title>" or "<Name> | <Contact>" format
        if let Some(separator_idx) = trimmed_first.find(" - ").or_else(|| trimmed_first.find(" | ")) {
            let potential_name = trimmed_first[..separator_idx].trim();
            let name_words: Vec<&str> = potential_name.split_whitespace()
                .filter(|w| w.chars().all(|c| c.is_alphabetic() || c == '-' || c == '\''))
                .collect();
            
            if name_words.len() >= 2 && name_words.len() <= 4 {
                let capitalized = name_words.iter()
                    .filter(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
                    .count();
                if capitalized >= 2 {
                    return Some(name_words.join(" "));
                }
            }
        }
        
        // Pattern 3: "Resume of <Name>"
        if let Some(idx) = first_lines.to_lowercase().find("resume of ") {
            let after = &first_lines[idx + 10..];
            let name_end = after.find('\n').unwrap_or(after.len());
            let name = after[..name_end].trim();
            if !name.is_empty() && name.split_whitespace().count() >= 2 {
                return Some(name.to_string());
            }
        }
        
        // SECOND: Check if document is clearly self-referential (first-person pronouns in main content)
        // Only count strong self-references, not casual mentions like "My Personal Website"
        let strong_self_refs = ["i am ", "i'm a ", "my name is ", "about me\n", 
                                "i work at ", "i work for ", "i have experience"];
        let is_self_referential = strong_self_refs.iter().any(|s| lower.contains(s));
        
        if is_self_referential {
            return user_identity.clone();
        }
        
        // No clear subject found
        None
    }

    /// Static pattern extraction (no self reference, can be used in parallel)
    /// `subject_name` is the person this document is ABOUT (may differ from logged-in user)
    fn extract_patterns_static(document: &str, subject_name: &Option<String>) -> (Vec<DocumentEntity>, Vec<DocumentRelationship>) {
        let mut entities = Vec::new();
        let mut relationships = Vec::new();
        let lower = document.to_lowercase();

        // Skills extraction
        let tech_keywords = [
            "rust", "python", "javascript", "typescript", "java", "c++", "go", "ruby",
            "react", "vue", "angular", "node", "django", "flask", "rails",
            "ai", "ml", "machine learning", "deep learning", "nlp", "natural language",
            "aws", "gcp", "azure", "docker", "kubernetes",
            "sql", "nosql", "postgresql", "mongodb", "redis",
            "api", "rest", "graphql", "microservices",
        ];
        for keyword in tech_keywords {
            if lower.contains(keyword) {
                let skill_name = Self::capitalize_static(keyword);
                entities.push(DocumentEntity {
                    name: skill_name.clone(),
                    entity_type: "skill".to_string(),
                    properties: std::collections::HashMap::new(),
                });
                if let Some(ref subject) = subject_name {
                    relationships.push(DocumentRelationship {
                        subject: subject.clone(),
                        predicate: "HAS_SKILL".to_string(),
                        object: skill_name,
                    });
                }
            }
        }

        // Technologies extraction
        let tech_frameworks = [
            ("react", "React"), ("next.js", "Next.js"), ("nextjs", "Next.js"),
            ("typescript", "TypeScript"), ("javascript", "JavaScript"),
            ("fastapi", "FastAPI"), ("django", "Django"), ("flask", "Flask"),
            ("docker", "Docker"), ("kubernetes", "Kubernetes"), ("k8s", "Kubernetes"),
            ("aws", "AWS"), ("azure", "Azure"), ("gcp", "GCP"),
            ("redis", "Redis"), ("postgresql", "PostgreSQL"), ("mongodb", "MongoDB"),
            ("pytorch", "PyTorch"), ("tensorflow", "TensorFlow"),
            ("scala", "Scala"), ("haskell", "Haskell"), ("kotlin", "Kotlin"),
        ];
        let mut seen_tech = std::collections::HashSet::new();
        for (keyword, tech) in tech_frameworks {
            if lower.contains(keyword) && !seen_tech.contains(tech) {
                seen_tech.insert(tech);
                entities.push(DocumentEntity {
                    name: tech.to_string(),
                    entity_type: "technology".to_string(),
                    properties: std::collections::HashMap::new(),
                });
            }
        }

        // Universities
        let universities = [
            ("imperial college", "Imperial College London"),
            ("oxford", "University of Oxford"),
            ("cambridge", "University of Cambridge"),
            ("mit", "MIT"),
            ("stanford", "Stanford University"),
        ];
        for (keyword, uni) in universities {
            if lower.contains(keyword) {
                entities.push(DocumentEntity {
                    name: uni.to_string(),
                    entity_type: "university".to_string(),
                    properties: std::collections::HashMap::new(),
                });
                if let Some(ref subject) = subject_name {
                    relationships.push(DocumentRelationship {
                        subject: subject.clone(),
                        predicate: "STUDIED_AT".to_string(),
                        object: uni.to_string(),
                    });
                }
            }
        }

        // Spoken languages (only if language context is present)
        let spoken_langs = [
            ("english", "English"), ("french", "French"), ("spanish", "Spanish"),
            ("german", "German"), ("hindi", "Hindi"), ("tamil", "Tamil"),
        ];
        let lang_context = lower.contains("language") || lower.contains("proficiency") 
            || lower.contains("native") || lower.contains("fluent");
        if lang_context {
            for (keyword, lang) in spoken_langs {
                if lower.contains(keyword) {
                    entities.push(DocumentEntity {
                        name: lang.to_string(),
                        entity_type: "spokenlanguage".to_string(),
                        properties: std::collections::HashMap::new(),
                    });
                    if let Some(ref subject) = subject_name {
                        relationships.push(DocumentRelationship {
                            subject: subject.clone(),
                            predicate: "SPEAKS".to_string(),
                            object: lang.to_string(),
                        });
                    }
                }
            }
        }

        // Industries
        let industry_keywords = [
            ("healthcare", "Healthcare"), ("medical", "Healthcare"),
            ("finance", "Finance"), ("trading", "Finance"),
            ("quant", "Quantitative Finance"),
            ("insurance", "Insurance"), ("legal", "Legal"),
        ];
        let mut seen_industries = std::collections::HashSet::new();
        for (keyword, industry) in industry_keywords {
            if lower.contains(keyword) && !seen_industries.contains(industry) {
                seen_industries.insert(industry);
                entities.push(DocumentEntity {
                    name: industry.to_string(),
                    entity_type: "industry".to_string(),
                    properties: std::collections::HashMap::new(),
                });
                if let Some(ref subject) = subject_name {
                    relationships.push(DocumentRelationship {
                        subject: subject.clone(),
                        predicate: "IN_INDUSTRY".to_string(),
                        object: industry.to_string(),
                    });
                }
            }
        }

        // Deduplicate entities by name
        let mut unique_entities: Vec<DocumentEntity> = Vec::new();
        for entity in entities {
            if !unique_entities.iter().any(|e| e.name.to_lowercase() == entity.name.to_lowercase()) {
                unique_entities.push(entity);
            }
        }

        (unique_entities, relationships)
    }

    /// Static capitalize function
    fn capitalize_static(s: &str) -> String {
        s.split_whitespace()
            .map(|word| {
                let mut chars: Vec<char> = word.chars().collect();
                if let Some(first) = chars.first_mut() {
                    *first = first.to_uppercase().next().unwrap_or(*first);
                }
                chars.into_iter().collect::<String>()
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Static fuzzy deduplication
    fn dedupe_entities_fuzzy(entities: &[DocumentEntity]) -> Vec<DocumentEntity> {
        let mut unique: Vec<DocumentEntity> = Vec::new();
        const SIMILARITY_THRESHOLD: f64 = 0.85;

        for entity in entities {
            let is_duplicate = unique.iter().any(|existing| {
                existing.entity_type == entity.entity_type
                    && jaro_winkler(&existing.name.to_lowercase(), &entity.name.to_lowercase()) > SIMILARITY_THRESHOLD
            });

            if !is_duplicate {
                unique.push(entity.clone());
            }
        }

        unique
    }

    /// Ingest a PDF file (extracts text using pdftotext)
    pub fn ingest_pdf(&self, path: &std::path::Path) -> Result<IngestionResult> {
        self.ingest_pdf_for(path, None)
    }
    
    /// Ingest a PDF file with explicit subject specification
    pub fn ingest_pdf_for(&self, path: &std::path::Path, subject: Option<String>) -> Result<IngestionResult> {
        // Extract text from PDF using pdftotext
        let output = std::process::Command::new("pdftotext")
            .args(["-layout", path.to_str().unwrap_or(""), "-"])
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to run pdftotext: {}. Install with 'brew install poppler'", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("pdftotext failed: {}", stderr));
        }

        let text = String::from_utf8(output.stdout)
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in PDF: {}", e))?;

        // Use parallel ingestion for the extracted text
        self.ingest_parallel_for(&text, subject)
    }

    /// Ingest any file - auto-detects type
    pub fn ingest_file(&self, path: &std::path::Path) -> Result<IngestionResult> {
        self.ingest_file_for(path, None)
    }
    
    /// Ingest any file with explicit subject specification
    pub fn ingest_file_for(&self, path: &std::path::Path, subject: Option<String>) -> Result<IngestionResult> {
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "pdf" => self.ingest_pdf_for(path, subject),
            "txt" | "md" | "markdown" => {
                let content = std::fs::read_to_string(path)?;
                self.ingest_parallel_for(&content, subject)
            }
            _ => {
                // Try to read as text
                match std::fs::read_to_string(path) {
                    Ok(content) => self.ingest_parallel_for(&content, subject),
                    Err(_) => Err(anyhow::anyhow!("Unsupported file type: {}", extension)),
                }
            }
        }
    }

    /// Extract entities using LLM
    async fn extract_with_llm(&self, document: &str) -> Result<String> {
        let system_prompt = r#"You are an entity extraction assistant. Extract entities and relationships from the given text.

Output format (use EXACTLY this format):
ENTITIES:
- [TYPE] Name | property1=value1 | property2=value2
...

RELATIONSHIPS:
- Subject | RELATIONSHIP_TYPE | Object
...

Entity types: PERSON, ORGANIZATION, PROJECT, LOCATION, TASK
Relationship types: 
- WORKS_AT (current employment)
- WORKED_AT (past employment)
- WORKS_ON (current project)
- WORKED_ON (past project)
- KNOWS, MANAGES, REPORTS_TO, LIVES_IN, MEMBER_OF, OWNS, ASSIGNED_TO, COLLABORATES_WITH

Use past tense (WORKED_AT, WORKED_ON) for former positions/projects.
Use present tense (WORKS_AT, WORKS_ON) for current positions/projects.

Example input: "I'm John Smith, I work at Google as a software engineer. I previously worked on Project Alpha. My email is john@google.com."

Example output:
ENTITIES:
- [PERSON] John Smith | email=john@google.com | role=software engineer
- [ORGANIZATION] Google
- [PROJECT] Project Alpha

RELATIONSHIPS:
- John Smith | WORKS_AT | Google
- John Smith | WORKED_ON | Project Alpha

Extract ALL entities and relationships. Be thorough."#;

        let messages = vec![
            Message::system(system_prompt),
            Message::user(&format!("Extract entities and relationships from this text:\n\n{}", document)),
        ];

        self.llm.chat(messages).await
    }

    /// Parse the LLM's structured response
    fn parse_llm_response(&self, response: &str) -> Result<(Vec<DocumentEntity>, Vec<DocumentRelationship>)> {
        let mut entities = Vec::new();
        let mut relationships = Vec::new();

        let mut in_entities = false;
        let mut in_relationships = false;

        for line in response.lines() {
            let line = line.trim();
            
            if line.to_uppercase().contains("ENTITIES:") {
                in_entities = true;
                in_relationships = false;
                continue;
            }
            
            if line.to_uppercase().contains("RELATIONSHIPS:") {
                in_entities = false;
                in_relationships = true;
                continue;
            }

            if line.starts_with('-') || line.starts_with('•') {
                let content = line.trim_start_matches(|c| c == '-' || c == '•' || c == ' ');
                
                if in_entities {
                    if let Some(entity) = self.parse_entity_line(content) {
                        entities.push(entity);
                    }
                } else if in_relationships {
                    if let Some(rel) = self.parse_relationship_line(content) {
                        relationships.push(rel);
                    }
                }
            }
        }

        Ok((entities, relationships))
    }

    /// Parse an entity line like "[PERSON] John Smith | email=john@example.com"
    fn parse_entity_line(&self, line: &str) -> Option<DocumentEntity> {
        // Extract type from brackets
        let type_start = line.find('[')?;
        let type_end = line.find(']')?;
        let entity_type = line[type_start + 1..type_end].trim().to_lowercase();

        // Get rest after the type
        let rest = line[type_end + 1..].trim();
        
        // Split by | to get name and properties
        let parts: Vec<&str> = rest.split('|').map(|s| s.trim()).collect();
        
        if parts.is_empty() {
            return None;
        }

        let name = parts[0].to_string();
        if name.is_empty() {
            return None;
        }

        let mut properties = std::collections::HashMap::new();
        for part in parts.iter().skip(1) {
            if let Some(eq_pos) = part.find('=') {
                let key = part[..eq_pos].trim().to_lowercase();
                let value = part[eq_pos + 1..].trim().to_string();
                if !key.is_empty() && !value.is_empty() {
                    properties.insert(key, value);
                }
            }
        }

        Some(DocumentEntity {
            name,
            entity_type,
            properties,
        })
    }

    /// Parse a relationship line like "John Smith | WORKS_AT | Google"
    fn parse_relationship_line(&self, line: &str) -> Option<DocumentRelationship> {
        let parts: Vec<&str> = line.split('|').map(|s| s.trim()).collect();
        
        if parts.len() >= 3 {
            Some(DocumentRelationship {
                subject: parts[0].to_string(),
                predicate: parts[1].to_uppercase().replace(' ', "_"),
                object: parts[2].to_string(),
            })
        } else {
            None
        }
    }

    /// Pattern-based extraction (fast, no LLM)
    fn extract_patterns(&self, document: &str) -> (Vec<DocumentEntity>, Vec<DocumentRelationship>) {
        let mut entities = Vec::new();
        let mut relationships = Vec::new();
        let lower = document.to_lowercase();

        // Get the user identity if set (for first-person references)
        let user_name = self.graph.get_user_identity_name().ok().flatten();

        // Pattern: "I am X" or "My name is X" - first person introduction
        if lower.contains("i am ") || lower.contains("my name is ") {
            if let Some(name) = self.extract_first_person_name(document) {
                entities.push(DocumentEntity {
                    name: name.clone(),
                    entity_type: "person".to_string(),
                    properties: std::collections::HashMap::new(),
                });

                // Look for properties about "I"
                if let Some(email) = self.extract_email_for(document, &["my email", "email is", "reach me at"]) {
                    if let Some(e) = entities.iter_mut().find(|e| e.name == name) {
                        e.properties.insert("email".to_string(), email);
                    }
                }

                if let Some(phone) = self.extract_phone_for(document, &["my phone", "phone is", "call me at", "number is"]) {
                    if let Some(e) = entities.iter_mut().find(|e| e.name == name) {
                        e.properties.insert("phone".to_string(), phone);
                    }
                }

                // "I work at X"
                if let Some(org) = self.extract_after_pattern(&lower, &["i work at ", "i work for ", "i'm at ", "employed at "]) {
                    let org_name = self.capitalize_words(&org);
                    entities.push(DocumentEntity {
                        name: org_name.clone(),
                        entity_type: "organization".to_string(),
                        properties: std::collections::HashMap::new(),
                    });
                    relationships.push(DocumentRelationship {
                        subject: name.clone(),
                        predicate: "WORKS_AT".to_string(),
                        object: org_name,
                    });
                }

                // "I live in X"
                if let Some(loc) = self.extract_after_pattern(&lower, &["i live in ", "i'm from ", "based in "]) {
                    let loc_name = self.capitalize_words(&loc);
                    entities.push(DocumentEntity {
                        name: loc_name.clone(),
                        entity_type: "location".to_string(),
                        properties: std::collections::HashMap::new(),
                    });
                    relationships.push(DocumentRelationship {
                        subject: name.clone(),
                        predicate: "LIVES_IN".to_string(),
                        object: loc_name,
                    });
                }

                // "I manage X" or "I know X"
                for pattern in &["i manage ", "i supervise "] {
                    if let Some(person) = self.extract_after_pattern(&lower, &[pattern]) {
                        let person_name = self.capitalize_words(&person);
                        entities.push(DocumentEntity {
                            name: person_name.clone(),
                            entity_type: "person".to_string(),
                            properties: std::collections::HashMap::new(),
                        });
                        relationships.push(DocumentRelationship {
                            subject: name.clone(),
                            predicate: "MANAGES".to_string(),
                            object: person_name,
                        });
                    }
                }

                // Past employment: "I worked at X", "I used to work at X", "I previously worked at X"
                if let Some(org) = self.extract_after_pattern(&lower, &[
                    "i worked at ", "i worked for ", "i used to work at ", "i used to work for ",
                    "i previously worked at ", "previously worked at ", "formerly at "
                ]) {
                    let org_name = self.capitalize_words(&org);
                    entities.push(DocumentEntity {
                        name: org_name.clone(),
                        entity_type: "organization".to_string(),
                        properties: std::collections::HashMap::new(),
                    });
                    relationships.push(DocumentRelationship {
                        subject: name.clone(),
                        predicate: "WORKED_AT".to_string(),
                        object: org_name,
                    });
                }

                // Past projects: "I worked on X", "I used to work on X"
                if let Some(project) = self.extract_after_pattern(&lower, &[
                    "i worked on ", "i used to work on ", "i previously worked on ",
                    "previously worked on ", "my former project ", "my previous project "
                ]) {
                    let project_name = self.capitalize_words(&project);
                    entities.push(DocumentEntity {
                        name: project_name.clone(),
                        entity_type: "project".to_string(),
                        properties: std::collections::HashMap::new(),
                    });
                    relationships.push(DocumentRelationship {
                        subject: name.clone(),
                        predicate: "WORKED_ON".to_string(),
                        object: project_name,
                    });
                }
            }
        }

        // Pattern: "X works at Y"
        for (pattern, rel) in &[
            ("works at", "WORKS_AT"),
            ("works for", "WORKS_AT"),
            ("worked at", "WORKED_AT"),
            ("worked for", "WORKED_AT"),
            ("works on", "WORKS_ON"),
            ("worked on", "WORKED_ON"),
            ("knows", "KNOWS"),
            ("manages", "MANAGES"),
            ("reports to", "REPORTS_TO"),
        ] {
            for (subject, object) in self.extract_subject_pattern_object(document, pattern) {
                entities.push(DocumentEntity {
                    name: subject.clone(),
                    entity_type: "person".to_string(),
                    properties: std::collections::HashMap::new(),
                });
                
                let obj_type = match *rel {
                    "WORKS_AT" | "WORKED_AT" => "organization",
                    "WORKS_ON" | "WORKED_ON" => "project",
                    _ => "person",
                };
                entities.push(DocumentEntity {
                    name: object.clone(),
                    entity_type: obj_type.to_string(),
                    properties: std::collections::HashMap::new(),
                });
                
                relationships.push(DocumentRelationship {
                    subject,
                    predicate: rel.to_string(),
                    object,
                });
            }
        }

        // Pattern: "X's email is Y"
        for (name, email) in self.extract_possessive_property(document, "email") {
            if let Some(e) = entities.iter_mut().find(|e| e.name.to_lowercase() == name.to_lowercase()) {
                e.properties.insert("email".to_string(), email);
            } else {
                let mut props = std::collections::HashMap::new();
                props.insert("email".to_string(), email);
                entities.push(DocumentEntity {
                    name: self.capitalize_words(&name),
                    entity_type: "person".to_string(),
                    properties: props,
                });
            }
        }

        // Pattern: "X's phone is Y"
        for (name, phone) in self.extract_possessive_property(document, "phone") {
            if let Some(e) = entities.iter_mut().find(|e| e.name.to_lowercase() == name.to_lowercase()) {
                e.properties.insert("phone".to_string(), phone);
            } else {
                let mut props = std::collections::HashMap::new();
                props.insert("phone".to_string(), phone);
                entities.push(DocumentEntity {
                    name: self.capitalize_words(&name),
                    entity_type: "person".to_string(),
                    properties: props,
                });
            }
        }

        // Pattern: "We're building X", "I'm building X", "I built X", "I created X", "I co-created X"
        // If user identity is set, create relationships too
        for pattern in &[
            "we're building ", "we are building ", "i'm building ", "i am building ",
        ] {
            if let Some(product) = self.extract_after_pattern(&lower, &[pattern]) {
                let product_name = self.extract_product_name(&product);
                if !product_name.is_empty() {
                    entities.push(DocumentEntity {
                        name: product_name.clone(),
                        entity_type: "product".to_string(),
                        properties: std::collections::HashMap::new(),
                    });
                    // Create relationship if user identity is set
                    if let Some(ref user) = user_name {
                        relationships.push(DocumentRelationship {
                            subject: user.clone(),
                            predicate: "WORKS_ON".to_string(),
                            object: product_name.clone(),
                        });
                        relationships.push(DocumentRelationship {
                            subject: user.clone(),
                            predicate: "FOUNDED".to_string(),
                            object: product_name,
                        });
                    }
                }
            }
        }

        // Past tense: "I built X", "I created X", "I co-created X", "I developed X"
        for pattern in &[
            "i built ", "i created ", "i co-created ", "i developed ", "i made ",
        ] {
            if let Some(product) = self.extract_after_pattern(&lower, &[pattern]) {
                let product_name = self.extract_product_name(&product);
                if !product_name.is_empty() {
                    entities.push(DocumentEntity {
                        name: product_name.clone(),
                        entity_type: "project".to_string(),
                        properties: std::collections::HashMap::new(),
                    });
                    // Create relationship if user identity is set
                    if let Some(ref user) = user_name {
                        relationships.push(DocumentRelationship {
                            subject: user.clone(),
                            predicate: "WORKED_ON".to_string(),
                            object: product_name,
                        });
                    }
                }
            }
        }

        // Also extract all products mentioned with "building X" without "we're"/"I'm"
        for pattern in &["building ", "developed ", "created "] {
            if let Some(product) = self.extract_after_pattern(&lower, &[pattern]) {
                let product_name = self.extract_product_name(&product);
                if !product_name.is_empty() {
                    entities.push(DocumentEntity {
                        name: product_name,
                        entity_type: "product".to_string(),
                        properties: std::collections::HashMap::new(),
                    });
                }
            }
        }

        // Pattern: "joined X", "I've joined X" (for organizations/teams)
        for pattern in &["i've joined ", "i joined ", "joined the ", "joined "] {
            if let Some(org) = self.extract_after_pattern(&lower, &[pattern]) {
                // Filter out common non-org words
                if !org.starts_with("a ") && !org.starts_with("the ") && org.len() > 2 {
                    let org_name = self.capitalize_words(&org);
                    entities.push(DocumentEntity {
                        name: org_name.clone(),
                        entity_type: "organization".to_string(),
                        properties: std::collections::HashMap::new(),
                    });
                    // Create relationship if user identity is set
                    if let Some(ref user) = user_name {
                        relationships.push(DocumentRelationship {
                            subject: user.clone(),
                            predicate: "MEMBER_OF".to_string(),
                            object: org_name,
                        });
                    }
                }
            }
        }

        // Pattern: "at the X" (for current team/org membership)
        for pattern in &["at the ", "part of "] {
            if let Some(org) = self.extract_after_pattern(&lower, &[pattern]) {
                // Filter out common non-org words
                if !org.starts_with("a ") && !org.starts_with("the ") && org.len() > 2 {
                    let org_name = self.capitalize_words(&org);
                    entities.push(DocumentEntity {
                        name: org_name,
                        entity_type: "organization".to_string(),
                        properties: std::collections::HashMap::new(),
                    });
                }
            }
        }

        // Pattern: Extract skills/technologies mentioned
        let tech_keywords = [
            "rust", "python", "javascript", "typescript", "java", "c++", "go", "ruby",
            "react", "vue", "angular", "node", "django", "flask", "rails",
            "ai", "ml", "machine learning", "deep learning", "nlp", "natural language",
            "aws", "gcp", "azure", "docker", "kubernetes",
            "sql", "nosql", "postgresql", "mongodb", "redis",
            "api", "rest", "graphql", "microservices",
        ];
        for keyword in tech_keywords {
            if lower.contains(keyword) {
                let skill_name = self.capitalize_words(keyword);
                entities.push(DocumentEntity {
                    name: skill_name.clone(),
                    entity_type: "skill".to_string(),
                    properties: std::collections::HashMap::new(),
                });
                // Create HAS_SKILL relationship if user identity is set
                if let Some(ref user) = user_name {
                    relationships.push(DocumentRelationship {
                        subject: user.clone(),
                        predicate: "HAS_SKILL".to_string(),
                        object: skill_name,
                    });
                }
            }
        }

        // Pattern: Extract industries mentioned
        let industry_keywords = [
            ("healthcare", "Healthcare"),
            ("medical", "Healthcare"),
            ("health systems", "Healthcare"),
            ("finance", "Finance"),
            ("financial", "Finance"),
            ("trading", "Finance"),
            ("quant", "Quantitative Finance"),
            ("insurance", "Insurance"),
            ("legal", "Legal"),
            ("law firm", "Legal"),
            ("education", "Education"),
            ("edtech", "Education Technology"),
        ];
        let mut seen_industries = std::collections::HashSet::new();
        for (keyword, industry) in industry_keywords {
            if lower.contains(keyword) && !seen_industries.contains(industry) {
                seen_industries.insert(industry);
                entities.push(DocumentEntity {
                    name: industry.to_string(),
                    entity_type: "industry".to_string(),
                    properties: std::collections::HashMap::new(),
                });
                // Create IN_INDUSTRY relationship if user identity is set
                if let Some(ref user) = user_name {
                    relationships.push(DocumentRelationship {
                        subject: user.clone(),
                        predicate: "IN_INDUSTRY".to_string(),
                        object: industry.to_string(),
                    });
                }
            }
        }

        // Pattern: Extract specific technologies/frameworks (more specific than skills)
        let tech_frameworks = [
            ("react", "React"), ("next.js", "Next.js"), ("nextjs", "Next.js"),
            ("typescript", "TypeScript"), ("javascript", "JavaScript"),
            ("fastapi", "FastAPI"), ("django", "Django"), ("flask", "Flask"),
            ("docker", "Docker"), ("kubernetes", "Kubernetes"), ("k8s", "Kubernetes"),
            ("aws", "AWS"), ("azure", "Azure"), ("gcp", "GCP"),
            ("redis", "Redis"), ("postgresql", "PostgreSQL"), ("mongodb", "MongoDB"),
            ("pytorch", "PyTorch"), ("tensorflow", "TensorFlow"),
            ("llama", "LLaMA"), ("openai", "OpenAI"), ("ollama", "Ollama"),
            ("ci/cd", "CI/CD"), ("gitlab", "GitLab"), ("github", "GitHub"),
            ("scala", "Scala"), ("haskell", "Haskell"), ("kotlin", "Kotlin"),
            ("llvm", "LLVM"), ("mediapipe", "MediaPipe"),
            ("discordjs", "DiscordJS"), ("discord.js", "DiscordJS"),
            ("vercel", "Vercel"), ("numpy", "NumPy"), ("pandas", "Pandas"),
        ];
        let mut seen_tech = std::collections::HashSet::new();
        for (keyword, tech) in tech_frameworks {
            if lower.contains(keyword) && !seen_tech.contains(tech) {
                seen_tech.insert(tech);
                entities.push(DocumentEntity {
                    name: tech.to_string(),
                    entity_type: "technology".to_string(),
                    properties: std::collections::HashMap::new(),
                });
            }
        }

        // Pattern: Extract universities/educational institutions
        let universities = [
            ("imperial college", "Imperial College London"),
            ("imperial college london", "Imperial College London"),
            ("oxford", "University of Oxford"),
            ("cambridge", "University of Cambridge"),
            ("mit", "MIT"),
            ("stanford", "Stanford University"),
            ("harvard", "Harvard University"),
            ("ucl", "University College London"),
            ("lse", "London School of Economics"),
        ];
        for (keyword, uni) in universities {
            if lower.contains(keyword) {
                entities.push(DocumentEntity {
                    name: uni.to_string(),
                    entity_type: "university".to_string(),
                    properties: std::collections::HashMap::new(),
                });
                // Create STUDIED_AT relationship if user identity is set
                if let Some(ref user) = user_name {
                    relationships.push(DocumentRelationship {
                        subject: user.clone(),
                        predicate: "STUDIED_AT".to_string(),
                        object: uni.to_string(),
                    });
                }
            }
        }

        // Pattern: Extract degrees
        let degree_patterns = [
            ("meng ", "MEng"), ("bsc ", "BSc"), ("beng ", "BEng"),
            ("msc ", "MSc"), ("phd ", "PhD"), ("ba ", "BA"), ("ma ", "MA"),
            ("mba ", "MBA"),
        ];
        for (keyword, degree_type) in degree_patterns {
            if lower.contains(keyword) {
                // Try to extract full degree name
                if let Some(pos) = lower.find(keyword) {
                    let rest = &document[pos..];
                    let degree_name: String = rest
                        .split(|c: char| c == '\n' || c == ';' || c == '|')
                        .next()
                        .unwrap_or("")
                        .trim()
                        .chars()
                        .take(50)
                        .collect();
                    
                    if !degree_name.is_empty() {
                        entities.push(DocumentEntity {
                            name: degree_name.clone(),
                            entity_type: "degree".to_string(),
                            properties: std::collections::HashMap::new(),
                        });
                        // Create HAS_DEGREE relationship if user identity is set
                        if let Some(ref user) = user_name {
                            relationships.push(DocumentRelationship {
                                subject: user.clone(),
                                predicate: "HAS_DEGREE".to_string(),
                                object: degree_name,
                            });
                        }
                    }
                }
            }
        }

        // Pattern: Extract spoken languages
        let spoken_langs = [
            ("english", "English"), ("french", "French"), ("spanish", "Spanish"),
            ("german", "German"), ("mandarin", "Mandarin"), ("chinese", "Chinese"),
            ("hindi", "Hindi"), ("tamil", "Tamil"), ("arabic", "Arabic"),
            ("japanese", "Japanese"), ("korean", "Korean"), ("portuguese", "Portuguese"),
            ("italian", "Italian"), ("russian", "Russian"),
        ];
        // Look for language patterns like "proficiency in X" or "native in X" or "spoken languages"
        let lang_context = lower.contains("language") || lower.contains("proficiency") 
            || lower.contains("native") || lower.contains("fluent") || lower.contains("spoken");
        if lang_context {
            for (keyword, lang) in spoken_langs {
                if lower.contains(keyword) {
                    entities.push(DocumentEntity {
                        name: lang.to_string(),
                        entity_type: "spokenlanguage".to_string(),
                        properties: std::collections::HashMap::new(),
                    });
                    // Create SPEAKS relationship if user identity is set
                    if let Some(ref user) = user_name {
                        relationships.push(DocumentRelationship {
                            subject: user.clone(),
                            predicate: "SPEAKS".to_string(),
                            object: lang.to_string(),
                        });
                    }
                }
            }
        }

        // Pattern: Extract achievements (hackathon wins, awards, places)
        let achievement_patterns = [
            "won ", "1st place", "first place", "2nd place", "second place",
            "3rd place", "third place", "award", "prize", "recognition",
        ];
        for pattern in achievement_patterns {
            if lower.contains(pattern) {
                // Try to extract the achievement context
                if let Some(pos) = lower.find(pattern) {
                    let start = pos.saturating_sub(30);
                    let end = (pos + 60).min(document.len());
                    let context = &document[start..end];
                    let achievement = context.trim().to_string();
                    
                    if achievement.len() > 10 {
                        entities.push(DocumentEntity {
                            name: achievement.chars().take(80).collect(),
                            entity_type: "achievement".to_string(),
                            properties: std::collections::HashMap::new(),
                        });
                    }
                }
            }
        }

        // Pattern: Extract projects mentioned (e.g., "Virtual Poker", "WACC Compiler")
        let project_indicators = [
            "created ", "built ", "developed ", "designed ", "programmed ",
            "implemented ", "architected ",
        ];
        for indicator in project_indicators {
            let mut search_start = 0;
            while let Some(pos) = lower[search_start..].find(indicator) {
                let abs_pos = search_start + pos;
                let rest = &document[abs_pos + indicator.len()..];
                
                // Extract project name (usually starts with capital letter or is quoted)
                let project_name: String = rest
                    .split(|c: char| c == ',' || c == '.' || c == ';' || c == '\n' || c == '–' || c == '-')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .split_whitespace()
                    .take(5)
                    .collect::<Vec<_>>()
                    .join(" ");

                if project_name.len() > 2 && project_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    entities.push(DocumentEntity {
                        name: project_name.clone(),
                        entity_type: "project".to_string(),
                        properties: std::collections::HashMap::new(),
                    });
                    // Create BUILT relationship if user identity is set
                    if let Some(ref user) = user_name {
                        relationships.push(DocumentRelationship {
                            subject: user.clone(),
                            predicate: "BUILT".to_string(),
                            object: project_name,
                        });
                    }
                }
                
                search_start = abs_pos + indicator.len();
            }
        }

        // Deduplicate entities by name
        let mut unique_entities: Vec<DocumentEntity> = Vec::new();
        for entity in entities {
            if let Some(existing) = unique_entities.iter_mut()
                .find(|e| e.name.to_lowercase() == entity.name.to_lowercase()) 
            {
                // Merge properties
                for (k, v) in entity.properties {
                    existing.properties.insert(k, v);
                }
            } else {
                unique_entities.push(entity);
            }
        }

        (unique_entities, relationships)
    }

    /// Extract first-person name from "I am X" or "My name is X"
    fn extract_first_person_name(&self, text: &str) -> Option<String> {
        let lower = text.to_lowercase();
        
        for pattern in &["i am ", "my name is ", "i'm ", "this is "] {
            if let Some(pos) = lower.find(pattern) {
                let rest = &text[pos + pattern.len()..];
                // Take words until we hit a comma, period, or common word
                let name: String = rest
                    .split(|c: char| c == ',' || c == '.' || c == '!' || c == '\n')
                    .next()?
                    .split_whitespace()
                    .take_while(|w| {
                        let lw = w.to_lowercase();
                        !["and", "i", "who", "working", "from", "living", "based"].contains(&lw.as_str())
                    })
                    .take(3) // Max 3 words for a name
                    .collect::<Vec<_>>()
                    .join(" ");
                
                if !name.is_empty() {
                    return Some(self.capitalize_words(&name));
                }
            }
        }
        
        None
    }

    /// Extract pattern after certain phrases
    fn extract_after_pattern(&self, lower_text: &str, patterns: &[&str]) -> Option<String> {
        for pattern in patterns {
            if let Some(pos) = lower_text.find(pattern) {
                let rest = &lower_text[pos + pattern.len()..];
                let value: String = rest
                    .split(|c: char| c == ',' || c == '.' || c == '!' || c == '\n' || c == ';')
                    .next()?
                    .split_whitespace()
                    .take_while(|w| !["and", "as", "where", "since", "for", "in"].contains(w))
                    .take(4)
                    .collect::<Vec<_>>()
                    .join(" ");
                
                if !value.is_empty() {
                    return Some(value);
                }
            }
        }
        None
    }

    /// Extract email from text
    fn extract_email_for(&self, text: &str, _patterns: &[&str]) -> Option<String> {
        // Simple email regex-like search
        for word in text.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '@' && c != '.' && c != '_' && c != '-');
            if clean.contains('@') && clean.contains('.') {
                return Some(clean.to_string());
            }
        }
        None
    }

    /// Extract phone from text
    fn extract_phone_for(&self, text: &str, _patterns: &[&str]) -> Option<String> {
        // Look for phone number patterns
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        
        while i < chars.len() {
            // Look for sequences of digits possibly with - ( ) or spaces
            if chars[i].is_ascii_digit() || chars[i] == '(' || chars[i] == '+' {
                let start = i;
                let mut digit_count = 0;
                
                while i < chars.len() {
                    if chars[i].is_ascii_digit() {
                        digit_count += 1;
                        i += 1;
                    } else if chars[i] == '-' || chars[i] == ' ' || chars[i] == '(' || chars[i] == ')' || chars[i] == '+' {
                        i += 1;
                    } else {
                        break;
                    }
                }
                
                if digit_count >= 7 && digit_count <= 15 {
                    let phone: String = chars[start..i].iter().collect();
                    return Some(phone.trim().to_string());
                }
            } else {
                i += 1;
            }
        }
        
        None
    }

    /// Extract "Subject pattern Object" patterns
    fn extract_subject_pattern_object(&self, text: &str, pattern: &str) -> Vec<(String, String)> {
        let lower = text.to_lowercase();
        let mut results = Vec::new();

        let mut search_start = 0;
        while let Some(pos) = lower[search_start..].find(pattern) {
            let abs_pos = search_start + pos;
            
            // Get subject (word before pattern)
            let before = &text[..abs_pos];
            let subject = before
                .split_whitespace()
                .last()
                .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
                .filter(|s| !s.is_empty() && s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false));

            // Get object (words after pattern)
            let after = &text[abs_pos + pattern.len()..];
            let object: String = after
                .split(|c: char| c == ',' || c == '.' || c == '!' || c == '\n' || c == ';')
                .next()
                .unwrap_or("")
                .trim()
                .split_whitespace()
                .take(3)
                .collect::<Vec<_>>()
                .join(" ");

            if let Some(subj) = subject {
                if !object.is_empty() {
                    results.push((
                        self.capitalize_words(subj),
                        self.capitalize_words(&object),
                    ));
                }
            }

            search_start = abs_pos + pattern.len();
        }

        results
    }

    /// Extract "X's property is Y" patterns
    fn extract_possessive_property(&self, text: &str, property: &str) -> Vec<(String, String)> {
        let lower = text.to_lowercase();
        let mut results = Vec::new();

        let patterns = [
            format!("'s {} is ", property),
            format!("'s {} ", property),
        ];

        for pattern in &patterns {
            let mut search_start = 0;
            while let Some(pos) = lower[search_start..].find(pattern.as_str()) {
                let abs_pos = search_start + pos;
                
                // Get name before 's
                let before = &text[..abs_pos];
                let name = before.split_whitespace().last().map(|s| s.to_string());

                // Get value after pattern
                let after_pos = abs_pos + pattern.len();
                let after = &text[after_pos..];
                let value = after
                    .split_whitespace()
                    .next()
                    .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric() && c != '@' && c != '.').to_string())
                    .filter(|s| !s.is_empty());

                if let (Some(n), Some(v)) = (name, value) {
                    results.push((n, v));
                }

                search_start = after_pos;
            }
        }

        results
    }

    /// Capitalize each word
    fn capitalize_words(&self, s: &str) -> String {
        s.split_whitespace()
            .map(|word| {
                let mut chars: Vec<char> = word.chars().collect();
                if let Some(first) = chars.first_mut() {
                    *first = first.to_uppercase().next().unwrap_or(*first);
                }
                chars.into_iter().collect::<String>()
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Extract a product/project name from text (handles "VaNI, an admin..." format)
    fn extract_product_name(&self, text: &str) -> String {
        // Take until comma, period, or common delimiters
        let name: String = text
            .split(|c: char| c == ',' || c == '.' || c == ':' || c == ';' || c == '!' || c == '\n')
            .next()
            .unwrap_or("")
            .trim()
            .split_whitespace()
            .take_while(|w| {
                let lw = w.to_lowercase();
                // Stop at common words that indicate description
                !["an", "a", "the", "which", "that", "for", "to", "with", "and"].contains(&lw.as_str())
            })
            .take(4) // Max 4 words for a name
            .collect::<Vec<_>>()
            .join(" ");
        
        self.capitalize_words(&name)
    }

    /// Apply extracted entities to the knowledge graph
    fn apply_entities(&self, entities: &[DocumentEntity], result: &mut IngestionResult) -> Result<()> {
        for entity in entities {
            let entity_type = EntityType::from_str(&entity.entity_type);
            
            // Check if entity already exists (works for all entity types)
            let existing = self.graph.find_by_name(&entity.name)?;

            let entity_id = if let Some(e) = existing {
                // Entity exists - just update properties if needed
                e.id
            } else {
                // Add new entity
                let new_entity = match entity_type {
                    EntityType::Person => self.graph.add_person(&entity.name)?,
                    EntityType::Organization => self.graph.add_organization(&entity.name)?,
                    EntityType::Project => self.graph.add_project(&entity.name)?,
                    EntityType::Location => self.graph.add_location(&entity.name)?,
                    EntityType::Task => self.graph.add_task_entity(&entity.name)?,
                    _ => self.graph.add_entity(entity_type.clone(), &entity.name)?,
                };
                result.entities_added.push(format!("{} ({})", entity.name, entity.entity_type));
                new_entity.id
            };

            // Set properties
            for (key, value) in &entity.properties {
                self.graph.set_property(entity_id, key, value)?;
                result.properties_set.push(format!("{}.{} = {}", entity.name, key, value));
            }
        }

        Ok(())
    }

    /// Apply extracted relationships to the knowledge graph
    fn apply_relationships(&self, relationships: &[DocumentRelationship], result: &mut IngestionResult) -> Result<()> {
        for rel in relationships {
            // Find subject and object
            let subject = self.find_entity_by_name(&rel.subject)?;
            let object = self.find_entity_by_name(&rel.object)?;

            if let (Some(subj), Some(obj)) = (subject, object) {
                let rel_type = RelationshipType::from_str(&rel.predicate);
                
                // Create the relationship
                match self.graph.create_relationship(subj.id, obj.id, rel_type) {
                    Ok(_) => {
                        result.relationships_added.push(format!(
                            "{} {} {}",
                            rel.subject, rel.predicate.to_lowercase().replace('_', " "), rel.object
                        ));
                    }
                    Err(e) => {
                        // Might be duplicate - that's ok
                        if !e.to_string().contains("UNIQUE") {
                            result.errors.push(format!("Failed to add relationship: {}", e));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Find entity by name (uses generic search for all entity types)
    fn find_entity_by_name(&self, name: &str) -> Result<Option<crate::knowledge::entities::Entity>> {
        self.graph.find_by_name(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Database;

    fn setup() -> KnowledgeGraph {
        let db = Database::in_memory().unwrap();
        KnowledgeGraph::new(db)
    }

    #[test]
    fn test_extract_first_person_name() {
        let graph = setup();
        let ingester = DocumentIngester::new(&graph);

        assert_eq!(
            ingester.extract_first_person_name("I am John Smith and I work at Google."),
            Some("John Smith".to_string())
        );
        
        assert_eq!(
            ingester.extract_first_person_name("My name is Alice Johnson, from Seattle."),
            Some("Alice Johnson".to_string())
        );
    }

    #[test]
    fn test_extract_patterns_personal_bio() {
        let graph = setup();
        let ingester = DocumentIngester::new(&graph);

        let bio = "I am John Smith. I work at Google. My email is john@google.com.";
        let (entities, relationships) = ingester.extract_patterns(bio);

        // Should find John Smith
        assert!(entities.iter().any(|e| e.name == "John Smith"));
        
        // Should find Google
        assert!(entities.iter().any(|e| e.name == "Google"));
        
        // Should have WORKS_AT relationship
        assert!(relationships.iter().any(|r| r.predicate == "WORKS_AT"));
    }

    #[test]
    fn test_extract_email() {
        let graph = setup();
        let ingester = DocumentIngester::new(&graph);

        let text = "Contact me at test@example.com for more info.";
        let email = ingester.extract_email_for(text, &[]);
        
        assert_eq!(email, Some("test@example.com".to_string()));
    }

    #[test]
    fn test_extract_phone() {
        let graph = setup();
        let ingester = DocumentIngester::new(&graph);

        let text = "Call me at 555-123-4567 anytime.";
        let phone = ingester.extract_phone_for(text, &[]);
        
        assert!(phone.is_some());
        assert!(phone.unwrap().contains("555"));
    }

    #[test]
    fn test_ingest_fast() {
        let graph = setup();
        let ingester = DocumentIngester::new(&graph);

        // Use a document with technologies and skills (what ingest_fast extracts)
        let doc = "Sarah Connor is a software engineer skilled in Rust, Python, and machine learning. She uses Docker and Kubernetes.";
        let result = ingester.ingest_fast_for(doc, Some("Sarah Connor".to_string())).unwrap();

        assert!(!result.entities_added.is_empty(), "Should extract skills/technologies");
        
        // Verify Sarah was added
        let sarah = graph.find_person("Sarah Connor").unwrap();
        assert!(sarah.is_some(), "Subject should be added to graph");
        
        // Verify skills/technologies were extracted
        assert!(result.entities_added.iter().any(|e| e.to_lowercase().contains("rust") || e.to_lowercase().contains("python")));
    }

    #[test]
    fn test_parse_entity_line() {
        let graph = setup();
        let ingester = DocumentIngester::new(&graph);

        let line = "[PERSON] John Smith | email=john@test.com | role=engineer";
        let entity = ingester.parse_entity_line(line);

        assert!(entity.is_some());
        let e = entity.unwrap();
        assert_eq!(e.name, "John Smith");
        assert_eq!(e.entity_type, "person");
        assert_eq!(e.properties.get("email"), Some(&"john@test.com".to_string()));
    }

    #[test]
    fn test_parse_relationship_line() {
        let graph = setup();
        let ingester = DocumentIngester::new(&graph);

        let line = "John Smith | WORKS_AT | Google";
        let rel = ingester.parse_relationship_line(line);

        assert!(rel.is_some());
        let r = rel.unwrap();
        assert_eq!(r.subject, "John Smith");
        assert_eq!(r.predicate, "WORKS_AT");
        assert_eq!(r.object, "Google");
    }

    #[test]
    fn test_capitalize_words() {
        let graph = setup();
        let ingester = DocumentIngester::new(&graph);

        assert_eq!(ingester.capitalize_words("john smith"), "John Smith");
        assert_eq!(ingester.capitalize_words("alice"), "Alice");
    }

    #[test]
    fn test_extract_subject_pattern_object() {
        let graph = setup();
        let ingester = DocumentIngester::new(&graph);

        let text = "Alice works at Google. Bob works at Microsoft.";
        let results = ingester.extract_subject_pattern_object(text, "works at");

        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|(s, o)| s == "Alice" && o == "Google"));
        assert!(results.iter().any(|(s, o)| s == "Bob" && o == "Microsoft"));
    }
}
