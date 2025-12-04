use anyhow::Result;
use crate::storage::Database;
use super::entities::{Entity, EntityType, RelationshipType};

const USER_IDENTITY_KEY: &str = "user_identity";

/// Knowledge graph operations built on top of the database
pub struct KnowledgeGraph {
    db: Database,
}

impl KnowledgeGraph {
    /// Create a new knowledge graph with the given database
    pub fn new(db: Database) -> Self {
        KnowledgeGraph { db }
    }

    // ========================================================================
    // User Identity
    // ========================================================================

    /// Set the current user identity (which person node represents "me")
    pub fn set_user_identity(&self, name: &str) -> Result<Option<Entity>> {
        // Verify the person exists
        if let Some(entity) = self.find_person(name)? {
            self.db.set_setting(USER_IDENTITY_KEY, name)?;
            Ok(Some(entity))
        } else {
            Ok(None)
        }
    }

    /// Get the current user identity
    pub fn get_user_identity(&self) -> Result<Option<Entity>> {
        if let Some(name) = self.db.get_setting(USER_IDENTITY_KEY)? {
            self.find_person(&name)
        } else {
            Ok(None)
        }
    }

    /// Get the user identity name (without loading full entity)
    pub fn get_user_identity_name(&self) -> Result<Option<String>> {
        self.db.get_setting(USER_IDENTITY_KEY)
    }

    /// Clear the user identity
    pub fn clear_user_identity(&self) -> Result<bool> {
        self.db.delete_setting(USER_IDENTITY_KEY)
    }

    /// Resolve first-person pronouns to the user's name
    /// Returns the original text with "I", "me", "my", "myself" replaced with user's name
    pub fn resolve_pronouns(&self, text: &str) -> Result<String> {
        if let Some(user_name) = self.get_user_identity_name()? {
            let mut result = text.to_string();
            
            // Replace pronouns with user's name (case-insensitive, word boundaries)
            // Order matters - check longer patterns first
            let replacements = [
                (r"\bI am\b", format!("{} is", user_name)),
                (r"\bI'm\b", format!("{} is", user_name)),
                (r"\bI have\b", format!("{} has", user_name)),
                (r"\bI've\b", format!("{} has", user_name)),
                (r"\bI do\b", format!("{} does", user_name)),
                (r"\bI don't\b", format!("{} doesn't", user_name)),
                (r"\bdo I\b", format!("does {}", user_name)),
                (r"\bam I\b", format!("is {}", user_name)),
                (r"\bmyself\b", user_name.clone()),
                (r"\bmy\b", format!("{}'s", user_name)),
                (r"\bmine\b", format!("{}'s", user_name)),
                (r"\bme\b", user_name.clone()),
                (r"\bI\b", user_name.clone()),
            ];
            
            for (pattern, replacement) in replacements {
                // Simple word boundary replacement (case-insensitive)
                let re = regex::RegexBuilder::new(pattern)
                    .case_insensitive(true)
                    .build();
                if let Ok(re) = re {
                    result = re.replace_all(&result, replacement.as_str()).to_string();
                }
            }
            
            Ok(result)
        } else {
            Ok(text.to_string())
        }
    }

    // ========================================================================
    // Entity Management
    // ========================================================================

    /// Add a person to the knowledge graph
    pub fn add_person(&self, name: &str) -> Result<Entity> {
        let id = self.db.insert_entity("person", name)?;
        Ok(Entity::new(id, EntityType::Person, name.to_string()))
    }

    /// Add an organization to the knowledge graph
    pub fn add_organization(&self, name: &str) -> Result<Entity> {
        let id = self.db.insert_entity("organization", name)?;
        Ok(Entity::new(id, EntityType::Organization, name.to_string()))
    }

    /// Add a project to the knowledge graph
    pub fn add_project(&self, name: &str) -> Result<Entity> {
        let id = self.db.insert_entity("project", name)?;
        Ok(Entity::new(id, EntityType::Project, name.to_string()))
    }

    /// Add a location to the knowledge graph
    pub fn add_location(&self, name: &str) -> Result<Entity> {
        let id = self.db.insert_entity("location", name)?;
        Ok(Entity::new(id, EntityType::Location, name.to_string()))
    }

    /// Add a generic entity
    pub fn add_entity(&self, entity_type: EntityType, name: &str) -> Result<Entity> {
        let id = self.db.insert_entity(entity_type.as_str(), name)?;
        Ok(Entity::new(id, entity_type, name.to_string()))
    }

    /// Set a property on an entity
    pub fn set_property(&self, entity_id: i64, key: &str, value: &str) -> Result<()> {
        self.db.set_entity_property(entity_id, key, value)
    }

    /// Delete a property from an entity
    pub fn delete_property(&self, entity_id: i64, key: &str) -> Result<bool> {
        self.db.delete_entity_property(entity_id, key)
    }

    /// Rename an entity
    pub fn rename_entity(&self, entity_id: i64, new_name: &str) -> Result<bool> {
        self.db.update_entity_name(entity_id, new_name)
    }

    /// Rename an entity by name (finds entity first)
    pub fn rename_by_name(&self, old_name: &str, new_name: &str) -> Result<Option<i64>> {
        // Try to find the entity by name in any type
        if let Some(entity) = self.find_person(old_name)? {
            self.rename_entity(entity.id, new_name)?;
            return Ok(Some(entity.id));
        }
        if let Some(entity) = self.find_organization(old_name)? {
            self.rename_entity(entity.id, new_name)?;
            return Ok(Some(entity.id));
        }
        if let Some(entity) = self.find_project(old_name)? {
            self.rename_entity(entity.id, new_name)?;
            return Ok(Some(entity.id));
        }
        if let Some(entity) = self.find_location(old_name)? {
            self.rename_entity(entity.id, new_name)?;
            return Ok(Some(entity.id));
        }
        Ok(None)
    }

    /// Delete an entity by ID
    pub fn delete_entity(&self, entity_id: i64) -> Result<bool> {
        self.db.delete_entity(entity_id)
    }

    /// Delete an entity by name (finds entity first)
    pub fn delete_by_name(&self, name: &str) -> Result<Option<i64>> {
        // Try to find the entity by name in any type
        if let Some(entity) = self.find_person(name)? {
            self.delete_entity(entity.id)?;
            return Ok(Some(entity.id));
        }
        if let Some(entity) = self.find_organization(name)? {
            self.delete_entity(entity.id)?;
            return Ok(Some(entity.id));
        }
        if let Some(entity) = self.find_project(name)? {
            self.delete_entity(entity.id)?;
            return Ok(Some(entity.id));
        }
        if let Some(entity) = self.find_location(name)? {
            self.delete_entity(entity.id)?;
            return Ok(Some(entity.id));
        }
        Ok(None)
    }

    /// Delete a relationship between two entities
    pub fn delete_relationship(&self, from_name: &str, to_name: &str, rel_type: &str) -> Result<bool> {
        // Find both entities
        let from_id = self.find_any_entity(from_name)?;
        let to_id = self.find_any_entity(to_name)?;
        
        match (from_id, to_id) {
            (Some(from), Some(to)) => self.db.delete_relationship(from.id, to.id, rel_type),
            _ => Ok(false),
        }
    }

    /// Find any entity by name (regardless of type)
    pub fn find_any_entity(&self, name: &str) -> Result<Option<Entity>> {
        if let Some(e) = self.find_person(name)? { return Ok(Some(e)); }
        if let Some(e) = self.find_organization(name)? { return Ok(Some(e)); }
        if let Some(e) = self.find_project(name)? { return Ok(Some(e)); }
        if let Some(e) = self.find_location(name)? { return Ok(Some(e)); }
        Ok(None)
    }

    /// Create a relationship between two entities
    pub fn create_relationship(
        &self,
        from_id: i64,
        to_id: i64,
        rel_type: RelationshipType,
    ) -> Result<i64> {
        self.db.create_relationship(from_id, to_id, rel_type.as_str())
    }

    /// Link a person to an organization (WORKS_AT relationship)
    pub fn link_works_at(&self, person_id: i64, org_id: i64) -> Result<i64> {
        self.create_relationship(person_id, org_id, RelationshipType::WorksAt)
    }

    /// Link two people as knowing each other
    pub fn link_knows(&self, person1_id: i64, person2_id: i64) -> Result<()> {
        // Create bidirectional relationship
        self.create_relationship(person1_id, person2_id, RelationshipType::Knows)?;
        self.create_relationship(person2_id, person1_id, RelationshipType::Knows)?;
        Ok(())
    }

    /// Link a person to a project (WORKS_ON relationship)
    pub fn link_works_on(&self, person_id: i64, project_id: i64) -> Result<i64> {
        self.create_relationship(person_id, project_id, RelationshipType::WorksOn)
    }

    /// Link a person or organization to a location (LOCATED_IN relationship)
    pub fn link_located_in(&self, entity_id: i64, location_id: i64) -> Result<i64> {
        self.create_relationship(entity_id, location_id, RelationshipType::LocatedIn)
    }

    /// Link a person as managing another entity (MANAGES relationship)
    pub fn link_manages(&self, person_id: i64, managed_id: i64) -> Result<i64> {
        self.create_relationship(person_id, managed_id, RelationshipType::Manages)
    }

    /// Link a person as reporting to another (REPORTS_TO relationship)
    pub fn link_reports_to(&self, person_id: i64, manager_id: i64) -> Result<i64> {
        self.create_relationship(person_id, manager_id, RelationshipType::ReportsTo)
    }

    /// Link an entity as owning another (OWNS relationship)
    pub fn link_owns(&self, owner_id: i64, owned_id: i64) -> Result<i64> {
        self.create_relationship(owner_id, owned_id, RelationshipType::Owns)
    }

    /// Link two people as collaborating (COLLABORATES_WITH relationship)
    pub fn link_collaborates(&self, person1_id: i64, person2_id: i64) -> Result<()> {
        // Create bidirectional relationship
        self.create_relationship(person1_id, person2_id, RelationshipType::CollaboratesWith)?;
        self.create_relationship(person2_id, person1_id, RelationshipType::CollaboratesWith)?;
        Ok(())
    }

    /// Find an entity by name
    pub fn find_by_name(&self, name: &str) -> Result<Option<Entity>> {
        if let Some((id, type_str)) = self.db.find_entity_by_name(name)? {
            let mut entity = Entity::new(id, EntityType::from_str(&type_str), name.to_string());
            
            // Load properties
            for (key, value) in self.db.get_entity_properties(id)? {
                entity.properties.insert(key, value);
            }
            
            Ok(Some(entity))
        } else {
            Ok(None)
        }
    }

    /// Find a person by name
    pub fn find_person(&self, name: &str) -> Result<Option<Entity>> {
        if let Some(id) = self.db.find_entity("person", name)? {
            let mut entity = Entity::new(id, EntityType::Person, name.to_string());
            for (key, value) in self.db.get_entity_properties(id)? {
                entity.properties.insert(key, value);
            }
            Ok(Some(entity))
        } else {
            Ok(None)
        }
    }

    /// Find an organization by name
    pub fn find_organization(&self, name: &str) -> Result<Option<Entity>> {
        if let Some(id) = self.db.find_entity("organization", name)? {
            let mut entity = Entity::new(id, EntityType::Organization, name.to_string());
            for (key, value) in self.db.get_entity_properties(id)? {
                entity.properties.insert(key, value);
            }
            Ok(Some(entity))
        } else {
            Ok(None)
        }
    }

    /// Find a project by name
    pub fn find_project(&self, name: &str) -> Result<Option<Entity>> {
        if let Some(id) = self.db.find_entity("project", name)? {
            let mut entity = Entity::new(id, EntityType::Project, name.to_string());
            for (key, value) in self.db.get_entity_properties(id)? {
                entity.properties.insert(key, value);
            }
            Ok(Some(entity))
        } else {
            Ok(None)
        }
    }

    /// Find a location by name
    pub fn find_location(&self, name: &str) -> Result<Option<Entity>> {
        if let Some(id) = self.db.find_entity("location", name)? {
            let mut entity = Entity::new(id, EntityType::Location, name.to_string());
            for (key, value) in self.db.get_entity_properties(id)? {
                entity.properties.insert(key, value);
            }
            Ok(Some(entity))
        } else {
            Ok(None)
        }
    }

    /// Find an entity by property value (e.g., email)
    pub fn find_by_property(&self, key: &str, value: &str) -> Result<Option<Entity>> {
        let entity_id = self.db.find_entity_by_property(key, value)?;
        if let Some(id) = entity_id {
            // Get entity name and type
            if let Some((name, type_str)) = self.db.get_entity_by_id(id)? {
                let entity_type = EntityType::from_str(&type_str);
                let mut entity = Entity::new(id, entity_type, name);
                for (k, v) in self.db.get_entity_properties(id)? {
                    entity.properties.insert(k, v);
                }
                return Ok(Some(entity));
            }
        }
        Ok(None)
    }

    /// List all people
    pub fn list_people(&self) -> Result<Vec<Entity>> {
        let people = self.db.list_entities("person")?;
        let mut entities = Vec::new();
        for (id, name) in people {
            let mut entity = Entity::new(id, EntityType::Person, name);
            for (key, value) in self.db.get_entity_properties(id)? {
                entity.properties.insert(key, value);
            }
            entities.push(entity);
        }
        Ok(entities)
    }

    /// List all organizations
    pub fn list_organizations(&self) -> Result<Vec<Entity>> {
        let orgs = self.db.list_entities("organization")?;
        let mut entities = Vec::new();
        for (id, name) in orgs {
            let mut entity = Entity::new(id, EntityType::Organization, name);
            for (key, value) in self.db.get_entity_properties(id)? {
                entity.properties.insert(key, value);
            }
            entities.push(entity);
        }
        Ok(entities)
    }

    /// List all projects
    pub fn list_projects(&self) -> Result<Vec<Entity>> {
        let projects = self.db.list_entities("project")?;
        let mut entities = Vec::new();
        for (id, name) in projects {
            let mut entity = Entity::new(id, EntityType::Project, name);
            for (key, value) in self.db.get_entity_properties(id)? {
                entity.properties.insert(key, value);
            }
            entities.push(entity);
        }
        Ok(entities)
    }

    /// List all locations
    pub fn list_locations(&self) -> Result<Vec<Entity>> {
        let locations = self.db.list_entities("location")?;
        let mut entities = Vec::new();
        for (id, name) in locations {
            let mut entity = Entity::new(id, EntityType::Location, name);
            for (key, value) in self.db.get_entity_properties(id)? {
                entity.properties.insert(key, value);
            }
            entities.push(entity);
        }
        Ok(entities)
    }

    // ========================================================================
    // Task-Knowledge Graph Integration
    // ========================================================================

    /// Add a task to the knowledge graph
    pub fn add_task_entity(&self, title: &str) -> Result<Entity> {
        let id = self.db.insert_entity("task", title)?;
        Ok(Entity::new(id, EntityType::Task, title.to_string()))
    }

    /// Find a task by title
    pub fn find_task(&self, title: &str) -> Result<Option<Entity>> {
        if let Some(id) = self.db.find_entity("task", title)? {
            let mut entity = Entity::new(id, EntityType::Task, title.to_string());
            for (key, value) in self.db.get_entity_properties(id)? {
                entity.properties.insert(key, value);
            }
            Ok(Some(entity))
        } else {
            Ok(None)
        }
    }

    /// List all tasks in the knowledge graph
    pub fn list_tasks(&self) -> Result<Vec<Entity>> {
        let tasks = self.db.list_entities("task")?;
        let mut entities = Vec::new();
        for (id, name) in tasks {
            let mut entity = Entity::new(id, EntityType::Task, name);
            for (key, value) in self.db.get_entity_properties(id)? {
                entity.properties.insert(key, value);
            }
            entities.push(entity);
        }
        Ok(entities)
    }

    /// Assign a task to a person (creates ASSIGNED_TO relationship)
    pub fn assign_task(&self, task_id: i64, person_id: i64) -> Result<i64> {
        self.create_relationship(task_id, person_id, RelationshipType::AssignedTo)
    }

    /// Give a person a task (creates HAS_TASK relationship)
    pub fn link_has_task(&self, person_id: i64, task_id: i64) -> Result<i64> {
        self.create_relationship(person_id, task_id, RelationshipType::HasTask)
    }

    /// Mark a task as blocked by another task
    pub fn link_blocked_by(&self, task_id: i64, blocking_task_id: i64) -> Result<i64> {
        self.create_relationship(task_id, blocking_task_id, RelationshipType::BlockedBy)
    }

    /// Mark a task as depending on another task (soft dependency)
    pub fn link_depends_on(&self, task_id: i64, dependency_id: i64) -> Result<i64> {
        self.create_relationship(task_id, dependency_id, RelationshipType::DependsOn)
    }

    /// Set a task's deadline (stores as property + creates temporal node)
    pub fn set_task_deadline(&self, task_id: i64, deadline: &str) -> Result<()> {
        // Store deadline as property
        self.set_property(task_id, "deadline", deadline)?;
        
        // Create a deadline entity if needed for temporal reasoning
        let deadline_name = format!("deadline:{}", deadline);
        let deadline_entity = if let Some(e) = self.find_by_name(&deadline_name)? {
            e
        } else {
            let id = self.db.insert_entity("deadline", &deadline_name)?;
            let mut e = Entity::new(id, EntityType::Deadline, deadline_name);
            e.properties.insert("timestamp".to_string(), deadline.to_string());
            self.set_property(id, "timestamp", deadline)?;
            e
        };
        
        // Link task to deadline
        self.create_relationship(task_id, deadline_entity.id, RelationshipType::DueBefore)?;
        Ok(())
    }

    /// Get all tasks assigned to a person
    pub fn get_person_tasks(&self, person_name: &str) -> Result<Vec<Entity>> {
        let person = match self.find_person(person_name)? {
            Some(p) => p,
            None => return Ok(vec![]),
        };
        
        // Get tasks via HAS_TASK relationship
        let rels = self.db.get_relationships_from(person.id)?;
        let mut tasks = Vec::new();
        
        for (target_id, rel_type, _) in rels {
            if rel_type == "HAS_TASK" {
                if let Some(task) = self.get_entity_by_id(target_id)? {
                    tasks.push(task);
                }
            }
        }
        
        // Also get tasks that are ASSIGNED_TO this person
        let incoming = self.db.get_relationships_to(person.id)?;
        for (source_id, rel_type, _) in incoming {
            if rel_type == "ASSIGNED_TO" {
                if let Some(task) = self.get_entity_by_id(source_id)? {
                    if !tasks.iter().any(|t| t.id == task.id) {
                        tasks.push(task);
                    }
                }
            }
        }
        
        Ok(tasks)
    }

    /// Get entity by ID
    pub fn get_entity_by_id(&self, id: i64) -> Result<Option<Entity>> {
        if let Some((name, type_str)) = self.db.get_entity_by_id(id)? {
            let mut entity = Entity::new(id, EntityType::from_str(&type_str), name);
            for (key, value) in self.db.get_entity_properties(id)? {
                entity.properties.insert(key, value);
            }
            Ok(Some(entity))
        } else {
            Ok(None)
        }
    }

    /// Get tasks blocked by a given task
    pub fn get_blocked_tasks(&self, blocking_task_name: &str) -> Result<Vec<Entity>> {
        let blocker = match self.find_task(blocking_task_name)? {
            Some(t) => t,
            None => return Ok(vec![]),
        };
        
        let incoming = self.db.get_relationships_to(blocker.id)?;
        let mut blocked = Vec::new();
        
        for (source_id, rel_type, _) in incoming {
            if rel_type == "BLOCKED_BY" {
                if let Some(task) = self.get_entity_by_id(source_id)? {
                    blocked.push(task);
                }
            }
        }
        
        Ok(blocked)
    }

    /// Get tasks that block a given task
    pub fn get_blocking_tasks(&self, task_name: &str) -> Result<Vec<Entity>> {
        let task = match self.find_task(task_name)? {
            Some(t) => t,
            None => return Ok(vec![]),
        };
        
        let rels = self.db.get_relationships_from(task.id)?;
        let mut blockers = Vec::new();
        
        for (target_id, rel_type, _) in rels {
            if rel_type == "BLOCKED_BY" {
                if let Some(blocking) = self.get_entity_by_id(target_id)? {
                    blockers.push(blocking);
                }
            }
        }
        
        Ok(blockers)
    }

    // ========================================================================
    // End Task-Knowledge Graph Integration
    // ========================================================================

    /// Get all information about an entity including relationships
    pub fn get_entity_info(&self, entity_id: i64) -> Result<String> {
        let name = self.db.get_entity_name(entity_id)?
            .ok_or_else(|| anyhow::anyhow!("Entity not found"))?;
        
        let properties = self.db.get_entity_properties(entity_id)?;
        let outgoing = self.db.get_relationships_from(entity_id)?;
        let incoming = self.db.get_relationships_to(entity_id)?;

        let mut info = format!("Entity: {}\n", name);
        
        if !properties.is_empty() {
            info.push_str("Properties:\n");
            for (key, value) in properties {
                info.push_str(&format!("  - {}: {}\n", key, value));
            }
        }

        if !outgoing.is_empty() {
            info.push_str("Relationships:\n");
            for (_, rel_type, target_name) in outgoing {
                info.push_str(&format!("  - {} -> {}\n", rel_type, target_name));
            }
        }

        if !incoming.is_empty() {
            info.push_str("Referenced by:\n");
            for (_, rel_type, source_name) in incoming {
                info.push_str(&format!("  - {} <- {}\n", rel_type, source_name));
            }
        }

        Ok(info)
    }

    /// Get a summary of the entire knowledge graph for LLM context
    pub fn get_summary(&self) -> Result<String> {
        let people = self.list_people()?;
        let orgs = self.list_organizations()?;
        let projects = self.list_projects()?;
        let locations = self.list_locations()?;

        let mut summary = String::new();
        
        if !people.is_empty() {
            summary.push_str("People:\n");
            for person in &people {
                summary.push_str(&format!("- {}", person.name));
                
                // Add relationships
                let rels = self.db.get_relationships_from(person.id)?;
                if !rels.is_empty() {
                    let rel_strs: Vec<String> = rels.iter()
                        .map(|(_, rel_type, target)| format!("{} {}", rel_type.to_lowercase().replace('_', " "), target))
                        .collect();
                    summary.push_str(&format!(" ({})", rel_strs.join(", ")));
                }
                
                // Add properties
                if !person.properties.is_empty() {
                    for (key, value) in &person.properties {
                        summary.push_str(&format!(" [{}={}]", key, value));
                    }
                }
                summary.push('\n');
            }
        }

        if !orgs.is_empty() {
            summary.push_str("\nOrganizations:\n");
            for org in &orgs {
                summary.push_str(&format!("- {}\n", org.name));
            }
        }

        if !projects.is_empty() {
            summary.push_str("\nProjects:\n");
            for project in &projects {
                summary.push_str(&format!("- {}", project.name));
                if !project.properties.is_empty() {
                    for (key, value) in &project.properties {
                        summary.push_str(&format!(" [{}={}]", key, value));
                    }
                }
                summary.push('\n');
            }
        }

        if !locations.is_empty() {
            summary.push_str("\nLocations:\n");
            for location in &locations {
                summary.push_str(&format!("- {}", location.name));
                if !location.properties.is_empty() {
                    for (key, value) in &location.properties {
                        summary.push_str(&format!(" [{}={}]", key, value));
                    }
                }
                summary.push('\n');
            }
        }

        if summary.is_empty() {
            summary = "No entities in knowledge graph yet.".to_string();
        }

        Ok(summary)
    }

    /// Get database reference (for advanced queries)
    pub fn database(&self) -> &Database {
        &self.db
    }

    /// Get all people who work at an organization
    pub fn get_employees(&self, org_name: &str) -> Result<Vec<Entity>> {
        let org = match self.find_organization(org_name)? {
            Some(o) => o,
            None => return Ok(vec![]),
        };
        
        // Get all relationships TO this org with type WORKS_AT
        let incoming = self.db.get_relationships_to(org.id)?;
        let mut employees = Vec::new();
        
        for (person_id, rel_type, person_name) in incoming {
            if rel_type == "WORKS_AT" {
                let mut entity = Entity::new(person_id, EntityType::Person, person_name);
                for (key, value) in self.db.get_entity_properties(person_id)? {
                    entity.properties.insert(key, value);
                }
                employees.push(entity);
            }
        }
        
        Ok(employees)
    }

    /// Get all people who live at/in a location
    pub fn get_residents(&self, location_name: &str) -> Result<Vec<Entity>> {
        let location = match self.find_location(location_name)? {
            Some(l) => l,
            None => return Ok(vec![]),
        };
        
        // Get all relationships TO this location with type LIVES_IN, LOCATED_IN, or BASED_IN
        let incoming = self.db.get_relationships_to(location.id)?;
        let mut residents = Vec::new();
        
        for (person_id, rel_type, person_name) in incoming {
            if rel_type == "LIVES_IN" || rel_type == "LOCATED_IN" || rel_type == "BASED_IN" {
                let mut entity = Entity::new(person_id, EntityType::Person, person_name);
                for (key, value) in self.db.get_entity_properties(person_id)? {
                    entity.properties.insert(key, value);
                }
                residents.push(entity);
            }
        }
        
        Ok(residents)
    }

    /// Get the organizations a person works at
    pub fn get_person_workplace(&self, person_name: &str) -> Result<Vec<Entity>> {
        let person = match self.find_person(person_name)? {
            Some(p) => p,
            None => return Ok(vec![]),
        };
        
        let outgoing = self.db.get_relationships_from(person.id)?;
        let mut workplaces = Vec::new();
        
        for (org_id, rel_type, org_name) in outgoing {
            if rel_type == "WORKS_AT" {
                let mut entity = Entity::new(org_id, EntityType::Organization, org_name);
                for (key, value) in self.db.get_entity_properties(org_id)? {
                    entity.properties.insert(key, value);
                }
                workplaces.push(entity);
            }
        }
        
        Ok(workplaces)
    }

    /// Get the locations a person lives in
    pub fn get_person_residence(&self, person_name: &str) -> Result<Vec<Entity>> {
        let person = match self.find_person(person_name)? {
            Some(p) => p,
            None => return Ok(vec![]),
        };
        
        let outgoing = self.db.get_relationships_from(person.id)?;
        let mut locations = Vec::new();
        
        for (loc_id, rel_type, loc_name) in outgoing {
            if rel_type == "LIVES_IN" || rel_type == "LOCATED_IN" || rel_type == "BASED_IN" {
                let mut entity = Entity::new(loc_id, EntityType::Location, loc_name);
                for (key, value) in self.db.get_entity_properties(loc_id)? {
                    entity.properties.insert(key, value);
                }
                locations.push(entity);
            }
        }
        
        Ok(locations)
    }

    /// Get all people connected to a person (via KNOWS relationship)
    /// Excludes the user themselves if they are the queried person
    pub fn get_connections(&self, person_name: &str) -> Result<Vec<Entity>> {
        let person = match self.find_person(person_name)? {
            Some(p) => p,
            None => return Ok(vec![]),
        };
        
        // Get user identity to filter out self-references
        let user_id = self.get_user_identity()?.map(|u| u.id);
        
        let outgoing = self.db.get_relationships_from(person.id)?;
        let incoming = self.db.get_relationships_to(person.id)?;
        
        let mut connections = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();
        
        // Skip the person themselves
        seen_ids.insert(person.id);
        
        // People this person knows
        for (target_id, rel_type, target_name) in outgoing {
            if rel_type == "KNOWS" && !seen_ids.contains(&target_id) {
                seen_ids.insert(target_id);
                let mut entity = Entity::new(target_id, EntityType::Person, target_name);
                for (key, value) in self.db.get_entity_properties(target_id)? {
                    entity.properties.insert(key, value);
                }
                connections.push(entity);
            }
        }
        
        // People who know this person
        for (source_id, rel_type, source_name) in incoming {
            if rel_type == "KNOWS" && !seen_ids.contains(&source_id) {
                seen_ids.insert(source_id);
                let mut entity = Entity::new(source_id, EntityType::Person, source_name);
                for (key, value) in self.db.get_entity_properties(source_id)? {
                    entity.properties.insert(key, value);
                }
                connections.push(entity);
            }
        }
        
        Ok(connections)
    }

    /// Get all people working on a project
    pub fn get_project_members(&self, project_name: &str) -> Result<Vec<Entity>> {
        let project = match self.find_project(project_name)? {
            Some(p) => p,
            None => return Ok(vec![]),
        };
        
        let incoming = self.db.get_relationships_to(project.id)?;
        let mut members = Vec::new();
        
        for (person_id, rel_type, person_name) in incoming {
            if rel_type == "WORKS_ON" {
                let mut entity = Entity::new(person_id, EntityType::Person, person_name);
                for (key, value) in self.db.get_entity_properties(person_id)? {
                    entity.properties.insert(key, value);
                }
                members.push(entity);
            }
        }
        
        Ok(members)
    }

    /// Get all projects a person works on
    pub fn get_person_projects(&self, person_name: &str) -> Result<Vec<Entity>> {
        let person = match self.find_person(person_name)? {
            Some(p) => p,
            None => return Ok(vec![]),
        };
        
        let outgoing = self.db.get_relationships_from(person.id)?;
        let mut projects = Vec::new();
        
        for (project_id, rel_type, project_name) in outgoing {
            if rel_type == "WORKS_ON" {
                let mut entity = Entity::new(project_id, EntityType::Project, project_name);
                for (key, value) in self.db.get_entity_properties(project_id)? {
                    entity.properties.insert(key, value);
                }
                projects.push(entity);
            }
        }
        
        Ok(projects)
    }

    /// Get all entities at a location
    pub fn get_entities_at_location(&self, location_name: &str) -> Result<Vec<Entity>> {
        let location = match self.find_location(location_name)? {
            Some(l) => l,
            None => return Ok(vec![]),
        };
        
        let incoming = self.db.get_relationships_to(location.id)?;
        let mut entities = Vec::new();
        
        for (entity_id, rel_type, entity_name) in incoming {
            if rel_type == "LOCATED_IN" {
                // Determine type from database
                if let Some((_, type_str)) = self.db.find_entity_by_name(&entity_name)? {
                    let entity_type = EntityType::from_str(&type_str);
                    let mut entity = Entity::new(entity_id, entity_type, entity_name);
                    for (key, value) in self.db.get_entity_properties(entity_id)? {
                        entity.properties.insert(key, value);
                    }
                    entities.push(entity);
                }
            }
        }
        
        Ok(entities)
    }

    /// Get the location of a person or organization
    pub fn get_location(&self, entity_name: &str) -> Result<Option<Entity>> {
        let entity = match self.find_by_name(entity_name)? {
            Some(e) => e,
            None => return Ok(None),
        };
        
        let outgoing = self.db.get_relationships_from(entity.id)?;
        
        for (location_id, rel_type, location_name) in outgoing {
            if rel_type == "LOCATED_IN" {
                let mut loc = Entity::new(location_id, EntityType::Location, location_name);
                for (key, value) in self.db.get_entity_properties(location_id)? {
                    loc.properties.insert(key, value);
                }
                return Ok(Some(loc));
            }
        }
        
        Ok(None)
    }

    /// Get all people managed by someone
    pub fn get_subordinates(&self, manager_name: &str) -> Result<Vec<Entity>> {
        let manager = match self.find_person(manager_name)? {
            Some(m) => m,
            None => return Ok(vec![]),
        };
        
        let outgoing = self.db.get_relationships_from(manager.id)?;
        let mut subordinates = Vec::new();
        
        for (person_id, rel_type, person_name) in outgoing {
            if rel_type == "MANAGES" {
                let mut entity = Entity::new(person_id, EntityType::Person, person_name);
                for (key, value) in self.db.get_entity_properties(person_id)? {
                    entity.properties.insert(key, value);
                }
                subordinates.push(entity);
            }
        }
        
        Ok(subordinates)
    }

    /// Get the manager of a person
    pub fn get_manager(&self, person_name: &str) -> Result<Option<Entity>> {
        let person = match self.find_person(person_name)? {
            Some(p) => p,
            None => return Ok(None),
        };
        
        let outgoing = self.db.get_relationships_from(person.id)?;
        
        for (manager_id, rel_type, manager_name) in outgoing {
            if rel_type == "REPORTS_TO" {
                let mut manager = Entity::new(manager_id, EntityType::Person, manager_name);
                for (key, value) in self.db.get_entity_properties(manager_id)? {
                    manager.properties.insert(key, value);
                }
                return Ok(Some(manager));
            }
        }
        
        Ok(None)
    }

    /// Get all collaborators of a person
    pub fn get_collaborators(&self, person_name: &str) -> Result<Vec<Entity>> {
        let person = match self.find_person(person_name)? {
            Some(p) => p,
            None => return Ok(vec![]),
        };
        
        let outgoing = self.db.get_relationships_from(person.id)?;
        let incoming = self.db.get_relationships_to(person.id)?;
        
        let mut collaborators = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();
        
        for (target_id, rel_type, target_name) in outgoing {
            if rel_type == "COLLABORATES_WITH" && !seen_ids.contains(&target_id) {
                seen_ids.insert(target_id);
                let mut entity = Entity::new(target_id, EntityType::Person, target_name);
                for (key, value) in self.db.get_entity_properties(target_id)? {
                    entity.properties.insert(key, value);
                }
                collaborators.push(entity);
            }
        }
        
        for (source_id, rel_type, source_name) in incoming {
            if rel_type == "COLLABORATES_WITH" && !seen_ids.contains(&source_id) {
                seen_ids.insert(source_id);
                let mut entity = Entity::new(source_id, EntityType::Person, source_name);
                for (key, value) in self.db.get_entity_properties(source_id)? {
                    entity.properties.insert(key, value);
                }
                collaborators.push(entity);
            }
        }
        
        Ok(collaborators)
    }

    /// Get properties for an entity as a HashMap
    pub fn get_entity_properties(&self, entity_id: i64) -> Result<std::collections::HashMap<String, String>> {
        let props = self.db.get_entity_properties(entity_id)?;
        Ok(props.into_iter().collect())
    }

    /// Get all relationships for an entity (formatted as tuples)
    pub fn get_entity_relationships(&self, entity_id: i64) -> Result<Vec<(String, String)>> {
        let outgoing = self.db.get_relationships_from(entity_id)?;
        let mut relationships = Vec::new();
        
        for (_, rel_type, target_name) in outgoing {
            relationships.push((rel_type, target_name));
        }
        
        Ok(relationships)
    }

    /// Get all organizations a person works at
    pub fn get_person_orgs(&self, person_name: &str) -> Result<Vec<Entity>> {
        let person = match self.find_person(person_name)? {
            Some(p) => p,
            None => return Ok(vec![]),
        };
        
        let outgoing = self.db.get_relationships_from(person.id)?;
        let mut orgs = Vec::new();
        
        for (org_id, rel_type, org_name) in outgoing {
            if rel_type == "WORKS_AT" {
                let mut entity = Entity::new(org_id, EntityType::Organization, org_name);
                for (key, value) in self.db.get_entity_properties(org_id)? {
                    entity.properties.insert(key, value);
                }
                orgs.push(entity);
            }
        }
        
        Ok(orgs)
    }

    /// List all entities of a given type
    pub fn list_entities_by_type(&self, entity_type: &str) -> Result<Vec<Entity>> {
        let all = self.db.list_all_entities()?;
        let mut entities = Vec::new();
        
        for (id, name, type_str) in all {
            if type_str.to_lowercase() == entity_type.to_lowercase() {
                let e_type = EntityType::from_str(&type_str);
                let mut entity = Entity::new(id, e_type, name);
                for (key, value) in self.db.get_entity_properties(id)? {
                    entity.properties.insert(key, value);
                }
                entities.push(entity);
            }
        }
        
        Ok(entities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_graph() -> KnowledgeGraph {
        let db = Database::in_memory().unwrap();
        KnowledgeGraph::new(db)
    }

    #[test]
    fn test_add_and_find_person() {
        let graph = test_graph();
        
        let person = graph.add_person("John Doe").unwrap();
        assert_eq!(person.name, "John Doe");
        
        let found = graph.find_person("John Doe").unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "John Doe");
    }

    #[test]
    fn test_works_at_relationship() {
        let graph = test_graph();
        
        let person = graph.add_person("Alice").unwrap();
        let org = graph.add_organization("TechCorp").unwrap();
        
        graph.link_works_at(person.id, org.id).unwrap();
        
        let info = graph.get_entity_info(person.id).unwrap();
        assert!(info.contains("WORKS_AT"));
        assert!(info.contains("TechCorp"));
    }

    #[test]
    fn test_summary() {
        let graph = test_graph();
        
        let person = graph.add_person("Bob").unwrap();
        let org = graph.add_organization("Acme Inc").unwrap();
        graph.link_works_at(person.id, org.id).unwrap();
        
        let summary = graph.get_summary().unwrap();
        assert!(summary.contains("Bob"));
        assert!(summary.contains("Acme Inc"));
    }

    #[test]
    fn test_add_and_find_project() {
        let graph = test_graph();
        
        let project = graph.add_project("Dosa AI").unwrap();
        assert_eq!(project.name, "Dosa AI");
        
        let found = graph.find_project("Dosa AI").unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "Dosa AI");
    }

    #[test]
    fn test_add_and_find_location() {
        let graph = test_graph();
        
        let location = graph.add_location("New York").unwrap();
        assert_eq!(location.name, "New York");
        
        let found = graph.find_location("New York").unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "New York");
    }

    #[test]
    fn test_works_on_relationship() {
        let graph = test_graph();
        
        let person = graph.add_person("Alice").unwrap();
        let project = graph.add_project("Project X").unwrap();
        
        graph.link_works_on(person.id, project.id).unwrap();
        
        let info = graph.get_entity_info(person.id).unwrap();
        assert!(info.contains("WORKS_ON"));
        assert!(info.contains("Project X"));
        
        let members = graph.get_project_members("Project X").unwrap();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].name, "Alice");
    }

    #[test]
    fn test_located_in_relationship() {
        let graph = test_graph();
        
        let person = graph.add_person("Bob").unwrap();
        let location = graph.add_location("San Francisco").unwrap();
        
        graph.link_located_in(person.id, location.id).unwrap();
        
        let loc = graph.get_location("Bob").unwrap();
        assert!(loc.is_some());
        assert_eq!(loc.unwrap().name, "San Francisco");
        
        let entities = graph.get_entities_at_location("San Francisco").unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "Bob");
    }

    #[test]
    fn test_manages_relationship() {
        let graph = test_graph();
        
        let manager = graph.add_person("Alice").unwrap();
        let employee = graph.add_person("Bob").unwrap();
        
        graph.link_manages(manager.id, employee.id).unwrap();
        graph.link_reports_to(employee.id, manager.id).unwrap();
        
        let subordinates = graph.get_subordinates("Alice").unwrap();
        assert_eq!(subordinates.len(), 1);
        assert_eq!(subordinates[0].name, "Bob");
        
        let mgr = graph.get_manager("Bob").unwrap();
        assert!(mgr.is_some());
        assert_eq!(mgr.unwrap().name, "Alice");
    }
}
