//! Task-Knowledge Graph Synchronization
//!
//! This module provides bidirectional sync between the tasks table
//! and the knowledge graph, enabling:
//! - Automatic task entity creation when tasks are added
//! - Person-task relationships (ASSIGNED_TO, HAS_TASK)
//! - Task dependencies (BLOCKED_BY, DEPENDS_ON)
//! - Temporal reasoning via deadline entities

use anyhow::Result;
use chrono::{DateTime, Local};

use crate::knowledge::KnowledgeGraph;
use crate::knowledge::entities::RelationshipType;
use crate::storage::Database;
use super::tasks::{Task, Reminders};

/// Synchronizes tasks with the knowledge graph
pub struct TaskGraphSync<'a> {
    graph: &'a KnowledgeGraph,
    reminders: Reminders<'a>,
}

impl<'a> TaskGraphSync<'a> {
    pub fn new(graph: &'a KnowledgeGraph, db: &'a Database) -> Self {
        TaskGraphSync {
            graph,
            reminders: Reminders::new(db),
        }
    }

    /// Sync all existing tasks to the knowledge graph
    pub fn sync_all_tasks(&self) -> Result<SyncResult> {
        let tasks = self.reminders.get_active_tasks()?;
        let mut result = SyncResult::default();

        for task in tasks {
            match self.sync_task(&task) {
                Ok(synced) => {
                    if synced {
                        result.synced += 1;
                    } else {
                        result.skipped += 1;
                    }
                }
                Err(e) => {
                    result.errors.push(format!("Task '{}': {}", task.title, e));
                }
            }
        }

        Ok(result)
    }

    /// Sync a single task to the knowledge graph
    pub fn sync_task(&self, task: &Task) -> Result<bool> {
        // Check if task entity already exists
        if self.graph.find_task(&task.title)?.is_some() {
            return Ok(false); // Already synced
        }

        // Create task entity in graph
        let task_entity = self.graph.add_task_entity(&task.title)?;

        // Set task properties
        self.graph.set_property(task_entity.id, "status", task.status.as_str())?;
        self.graph.set_property(task_entity.id, "priority", task.priority.as_str())?;
        self.graph.set_property(task_entity.id, "task_id", &task.id.to_string())?;

        // Link to assigned person if present
        if let Some(ref assigned) = task.assigned_to {
            // Find or create person
            let person = if let Some(p) = self.graph.find_person(assigned)? {
                p
            } else {
                self.graph.add_person(assigned)?
            };

            // Create both relationships
            self.graph.assign_task(task_entity.id, person.id)?;
            self.graph.link_has_task(person.id, task_entity.id)?;
        }

        // Set deadline if present
        if let Some(deadline) = task.deadline {
            let deadline_str = deadline.format("%Y-%m-%dT%H:%M:%S").to_string();
            self.graph.set_task_deadline(task_entity.id, &deadline_str)?;
        }

        Ok(true)
    }

    /// Create a task with graph sync
    pub fn create_task_with_sync(
        &self,
        title: &str,
        description: Option<&str>,
        deadline: Option<DateTime<Local>>,
        priority: super::tasks::TaskPriority,
        assigned_to: Option<&str>,
    ) -> Result<Task> {
        // Create task in reminders table
        let mut task = self.reminders.add_task(title, description, deadline, priority)?;
        
        // Set assigned_to in the task struct (we'll sync this to the graph)
        task.assigned_to = assigned_to.map(|s| s.to_string());
        
        // Sync to graph
        self.sync_task(&task)?;
        
        Ok(task)
    }

    /// Link a task to a project
    pub fn link_task_to_project(&self, task_title: &str, project_name: &str) -> Result<bool> {
        let task = match self.graph.find_task(task_title)? {
            Some(t) => t,
            None => return Ok(false),
        };

        let project = if let Some(p) = self.graph.find_project(project_name)? {
            p
        } else {
            self.graph.add_project(project_name)?
        };

        self.graph.create_relationship(task.id, project.id, RelationshipType::RelatedTo)?;
        Ok(true)
    }

    /// Mark a task as blocked by another task
    pub fn block_task(&self, blocked_task: &str, blocking_task: &str) -> Result<bool> {
        let blocked = match self.graph.find_task(blocked_task)? {
            Some(t) => t,
            None => return Ok(false),
        };

        let blocker = match self.graph.find_task(blocking_task)? {
            Some(t) => t,
            None => return Ok(false),
        };

        self.graph.link_blocked_by(blocked.id, blocker.id)?;
        Ok(true)
    }

    /// Get all tasks for a person (from graph)
    pub fn get_person_tasks(&self, person_name: &str) -> Result<Vec<String>> {
        let tasks = self.graph.get_person_tasks(person_name)?;
        Ok(tasks.into_iter().map(|t| t.name).collect())
    }

    /// Get tasks blocking other tasks
    pub fn get_blocking_chain(&self, task_name: &str) -> Result<Vec<String>> {
        let mut chain = Vec::new();
        let mut current = task_name.to_string();
        let mut visited = std::collections::HashSet::new();

        while !visited.contains(&current) {
            visited.insert(current.clone());
            
            let blockers = self.graph.get_blocking_tasks(&current)?;
            if blockers.is_empty() {
                break;
            }

            for blocker in blockers {
                chain.push(blocker.name.clone());
                current = blocker.name;
                break; // Follow first blocker for now
            }
        }

        Ok(chain)
    }

    /// Get tasks that would be unblocked by completing a task
    pub fn get_unblocked_by_completing(&self, task_name: &str) -> Result<Vec<String>> {
        let blocked = self.graph.get_blocked_tasks(task_name)?;
        Ok(blocked.into_iter().map(|t| t.name).collect())
    }

    /// Calculate workload for a person (number of active tasks)
    pub fn get_person_workload(&self, person_name: &str) -> Result<WorkloadInfo> {
        let tasks = self.graph.get_person_tasks(person_name)?;
        
        let mut workload = WorkloadInfo {
            person: person_name.to_string(),
            total_tasks: tasks.len(),
            pending: 0,
            in_progress: 0,
            blocked: 0,
            overdue: 0,
        };

        for task in &tasks {
            let status = task.properties.get("status").map(|s| s.as_str()).unwrap_or("pending");
            match status {
                "pending" => workload.pending += 1,
                "in_progress" => workload.in_progress += 1,
                _ => {}
            }

            // Check if blocked
            let blockers = self.graph.get_blocking_tasks(&task.name)?;
            if !blockers.is_empty() {
                workload.blocked += 1;
            }

            // Check if overdue
            if let Some(deadline) = task.properties.get("deadline") {
                if let Ok(dt) = deadline.parse::<DateTime<Local>>() {
                    if dt < Local::now() && status != "done" {
                        workload.overdue += 1;
                    }
                }
            }
        }

        Ok(workload)
    }
}

/// Result of a sync operation
#[derive(Debug, Default)]
pub struct SyncResult {
    pub synced: usize,
    pub skipped: usize,
    pub errors: Vec<String>,
}

impl SyncResult {
    pub fn summary(&self) -> String {
        let mut msg = format!("Synced: {}, Skipped: {}", self.synced, self.skipped);
        if !self.errors.is_empty() {
            msg.push_str(&format!(", Errors: {}", self.errors.len()));
        }
        msg
    }
}

/// Workload information for a person
#[derive(Debug)]
pub struct WorkloadInfo {
    pub person: String,
    pub total_tasks: usize,
    pub pending: usize,
    pub in_progress: usize,
    pub blocked: usize,
    pub overdue: usize,
}

impl WorkloadInfo {
    pub fn display(&self) -> String {
        format!(
            "📊 Workload for {}:\n  Total: {} tasks\n  ⬜ Pending: {}\n  🔄 In Progress: {}\n  🚫 Blocked: {}\n  ⚠️ Overdue: {}",
            self.person,
            self.total_tasks,
            self.pending,
            self.in_progress,
            self.blocked,
            self.overdue
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Database;
    use crate::reminders::TaskPriority;

    fn setup() -> (Database, KnowledgeGraph) {
        let db = Database::in_memory().unwrap();
        Reminders::init_tables(&db).unwrap();
        let graph = KnowledgeGraph::new(Database::in_memory().unwrap());
        (db, graph)
    }

    #[test]
    fn test_sync_task_to_graph() {
        let (db, graph) = setup();
        let reminders = Reminders::new(&db);
        
        // Create a task
        let mut task = reminders.add_task(
            "Test Task",
            Some("A test"),
            None,
            TaskPriority::High,
        ).unwrap();
        task.assigned_to = Some("Alice".to_string());

        // Sync
        let sync = TaskGraphSync::new(&graph, &db);
        let synced = sync.sync_task(&task).unwrap();
        assert!(synced);

        // Verify in graph
        let found = graph.find_task("Test Task").unwrap();
        assert!(found.is_some());
    }

    #[test]
    fn test_task_assignment() {
        let (db, graph) = setup();
        let reminders = Reminders::new(&db);
        
        // Add person first
        graph.add_person("Bob").unwrap();
        
        // Create and sync task
        let mut task = reminders.add_task("Bob's Task", None, None, TaskPriority::Medium).unwrap();
        task.assigned_to = Some("Bob".to_string());
        
        let sync = TaskGraphSync::new(&graph, &db);
        sync.sync_task(&task).unwrap();

        // Check person's tasks
        let tasks = sync.get_person_tasks("Bob").unwrap();
        assert!(tasks.contains(&"Bob's Task".to_string()));
    }

    #[test]
    fn test_task_blocking() {
        let (db, graph) = setup();
        
        // Create tasks in graph
        graph.add_task_entity("Task A").unwrap();
        graph.add_task_entity("Task B").unwrap();

        let sync = TaskGraphSync::new(&graph, &db);
        
        // Block Task B with Task A
        sync.block_task("Task B", "Task A").unwrap();

        // Verify
        let blockers = graph.get_blocking_tasks("Task B").unwrap();
        assert_eq!(blockers.len(), 1);
        assert_eq!(blockers[0].name, "Task A");
    }
}
