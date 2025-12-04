use anyhow::Result;
use chrono::{DateTime, Local, Duration};
use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::storage::Database;

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Urgent,
}

impl TaskPriority {
    pub fn as_str(&self) -> &str {
        match self {
            TaskPriority::Low => "low",
            TaskPriority::Medium => "medium",
            TaskPriority::High => "high",
            TaskPriority::Urgent => "urgent",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "low" | "l" => TaskPriority::Low,
            "high" | "h" => TaskPriority::High,
            "urgent" | "u" | "!" => TaskPriority::Urgent,
            _ => TaskPriority::Medium,
        }
    }

    pub fn emoji(&self) -> &str {
        match self {
            TaskPriority::Low => "🔵",
            TaskPriority::Medium => "🟡",
            TaskPriority::High => "🟠",
            TaskPriority::Urgent => "🔴",
        }
    }
}

/// Task status
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Done,
    Cancelled,
}

impl TaskStatus {
    pub fn as_str(&self) -> &str {
        match self {
            TaskStatus::Pending => "pending",
            TaskStatus::InProgress => "in_progress",
            TaskStatus::Done => "done",
            TaskStatus::Cancelled => "cancelled",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "in_progress" | "inprogress" | "working" => TaskStatus::InProgress,
            "done" | "complete" | "completed" => TaskStatus::Done,
            "cancelled" | "canceled" => TaskStatus::Cancelled,
            _ => TaskStatus::Pending,
        }
    }

    pub fn emoji(&self) -> &str {
        match self {
            TaskStatus::Pending => "⬜",
            TaskStatus::InProgress => "🔄",
            TaskStatus::Done => "✅",
            TaskStatus::Cancelled => "❌",
        }
    }
}

/// A task/reminder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: i64,
    pub title: String,
    pub description: Option<String>,
    pub deadline: Option<DateTime<Local>>,
    pub priority: TaskPriority,
    pub status: TaskStatus,
    pub assigned_to: Option<String>, // Person name (can link to KG)
    pub created_at: DateTime<Local>,
}

impl Task {
    /// Format task for display
    pub fn display(&self) -> String {
        let mut output = format!(
            "{} {} {}",
            self.status.emoji(),
            self.priority.emoji(),
            self.title
        );

        if let Some(deadline) = &self.deadline {
            let now = Local::now();
            let is_overdue = *deadline < now && self.status != TaskStatus::Done;
            
            if is_overdue {
                output.push_str(&format!(" ⚠️ OVERDUE (was {})", deadline.format("%b %d")));
            } else {
                output.push_str(&format!(" 📅 {}", deadline.format("%b %d, %I:%M %p")));
            }
        }

        if let Some(assigned) = &self.assigned_to {
            output.push_str(&format!(" 👤 {}", assigned));
        }

        if let Some(desc) = &self.description {
            output.push_str(&format!("\n   {}", desc));
        }

        output
    }

    /// Short display for lists
    pub fn display_short(&self) -> String {
        let deadline_str = self.deadline
            .map(|d| format!(" ({})", d.format("%b %d")))
            .unwrap_or_default();
        
        format!(
            "{} {} {}{}",
            self.status.emoji(),
            self.priority.emoji(),
            self.title,
            deadline_str
        )
    }

    /// Check if task is overdue
    pub fn is_overdue(&self) -> bool {
        if self.status == TaskStatus::Done || self.status == TaskStatus::Cancelled {
            return false;
        }
        self.deadline.map(|d| d < Local::now()).unwrap_or(false)
    }
}

/// Reminders/Tasks manager
pub struct Reminders<'a> {
    db: &'a Database,
}

impl<'a> Reminders<'a> {
    pub fn new(db: &'a Database) -> Self {
        Reminders { db }
    }

    /// Initialize reminders tables
    pub fn init_tables(db: &Database) -> Result<()> {
        db.connection().execute(
            "CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                deadline TEXT,
                priority TEXT NOT NULL DEFAULT 'medium',
                status TEXT NOT NULL DEFAULT 'pending',
                assigned_to TEXT,
                assigned_entity_id INTEGER,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )",
            [],
        )?;

        db.connection().execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
            [],
        )?;

        db.connection().execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_deadline ON tasks(deadline)",
            [],
        )?;

        Ok(())
    }

    /// Add a new task
    pub fn add_task(
        &self,
        title: &str,
        description: Option<&str>,
        deadline: Option<DateTime<Local>>,
        priority: TaskPriority,
    ) -> Result<Task> {
        let now = Local::now();
        
        self.db.connection().execute(
            "INSERT INTO tasks (title, description, deadline, priority, status, created_at) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                title,
                description,
                deadline.map(|d| d.to_rfc3339()),
                priority.as_str(),
                TaskStatus::Pending.as_str(),
                now.to_rfc3339()
            ],
        )?;

        let id = self.db.connection().last_insert_rowid();

        Ok(Task {
            id,
            title: title.to_string(),
            description: description.map(|s| s.to_string()),
            deadline,
            priority,
            status: TaskStatus::Pending,
            assigned_to: None,
            created_at: now,
        })
    }

    /// Quick reminder - add a task with just title and optional deadline
    pub fn remind(&self, title: &str, deadline: Option<DateTime<Local>>) -> Result<Task> {
        self.add_task(title, None, deadline, TaskPriority::Medium)
    }

    /// Mark a task as done
    pub fn complete_task(&self, task_id: i64) -> Result<bool> {
        let now = Local::now().to_rfc3339();
        let rows = self.db.connection().execute(
            "UPDATE tasks SET status = 'done', completed_at = ?1 WHERE id = ?2",
            params![now, task_id],
        )?;
        Ok(rows > 0)
    }

    /// Update task status
    pub fn update_status(&self, task_id: i64, status: TaskStatus) -> Result<bool> {
        let completed_at = if status == TaskStatus::Done {
            Some(Local::now().to_rfc3339())
        } else {
            None
        };

        let rows = self.db.connection().execute(
            "UPDATE tasks SET status = ?1, completed_at = ?2 WHERE id = ?3",
            params![status.as_str(), completed_at, task_id],
        )?;
        Ok(rows > 0)
    }

    /// Get all pending tasks
    pub fn get_pending_tasks(&self) -> Result<Vec<Task>> {
        self.get_tasks_by_status(TaskStatus::Pending)
    }

    /// Get tasks by status
    fn get_tasks_by_status(&self, status: TaskStatus) -> Result<Vec<Task>> {
        let mut stmt = self.db.connection().prepare(
            "SELECT id, title, description, deadline, priority, status, assigned_to, created_at 
             FROM tasks 
             WHERE status = ?1
             ORDER BY 
                CASE priority 
                    WHEN 'urgent' THEN 1 
                    WHEN 'high' THEN 2 
                    WHEN 'medium' THEN 3 
                    ELSE 4 
                END,
                deadline NULLS LAST"
        )?;

        self.collect_tasks(&mut stmt, params![status.as_str()])
    }

    /// Get all active tasks (pending or in progress)
    pub fn get_active_tasks(&self) -> Result<Vec<Task>> {
        let mut stmt = self.db.connection().prepare(
            "SELECT id, title, description, deadline, priority, status, assigned_to, created_at 
             FROM tasks 
             WHERE status IN ('pending', 'in_progress')
             ORDER BY 
                CASE priority 
                    WHEN 'urgent' THEN 1 
                    WHEN 'high' THEN 2 
                    WHEN 'medium' THEN 3 
                    ELSE 4 
                END,
                deadline NULLS LAST"
        )?;

        self.collect_tasks(&mut stmt, params![])
    }

    /// Get overdue tasks
    pub fn get_overdue_tasks(&self) -> Result<Vec<Task>> {
        let now = Local::now().to_rfc3339();
        let mut stmt = self.db.connection().prepare(
            "SELECT id, title, description, deadline, priority, status, assigned_to, created_at 
             FROM tasks 
             WHERE status IN ('pending', 'in_progress') AND deadline < ?1
             ORDER BY deadline"
        )?;

        self.collect_tasks(&mut stmt, params![now])
    }

    /// Get tasks due soon (within next 24 hours)
    pub fn get_tasks_due_soon(&self) -> Result<Vec<Task>> {
        let now = Local::now();
        let soon = now + Duration::hours(24);

        let mut stmt = self.db.connection().prepare(
            "SELECT id, title, description, deadline, priority, status, assigned_to, created_at 
             FROM tasks 
             WHERE status IN ('pending', 'in_progress') 
                AND deadline >= ?1 AND deadline <= ?2
             ORDER BY deadline"
        )?;

        self.collect_tasks(&mut stmt, params![now.to_rfc3339(), soon.to_rfc3339()])
    }

    /// Find task by title (partial match)
    pub fn find_task_by_title(&self, title: &str) -> Result<Option<Task>> {
        let pattern = format!("%{}%", title);
        let mut stmt = self.db.connection().prepare(
            "SELECT id, title, description, deadline, priority, status, assigned_to, created_at 
             FROM tasks 
             WHERE LOWER(title) LIKE LOWER(?1)
             ORDER BY created_at DESC
             LIMIT 1"
        )?;

        let tasks = self.collect_tasks(&mut stmt, params![pattern])?;
        Ok(tasks.into_iter().next())
    }

    /// Delete a task
    pub fn delete_task(&self, task_id: i64) -> Result<bool> {
        let rows = self.db.connection().execute(
            "DELETE FROM tasks WHERE id = ?1",
            params![task_id],
        )?;
        Ok(rows > 0)
    }

    /// Collect tasks from a prepared statement
    fn collect_tasks(&self, stmt: &mut rusqlite::Statement, params: impl rusqlite::Params) -> Result<Vec<Task>> {
        let rows = stmt.query_map(params, |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, Option<String>>(6)?,
                row.get::<_, String>(7)?,
            ))
        })?;

        let mut tasks = Vec::new();
        for row in rows {
            let (id, title, description, deadline_str, priority_str, status_str, assigned_to, created_str) = row?;

            let deadline = deadline_str.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .map(|dt| dt.with_timezone(&Local))
                    .ok()
            });

            let created_at = DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Local))
                .unwrap_or_else(|_| Local::now());

            tasks.push(Task {
                id,
                title,
                description,
                deadline,
                priority: TaskPriority::from_str(&priority_str),
                status: TaskStatus::from_str(&status_str),
                assigned_to,
                created_at,
            });
        }

        Ok(tasks)
    }

    /// Get a summary for LLM context
    pub fn get_summary(&self) -> Result<String> {
        let overdue = self.get_overdue_tasks()?;
        let due_soon = self.get_tasks_due_soon()?;
        let pending = self.get_pending_tasks()?;

        let mut summary = String::new();

        if !overdue.is_empty() {
            summary.push_str(&format!("⚠️ {} OVERDUE task(s):\n", overdue.len()));
            for task in &overdue {
                summary.push_str(&format!("- {}\n", task.title));
            }
        }

        if !due_soon.is_empty() {
            if !summary.is_empty() { summary.push('\n'); }
            summary.push_str("Due in next 24 hours:\n");
            for task in &due_soon {
                summary.push_str(&format!("- {} ({})\n", 
                    task.title,
                    task.deadline.map(|d| d.format("%I:%M %p").to_string()).unwrap_or_default()
                ));
            }
        }

        if !pending.is_empty() && pending.len() <= 10 {
            if !summary.is_empty() { summary.push('\n'); }
            summary.push_str("Pending tasks:\n");
            for task in pending.iter().take(5) {
                summary.push_str(&format!("- {} {}\n", task.priority.emoji(), task.title));
            }
            if pending.len() > 5 {
                summary.push_str(&format!("  ...and {} more\n", pending.len() - 5));
            }
        }

        if summary.is_empty() {
            summary = "No pending tasks.".to_string();
        }

        Ok(summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_from_str() {
        assert_eq!(TaskPriority::from_str("high"), TaskPriority::High);
        assert_eq!(TaskPriority::from_str("H"), TaskPriority::High);
        assert_eq!(TaskPriority::from_str("!"), TaskPriority::Urgent);
    }

    #[test]
    fn test_status_from_str() {
        assert_eq!(TaskStatus::from_str("done"), TaskStatus::Done);
        assert_eq!(TaskStatus::from_str("completed"), TaskStatus::Done);
    }
}
