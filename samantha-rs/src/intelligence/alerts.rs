//! Proactive alert system for reminders and upcoming events
//! 
//! This module checks for:
//! - Overdue tasks
//! - Tasks due soon (within 24 hours)
//! - Upcoming events today
//! - Events starting soon (within 1 hour)

use anyhow::Result;
use chrono::{DateTime, Local, Duration, Timelike};

use crate::storage::Database;
use crate::calendar::Calendar;
use crate::reminders::Reminders;

/// Alert types for different urgency levels
#[derive(Debug, Clone)]
pub enum AlertLevel {
    Urgent,    // Overdue or happening now
    Warning,   // Due/happening very soon
    Info,      // Upcoming reminder
}

/// A single alert item
#[derive(Debug, Clone)]
pub struct Alert {
    pub level: AlertLevel,
    pub title: String,
    pub message: String,
    pub timestamp: Option<DateTime<Local>>,
}

impl Alert {
    pub fn urgent(title: &str, message: &str) -> Self {
        Self {
            level: AlertLevel::Urgent,
            title: title.to_string(),
            message: message.to_string(),
            timestamp: Some(Local::now()),
        }
    }

    pub fn warning(title: &str, message: &str) -> Self {
        Self {
            level: AlertLevel::Warning,
            title: title.to_string(),
            message: message.to_string(),
            timestamp: Some(Local::now()),
        }
    }

    pub fn info(title: &str, message: &str) -> Self {
        Self {
            level: AlertLevel::Info,
            title: title.to_string(),
            message: message.to_string(),
            timestamp: Some(Local::now()),
        }
    }

    /// Format alert for display
    pub fn display(&self) -> String {
        let icon = match self.level {
            AlertLevel::Urgent => "🚨",
            AlertLevel::Warning => "⚠️",
            AlertLevel::Info => "ℹ️",
        };
        format!("{} {}: {}", icon, self.title, self.message)
    }
}

/// The alert system that checks for proactive notifications
pub struct AlertSystem<'a> {
    db: &'a Database,
}

impl<'a> AlertSystem<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Check all alert conditions and return any active alerts
    pub fn check_alerts(&self) -> Result<Vec<Alert>> {
        let mut alerts = Vec::new();

        // Check task-related alerts
        self.check_task_alerts(&mut alerts)?;
        
        // Check event-related alerts
        self.check_event_alerts(&mut alerts)?;

        // Sort by urgency (urgent first)
        alerts.sort_by(|a, b| {
            let priority_a = match a.level {
                AlertLevel::Urgent => 0,
                AlertLevel::Warning => 1,
                AlertLevel::Info => 2,
            };
            let priority_b = match b.level {
                AlertLevel::Urgent => 0,
                AlertLevel::Warning => 1,
                AlertLevel::Info => 2,
            };
            priority_a.cmp(&priority_b)
        });

        Ok(alerts)
    }

    /// Check for overdue and upcoming tasks
    fn check_task_alerts(&self, alerts: &mut Vec<Alert>) -> Result<()> {
        let reminders = Reminders::new(self.db);
        let now = Local::now();
        
        // Get overdue tasks
        let overdue = reminders.get_overdue_tasks()?;
        for task in overdue {
            let overdue_by = if let Some(deadline) = task.deadline {
                let duration = now.signed_duration_since(deadline);
                if duration.num_days() > 0 {
                    format!("{} days overdue", duration.num_days())
                } else if duration.num_hours() > 0 {
                    format!("{} hours overdue", duration.num_hours())
                } else {
                    "just overdue".to_string()
                }
            } else {
                "overdue".to_string()
            };
            
            alerts.push(Alert::urgent(
                "Overdue Task",
                &format!("'{}' is {} (Priority: {:?})", task.title, overdue_by, task.priority)
            ));
        }

        // Get tasks due soon (within 24 hours)
        let upcoming = reminders.get_tasks_due_soon()?;
        let in_24_hours = now + Duration::hours(24);
        
        for task in upcoming {
            if let Some(deadline) = task.deadline {
                // Skip if overdue (already handled above)
                if deadline <= now {
                    continue;
                }
                
                if deadline <= in_24_hours {
                    let time_left = deadline.signed_duration_since(now);
                    let time_str = if time_left.num_hours() > 0 {
                        format!("in {} hours", time_left.num_hours())
                    } else {
                        format!("in {} minutes", time_left.num_minutes())
                    };
                    
                    alerts.push(Alert::warning(
                        "Task Due Soon",
                        &format!("'{}' is due {}", task.title, time_str)
                    ));
                }
            }
        }

        Ok(())
    }

    /// Check for upcoming events
    fn check_event_alerts(&self, alerts: &mut Vec<Alert>) -> Result<()> {
        let calendar = Calendar::new(self.db);
        let now = Local::now();
        let in_1_hour = now + Duration::hours(1);
        
        // Get today's events
        let events = calendar.get_today_events()?;
        
        for event in events {
            // Event starting in the next hour
            if event.start_time > now && event.start_time <= in_1_hour {
                let time_left = event.start_time.signed_duration_since(now);
                let time_str = if time_left.num_minutes() > 0 {
                    format!("in {} minutes", time_left.num_minutes())
                } else {
                    "starting now".to_string()
                };
                
                let location = event.location
                    .map(|l| format!(" at {}", l))
                    .unwrap_or_default();
                
                alerts.push(Alert::warning(
                    "Upcoming Event",
                    &format!("'{}' starts {}{}", event.title, time_str, location)
                ));
            }
            // Event already started (in progress)
            else if let Some(end_time) = event.end_time {
                if event.start_time <= now && now < end_time {
                    alerts.push(Alert::info(
                        "In Progress",
                        &format!("'{}' is currently happening", event.title)
                    ));
                }
            }
        }

        Ok(())
    }

    /// Get a summary of today's agenda
    pub fn get_daily_briefing(&self) -> Result<String> {
        let calendar = Calendar::new(self.db);
        let reminders = Reminders::new(self.db);
        let now = Local::now();
        
        let events = calendar.get_today_events()?;
        let tasks_due = reminders.get_tasks_due_soon()?;
        let overdue = reminders.get_overdue_tasks()?;
        
        let mut briefing = format!(
            "📅 Good {}! Here's your briefing for {}:\n\n",
            get_time_of_day(),
            now.format("%A, %B %d")
        );
        
        // Overdue items first
        if !overdue.is_empty() {
            briefing.push_str(&format!("🚨 {} overdue task(s):\n", overdue.len()));
            for task in overdue.iter().take(3) {
                briefing.push_str(&format!("   • {}\n", task.title));
            }
            if overdue.len() > 3 {
                briefing.push_str(&format!("   ... and {} more\n", overdue.len() - 3));
            }
            briefing.push('\n');
        }
        
        // Today's events
        if events.is_empty() {
            briefing.push_str("📆 No events scheduled for today.\n");
        } else {
            briefing.push_str(&format!("📆 {} event(s) today:\n", events.len()));
            for event in events.iter().take(5) {
                let time = event.start_time.format("%I:%M %p");
                briefing.push_str(&format!("   • {} - {}\n", time, event.title));
            }
            if events.len() > 5 {
                briefing.push_str(&format!("   ... and {} more\n", events.len() - 5));
            }
        }
        briefing.push('\n');
        
        // Tasks due today
        let tasks_due_today: Vec<_> = tasks_due.iter()
            .filter(|t| {
                t.deadline.map(|d| d.date_naive() == now.date_naive()).unwrap_or(false)
            })
            .collect();
        
        if !tasks_due_today.is_empty() {
            briefing.push_str(&format!("✅ {} task(s) due today:\n", tasks_due_today.len()));
            for task in tasks_due_today.iter().take(5) {
                briefing.push_str(&format!("   • {}\n", task.title));
            }
        } else {
            briefing.push_str("✅ No tasks due today.\n");
        }
        
        Ok(briefing)
    }

    /// Display all alerts in a formatted way
    pub fn display_alerts(&self) -> Result<Option<String>> {
        let alerts = self.check_alerts()?;
        
        if alerts.is_empty() {
            return Ok(None);
        }

        let mut output = String::from("\n╔══════════════════════════════════════════════╗\n");
        output.push_str("║              📢 NOTIFICATIONS                 ║\n");
        output.push_str("╚══════════════════════════════════════════════╝\n\n");

        for alert in &alerts {
            output.push_str(&alert.display());
            output.push('\n');
        }

        Ok(Some(output))
    }
}

/// Get time of day greeting
fn get_time_of_day() -> &'static str {
    let hour = Local::now().hour();
    match hour {
        5..=11 => "morning",
        12..=16 => "afternoon",
        17..=20 => "evening",
        _ => "night",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_display() {
        let alert = Alert::urgent("Test", "This is urgent");
        assert!(alert.display().contains("🚨"));
        assert!(alert.display().contains("Test"));
        
        let warning = Alert::warning("Warning", "Be careful");
        assert!(warning.display().contains("⚠️"));
    }

    #[test]
    fn test_alert_levels() {
        let urgent = Alert::urgent("U", "msg");
        let warning = Alert::warning("W", "msg");
        let info = Alert::info("I", "msg");
        
        assert!(matches!(urgent.level, AlertLevel::Urgent));
        assert!(matches!(warning.level, AlertLevel::Warning));
        assert!(matches!(info.level, AlertLevel::Info));
    }
}
