//! Notification types and display formatting
//! 
//! Defines how proactive notifications are presented to the user.

use chrono::{DateTime, Local};

/// Type of notification
#[derive(Clone, Debug, PartialEq)]
pub enum NotificationType {
    /// Upcoming meeting reminder
    MeetingReminder {
        title: String,
        starts_in_minutes: u32,
        location: Option<String>,
        attendees: Vec<String>,
    },
    /// Morning daily brief
    MorningBrief {
        meetings_today: usize,
        unread_emails: usize,
        tasks_due: usize,
    },
    /// Suggestion to follow up on email
    EmailFollowUp {
        person: String,
        subject: String,
        days_since: u32,
    },
    /// Free time suggestion
    FreeTimeSuggestion {
        duration_hours: f32,
        suggested_tasks: Vec<String>,
    },
    /// Pattern-based reminder
    PatternReminder {
        pattern_name: String,
        suggestion: String,
    },
    /// Generic notification
    Generic {
        message: String,
    },
}

/// A notification to display to the user
#[derive(Clone, Debug)]
pub struct Notification {
    /// Unique identifier
    pub id: String,
    /// Type of notification
    pub notification_type: NotificationType,
    /// When the notification was created
    pub created_at: DateTime<Local>,
    /// Whether the notification has been seen
    pub seen: bool,
    /// Whether the notification has been dismissed
    pub dismissed: bool,
    /// Priority (higher = more important)
    pub priority: u8,
}

impl Notification {
    /// Create a new notification
    pub fn new(id: &str, notification_type: NotificationType, priority: u8) -> Self {
        Self {
            id: id.to_string(),
            notification_type,
            created_at: Local::now(),
            seen: false,
            dismissed: false,
            priority,
        }
    }
    
    /// Create a meeting reminder notification
    pub fn meeting_reminder(
        id: &str,
        title: &str,
        starts_in: u32,
        location: Option<&str>,
        attendees: Vec<String>,
    ) -> Self {
        Self::new(
            id,
            NotificationType::MeetingReminder {
                title: title.to_string(),
                starts_in_minutes: starts_in,
                location: location.map(|s| s.to_string()),
                attendees,
            },
            8, // High priority
        )
    }
    
    /// Create a morning brief notification
    pub fn morning_brief(id: &str, meetings: usize, emails: usize, tasks: usize) -> Self {
        Self::new(
            id,
            NotificationType::MorningBrief {
                meetings_today: meetings,
                unread_emails: emails,
                tasks_due: tasks,
            },
            5, // Medium priority
        )
    }
    
    /// Create an email follow-up notification
    pub fn email_followup(id: &str, person: &str, subject: &str, days: u32) -> Self {
        Self::new(
            id,
            NotificationType::EmailFollowUp {
                person: person.to_string(),
                subject: subject.to_string(),
                days_since: days,
            },
            4, // Medium-low priority
        )
    }
    
    /// Create a generic notification
    pub fn generic(id: &str, message: &str) -> Self {
        Self::new(
            id,
            NotificationType::Generic {
                message: message.to_string(),
            },
            3, // Low priority
        )
    }
    
    /// Mark as seen
    pub fn mark_seen(&mut self) {
        self.seen = true;
    }
    
    /// Dismiss the notification
    pub fn dismiss(&mut self) {
        self.dismissed = true;
    }
    
    /// Format the notification for display
    pub fn display(&self) -> String {
        match &self.notification_type {
            NotificationType::MeetingReminder { title, starts_in_minutes, location, attendees } => {
                let mut msg = format!("📅 Meeting in {} min: {}", starts_in_minutes, title);
                if let Some(loc) = location {
                    msg.push_str(&format!("\n   📍 {}", loc));
                }
                if !attendees.is_empty() {
                    msg.push_str(&format!("\n   👥 {}", attendees.join(", ")));
                }
                msg
            }
            NotificationType::MorningBrief { meetings_today, unread_emails, tasks_due } => {
                let mut msg = "☀️ Good morning! Today you have:\n".to_string();
                msg.push_str(&format!("   📅 {} meeting(s)\n", meetings_today));
                msg.push_str(&format!("   📧 {} unread email(s)\n", unread_emails));
                msg.push_str(&format!("   ✅ {} task(s) due", tasks_due));
                msg
            }
            NotificationType::EmailFollowUp { person, subject, days_since } => {
                format!(
                    "📧 No response from {} in {} days\n   Subject: {}\n   💡 Would you like to follow up?",
                    person, days_since, subject
                )
            }
            NotificationType::FreeTimeSuggestion { duration_hours, suggested_tasks } => {
                let mut msg = format!("💡 You have {:.1} hours free", duration_hours);
                if !suggested_tasks.is_empty() {
                    msg.push_str("\n   Suggestions:");
                    for task in suggested_tasks {
                        msg.push_str(&format!("\n   • {}", task));
                    }
                }
                msg
            }
            NotificationType::PatternReminder { pattern_name, suggestion } => {
                format!("💡 {} - {}", pattern_name, suggestion)
            }
            NotificationType::Generic { message } => {
                message.clone()
            }
        }
    }
    
    /// Get emoji icon for the notification type
    pub fn icon(&self) -> &str {
        match &self.notification_type {
            NotificationType::MeetingReminder { .. } => "📅",
            NotificationType::MorningBrief { .. } => "☀️",
            NotificationType::EmailFollowUp { .. } => "📧",
            NotificationType::FreeTimeSuggestion { .. } => "💡",
            NotificationType::PatternReminder { .. } => "💡",
            NotificationType::Generic { .. } => "🔔",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_meeting_reminder_display() {
        let notif = Notification::meeting_reminder(
            "test-1",
            "Team Standup",
            30,
            Some("Conference Room A"),
            vec!["Alice".to_string(), "Bob".to_string()],
        );
        
        let display = notif.display();
        assert!(display.contains("30 min"));
        assert!(display.contains("Team Standup"));
        assert!(display.contains("Conference Room A"));
        assert!(display.contains("Alice"));
    }
    
    #[test]
    fn test_morning_brief_display() {
        let notif = Notification::morning_brief("test-2", 3, 12, 2);
        
        let display = notif.display();
        assert!(display.contains("3 meeting"));
        assert!(display.contains("12 unread"));
        assert!(display.contains("2 task"));
    }
    
    #[test]
    fn test_notification_state() {
        let mut notif = Notification::generic("test-3", "Test message");
        
        assert!(!notif.seen);
        assert!(!notif.dismissed);
        
        notif.mark_seen();
        assert!(notif.seen);
        
        notif.dismiss();
        assert!(notif.dismissed);
    }
}
