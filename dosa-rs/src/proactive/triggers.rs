//! Trigger definitions for proactive actions
//! 
//! Defines when and why proactive notifications should be sent.

use chrono::{DateTime, Local, Duration};

/// Type of trigger that initiated the notification
#[derive(Clone, Debug, PartialEq)]
pub enum TriggerType {
    /// Time-based trigger (e.g., meeting reminders)
    TimeBased,
    /// Event-based trigger (e.g., new important email)
    EventBased,
    /// Pattern-based trigger (e.g., "you usually email X on Mondays")
    PatternBased,
    /// Gap-based trigger (e.g., no response in 3 days)
    GapBased,
}

/// Condition that must be met for a trigger to fire
#[derive(Clone, Debug)]
pub enum TriggerCondition {
    /// Time until event (for reminders)
    TimeBefore { 
        minutes: u32,
        event_type: String,
    },
    /// Time since last event
    TimeSince {
        days: u32,
        event_type: String,
    },
    /// Specific time of day
    TimeOfDay {
        hour: u8,
        minute: u8,
    },
    /// Day of week pattern
    DayOfWeek {
        day: chrono::Weekday,
        pattern_name: String,
    },
    /// Email from specific person
    EmailFrom {
        person: String,
        importance: Importance,
    },
    /// No response to email
    NoEmailResponse {
        days: u32,
        person: String,
    },
    /// Calendar gap (free time)
    FreeTime {
        duration_minutes: u32,
    },
}

/// Importance level for filtering
#[derive(Clone, Debug, PartialEq)]
pub enum Importance {
    High,
    Medium,
    Low,
    Any,
}

/// A trigger definition
#[derive(Clone, Debug)]
pub struct Trigger {
    /// Unique identifier
    pub id: String,
    /// Type of trigger
    pub trigger_type: TriggerType,
    /// Condition to evaluate
    pub condition: TriggerCondition,
    /// Message template to show
    pub message_template: String,
    /// Whether this trigger is enabled
    pub enabled: bool,
    /// Last time this trigger fired
    pub last_fired: Option<DateTime<Local>>,
}

impl Trigger {
    /// Create a new trigger
    pub fn new(id: &str, trigger_type: TriggerType, condition: TriggerCondition, message: &str) -> Self {
        Self {
            id: id.to_string(),
            trigger_type,
            condition,
            message_template: message.to_string(),
            enabled: true,
            last_fired: None,
        }
    }
    
    /// Create a meeting reminder trigger
    pub fn meeting_reminder(minutes_before: u32) -> Self {
        Self::new(
            "meeting_reminder",
            TriggerType::TimeBased,
            TriggerCondition::TimeBefore {
                minutes: minutes_before,
                event_type: "meeting".to_string(),
            },
            "📅 You have a meeting in {minutes} minutes: {title}",
        )
    }
    
    /// Create an email follow-up trigger
    pub fn email_followup(days: u32) -> Self {
        Self::new(
            "email_followup",
            TriggerType::GapBased,
            TriggerCondition::NoEmailResponse {
                days,
                person: "*".to_string(), // Any person
            },
            "📧 No response from {person} in {days} days - want to follow up?",
        )
    }
    
    /// Create a morning brief trigger
    pub fn morning_brief(hour: u8) -> Self {
        Self::new(
            "morning_brief",
            TriggerType::TimeBased,
            TriggerCondition::TimeOfDay {
                hour,
                minute: 0,
            },
            "☀️ Good morning! Here's your day ahead...",
        )
    }
    
    /// Create a free time suggestion trigger
    pub fn free_time_suggestion(min_duration_minutes: u32) -> Self {
        Self::new(
            "free_time",
            TriggerType::PatternBased,
            TriggerCondition::FreeTime {
                duration_minutes: min_duration_minutes,
            },
            "💡 You have {duration} hours free - good time for deep work or catch-up tasks",
        )
    }
    
    /// Check if enough time has passed since last fire (debounce)
    pub fn can_fire(&self, debounce_minutes: u32) -> bool {
        if !self.enabled {
            return false;
        }
        
        match self.last_fired {
            None => true,
            Some(last) => {
                let now = Local::now();
                let elapsed = now - last;
                elapsed > Duration::minutes(debounce_minutes as i64)
            }
        }
    }
    
    /// Mark as fired
    pub fn mark_fired(&mut self) {
        self.last_fired = Some(Local::now());
    }
    
    /// Format the message with context
    pub fn format_message(&self, context: &TriggerContext) -> String {
        let mut msg = self.message_template.clone();
        
        msg = msg.replace("{minutes}", &context.minutes.unwrap_or(0).to_string());
        msg = msg.replace("{hours}", &context.hours.unwrap_or(0).to_string());
        msg = msg.replace("{days}", &context.days.unwrap_or(0).to_string());
        msg = msg.replace("{duration}", &context.duration.clone().unwrap_or_default());
        msg = msg.replace("{title}", &context.title.clone().unwrap_or_default());
        msg = msg.replace("{person}", &context.person.clone().unwrap_or_default());
        
        msg
    }
}

/// Context for formatting trigger messages
#[derive(Clone, Debug, Default)]
pub struct TriggerContext {
    pub minutes: Option<u32>,
    pub hours: Option<u32>,
    pub days: Option<u32>,
    pub duration: Option<String>,
    pub title: Option<String>,
    pub person: Option<String>,
    pub extra: Option<String>,
}

impl TriggerContext {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_minutes(mut self, minutes: u32) -> Self {
        self.minutes = Some(minutes);
        self
    }
    
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = Some(title.to_string());
        self
    }
    
    pub fn with_person(mut self, person: &str) -> Self {
        self.person = Some(person.to_string());
        self
    }
    
    pub fn with_days(mut self, days: u32) -> Self {
        self.days = Some(days);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_meeting_reminder_trigger() {
        let trigger = Trigger::meeting_reminder(30);
        assert_eq!(trigger.id, "meeting_reminder");
        assert_eq!(trigger.trigger_type, TriggerType::TimeBased);
        assert!(trigger.enabled);
    }
    
    #[test]
    fn test_trigger_message_formatting() {
        let trigger = Trigger::meeting_reminder(30);
        let context = TriggerContext::new()
            .with_minutes(30)
            .with_title("Team Standup");
        
        let msg = trigger.format_message(&context);
        assert!(msg.contains("30"));
        assert!(msg.contains("Team Standup"));
    }
    
    #[test]
    fn test_can_fire_debounce() {
        let mut trigger = Trigger::meeting_reminder(30);
        assert!(trigger.can_fire(5)); // First time, can fire
        
        trigger.mark_fired();
        assert!(!trigger.can_fire(60)); // Just fired, can't fire within 60 min
    }
}
