//! Proactive Assistant - Background task scheduling and intelligent alerts
//! 
//! This module provides proactive notifications and suggestions:
//! - Meeting reminders with context
//! - Email follow-up suggestions  
//! - Morning brief / daily digest
//! - Pattern-based suggestions

mod scheduler;
mod triggers;
mod notifications;

pub use scheduler::ProactiveScheduler;
pub use triggers::{Trigger, TriggerType, TriggerCondition};
pub use notifications::{Notification, NotificationType};

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for the proactive assistant
#[derive(Clone, Debug)]
pub struct ProactiveConfig {
    /// Enable meeting reminders (default: true)
    pub meeting_reminders: bool,
    /// Minutes before meeting to send reminder (default: 30)
    pub reminder_minutes: u32,
    /// Enable morning brief (default: true)
    pub morning_brief: bool,
    /// Hour to send morning brief (24h format, default: 8)
    pub morning_brief_hour: u8,
    /// Enable follow-up suggestions (default: true)
    pub follow_up_suggestions: bool,
    /// Days without response to suggest follow-up (default: 3)
    pub follow_up_days: u32,
}

impl Default for ProactiveConfig {
    fn default() -> Self {
        Self {
            meeting_reminders: true,
            reminder_minutes: 30,
            morning_brief: true,
            morning_brief_hour: 8,
            follow_up_suggestions: true,
            follow_up_days: 3,
        }
    }
}

/// Main proactive assistant interface
pub struct ProactiveAssistant {
    config: ProactiveConfig,
    scheduler: Arc<RwLock<ProactiveScheduler>>,
    running: Arc<RwLock<bool>>,
}

impl ProactiveAssistant {
    /// Create a new proactive assistant
    pub async fn new(config: ProactiveConfig) -> Result<Self> {
        let scheduler = ProactiveScheduler::new()?;
        
        Ok(Self {
            config,
            scheduler: Arc::new(RwLock::new(scheduler)),
            running: Arc::new(RwLock::new(false)),
        })
    }
    
    /// Create with default configuration
    pub async fn default_config() -> Result<Self> {
        Self::new(ProactiveConfig::default()).await
    }
    
    /// Start the proactive assistant
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Ok(()); // Already running
        }
        
        let mut scheduler = self.scheduler.write().await;
        
        // Set up morning brief if enabled
        if self.config.morning_brief {
            scheduler.schedule_morning_brief(self.config.morning_brief_hour)?;
        }
        
        // Start the scheduler
        scheduler.start().await?;
        *running = true;
        
        eprintln!("🔔 Proactive assistant started");
        Ok(())
    }
    
    /// Stop the proactive assistant
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Ok(()); // Not running
        }
        
        let mut scheduler = self.scheduler.write().await;
        scheduler.shutdown().await?;
        *running = false;
        
        eprintln!("⏹️ Proactive assistant stopped");
        Ok(())
    }
    
    /// Check if the assistant is running
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }
    
    /// Get the current configuration
    pub fn config(&self) -> &ProactiveConfig {
        &self.config
    }
    
    /// Schedule a one-time reminder
    pub async fn schedule_reminder(&self, in_minutes: u32, message: &str) -> Result<()> {
        let mut scheduler = self.scheduler.write().await;
        scheduler.schedule_one_time(in_minutes, message.to_string())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_proactive_config_default() {
        let config = ProactiveConfig::default();
        assert!(config.meeting_reminders);
        assert_eq!(config.reminder_minutes, 30);
        assert!(config.morning_brief);
        assert_eq!(config.morning_brief_hour, 8);
    }
}
