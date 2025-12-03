//! Background task scheduler using tokio-cron-scheduler
//! 
//! Handles scheduling of recurring and one-time tasks.

use anyhow::{anyhow, Result};
use tokio_cron_scheduler::{Job, JobScheduler, JobSchedulerError};
use std::sync::Arc;
use tokio::sync::mpsc;

/// A scheduled notification
#[derive(Clone, Debug)]
pub struct ScheduledNotification {
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Local>,
}

/// Proactive task scheduler
pub struct ProactiveScheduler {
    scheduler: JobScheduler,
    notification_tx: mpsc::UnboundedSender<ScheduledNotification>,
    notification_rx: Option<mpsc::UnboundedReceiver<ScheduledNotification>>,
}

impl ProactiveScheduler {
    /// Create a new scheduler
    pub fn new() -> Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel();
        
        let scheduler = tokio::runtime::Handle::current().block_on(async {
            JobScheduler::new().await
        }).map_err(|e| anyhow!("Failed to create scheduler: {:?}", e))?;
        
        Ok(Self {
            scheduler,
            notification_tx: tx,
            notification_rx: Some(rx),
        })
    }
    
    /// Start the scheduler
    pub async fn start(&mut self) -> Result<()> {
        self.scheduler.start().await
            .map_err(|e| anyhow!("Failed to start scheduler: {:?}", e))
    }
    
    /// Shutdown the scheduler
    pub async fn shutdown(&mut self) -> Result<()> {
        self.scheduler.shutdown().await
            .map_err(|e| anyhow!("Failed to shutdown scheduler: {:?}", e))
    }
    
    /// Take the notification receiver (for consuming notifications)
    pub fn take_notification_rx(&mut self) -> Option<mpsc::UnboundedReceiver<ScheduledNotification>> {
        self.notification_rx.take()
    }
    
    /// Schedule morning brief at a specific hour
    pub fn schedule_morning_brief(&mut self, hour: u8) -> Result<()> {
        let tx = self.notification_tx.clone();
        let cron_expr = format!("0 0 {} * * *", hour); // At hour:00:00 every day
        
        let job = tokio::runtime::Handle::current().block_on(async {
            Job::new_async(cron_expr.as_str(), move |_uuid, _lock| {
                let tx = tx.clone();
                Box::pin(async move {
                    let notification = ScheduledNotification {
                        message: "☀️ Good morning! Here's your daily brief...".to_string(),
                        timestamp: chrono::Local::now(),
                    };
                    let _ = tx.send(notification);
                })
            })
        }).map_err(|e: JobSchedulerError| anyhow!("Failed to create morning brief job: {:?}", e))?;
        
        tokio::runtime::Handle::current().block_on(async {
            self.scheduler.add(job).await
        }).map_err(|e| anyhow!("Failed to schedule morning brief: {:?}", e))?;
        
        eprintln!("📅 Scheduled morning brief at {}:00", hour);
        Ok(())
    }
    
    /// Schedule a meeting reminder
    pub fn schedule_meeting_reminder(&mut self, meeting_title: &str, at_time: chrono::DateTime<chrono::Local>) -> Result<()> {
        let tx = self.notification_tx.clone();
        let title = meeting_title.to_string();
        
        // Convert to cron expression for one-time execution
        let datetime = at_time;
        let cron_expr = format!(
            "{} {} {} {} * *",
            datetime.format("%M"),
            datetime.format("%H"),
            datetime.format("%d"),
            datetime.format("%m")
        );
        
        let job = tokio::runtime::Handle::current().block_on(async {
            Job::new_one_shot_async(
                std::time::Duration::from_secs(0), // We'll calculate the duration
                move |_uuid, _lock| {
                    let tx = tx.clone();
                    let title = title.clone();
                    Box::pin(async move {
                        let notification = ScheduledNotification {
                            message: format!("📅 Upcoming meeting: {}", title),
                            timestamp: chrono::Local::now(),
                        };
                        let _ = tx.send(notification);
                    })
                }
            )
        }).map_err(|e: JobSchedulerError| anyhow!("Failed to create meeting reminder job: {:?}", e))?;
        
        tokio::runtime::Handle::current().block_on(async {
            self.scheduler.add(job).await
        }).map_err(|e| anyhow!("Failed to schedule meeting reminder: {:?}", e))?;
        
        eprintln!("⏰ Scheduled reminder for: {}", meeting_title);
        Ok(())
    }
    
    /// Schedule a one-time notification
    pub fn schedule_one_time(&mut self, in_minutes: u32, message: String) -> Result<()> {
        let tx = self.notification_tx.clone();
        let duration = std::time::Duration::from_secs(in_minutes as u64 * 60);
        
        let job = tokio::runtime::Handle::current().block_on(async {
            Job::new_one_shot_async(
                duration,
                move |_uuid, _lock| {
                    let tx = tx.clone();
                    let msg = message.clone();
                    Box::pin(async move {
                        let notification = ScheduledNotification {
                            message: msg,
                            timestamp: chrono::Local::now(),
                        };
                        let _ = tx.send(notification);
                    })
                }
            )
        }).map_err(|e: JobSchedulerError| anyhow!("Failed to create one-shot job: {:?}", e))?;
        
        tokio::runtime::Handle::current().block_on(async {
            self.scheduler.add(job).await
        }).map_err(|e| anyhow!("Failed to schedule notification: {:?}", e))?;
        
        eprintln!("⏰ Notification scheduled in {} minutes", in_minutes);
        Ok(())
    }
    
    /// Schedule a recurring check (e.g., every 15 minutes)
    pub fn schedule_periodic_check(&mut self, interval_minutes: u32, check_name: &str) -> Result<()> {
        let tx = self.notification_tx.clone();
        let name = check_name.to_string();
        
        // Run every N minutes
        let cron_expr = format!("0 */{} * * * *", interval_minutes);
        
        let job = tokio::runtime::Handle::current().block_on(async {
            Job::new_async(cron_expr.as_str(), move |_uuid, _lock| {
                let tx = tx.clone();
                let name = name.clone();
                Box::pin(async move {
                    // This triggers a check - the actual check logic happens elsewhere
                    let notification = ScheduledNotification {
                        message: format!("🔍 Running check: {}", name),
                        timestamp: chrono::Local::now(),
                    };
                    let _ = tx.send(notification);
                })
            })
        }).map_err(|e: JobSchedulerError| anyhow!("Failed to create periodic job: {:?}", e))?;
        
        tokio::runtime::Handle::current().block_on(async {
            self.scheduler.add(job).await
        }).map_err(|e| anyhow!("Failed to schedule periodic check: {:?}", e))?;
        
        eprintln!("🔄 Scheduled {} every {} minutes", check_name, interval_minutes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_scheduler_creation() {
        // Note: This test requires tokio runtime
        // In non-async context it won't work
    }
}
