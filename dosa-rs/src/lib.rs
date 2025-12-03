// Library exports for testing and external use
pub mod llm;
pub mod knowledge;
pub mod router;
pub mod storage;
pub mod calendar;
pub mod reminders;
pub mod contacts;
pub mod intelligence;
pub mod completer;
pub mod integrations;
pub mod intent;
pub mod proactive;

#[cfg(feature = "voice")]
pub mod voice;
