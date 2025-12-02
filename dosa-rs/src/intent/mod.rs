//! Intent Classification System
//!
//! Modular intent detection with swappable classifiers.
//! Supports keyword-based, LLM-based, ONNX-based, and hybrid approaches.

pub mod types;
pub mod keyword;
pub mod hybrid;
pub mod handler;
pub mod onnx;
#[cfg(test)]
mod tests;

pub use types::{Intent, IntentResult, IntentContext, CalendarQueryType, EmailQueryType};
pub use keyword::KeywordClassifier;
pub use hybrid::HybridClassifier;
pub use handler::{IntentHandler, HandleResult};
pub use onnx::{OnnxClassifier, OnnxClassifyResult, ExtractedSlot, NluConfig};

use anyhow::Result;
use async_trait::async_trait;

/// Trait for intent classifiers - implement this to add new classification strategies
#[async_trait]
pub trait IntentClassifier: Send + Sync {
    /// Classify the user input into an intent
    async fn classify(&self, input: &str, ctx: &IntentContext) -> Result<IntentResult>;
    
    /// Get classifier name for debugging
    fn name(&self) -> &str;
}
