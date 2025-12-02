//! Intent types and results

use chrono::NaiveDate;

/// Unified intent enum covering all possible user intents
#[derive(Debug, Clone, PartialEq)]
pub enum Intent {
    // Commands (explicit /command)
    Command(String),
    
    // Calendar intents
    CalendarCreate { description: String },
    CalendarQuery { query_type: CalendarQueryType },
    CalendarUpdate { description: String },
    CalendarDelete { description: String },
    
    // Email intents
    EmailCompose { description: String },
    EmailReply { description: String },
    EmailAttendees { description: String },
    EmailQuery { query_type: EmailQueryType },
    
    // Contact/Knowledge management
    ContactUpdate { description: String },  // Add person, set email, etc.
    KnowledgeQuery { query: String },
    
    // Fallback
    Conversation { input: String },
    
    // Unknown/ambiguous - needs clarification
    Ambiguous { candidates: Vec<(Intent, f32)> },
}

/// Calendar query subtypes
#[derive(Debug, Clone, PartialEq)]
pub enum CalendarQueryType {
    Today,
    Tomorrow,
    Week,
    Upcoming,
    Date(NaiveDate),
}

/// Email query subtypes
#[derive(Debug, Clone, PartialEq)]
pub enum EmailQueryType {
    Unread,
    List,
    From(String),
    Summary,
}

/// Result of intent classification
#[derive(Debug, Clone)]
pub struct IntentResult {
    /// The primary detected intent
    pub intent: Intent,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Alternative interpretations with their confidence
    pub alternatives: Vec<(Intent, f32)>,
    /// Raw input preserved
    pub raw_input: String,
}

impl IntentResult {
    pub fn new(intent: Intent, confidence: f32, raw_input: &str) -> Self {
        IntentResult {
            intent,
            confidence,
            alternatives: Vec::new(),
            raw_input: raw_input.to_string(),
        }
    }
    
    pub fn with_alternatives(mut self, alternatives: Vec<(Intent, f32)>) -> Self {
        self.alternatives = alternatives;
        self
    }
    
    /// Check if classification is confident enough to proceed
    pub fn is_confident(&self) -> bool {
        self.confidence >= 0.7
    }
    
    /// Check if there are ambiguous alternatives
    pub fn is_ambiguous(&self) -> bool {
        !self.alternatives.is_empty() && 
        self.alternatives.iter().any(|(_, conf)| *conf > 0.5)
    }
}

/// Context available during classification
pub struct IntentContext<'a> {
    /// Is user authenticated with Google
    pub is_google_auth: bool,
    /// Recent conversation context (last few exchanges)
    pub recent_context: Option<&'a [String]>,
    /// User's timezone
    pub timezone: Option<String>,
}

impl<'a> IntentContext<'a> {
    pub fn new(is_google_auth: bool) -> Self {
        IntentContext {
            is_google_auth,
            recent_context: None,
            timezone: None,
        }
    }
    
    pub fn with_context(mut self, context: &'a [String]) -> Self {
        self.recent_context = Some(context);
        self
    }
}
