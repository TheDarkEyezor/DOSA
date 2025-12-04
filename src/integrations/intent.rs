//! Natural language intent detection for calendar operations
//!
//! Detects when users want to create events or query calendar without explicit commands.

use chrono::{NaiveDate, Local, Duration, Datelike};

/// Calendar-related action keywords
const CALENDAR_CREATE_KEYWORDS: &[&str] = &[
    "schedule", "book", "set up", "arrange", "plan",
    "create a meeting", "create an event", "add to calendar",
    "put on calendar", "block time", "block off",
];

/// Calendar update keywords
const CALENDAR_UPDATE_KEYWORDS: &[&str] = &[
    "update", "change", "move", "reschedule", "modify",
    "push back", "push to", "make it", "switch to",
];

/// Calendar delete keywords
const CALENDAR_DELETE_KEYWORDS: &[&str] = &[
    "cancel", "delete", "remove", "drop",
];

/// Calendar query keywords
const CALENDAR_QUERY_KEYWORDS: &[&str] = &[
    "what's on my calendar", "whats on my calendar",
    "what is on my calendar", "show my calendar",
    "my schedule", "my events", "my meetings",
    "do i have", "am i free", "am i busy",
    "what events", "any events", "any meetings",
    "calendar for", "schedule for",
    "what do i have", "what have i got",
    "coming up", "upcoming",
];

/// Time-related patterns that suggest calendar intent
const TIME_PATTERNS: &[&str] = &[
    "at ", "on monday", "on tuesday", "on wednesday", "on thursday", 
    "on friday", "on saturday", "on sunday",
    "tomorrow", "next week", "this week", "today",
    "morning", "afternoon", "evening",
    " pm", " am", "o'clock", "noon", "midnight",
];

/// Event type keywords
const EVENT_KEYWORDS: &[&str] = &[
    "meeting", "call", "lunch", "dinner", "breakfast", "coffee",
    "appointment", "interview", "review", "sync", "standup",
    "1:1", "one-on-one", "catch up", "catchup", "session",
];

/// Preposition patterns that suggest events with people
const WITH_PATTERNS: &[&str] = &[
    "with ", "and ", "invite ",
];

/// Type of calendar intent detected
#[derive(Debug, Clone, PartialEq)]
pub enum CalendarIntent {
    /// User wants to create a new event
    CreateEvent(String),
    /// User wants to update an existing event
    UpdateEvent(String),
    /// User wants to delete/cancel an event
    DeleteEvent(String),
    /// User wants to see today's events
    QueryToday,
    /// User wants to see tomorrow's events
    QueryTomorrow,
    /// User wants to see this week's events
    QueryWeek,
    /// User wants to see events on a specific date
    QueryDate(NaiveDate),
    /// User wants to see upcoming events
    QueryUpcoming,
    /// Not a calendar intent
    None,
}

/// Detect if the input looks like a calendar event creation request
pub fn is_calendar_intent(input: &str) -> bool {
    let lower = input.to_lowercase();
    
    // Must not start with / (that's a command)
    if lower.starts_with('/') {
        return false;
    }
    
    // Check for explicit calendar creation keywords
    for keyword in CALENDAR_CREATE_KEYWORDS {
        if lower.contains(keyword) {
            return true;
        }
    }
    
    // Check for event keywords + time patterns
    let has_event_keyword = EVENT_KEYWORDS.iter().any(|k| lower.contains(k));
    let has_time_pattern = TIME_PATTERNS.iter().any(|t| lower.contains(t));
    let has_with_pattern = WITH_PATTERNS.iter().any(|w| lower.contains(w));
    
    // Strong signal: event keyword + time
    if has_event_keyword && has_time_pattern {
        return true;
    }
    
    // Medium signal: "with someone" + time (e.g., "lunch with John tomorrow")
    if has_with_pattern && has_time_pattern && has_event_keyword {
        return true;
    }
    
    // Check for common phrases
    let calendar_phrases = [
        "remind me to",  // Could be task or calendar
        "don't forget",
        "let's meet",
        "can we meet",
        "let's have",
        "i have a",
        "i've got a",
    ];
    
    for phrase in calendar_phrases {
        if lower.contains(phrase) && has_time_pattern {
            return true;
        }
    }
    
    false
}

/// Detect if the input is a calendar query
pub fn is_calendar_query(input: &str) -> bool {
    let lower = input.to_lowercase();
    
    // Must not start with / (that's a command)
    if lower.starts_with('/') {
        return false;
    }
    
    // Check for explicit query keywords
    for keyword in CALENDAR_QUERY_KEYWORDS {
        if lower.contains(keyword) {
            return true;
        }
    }
    
    false
}

/// Parse the specific calendar query intent
pub fn parse_calendar_query(input: &str) -> CalendarIntent {
    let lower = input.to_lowercase();
    
    // Check for specific day mentions
    if lower.contains("today") {
        return CalendarIntent::QueryToday;
    }
    
    if lower.contains("tomorrow") {
        return CalendarIntent::QueryTomorrow;
    }
    
    if lower.contains("this week") || lower.contains("week ahead") {
        return CalendarIntent::QueryWeek;
    }
    
    if lower.contains("next week") {
        return CalendarIntent::QueryWeek;
    }
    
    // Check for weekday names
    let today = Local::now().date_naive();
    let weekdays = [
        ("monday", chrono::Weekday::Mon),
        ("tuesday", chrono::Weekday::Tue),
        ("wednesday", chrono::Weekday::Wed),
        ("thursday", chrono::Weekday::Thu),
        ("friday", chrono::Weekday::Fri),
        ("saturday", chrono::Weekday::Sat),
        ("sunday", chrono::Weekday::Sun),
    ];
    
    for (name, weekday) in weekdays {
        if lower.contains(name) {
            let current = today.weekday().num_days_from_monday() as i64;
            let target = weekday.num_days_from_monday() as i64;
            let mut days_ahead = target - current;
            if days_ahead <= 0 {
                days_ahead += 7;
            }
            return CalendarIntent::QueryDate(today + Duration::days(days_ahead));
        }
    }
    
    // Default to upcoming if it's a general query
    if lower.contains("coming up") || lower.contains("upcoming") {
        return CalendarIntent::QueryUpcoming;
    }
    
    // For generic "what's on my calendar" type queries, show week
    CalendarIntent::QueryWeek
}

/// Detect if the input is an event update request
pub fn is_update_intent(input: &str) -> bool {
    let lower = input.to_lowercase();
    
    if lower.starts_with('/') {
        return false;
    }
    
    // If talking about emailing/notifying attendees, this is NOT an update intent
    if lower.contains("email") || lower.contains("notify") || lower.contains("attendee") 
        || lower.contains("let them know") || lower.contains("letting them know")
        || lower.contains("tell them") || lower.contains("message") {
        return false;
    }
    
    // Must have an update keyword
    let has_update_keyword = CALENDAR_UPDATE_KEYWORDS.iter().any(|k| lower.contains(k));
    
    // And should reference an event or time
    let has_event_ref = EVENT_KEYWORDS.iter().any(|k| lower.contains(k))
        || lower.contains("the ")  // "the meeting", "the dinner"
        || lower.contains("my ");  // "my meeting"
    
    let has_time = TIME_PATTERNS.iter().any(|t| lower.contains(t));
    
    has_update_keyword && (has_event_ref || has_time)
}

/// Detect if the input is an event delete request
pub fn is_delete_intent(input: &str) -> bool {
    let lower = input.to_lowercase();
    
    if lower.starts_with('/') {
        return false;
    }
    
    // Must have a delete keyword
    let has_delete_keyword = CALENDAR_DELETE_KEYWORDS.iter().any(|k| lower.contains(k));
    
    // And should reference an event
    let has_event_ref = EVENT_KEYWORDS.iter().any(|k| lower.contains(k))
        || lower.contains("the ")
        || lower.contains("my ")
        || lower.contains("with ");  // "cancel dinner with Charlie"
    
    has_delete_keyword && has_event_ref
}

/// Get the full calendar intent (create, update, delete, or query)
pub fn detect_calendar_intent(input: &str) -> CalendarIntent {
    // First check if it's a query (takes precedence since it's less invasive)
    if is_calendar_query(input) {
        return parse_calendar_query(input);
    }
    
    // Check for update intent
    if is_update_intent(input) {
        return CalendarIntent::UpdateEvent(input.to_string());
    }
    
    // Check for delete intent
    if is_delete_intent(input) {
        return CalendarIntent::DeleteEvent(input.to_string());
    }
    
    // Then check if it's a create intent
    if is_calendar_intent(input) {
        return CalendarIntent::CreateEvent(input.to_string());
    }
    
    CalendarIntent::None
}

/// Get the confidence level of calendar intent (0.0 - 1.0)
pub fn calendar_intent_confidence(input: &str) -> f32 {
    let lower = input.to_lowercase();
    let mut score = 0.0f32;
    
    // Explicit keywords are strong signals
    for keyword in CALENDAR_CREATE_KEYWORDS {
        if lower.contains(keyword) {
            score += 0.4;
            break;
        }
    }
    
    // Event keywords
    for keyword in EVENT_KEYWORDS {
        if lower.contains(keyword) {
            score += 0.2;
            break;
        }
    }
    
    // Time patterns
    let time_count = TIME_PATTERNS.iter().filter(|t| lower.contains(*t)).count();
    score += (time_count as f32 * 0.15).min(0.3);
    
    // With patterns (suggests attendees)
    if WITH_PATTERNS.iter().any(|w| lower.contains(w)) {
        score += 0.1;
    }
    
    score.min(1.0)
}

// ============================================================================
// Email Intent Detection
// ============================================================================

/// Email query keywords
const EMAIL_QUERY_KEYWORDS: &[&str] = &[
    "my emails", "my inbox", "my mail",
    "unread emails", "unread mail", "unread messages",
    "any emails", "any mail", "any messages",
    "new emails", "new mail", "new messages",
    "check email", "check mail", "check my email",
    "show email", "show mail", "show inbox",
    "email from", "mail from", "messages from",
    "what emails", "what mail",
    "do i have email", "do i have mail",
    "emails today", "mail today",
];

/// Email compose keywords
const EMAIL_COMPOSE_KEYWORDS: &[&str] = &[
    "send an email", "send email", "send a message",
    "write an email", "write email", "write a message",
    "compose an email", "compose email", "compose a message",
    "email to", "mail to", "message to",
    "draft an email", "draft email", "draft a message",
    "send a note", "send note",
    "let me email", "let me send",
    "can you email", "can you send an email",
    "could you email", "please email",
    "an email saying", "an email to",
    "email the attendees", "email attendees",
    "notify the attendees", "message the attendees",
];

/// Patterns that suggest sending email to someone (name + email action)
const EMAIL_ACTION_PATTERNS: &[&str] = &[
    "email saying", "email telling", "email letting",
    "message saying", "message telling", "message letting",
];

/// Type of email intent detected
#[derive(Debug, Clone, PartialEq)]
pub enum EmailIntent {
    /// Check unread emails
    Unread,
    /// List recent emails
    List,
    /// Emails from a specific sender
    From(String),
    /// Email summary
    Summary,
    /// Compose/send an email
    Compose(String),
    /// Email attendees of a calendar event (multi-hop)
    EmailEventAttendees(String),
    /// Not an email intent
    None,
}

/// Check if this is a multi-hop email to event attendees
pub fn is_attendee_email(input: &str) -> bool {
    let lower = input.to_lowercase();
    (lower.contains("attendee") || lower.contains("participant") || lower.contains("invitee")) &&
    (lower.contains("email") || lower.contains("message") || lower.contains("notify") || lower.contains("send") || lower.contains("let") || lower.contains("tell"))
}

/// Detect if the input is an email query
pub fn is_email_query(input: &str) -> bool {
    let lower = input.to_lowercase();
    
    if lower.starts_with('/') {
        return false;
    }
    
    EMAIL_QUERY_KEYWORDS.iter().any(|k| lower.contains(k))
}

/// Detect if the input is an email compose request
pub fn is_email_compose(input: &str) -> bool {
    let lower = input.to_lowercase();
    
    if lower.starts_with('/') {
        return false;
    }
    
    // Check explicit keywords
    if EMAIL_COMPOSE_KEYWORDS.iter().any(|k| lower.contains(k)) {
        return true;
    }
    
    // Check action patterns
    if EMAIL_ACTION_PATTERNS.iter().any(|k| lower.contains(k)) {
        return true;
    }
    
    // Pattern: "send/email <name> ..." - check for name followed by email-related words
    // e.g., "send charlie an email", "email john about the meeting"
    let words: Vec<&str> = lower.split_whitespace().collect();
    for (i, word) in words.iter().enumerate() {
        if (*word == "send" || *word == "email" || *word == "message") && i + 1 < words.len() {
            // Check if this is followed by a name and then email-related words
            let rest = words[i+1..].join(" ");
            if rest.contains("email") || rest.contains("message") || rest.contains("saying") || rest.contains("about") {
                return true;
            }
        }
    }
    
    false
}

/// Parse the specific email query intent
pub fn parse_email_query(input: &str) -> EmailIntent {
    let lower = input.to_lowercase();
    
    // Check for "from" queries
    if lower.contains("from ") {
        // Extract the sender name
        if let Some(idx) = lower.find("from ") {
            let after_from = &input[idx + 5..];
            let sender = after_from.split_whitespace().next().unwrap_or("");
            if !sender.is_empty() {
                return EmailIntent::From(sender.to_string());
            }
        }
    }
    
    // Check for unread
    if lower.contains("unread") || lower.contains("new ") {
        return EmailIntent::Unread;
    }
    
    // Check for summary
    if lower.contains("summary") || lower.contains("summarize") || lower.contains("important") {
        return EmailIntent::Summary;
    }
    
    // Default to list
    EmailIntent::List
}

/// Detect email intent (both query and compose)
pub fn detect_email_intent(input: &str) -> EmailIntent {
    // Check for attendee-based email first (multi-hop)
    if is_attendee_email(input) {
        return EmailIntent::EmailEventAttendees(input.to_string());
    }
    
    // Check compose next
    if is_email_compose(input) {
        return EmailIntent::Compose(input.to_string());
    }
    
    if is_email_query(input) {
        return parse_email_query(input);
    }
    
    EmailIntent::None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calendar_intent_detection() {
        // Should detect as create
        assert!(is_calendar_intent("schedule a meeting with John tomorrow at 2pm"));
        assert!(is_calendar_intent("book lunch with Sarah on Friday"));
        assert!(is_calendar_intent("set up a call with the team next week"));
        assert!(is_calendar_intent("lunch with Charlie tomorrow at noon"));
        assert!(is_calendar_intent("meeting with Bob at 3pm"));
        assert!(is_calendar_intent("coffee with Alice on Monday morning"));
        
        // Should not detect as create
        assert!(!is_calendar_intent("what's on my calendar?"));
        assert!(!is_calendar_intent("who is John?"));
        assert!(!is_calendar_intent("/gcal create meeting"));
        assert!(!is_calendar_intent("tell me about tomorrow"));
    }

    #[test]
    fn test_calendar_query_detection() {
        // Should detect as query
        assert!(is_calendar_query("what's on my calendar tomorrow?"));
        assert!(is_calendar_query("what events do I have coming up?"));
        assert!(is_calendar_query("show my schedule for this week"));
        assert!(is_calendar_query("am I free tomorrow afternoon?"));
        assert!(is_calendar_query("do I have any meetings today?"));
        assert!(is_calendar_query("what do I have on Monday?"));
        
        // Should not detect as query
        assert!(!is_calendar_query("schedule lunch tomorrow"));
        assert!(!is_calendar_query("who is Bob?"));
        assert!(!is_calendar_query("/gcal today"));
    }

    #[test]
    fn test_calendar_query_parsing() {
        assert_eq!(parse_calendar_query("what's on today?"), CalendarIntent::QueryToday);
        assert_eq!(parse_calendar_query("what events tomorrow?"), CalendarIntent::QueryTomorrow);
        assert_eq!(parse_calendar_query("show me this week"), CalendarIntent::QueryWeek);
        assert_eq!(parse_calendar_query("what's coming up?"), CalendarIntent::QueryUpcoming);
    }
}

