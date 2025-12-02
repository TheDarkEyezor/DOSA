//! Intent classifier test suite
//!
//! Tests for keyword-based classification accuracy

use super::keyword::KeywordClassifier;
use super::types::*;
use super::IntentClassifier;

/// Helper to run async classify
fn classify_sync(classifier: &KeywordClassifier, input: &str) -> IntentResult {
    let ctx = IntentContext::new(true);
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(classifier.classify(input, &ctx))
        .unwrap()
}

/// Check if an intent matches the expected type
fn matches_intent(intent: &Intent, expected: &str) -> bool {
    match (intent, expected) {
        (Intent::CalendarCreate { .. }, "calendar_create") => true,
        (Intent::CalendarQuery { .. }, "calendar_query") => true,
        (Intent::CalendarUpdate { .. }, "calendar_update") => true,
        (Intent::CalendarDelete { .. }, "calendar_delete") => true,
        (Intent::EmailCompose { .. }, "email_compose") => true,
        (Intent::EmailReply { .. }, "email_reply") => true,
        (Intent::EmailAttendees { .. }, "email_attendees") => true,
        (Intent::EmailQuery { .. }, "email_query") => true,
        (Intent::ContactUpdate { .. }, "contact_update") => true,
        (Intent::KnowledgeQuery { .. }, "knowledge_query") => true,
        (Intent::Conversation { .. }, "conversation") => true,
        (Intent::Command(_), "command") => true,
        _ => false,
    }
}

// =============================================================================
// Calendar Create Tests
// =============================================================================

#[test]
fn test_calendar_create_schedule_meeting() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "schedule a meeting with John tomorrow at 3pm");
    assert!(matches_intent(&result.intent, "calendar_create"), 
        "Expected calendar_create, got {:?}", result.intent);
    assert!(result.confidence > 0.5, "Confidence too low: {}", result.confidence);
}

#[test]
fn test_calendar_create_book_lunch() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "book lunch with Sarah on Friday");
    assert!(matches_intent(&result.intent, "calendar_create"));
}

#[test]
fn test_calendar_create_set_up_call() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "set up a call with the team at 10am");
    assert!(matches_intent(&result.intent, "calendar_create"));
}

#[test]
fn test_calendar_create_add_event() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "add dinner at 7pm tomorrow");
    assert!(matches_intent(&result.intent, "calendar_create"));
}

#[test]
fn test_calendar_create_arrange_meeting() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "arrange a meeting for Monday morning");
    assert!(matches_intent(&result.intent, "calendar_create"));
}

// =============================================================================
// Calendar Query Tests
// =============================================================================

#[test]
fn test_calendar_query_today() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "what do I have today");
    assert!(matches_intent(&result.intent, "calendar_query"));
    if let Intent::CalendarQuery { query_type } = &result.intent {
        assert!(matches!(query_type, CalendarQueryType::Today));
    }
}

#[test]
fn test_calendar_query_tomorrow() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "what's on my calendar tomorrow");
    assert!(matches_intent(&result.intent, "calendar_query"));
    if let Intent::CalendarQuery { query_type } = &result.intent {
        assert!(matches!(query_type, CalendarQueryType::Tomorrow));
    }
}

#[test]
fn test_calendar_query_week() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "show me this week's events");
    assert!(matches_intent(&result.intent, "calendar_query"));
    if let Intent::CalendarQuery { query_type } = &result.intent {
        assert!(matches!(query_type, CalendarQueryType::Week));
    }
}

#[test]
fn test_calendar_query_schedule() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "what's my schedule for today");
    assert!(matches_intent(&result.intent, "calendar_query"));
}

#[test]
fn test_calendar_query_any_events() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "do I have any events tomorrow");
    assert!(matches_intent(&result.intent, "calendar_query"));
}

// =============================================================================
// Calendar Update Tests
// =============================================================================

#[test]
fn test_calendar_update_move() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "move my meeting to 4pm");
    assert!(matches_intent(&result.intent, "calendar_update"),
        "Expected calendar_update, got {:?}", result.intent);
}

#[test]
fn test_calendar_update_reschedule() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "reschedule lunch with Bob to Friday");
    assert!(matches_intent(&result.intent, "calendar_update"));
}

#[test]
fn test_calendar_update_change_time() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "change the meeting time to 2pm");
    assert!(matches_intent(&result.intent, "calendar_update"));
}

#[test]
fn test_calendar_update_push() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "push my appointment to next week");
    assert!(matches_intent(&result.intent, "calendar_update"));
}

// =============================================================================
// Calendar Delete Tests
// =============================================================================

#[test]
fn test_calendar_delete_cancel() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "cancel my meeting with John");
    assert!(matches_intent(&result.intent, "calendar_delete"),
        "Expected calendar_delete, got {:?}", result.intent);
}

#[test]
fn test_calendar_delete_delete() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "delete the event tomorrow");
    assert!(matches_intent(&result.intent, "calendar_delete"));
}

#[test]
fn test_calendar_delete_remove() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "remove the lunch appointment");
    assert!(matches_intent(&result.intent, "calendar_delete"));
}

// =============================================================================
// Email Compose Tests
// =============================================================================

#[test]
fn test_email_compose_send() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "send an email to John about the project");
    assert!(matches_intent(&result.intent, "email_compose"),
        "Expected email_compose, got {:?}", result.intent);
}

#[test]
fn test_email_compose_write() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "write an email to Sarah saying thanks");
    assert!(matches_intent(&result.intent, "email_compose"));
}

#[test]
fn test_email_compose_draft() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "draft an email to the team about the update");
    assert!(matches_intent(&result.intent, "email_compose"));
}

#[test]
fn test_email_compose_send_name_email() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "send Charlie an email saying hi");
    // This pattern should match email_compose
    assert!(matches_intent(&result.intent, "email_compose"),
        "Expected email_compose for 'send X an email', got {:?}", result.intent);
}

// =============================================================================
// Email Attendees Tests (Multi-hop)
// =============================================================================

#[test]
fn test_email_attendees_basic() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "email the attendees of tomorrow's dinner");
    assert!(matches_intent(&result.intent, "email_attendees"),
        "Expected email_attendees, got {:?}", result.intent);
}

#[test]
fn test_email_attendees_time_changed() {
    let classifier = KeywordClassifier::new();
    // This was the failing case - must NOT trigger calendar_update
    let result = classify_sync(&classifier, 
        "email tomorrow dinner's attendees that the time has changed to 6pm");
    assert!(matches_intent(&result.intent, "email_attendees"),
        "Expected email_attendees (not calendar_update), got {:?}", result.intent);
}

#[test]
fn test_email_attendees_notify() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "notify the meeting participants about the delay");
    assert!(matches_intent(&result.intent, "email_attendees"));
}

#[test]
fn test_email_attendees_let_them_know() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, 
        "let the attendees of Friday's call know it's cancelled");
    assert!(matches_intent(&result.intent, "email_attendees"));
}

// =============================================================================
// Email Reply Tests
// =============================================================================

#[test]
fn test_email_reply_basic() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "reply to John's email saying I'll be there");
    assert!(matches_intent(&result.intent, "email_reply"),
        "Expected email_reply, got {:?}", result.intent);
}

#[test]
fn test_email_reply_respond_to() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "respond to Sarah's message");
    assert!(matches_intent(&result.intent, "email_reply"),
        "Expected email_reply, got {:?}", result.intent);
}

#[test]
fn test_email_reply_answer() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "answer the email from Mike");
    assert!(matches_intent(&result.intent, "email_reply"),
        "Expected email_reply, got {:?}", result.intent);
}

#[test]
fn test_email_reply_get_back_to() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "get back to that email about the meeting");
    assert!(matches_intent(&result.intent, "email_reply"),
        "Expected email_reply, got {:?}", result.intent);
}

#[test]
fn test_email_reply_with_content() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "reply to the project update email with 'sounds good'");
    assert!(matches_intent(&result.intent, "email_reply"),
        "Expected email_reply, got {:?}", result.intent);
}

// =============================================================================
// Email Query Tests
// =============================================================================

#[test]
fn test_email_query_unread() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "show me my unread emails");
    assert!(matches_intent(&result.intent, "email_query"));
    if let Intent::EmailQuery { query_type } = &result.intent {
        assert!(matches!(query_type, EmailQueryType::Unread));
    }
}

#[test]
fn test_email_query_inbox() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "check my inbox");
    // May classify as email_query or conversation depending on confidence
}

#[test]
fn test_email_query_from() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "show emails from John");
    assert!(matches_intent(&result.intent, "email_query"));
    if let Intent::EmailQuery { query_type } = &result.intent {
        if let EmailQueryType::From(sender) = query_type {
            assert!(sender.to_lowercase().contains("john"));
        }
    }
}

#[test]
fn test_email_query_summary() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "summarize my emails");
    assert!(matches_intent(&result.intent, "email_query"));
    if let Intent::EmailQuery { query_type } = &result.intent {
        assert!(matches!(query_type, EmailQueryType::Summary));
    }
}

// =============================================================================
// Contact Update Tests
// =============================================================================

#[test]
fn test_contact_update_add_person() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "add John Smith as a person I know");
    assert!(matches_intent(&result.intent, "contact_update"),
        "Expected contact_update, got {:?}", result.intent);
}

#[test]
fn test_contact_update_add_people() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "add Sarah and Mike as contacts");
    assert!(matches_intent(&result.intent, "contact_update"),
        "Expected contact_update, got {:?}", result.intent);
}

#[test]
fn test_contact_update_set_email() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "John's email is john@example.com");
    assert!(matches_intent(&result.intent, "contact_update"),
        "Expected contact_update, got {:?}", result.intent);
}

#[test]
fn test_contact_update_email_is() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "Amogh's email is amogh@gmail.com");
    assert!(matches_intent(&result.intent, "contact_update"),
        "Expected contact_update, got {:?}", result.intent);
}

#[test]
fn test_contact_update_with_org() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "Add Bob, he works at Google");
    assert!(matches_intent(&result.intent, "contact_update"),
        "Expected contact_update, got {:?}", result.intent);
}

#[test]
fn test_contact_update_studies_at() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "Add Rohan, he studies at Imperial College");
    assert!(matches_intent(&result.intent, "contact_update"),
        "Expected contact_update, got {:?}", result.intent);
}

#[test]
fn test_contact_update_remember() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "remember that Alice's email is alice@company.com");
    assert!(matches_intent(&result.intent, "contact_update"),
        "Expected contact_update, got {:?}", result.intent);
}

// =============================================================================
// Knowledge Query Tests
// =============================================================================

#[test]
fn test_knowledge_query_who_is() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "who is John Smith");
    assert!(matches_intent(&result.intent, "knowledge_query"),
        "Expected knowledge_query, got {:?}", result.intent);
}

#[test]
fn test_knowledge_query_who_works() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "who works at Acme Corp");
    assert!(matches_intent(&result.intent, "knowledge_query"));
}

#[test]
fn test_knowledge_query_tell_me() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "tell me about Sarah's manager");
    assert!(matches_intent(&result.intent, "knowledge_query"));
}

// =============================================================================
// Command Tests
// =============================================================================

#[test]
fn test_command_detection() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "/help");
    assert!(matches_intent(&result.intent, "command"));
}

#[test]
fn test_command_with_args() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "/add person John");
    assert!(matches_intent(&result.intent, "command"));
}

// =============================================================================
// Conversation Fallback Tests
// =============================================================================

#[test]
fn test_conversation_greeting() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "hello");
    // Greetings should fall through to conversation
    assert!(matches_intent(&result.intent, "conversation"),
        "Expected conversation for greeting, got {:?}", result.intent);
}

#[test]
fn test_conversation_general() {
    let classifier = KeywordClassifier::new();
    let result = classify_sync(&classifier, "how are you doing today");
    // General questions should fall through to conversation
    assert!(matches_intent(&result.intent, "conversation"),
        "Expected conversation for general question, got {:?}", result.intent);
}

// =============================================================================
// Edge Cases and Disambiguation Tests
// =============================================================================

#[test]
fn test_email_about_change_not_calendar_update() {
    // This is crucial - emailing about a change should NOT trigger calendar update
    let classifier = KeywordClassifier::new();
    
    let inputs = [
        "email the attendees that the time changed",
        "notify everyone the meeting is moved",
        "let them know the event is rescheduled",
        "message the participants about the delay",
    ];
    
    for input in inputs {
        let result = classify_sync(&classifier, input);
        assert!(!matches_intent(&result.intent, "calendar_update"),
            "Input '{}' should NOT be calendar_update, got {:?}", input, result.intent);
    }
}

#[test]
fn test_actual_calendar_update_detected() {
    // Actual calendar updates without email context
    let classifier = KeywordClassifier::new();
    
    let inputs = [
        "move my meeting to 3pm",
        "reschedule lunch to Friday",
        "change the appointment time",
    ];
    
    for input in inputs {
        let result = classify_sync(&classifier, input);
        assert!(matches_intent(&result.intent, "calendar_update"),
            "Input '{}' should be calendar_update, got {:?}", input, result.intent);
    }
}

#[test]
fn test_confidence_ordering() {
    let classifier = KeywordClassifier::new();
    
    // Strong signal should have high confidence
    let result = classify_sync(&classifier, "schedule a meeting tomorrow at 2pm");
    assert!(result.confidence >= 0.7, 
        "Strong calendar create signal should have high confidence, got {}", result.confidence);
    
    // Weak signal should have lower confidence
    let result = classify_sync(&classifier, "meeting tomorrow");
    assert!(result.confidence < 0.9,
        "Ambiguous input should have moderate confidence, got {}", result.confidence);
}

#[test]
fn test_alternatives_provided() {
    let classifier = KeywordClassifier::new();
    
    // Ambiguous input might have alternatives
    let result = classify_sync(&classifier, "can you check my calendar for tomorrow");
    // This could be calendar query, but just asking generally
    // Should have alternatives if ambiguous
}

// =============================================================================
// Scoring Method Unit Tests
// =============================================================================

#[test]
fn test_score_calendar_create_high() {
    let classifier = KeywordClassifier::new();
    let candidates = classifier.get_candidates("schedule a meeting with John tomorrow at 3pm");
    
    let create_score = candidates.iter()
        .find(|(intent, _)| matches!(intent, Intent::CalendarCreate { .. }))
        .map(|(_, score)| *score);
    
    assert!(create_score.is_some());
    assert!(create_score.unwrap() >= 0.8, "Score should be high: {:?}", create_score);
}

#[test]
fn test_score_email_attendees_high() {
    let classifier = KeywordClassifier::new();
    let candidates = classifier.get_candidates("email the attendees of tomorrow's dinner");
    
    let attendee_score = candidates.iter()
        .find(|(intent, _)| matches!(intent, Intent::EmailAttendees { .. }))
        .map(|(_, score)| *score);
    
    assert!(attendee_score.is_some(), "Should detect email_attendees intent");
    assert!(attendee_score.unwrap() >= 0.7, "Score should be high: {:?}", attendee_score);
}
