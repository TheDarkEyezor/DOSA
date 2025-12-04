// Integration tests for intent detection and classification

use samantha::intent::keyword::KeywordClassifier;
use samantha::intent::types::*;
use samantha::intent::{IntentClassifier, IntentContext};

// ============================================
// Email Conversation/Threading Tests
// ============================================

#[test]
fn test_conversation_with_detection() {
    let classifier = KeywordClassifier::new();
    
    let test_cases = vec![
        ("summarize my conversation with Sarah", true),
        ("summarize conversation with John", true),
        ("my emails with Alice", true),
        ("email thread with Bob", true),
        ("discussion with Charlie", true),
        ("correspondence with Dave", true),
        ("emails between me and Eve", true),
        // Negative cases
        ("send email to John", false),
        ("check my emails", false),
        ("what emails do I have", false),
    ];
    
    for (input, should_match) in test_cases {
        let candidates = classifier.get_candidates(input);
        let is_conversation = candidates.iter().any(|(intent, _)| {
            matches!(intent, Intent::EmailQuery { query_type: EmailQueryType::ConversationWith(_) })
        });
        
        assert_eq!(is_conversation, should_match, 
            "Input '{}' should {} match conversation pattern", 
            input, 
            if should_match { "" } else { "not" }
        );
    }
    
    println!("✅ Conversation detection tests passed");
}

#[test]
fn test_conversation_person_extraction() {
    let classifier = KeywordClassifier::new();
    
    let test_cases = vec![
        ("summarize my conversation with Sarah", "sarah"),
        ("conversation with John Smith", "john smith"),
        // Email addresses are part of the extracted name, not separate
        ("my emails with alice", "alice"),
    ];
    
    for (input, expected_person) in test_cases {
        let candidates = classifier.get_candidates(input);
        let person = candidates.iter()
            .find_map(|(intent, _)| {
                if let Intent::EmailQuery { query_type: EmailQueryType::ConversationWith(p) } = intent {
                    Some(p.clone())
                } else {
                    None
                }
            });
        
        assert!(person.is_some(), "Should extract person from '{}'", input);
        assert!(person.unwrap().to_lowercase().contains(expected_person), 
            "Extracted person should contain '{}'", expected_person);
    }
    
    println!("✅ Conversation person extraction tests passed");
}

// ============================================
// Calendar Intent Tests
// ============================================

#[test]
fn test_calendar_create_detection() {
    let classifier = KeywordClassifier::new();
    
    let test_cases = vec![
        ("schedule lunch with John tomorrow", true),
        ("create a meeting on Friday", true),
        ("add dinner to my calendar", true),
        ("book a call with Sarah", true),
        ("set up an appointment for next week", true),
        // Negative cases - these should NOT trigger create
        ("what's on my calendar tomorrow", false),
        ("show my meetings", false),
        ("cancel the lunch", false),
    ];
    
    for (input, should_match) in test_cases {
        let candidates = classifier.get_candidates(input);
        let is_create = candidates.iter().any(|(intent, _)| {
            matches!(intent, Intent::CalendarCreate { .. })
        });
        
        assert_eq!(is_create, should_match,
            "Input '{}' should {} match calendar create", 
            input, 
            if should_match { "" } else { "not" }
        );
    }
    
    println!("✅ Calendar create detection tests passed");
}

#[test]
fn test_calendar_query_types() {
    let classifier = KeywordClassifier::new();
    
    // Test today query
    let candidates = classifier.get_candidates("what's on my calendar today");
    let has_today = candidates.iter().any(|(intent, _)| {
        matches!(intent, Intent::CalendarQuery { query_type: CalendarQueryType::Today })
    });
    assert!(has_today, "Should detect 'today' calendar query");
    
    // Test tomorrow query
    let candidates = classifier.get_candidates("what do I have tomorrow");
    let has_tomorrow = candidates.iter().any(|(intent, _)| {
        matches!(intent, Intent::CalendarQuery { query_type: CalendarQueryType::Tomorrow })
    });
    assert!(has_tomorrow, "Should detect 'tomorrow' calendar query");
    
    // Test week query
    let candidates = classifier.get_candidates("show my schedule this week");
    let has_week = candidates.iter().any(|(intent, _)| {
        matches!(intent, Intent::CalendarQuery { query_type: CalendarQueryType::Week })
    });
    assert!(has_week, "Should detect 'week' calendar query");
    
    // Test person query
    let candidates = classifier.get_candidates("meetings with John");
    let has_person = candidates.iter().any(|(intent, _)| {
        matches!(intent, Intent::CalendarQuery { query_type: CalendarQueryType::WithPerson(_) })
    });
    assert!(has_person, "Should detect 'with person' calendar query");
    
    println!("✅ Calendar query type detection tests passed");
}

#[test]
fn test_calendar_update_detection() {
    let classifier = KeywordClassifier::new();
    
    let test_cases = vec![
        ("reschedule the meeting to 3pm", true),
        ("move lunch to Friday", true),
        ("change the call time to 2pm", true),
        ("push the standup to 11am", true),
        // Negative - emailing about changes should NOT trigger update
        ("email attendees about the time change", false),
        ("let them know the meeting moved", false),
    ];
    
    for (input, should_match) in test_cases {
        let candidates = classifier.get_candidates(input);
        let is_update = candidates.iter().any(|(intent, _)| {
            matches!(intent, Intent::CalendarUpdate { .. })
        });
        
        assert_eq!(is_update, should_match,
            "Input '{}' should {} match calendar update", 
            input, 
            if should_match { "" } else { "not" }
        );
    }
    
    println!("✅ Calendar update detection tests passed");
}

// ============================================
// Email Intent Tests
// ============================================

#[test]
fn test_email_compose_detection() {
    let classifier = KeywordClassifier::new();
    
    let test_cases = vec![
        ("send an email to John", true),
        ("write an email to Sarah about the project", true),
        ("email saying I'll be late", true),  // Direct pattern without name in between
        ("compose email to the team", true),
        ("draft an email to Bob", true),
        // Negative cases
        ("check my emails", false),
        ("reply to John's email", false),
    ];
    
    for (input, should_match) in test_cases {
        let candidates = classifier.get_candidates(input);
        let is_compose = candidates.iter().any(|(intent, _)| {
            matches!(intent, Intent::EmailCompose { .. })
        });
        
        assert_eq!(is_compose, should_match,
            "Input '{}' should {} match email compose", 
            input, 
            if should_match { "" } else { "not" }
        );
    }
    
    println!("✅ Email compose detection tests passed");
}

#[test]
fn test_email_attendees_detection() {
    let classifier = KeywordClassifier::new();
    
    let test_cases = vec![
        ("email the attendees of tomorrow's dinner", true),
        ("notify participants about the meeting change", true),
        ("let them know about the venue change", true),
        ("message everyone in the meeting", true),
    ];
    
    for (input, should_match) in test_cases {
        let candidates = classifier.get_candidates(input);
        let is_attendees = candidates.iter().any(|(intent, score)| {
            matches!(intent, Intent::EmailAttendees { .. }) && *score > 0.5
        });
        
        assert_eq!(is_attendees, should_match,
            "Input '{}' should {} match email attendees", 
            input, 
            if should_match { "" } else { "not" }
        );
    }
    
    println!("✅ Email attendees detection tests passed");
}

#[test]
fn test_email_reply_detection() {
    let classifier = KeywordClassifier::new();
    
    let test_cases = vec![
        ("reply to John's email", true),
        ("respond to Sarah's last message", true),
        ("answer Bob's email", true),
        ("write back to the last email", true),
    ];
    
    for (input, should_match) in test_cases {
        let candidates = classifier.get_candidates(input);
        let is_reply = candidates.iter().any(|(intent, score)| {
            matches!(intent, Intent::EmailReply { .. }) && *score > 0.5
        });
        
        assert_eq!(is_reply, should_match,
            "Input '{}' should {} match email reply", 
            input, 
            if should_match { "" } else { "not" }
        );
    }
    
    println!("✅ Email reply detection tests passed");
}

// ============================================
// Contact Update Tests
// ============================================

#[test]
fn test_contact_update_detection() {
    let classifier = KeywordClassifier::new();
    
    let test_cases = vec![
        ("add John, he works at Google", true),
        ("Sarah's email is sarah@example.com", true),
        ("remember that Bob works at Meta", true),
        ("add a contact for Alice", true),
        // Negative cases
        ("schedule lunch with John", false),
        ("email Sarah", false),
    ];
    
    for (input, should_match) in test_cases {
        let candidates = classifier.get_candidates(input);
        let is_contact = candidates.iter().any(|(intent, score)| {
            matches!(intent, Intent::ContactUpdate { .. }) && *score > 0.4
        });
        
        assert_eq!(is_contact, should_match,
            "Input '{}' should {} match contact update", 
            input, 
            if should_match { "" } else { "not" }
        );
    }
    
    println!("✅ Contact update detection tests passed");
}

// ============================================
// Web Search vs Knowledge Query Tests
// ============================================

#[test]
fn test_web_search_detection() {
    let classifier = KeywordClassifier::new();
    
    let test_cases = vec![
        ("what's the weather today", true),
        ("search for latest AI news", true),
        ("who is the CEO of Apple", true),
        ("stock price of Tesla", true),
        ("how to make pasta", true),
        // Negative - should be knowledge query, not web search
        ("who do I know at Google", false),
        ("what's on my calendar", false),
    ];
    
    for (input, should_match) in test_cases {
        let candidates = classifier.get_candidates(input);
        let is_web = candidates.iter().any(|(intent, score)| {
            matches!(intent, Intent::WebSearch { .. }) && *score > 0.5
        });
        
        assert_eq!(is_web, should_match,
            "Input '{}' should {} match web search", 
            input, 
            if should_match { "" } else { "not" }
        );
    }
    
    println!("✅ Web search detection tests passed");
}

#[test]
fn test_knowledge_query_detection() {
    let classifier = KeywordClassifier::new();
    
    let test_cases = vec![
        ("who do I know at Google", true),
        ("tell me about John Smith", true),
        ("who knows Python", true),  // Uses "knows" pattern
        ("who works with Bob", true),
        ("find someone who knows Rust", true),  // Uses "find someone" + "knows" patterns
        // Negative - should be web search, not knowledge
        ("what's the weather", false),
        // Note: "who is X" triggers KG query - famous person distinction requires more context
    ];
    
    for (input, should_match) in test_cases {
        let candidates = classifier.get_candidates(input);
        let is_kg = candidates.iter().any(|(intent, score)| {
            matches!(intent, Intent::KnowledgeQuery { .. }) && *score > 0.5
        });
        
        assert_eq!(is_kg, should_match,
            "Input '{}' should {} match knowledge query", 
            input, 
            if should_match { "" } else { "not" }
        );
    }
    
    println!("✅ Knowledge query detection tests passed");
}

// ============================================
// Intent Result Tests
// ============================================

#[test]
fn test_intent_result_confidence() {
    let result = IntentResult::new(
        Intent::CalendarCreate { description: "lunch".to_string() },
        0.8,
        "schedule lunch"
    );
    
    assert!(result.is_confident(), "0.8 confidence should be confident");
    
    let low_confidence = IntentResult::new(
        Intent::Conversation { input: "hello".to_string() },
        0.3,
        "hello"
    );
    
    assert!(!low_confidence.is_confident(), "0.3 confidence should not be confident");
    
    println!("✅ Intent confidence tests passed");
}

#[test]
fn test_intent_result_ambiguity() {
    let mut result = IntentResult::new(
        Intent::CalendarCreate { description: "lunch".to_string() },
        0.7,
        "add lunch"
    );
    
    // No alternatives - not ambiguous
    assert!(!result.is_ambiguous(), "Should not be ambiguous without alternatives");
    
    // Add a high-confidence alternative
    result = result.with_alternatives(vec![
        (Intent::ContactUpdate { description: "add lunch".to_string() }, 0.6)
    ]);
    
    assert!(result.is_ambiguous(), "Should be ambiguous with high-confidence alternative");
    
    println!("✅ Intent ambiguity tests passed");
}

// ============================================
// Command Detection Tests
// ============================================

#[tokio::test]
async fn test_command_detection() {
    let classifier = KeywordClassifier::new();
    let ctx = IntentContext::new(true);
    
    let commands = vec!["/help", "/gcal", "/email", "/kg"];
    
    for cmd in commands {
        let result = classifier.classify(cmd, &ctx).await.unwrap();
        assert!(matches!(result.intent, Intent::Command(_)), 
            "/{} should be detected as a command", cmd);
        assert_eq!(result.confidence, 1.0, "Commands should have confidence 1.0");
    }
    
    println!("✅ Command detection tests passed");
}
