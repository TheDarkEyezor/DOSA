// Integration tests for calendar and date parsing features

use chrono::{Local, Duration, Datelike, Weekday};

// ============================================
// Date Parsing Tests
// ============================================

#[test]
fn test_parse_relative_dates() {
    use samantha::calendar::parsing::parse_datetime;
    
    let now = Local::now();
    
    // Test "tomorrow"
    let result = parse_datetime("tomorrow at 2pm");
    assert!(result.is_ok(), "Should parse 'tomorrow at 2pm'");
    let parsed = result.unwrap();
    assert!(parsed.start.date_naive() > now.date_naive(), 
        "Tomorrow should be after today");
    
    // Test "next week"
    let result = parse_datetime("next monday");
    assert!(result.is_ok(), "Should parse 'next monday'");
    
    println!("✅ Relative date parsing tests passed");
}

#[test]
fn test_parse_special_patterns() {
    use samantha::calendar::parsing::parse_datetime;
    
    // Test "end of month"
    let result = parse_datetime("end of month");
    assert!(result.is_ok(), "Should parse 'end of month'");
    let parsed = result.unwrap();
    
    // Should be near end of current month
    let now = Local::now();
    let days_in_month = get_days_in_month(now.year(), now.month());
    assert!(parsed.start.day() >= days_in_month - 3, 
        "End of month should be near month end");
    
    // Test "end of week"
    let result = parse_datetime("end of week");
    assert!(result.is_ok(), "Should parse 'end of week'");
    let parsed = result.unwrap();
    
    // End of week should be Friday
    assert_eq!(parsed.start.weekday(), Weekday::Fri, 
        "End of week should be Friday");
    
    println!("✅ Special pattern parsing tests passed");
}

// ============================================
// Recurrence Parsing Tests
// ============================================

#[test]
fn test_parse_recurrence_rules() {
    use samantha::calendar::parsing::parse_recurrence;
    
    // Test daily
    let result = parse_recurrence("daily");
    assert!(result.is_ok(), "Should parse 'daily'");
    if let Ok(Some(rule)) = result {
        assert!(rule.rrule.contains("FREQ=DAILY"), "Should have DAILY frequency");
    }
    
    // Test weekly
    let result = parse_recurrence("weekly");
    assert!(result.is_ok(), "Should parse 'weekly'");
    if let Ok(Some(rule)) = result {
        assert!(rule.rrule.contains("FREQ=WEEKLY"), "Should have WEEKLY frequency");
    }
    
    println!("✅ Recurrence parsing tests passed");
}

#[test]
fn test_parse_weekday_recurrence() {
    use samantha::calendar::parsing::parse_recurrence;
    
    // Test "every Monday"
    let result = parse_recurrence("every Monday");
    assert!(result.is_ok(), "Should parse 'every Monday'");
    if let Ok(Some(rule)) = result {
        assert!(rule.rrule.contains("BYDAY=MO"), "Should have BYDAY=MO");
    }
    
    println!("✅ Weekday recurrence parsing tests passed");
}

// ============================================
// Calendar Event Builder Tests
// ============================================

#[test]
fn test_new_calendar_event_builder() {
    use samantha::integrations::google::calendar::NewCalendarEvent;
    use chrono::Local;
    
    let start = Local::now();
    let end = start + Duration::hours(1);
    
    let event = NewCalendarEvent::new("Test Meeting", start, end)
        .with_description("A test meeting")
        .with_location("Conference Room A");
    
    assert_eq!(event.title, "Test Meeting");
    assert_eq!(event.description.as_deref(), Some("A test meeting"));
    assert_eq!(event.location.as_deref(), Some("Conference Room A"));
    
    println!("✅ Calendar event builder tests passed");
}

#[test]
fn test_event_with_recurrence() {
    use samantha::integrations::google::calendar::NewCalendarEvent;
    use chrono::Local;
    
    let start = Local::now();
    let end = start + Duration::hours(1);
    
    let event = NewCalendarEvent::new("Weekly Standup", start, end)
        .with_recurrence("RRULE:FREQ=WEEKLY;BYDAY=MO");
    
    assert!(event.recurrence.is_some(), "Should have recurrence");
    assert!(event.recurrence.unwrap().contains("WEEKLY"), 
        "Should have weekly recurrence");
    
    println!("✅ Event with recurrence tests passed");
}

#[test]
fn test_event_with_attendees() {
    use samantha::integrations::google::calendar::NewCalendarEvent;
    use chrono::Local;
    
    let start = Local::now();
    let end = start + Duration::hours(1);
    
    let event = NewCalendarEvent::new("Team Meeting", start, end)
        .with_attendees(vec!["alice@example.com".to_string(), "bob@example.com".to_string()]);
    
    assert_eq!(event.attendees.len(), 2, "Should have 2 attendees");
    assert!(event.attendees.contains(&"alice@example.com".to_string()));
    assert!(event.attendees.contains(&"bob@example.com".to_string()));
    
    println!("✅ Event with attendees tests passed");
}

// ============================================
// Helper Functions
// ============================================

fn get_days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0) {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}
