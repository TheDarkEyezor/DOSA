// Integration tests for email features

use chrono::Utc;

// ============================================
// Email Summary Tests
// ============================================

#[test]
fn test_email_summary_display_short() {
    use samantha::integrations::google::gmail::EmailSummary;
    
    let email = EmailSummary {
        id: "123".to_string(),
        thread_id: "thread123".to_string(),
        from: "John Doe <john@example.com>".to_string(),
        to: Some("me@example.com".to_string()),
        subject: "Test Subject".to_string(),
        snippet: "This is a preview of the email content...".to_string(),
        date: Utc::now(),
        is_unread: true,
        labels: vec!["INBOX".to_string()],
    };
    
    let display = email.display_short();
    assert!(display.contains("●"), "Unread email should have ● marker");
    assert!(display.contains("John Doe"), "Should show sender name");
    assert!(display.contains("Test Subject"), "Should show subject");
    
    println!("✅ Email summary display tests passed");
}

#[test]
fn test_email_from_short_extraction() {
    use samantha::integrations::google::gmail::EmailSummary;
    
    // Test "Name <email>" format
    let email1 = EmailSummary {
        id: "1".to_string(),
        thread_id: "t1".to_string(),
        from: "John Doe <john@example.com>".to_string(),
        to: None,
        subject: "Test".to_string(),
        snippet: "".to_string(),
        date: Utc::now(),
        is_unread: false,
        labels: vec![],
    };
    assert_eq!(email1.from_short(), "John Doe");
    
    // Test email-only format
    let email2 = EmailSummary {
        id: "2".to_string(),
        thread_id: "t2".to_string(),
        from: "sarah@company.com".to_string(),
        to: None,
        subject: "Test".to_string(),
        snippet: "".to_string(),
        date: Utc::now(),
        is_unread: false,
        labels: vec![],
    };
    assert_eq!(email2.from_short(), "sarah");
    
    println!("✅ Email from_short extraction tests passed");
}

#[test]
fn test_email_unread_marker() {
    use samantha::integrations::google::gmail::EmailSummary;
    
    let unread = EmailSummary {
        id: "1".to_string(),
        thread_id: "t1".to_string(),
        from: "test@example.com".to_string(),
        to: None,
        subject: "Unread Test".to_string(),
        snippet: "".to_string(),
        date: Utc::now(),
        is_unread: true,
        labels: vec!["UNREAD".to_string()],
    };
    
    let read = EmailSummary {
        id: "2".to_string(),
        thread_id: "t2".to_string(),
        from: "test@example.com".to_string(),
        to: None,
        subject: "Read Test".to_string(),
        snippet: "".to_string(),
        date: Utc::now(),
        is_unread: false,
        labels: vec![],
    };
    
    let unread_display = unread.display_short();
    let read_display = read.display_short();
    
    assert!(unread_display.contains("●"), "Unread should have filled marker");
    assert!(read_display.contains("○"), "Read should have empty marker");
    
    println!("✅ Email unread marker tests passed");
}

// ============================================
// Email Thread Tests
// ============================================

#[test]
fn test_email_thread_structure() {
    use samantha::integrations::google::gmail::{Email, EmailThread};
    
    let email1 = Email {
        id: "msg1".to_string(),
        thread_id: "thread1".to_string(),
        from: "alice@example.com".to_string(),
        to: vec!["bob@example.com".to_string()],
        cc: vec![],
        subject: "Project Discussion".to_string(),
        body_text: Some("Hi Bob, let's discuss the project.".to_string()),
        body_html: None,
        date: Utc::now() - chrono::Duration::hours(2),
        is_unread: false,
        labels: vec![],
    };
    
    let email2 = Email {
        id: "msg2".to_string(),
        thread_id: "thread1".to_string(),
        from: "bob@example.com".to_string(),
        to: vec!["alice@example.com".to_string()],
        cc: vec![],
        subject: "Re: Project Discussion".to_string(),
        body_text: Some("Sure, when works for you?".to_string()),
        body_html: None,
        date: Utc::now() - chrono::Duration::hours(1),
        is_unread: true,
        labels: vec!["UNREAD".to_string()],
    };
    
    let thread = EmailThread {
        id: "thread1".to_string(),
        subject: "Project Discussion".to_string(),
        participants: vec!["alice@example.com".to_string(), "bob@example.com".to_string()],
        messages: vec![email1, email2],
    };
    
    assert_eq!(thread.messages.len(), 2, "Thread should have 2 messages");
    assert_eq!(thread.participants.len(), 2, "Thread should have 2 participants");
    assert!(thread.subject.contains("Project"), "Subject should be preserved");
    
    println!("✅ Email thread structure tests passed");
}

#[test]
fn test_email_thread_display() {
    use samantha::integrations::google::gmail::{Email, EmailThread};
    
    let email = Email {
        id: "msg1".to_string(),
        thread_id: "thread1".to_string(),
        from: "alice@example.com".to_string(),
        to: vec!["bob@example.com".to_string()],
        cc: vec![],
        subject: "Test Thread".to_string(),
        body_text: Some("This is the message body.".to_string()),
        body_html: None,
        date: Utc::now(),
        is_unread: false,
        labels: vec![],
    };
    
    let thread = EmailThread {
        id: "thread1".to_string(),
        subject: "Test Thread".to_string(),
        participants: vec!["alice@example.com".to_string(), "bob@example.com".to_string()],
        messages: vec![email],
    };
    
    let display = thread.display();
    assert!(display.contains("Thread:"), "Should show thread header");
    assert!(display.contains("Test Thread"), "Should show subject");
    assert!(display.contains("alice@example.com"), "Should show participants");
    assert!(display.contains("Message 1"), "Should show message number");
    
    println!("✅ Email thread display tests passed");
}

// ============================================
// Email Draft Tests
// ============================================

#[test]
fn test_email_draft_creation() {
    use samantha::integrations::google::gmail::EmailDraft;
    
    let draft = EmailDraft::new(
        vec!["alice@example.com".to_string()],
        "Test Subject",
        "Test body content"
    );
    
    assert_eq!(draft.to.len(), 1, "Should have 1 recipient");
    assert_eq!(draft.subject, "Test Subject");
    assert_eq!(draft.body, "Test body content");
    
    println!("✅ Email draft creation tests passed");
}

#[test]
fn test_email_draft_with_cc() {
    use samantha::integrations::google::gmail::EmailDraft;
    
    let draft = EmailDraft::new(
        vec!["alice@example.com".to_string()],
        "Subject",
        "Body"
    ).with_cc(vec!["bob@example.com".to_string()]);
    
    assert_eq!(draft.cc.len(), 1, "Should have 1 CC recipient");
    assert!(draft.cc.contains(&"bob@example.com".to_string()));
    
    println!("✅ Email draft with CC tests passed");
}
