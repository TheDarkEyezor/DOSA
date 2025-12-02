//! Google Calendar API client
//!
//! Provides access to Google Calendar events for display, summarization, and creation.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Local, Utc, Duration, NaiveDate, NaiveTime, Datelike};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::integrations::oauth::{OAuthManager, Provider};

const CALENDAR_API_BASE: &str = "https://www.googleapis.com/calendar/v3";

/// Google Calendar event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCalEvent {
    pub id: String,
    pub summary: Option<String>,
    pub description: Option<String>,
    pub location: Option<String>,
    pub start: GCalDateTime,
    pub end: GCalDateTime,
    pub attendees: Option<Vec<GCalAttendee>>,
    pub organizer: Option<GCalOrganizer>,
    pub status: Option<String>,
    #[serde(rename = "htmlLink")]
    pub html_link: Option<String>,
    #[serde(rename = "hangoutLink")]
    pub hangout_link: Option<String>,
    #[serde(rename = "conferenceData")]
    pub conference_data: Option<ConferenceData>,
}

impl GCalEvent {
    /// Get the title (summary) or a default
    pub fn title(&self) -> &str {
        self.summary.as_deref().unwrap_or("(No title)")
    }

    /// Get the start time as DateTime<Local>
    pub fn start_time(&self) -> Option<DateTime<Local>> {
        self.start.to_local_datetime()
    }

    /// Get the end time as DateTime<Local>
    pub fn end_time(&self) -> Option<DateTime<Local>> {
        self.end.to_local_datetime()
    }

    /// Check if this is an all-day event
    pub fn is_all_day(&self) -> bool {
        self.start.date.is_some()
    }

    /// Get meeting link (Google Meet, Zoom, etc.)
    pub fn meeting_link(&self) -> Option<&str> {
        // Try hangout link first
        if let Some(ref link) = self.hangout_link {
            return Some(link);
        }
        // Then try conference data
        if let Some(ref conf) = self.conference_data {
            if let Some(ref entry_points) = conf.entry_points {
                for ep in entry_points {
                    if ep.entry_point_type == "video" {
                        return Some(&ep.uri);
                    }
                }
            }
        }
        None
    }

    /// Format for display
    pub fn display(&self) -> String {
        let mut output = String::new();

        // Time
        if self.is_all_day() {
            output.push_str("📅 All day - ");
        } else if let Some(start) = self.start_time() {
            output.push_str(&format!("📅 {} - ", start.format("%I:%M %p")));
        }

        // Title
        output.push_str(self.title());

        // Location
        if let Some(ref loc) = self.location {
            output.push_str(&format!("\n   📍 {}", loc));
        }

        // Meeting link
        if let Some(link) = self.meeting_link() {
            output.push_str(&format!("\n   🔗 {}", link));
        }

        // Attendees
        if let Some(ref attendees) = self.attendees {
            if !attendees.is_empty() {
                let names: Vec<_> = attendees.iter()
                    .filter_map(|a| a.display_name.as_ref().or(a.email.as_ref()))
                    .take(5)
                    .map(|s| s.as_str())
                    .collect();
                if !names.is_empty() {
                    output.push_str(&format!("\n   👥 {}", names.join(", ")));
                    if attendees.len() > 5 {
                        output.push_str(&format!(" +{} more", attendees.len() - 5));
                    }
                }
            }
        }

        output
    }

    /// Short display for lists
    pub fn display_short(&self) -> String {
        if self.is_all_day() {
            format!("All day - {}", self.title())
        } else if let Some(start) = self.start_time() {
            format!("{} - {}", start.format("%I:%M %p"), self.title())
        } else {
            self.title().to_string()
        }
    }
}

/// DateTime with timezone or date-only (all-day events)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCalDateTime {
    #[serde(rename = "dateTime")]
    pub date_time: Option<String>,
    pub date: Option<String>,
    #[serde(rename = "timeZone")]
    pub time_zone: Option<String>,
}

impl GCalDateTime {
    pub fn to_local_datetime(&self) -> Option<DateTime<Local>> {
        if let Some(ref dt) = self.date_time {
            DateTime::parse_from_rfc3339(dt)
                .ok()
                .map(|d| d.with_timezone(&Local))
        } else if let Some(ref d) = self.date {
            NaiveDate::parse_from_str(d, "%Y-%m-%d")
                .ok()
                .map(|nd| nd.and_hms_opt(0, 0, 0).unwrap())
                .map(|ndt| ndt.and_local_timezone(Local).unwrap())
        } else {
            None
        }
    }
}

/// Event attendee
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCalAttendee {
    pub email: Option<String>,
    #[serde(rename = "displayName")]
    pub display_name: Option<String>,
    #[serde(rename = "responseStatus")]
    pub response_status: Option<String>,
    #[serde(rename = "self")]
    pub is_self: Option<bool>,
    pub organizer: Option<bool>,
}

/// Event organizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCalOrganizer {
    pub email: Option<String>,
    #[serde(rename = "displayName")]
    pub display_name: Option<String>,
    #[serde(rename = "self")]
    pub is_self: Option<bool>,
}

/// Conference data (Google Meet, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConferenceData {
    #[serde(rename = "entryPoints")]
    pub entry_points: Option<Vec<EntryPoint>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryPoint {
    #[serde(rename = "entryPointType")]
    pub entry_point_type: String,
    pub uri: String,
}

/// API response for events list
#[derive(Debug, Deserialize)]
struct EventsListResponse {
    items: Option<Vec<GCalEvent>>,
    #[serde(rename = "nextPageToken")]
    next_page_token: Option<String>,
}

/// Google Calendar client
pub struct GoogleCalendarClient {
    oauth: Arc<OAuthManager>,
    http: reqwest::Client,
}

impl GoogleCalendarClient {
    pub fn new(oauth: Arc<OAuthManager>) -> Self {
        GoogleCalendarClient {
            oauth,
            http: reqwest::Client::new(),
        }
    }

    /// List events in a time range
    pub async fn list_events(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        max_results: Option<usize>,
    ) -> Result<Vec<GCalEvent>> {
        let token = self.oauth.get_token(Provider::Google).await?;

        let mut url = format!(
            "{}/calendars/primary/events?timeMin={}&timeMax={}&singleEvents=true&orderBy=startTime",
            CALENDAR_API_BASE,
            urlencoding::encode(&start.to_rfc3339()),
            urlencoding::encode(&end.to_rfc3339()),
        );

        if let Some(max) = max_results {
            url.push_str(&format!("&maxResults={}", max));
        }

        let resp = self.http
            .get(&url)
            .bearer_auth(&token)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Calendar API error {}: {}", status, body));
        }

        let data: EventsListResponse = resp.json().await?;
        Ok(data.items.unwrap_or_default())
    }

    /// Get today's events
    pub async fn get_today_events(&self) -> Result<Vec<GCalEvent>> {
        let now = Local::now();
        let start_of_day = now.date_naive().and_hms_opt(0, 0, 0).unwrap()
            .and_local_timezone(Local).unwrap()
            .with_timezone(&Utc);
        let end_of_day = now.date_naive().and_hms_opt(23, 59, 59).unwrap()
            .and_local_timezone(Local).unwrap()
            .with_timezone(&Utc);

        self.list_events(start_of_day, end_of_day, None).await
    }

    /// Get tomorrow's events
    pub async fn get_tomorrow_events(&self) -> Result<Vec<GCalEvent>> {
        let tomorrow = Local::now().date_naive() + Duration::days(1);
        let start = tomorrow.and_hms_opt(0, 0, 0).unwrap()
            .and_local_timezone(Local).unwrap()
            .with_timezone(&Utc);
        let end = tomorrow.and_hms_opt(23, 59, 59).unwrap()
            .and_local_timezone(Local).unwrap()
            .with_timezone(&Utc);

        self.list_events(start, end, None).await
    }

    /// Get this week's events
    pub async fn get_week_events(&self) -> Result<Vec<GCalEvent>> {
        let now = Local::now();
        let start = now.with_timezone(&Utc);
        let end = (now + Duration::days(7)).with_timezone(&Utc);

        self.list_events(start, end, None).await
    }

    /// Get upcoming events (next N)
    pub async fn get_upcoming_events(&self, count: usize) -> Result<Vec<GCalEvent>> {
        let now = Utc::now();
        let end = now + Duration::days(30);

        self.list_events(now, end, Some(count)).await
    }

    /// Get a single event by ID
    pub async fn get_event(&self, event_id: &str) -> Result<GCalEvent> {
        let token = self.oauth.get_token(Provider::Google).await?;

        let url = format!(
            "{}/calendars/primary/events/{}",
            CALENDAR_API_BASE,
            urlencoding::encode(event_id),
        );

        let resp = self.http
            .get(&url)
            .bearer_auth(&token)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Calendar API error {}: {}", status, body));
        }

        Ok(resp.json().await?)
    }

    /// Get summary text for LLM context
    pub async fn get_summary(&self) -> Result<String> {
        let today = self.get_today_events().await.unwrap_or_default();
        let upcoming = self.get_upcoming_events(5).await.unwrap_or_default();

        let mut summary = String::new();

        if !today.is_empty() {
            summary.push_str("Today's Google Calendar events:\n");
            for event in &today {
                summary.push_str(&format!("- {}\n", event.display_short()));
            }
        }

        if !upcoming.is_empty() {
            if !summary.is_empty() {
                summary.push('\n');
            }
            summary.push_str("Upcoming events:\n");
            for event in &upcoming {
                if let Some(start) = event.start_time() {
                    summary.push_str(&format!("- {} on {}\n", 
                        event.title(), 
                        start.format("%a %b %d at %I:%M %p")
                    ));
                }
            }
        }

        if summary.is_empty() {
            summary = "No upcoming Google Calendar events.".to_string();
        }

        Ok(summary)
    }

    /// Create a new calendar event
    pub async fn create_event(&self, event: &NewCalendarEvent) -> Result<GCalEvent> {
        let token = self.oauth.get_token(Provider::Google).await?;

        let url = format!("{}/calendars/primary/events", CALENDAR_API_BASE);

        // Build the request body
        let body = event.to_api_request();

        let resp = self.http
            .post(&url)
            .bearer_auth(&token)
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Failed to create event ({}): {}", status, body_text));
        }

        Ok(resp.json().await?)
    }

    /// Create event with attendees (sends invites)
    pub async fn create_event_with_invites(&self, event: &NewCalendarEvent) -> Result<GCalEvent> {
        let token = self.oauth.get_token(Provider::Google).await?;

        // sendUpdates=all sends email notifications to attendees
        let url = format!("{}/calendars/primary/events?sendUpdates=all", CALENDAR_API_BASE);

        let body = event.to_api_request();

        let resp = self.http
            .post(&url)
            .bearer_auth(&token)
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Failed to create event ({}): {}", status, body_text));
        }

        Ok(resp.json().await?)
    }

    /// Update an existing event
    pub async fn update_event(&self, event_id: &str, updates: &EventUpdate) -> Result<GCalEvent> {
        let token = self.oauth.get_token(Provider::Google).await?;

        let url = format!(
            "{}/calendars/primary/events/{}?sendUpdates=all",
            CALENDAR_API_BASE,
            urlencoding::encode(event_id),
        );

        let body = updates.to_api_request();

        let resp = self.http
            .patch(&url)
            .bearer_auth(&token)
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Failed to update event ({}): {}", status, body_text));
        }

        Ok(resp.json().await?)
    }

    /// Delete an event
    pub async fn delete_event(&self, event_id: &str) -> Result<()> {
        let token = self.oauth.get_token(Provider::Google).await?;

        let url = format!(
            "{}/calendars/primary/events/{}?sendUpdates=all",
            CALENDAR_API_BASE,
            urlencoding::encode(event_id),
        );

        let resp = self.http
            .delete(&url)
            .bearer_auth(&token)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Failed to delete event ({}): {}", status, body_text));
        }

        Ok(())
    }

    /// Search for events by title (fuzzy match)
    pub async fn find_event_by_title(&self, title: &str, days_ahead: i64) -> Result<Vec<GCalEvent>> {
        let now = Utc::now();
        let end = now + Duration::days(days_ahead);
        
        let events = self.list_events(now, end, None).await?;
        let lower_title = title.to_lowercase();
        
        // Filter events that match the title
        let matches: Vec<_> = events.into_iter()
            .filter(|e| {
                e.summary.as_ref()
                    .map(|s| s.to_lowercase().contains(&lower_title))
                    .unwrap_or(false)
            })
            .collect();
        
        Ok(matches)
    }

    /// Get detailed day summary with full event info
    pub async fn get_day_summary(&self, date: NaiveDate) -> Result<String> {
        let start = date.and_hms_opt(0, 0, 0).unwrap()
            .and_local_timezone(Local).unwrap()
            .with_timezone(&Utc);
        let end = date.and_hms_opt(23, 59, 59).unwrap()
            .and_local_timezone(Local).unwrap()
            .with_timezone(&Utc);

        let events = self.list_events(start, end, None).await?;

        if events.is_empty() {
            return Ok(format!("📅 No events scheduled for {}", date.format("%A, %B %d")));
        }

        let mut output = format!("📅 Schedule for {} ({} events)\n\n", 
            date.format("%A, %B %d"), events.len());

        for event in &events {
            output.push_str(&event.display());
            output.push_str("\n\n");
        }

        Ok(output.trim().to_string())
    }

    /// Get detailed week summary grouped by day
    pub async fn get_week_summary(&self) -> Result<String> {
        let now = Local::now();
        let events = self.get_week_events().await?;

        if events.is_empty() {
            return Ok("📅 No events scheduled for the next 7 days".to_string());
        }

        let mut output = format!("📅 Week Ahead ({} events)\n", events.len());
        output.push_str(&format!("{} - {}\n\n", 
            now.format("%b %d"), 
            (now + Duration::days(7)).format("%b %d")
        ));

        // Group events by date
        let mut current_date: Option<NaiveDate> = None;
        
        for event in &events {
            if let Some(start) = event.start_time() {
                let event_date = start.date_naive();
                
                // Print date header if new day
                if current_date != Some(event_date) {
                    if current_date.is_some() {
                        output.push('\n');
                    }
                    output.push_str(&format!("📆 {}\n", start.format("%A, %B %d")));
                    current_date = Some(event_date);
                }

                // Event details
                let time_str = if event.is_all_day() {
                    "   All day".to_string()
                } else {
                    format!("   {}", start.format("%I:%M %p"))
                };
                
                output.push_str(&format!("{} - {}\n", time_str, event.title()));
                
                // Location (if present)
                if let Some(ref loc) = event.location {
                    output.push_str(&format!("      📍 {}\n", loc));
                }
                
                // Attendees (if present)
                if let Some(ref attendees) = event.attendees {
                    let others: Vec<_> = attendees.iter()
                        .filter(|a| a.is_self != Some(true))
                        .filter_map(|a| a.display_name.as_ref().or(a.email.as_ref()))
                        .take(3)
                        .collect();
                    if !others.is_empty() {
                        let names = others.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ");
                        let more = if attendees.len() > 4 {
                            format!(" +{}", attendees.len() - 4)
                        } else {
                            String::new()
                        };
                        output.push_str(&format!("      👥 {}{}\n", names, more));
                    }
                }
            }
        }

        Ok(output.trim().to_string())
    }
}

// ============================================================================
// Event Creation Types
// ============================================================================

/// Represents a new calendar event to be created
#[derive(Debug, Clone, Default)]
pub struct NewCalendarEvent {
    pub title: String,
    pub description: Option<String>,
    pub location: Option<String>,
    pub start: DateTime<Local>,
    pub end: DateTime<Local>,
    pub attendees: Vec<String>,  // Email addresses
    pub all_day: bool,
}

impl NewCalendarEvent {
    pub fn new(title: &str, start: DateTime<Local>, end: DateTime<Local>) -> Self {
        NewCalendarEvent {
            title: title.to_string(),
            description: None,
            location: None,
            start,
            end,
            attendees: Vec::new(),
            all_day: false,
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    pub fn with_location(mut self, loc: &str) -> Self {
        self.location = Some(loc.to_string());
        self
    }

    pub fn with_attendee(mut self, email: &str) -> Self {
        self.attendees.push(email.to_string());
        self
    }

    pub fn with_attendees(mut self, emails: Vec<String>) -> Self {
        self.attendees.extend(emails);
        self
    }

    pub fn all_day(mut self) -> Self {
        self.all_day = true;
        self
    }

    /// Convert to Google Calendar API request format
    fn to_api_request(&self) -> serde_json::Value {
        let mut event = serde_json::json!({
            "summary": self.title,
        });

        if let Some(ref desc) = self.description {
            event["description"] = serde_json::json!(desc);
        }

        if let Some(ref loc) = self.location {
            event["location"] = serde_json::json!(loc);
        }

        if self.all_day {
            event["start"] = serde_json::json!({
                "date": self.start.format("%Y-%m-%d").to_string(),
            });
            event["end"] = serde_json::json!({
                "date": self.end.format("%Y-%m-%d").to_string(),
            });
        } else {
            // Use RFC3339 format with timezone offset
            // Convert to UTC for Google Calendar API compatibility
            let start_utc = self.start.with_timezone(&chrono::Utc);
            let end_utc = self.end.with_timezone(&chrono::Utc);
            event["start"] = serde_json::json!({
                "dateTime": start_utc.to_rfc3339(),
            });
            event["end"] = serde_json::json!({
                "dateTime": end_utc.to_rfc3339(),
            });
        }

        if !self.attendees.is_empty() {
            let attendees: Vec<_> = self.attendees.iter()
                .map(|email| serde_json::json!({ "email": email }))
                .collect();
            event["attendees"] = serde_json::json!(attendees);
        }

        event
    }

    /// Display preview before creation
    pub fn preview(&self) -> String {
        let mut output = format!("📅 {}\n", self.title);
        
        if self.all_day {
            output.push_str(&format!("   📆 {} (all day)\n", self.start.format("%A, %B %d")));
        } else {
            output.push_str(&format!("   🕐 {} - {}\n", 
                self.start.format("%a %b %d, %I:%M %p"),
                self.end.format("%I:%M %p")
            ));
        }

        if let Some(ref loc) = self.location {
            output.push_str(&format!("   📍 {}\n", loc));
        }

        if !self.attendees.is_empty() {
            output.push_str(&format!("   👥 {}\n", self.attendees.join(", ")));
        }

        if let Some(ref desc) = self.description {
            output.push_str(&format!("   📝 {}\n", desc));
        }

        output
    }
}

// ============================================================================
// Event Updates
// ============================================================================

/// Represents updates to an existing calendar event
#[derive(Debug, Clone, Default)]
pub struct EventUpdate {
    pub title: Option<String>,
    pub description: Option<String>,
    pub location: Option<String>,
    pub start: Option<DateTime<Local>>,
    pub end: Option<DateTime<Local>>,
    pub add_attendees: Vec<String>,
}

impl EventUpdate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_title(mut self, title: &str) -> Self {
        self.title = Some(title.to_string());
        self
    }

    pub fn with_time(mut self, start: DateTime<Local>, end: DateTime<Local>) -> Self {
        self.start = Some(start);
        self.end = Some(end);
        self
    }

    pub fn with_location(mut self, loc: &str) -> Self {
        self.location = Some(loc.to_string());
        self
    }

    pub fn with_attendee(mut self, email: &str) -> Self {
        self.add_attendees.push(email.to_string());
        self
    }

    /// Convert to Google Calendar API PATCH request format
    fn to_api_request(&self) -> serde_json::Value {
        let mut update = serde_json::json!({});

        if let Some(ref title) = self.title {
            update["summary"] = serde_json::json!(title);
        }

        if let Some(ref desc) = self.description {
            update["description"] = serde_json::json!(desc);
        }

        if let Some(ref loc) = self.location {
            update["location"] = serde_json::json!(loc);
        }

        if let Some(start) = self.start {
            let start_utc = start.with_timezone(&chrono::Utc);
            update["start"] = serde_json::json!({
                "dateTime": start_utc.to_rfc3339(),
            });
        }

        if let Some(end) = self.end {
            let end_utc = end.with_timezone(&chrono::Utc);
            update["end"] = serde_json::json!({
                "dateTime": end_utc.to_rfc3339(),
            });
        }

        if !self.add_attendees.is_empty() {
            let attendees: Vec<_> = self.add_attendees.iter()
                .map(|email| serde_json::json!({ "email": email }))
                .collect();
            update["attendees"] = serde_json::json!(attendees);
        }

        update
    }

    pub fn is_empty(&self) -> bool {
        self.title.is_none() 
            && self.description.is_none() 
            && self.location.is_none()
            && self.start.is_none()
            && self.end.is_none()
            && self.add_attendees.is_empty()
    }

    /// Display what will be changed
    pub fn preview(&self, original_title: &str) -> String {
        let mut output = format!("📝 Updating: {}\n\n", original_title);
        output.push_str("Changes:\n");

        if let Some(ref title) = self.title {
            output.push_str(&format!("   📌 Title → {}\n", title));
        }

        if let Some(start) = self.start {
            if let Some(end) = self.end {
                output.push_str(&format!("   🕐 Time → {} - {}\n", 
                    start.format("%a %b %d, %I:%M %p"),
                    end.format("%I:%M %p")
                ));
            } else {
                output.push_str(&format!("   🕐 Start → {}\n", start.format("%a %b %d, %I:%M %p")));
            }
        }

        if let Some(ref loc) = self.location {
            output.push_str(&format!("   📍 Location → {}\n", loc));
        }

        if !self.add_attendees.is_empty() {
            output.push_str(&format!("   👥 Adding → {}\n", self.add_attendees.join(", ")));
        }

        output
    }
}

// ============================================================================
// AI Event Parsing
// ============================================================================

/// Parsed event from natural language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedEvent {
    pub title: String,
    pub date: Option<String>,      // "2024-12-15" or "tomorrow" etc.
    pub time: Option<String>,      // "14:00" or "2pm" etc.
    pub duration_minutes: Option<i64>,
    pub location: Option<String>,
    pub attendees: Vec<String>,    // Names that need to be resolved
    pub description: Option<String>,
    pub is_all_day: bool,
}

impl ParsedEvent {
    /// Convert to NewCalendarEvent, resolving relative dates
    pub fn to_calendar_event(&self, resolved_emails: Vec<String>) -> Result<NewCalendarEvent> {
        let now = Local::now();
        
        // Parse date
        let date = if let Some(ref d) = self.date {
            parse_relative_date(d, now.date_naive())?
        } else {
            now.date_naive()
        };

        // Parse time
        let (start_time, is_all_day) = if let Some(ref t) = self.time {
            (parse_time(t)?, false)
        } else if self.is_all_day {
            (NaiveTime::from_hms_opt(9, 0, 0).unwrap(), true)
        } else {
            // Default to 9am if no time specified
            (NaiveTime::from_hms_opt(9, 0, 0).unwrap(), false)
        };

        let start = date.and_time(start_time)
            .and_local_timezone(Local)
            .single()
            .ok_or_else(|| anyhow!("Invalid datetime"))?;

        let duration = Duration::minutes(self.duration_minutes.unwrap_or(60));
        let end = start + duration;

        let mut event = NewCalendarEvent::new(&self.title, start, end)
            .with_attendees(resolved_emails);

        if let Some(ref loc) = self.location {
            event = event.with_location(loc);
        }

        if let Some(ref desc) = self.description {
            event = event.with_description(desc);
        }

        if is_all_day || self.is_all_day {
            event = event.all_day();
        }

        Ok(event)
    }
}

/// Parse relative date strings
pub fn parse_relative_date(s: &str, today: NaiveDate) -> Result<NaiveDate> {
    let lower = s.to_lowercase();
    
    match lower.as_str() {
        "today" => Ok(today),
        "tomorrow" => Ok(today + Duration::days(1)),
        "day after tomorrow" => Ok(today + Duration::days(2)),
        _ => {
            // Try parsing weekday names
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
                    let current_weekday = today.weekday();
                    let target = weekday.num_days_from_monday() as i64;
                    let current = current_weekday.num_days_from_monday() as i64;
                    let mut days_ahead = target - current;
                    if days_ahead <= 0 {
                        days_ahead += 7; // Next week
                    }
                    return Ok(today + Duration::days(days_ahead));
                }
            }
            
            // Try parsing as YYYY-MM-DD
            if let Ok(date) = NaiveDate::parse_from_str(&lower, "%Y-%m-%d") {
                return Ok(date);
            }
            
            // Try other formats
            for fmt in ["%B %d", "%b %d", "%m/%d", "%d/%m"] {
                if let Ok(parsed) = NaiveDate::parse_from_str(&format!("{} {}", lower, today.year()), &format!("{} %Y", fmt)) {
                    return Ok(parsed);
                }
            }
            
            Err(anyhow!("Could not parse date: {}", s))
        }
    }
}

/// Parse time strings
pub fn parse_time(s: &str) -> Result<NaiveTime> {
    let lower = s.to_lowercase().replace(" ", "");
    
    // Handle noon/midnight
    if lower == "noon" {
        return Ok(NaiveTime::from_hms_opt(12, 0, 0).unwrap());
    }
    if lower == "midnight" {
        return Ok(NaiveTime::from_hms_opt(0, 0, 0).unwrap());
    }
    
    // Try various formats
    let formats = [
        "%H:%M",      // 14:30
        "%I:%M%p",    // 2:30pm
        "%I%p",       // 2pm
        "%I:%M %p",   // 2:30 pm
        "%I %p",      // 2 pm
    ];
    
    for fmt in formats {
        if let Ok(time) = NaiveTime::parse_from_str(&lower, fmt) {
            return Ok(time);
        }
    }
    
    // Try extracting hour and am/pm
    let is_pm = lower.contains("pm");
    let is_am = lower.contains("am");
    let digits: String = lower.chars().filter(|c| c.is_ascii_digit() || *c == ':').collect();
    
    if let Some(hour) = digits.split(':').next().and_then(|h| h.parse::<u32>().ok()) {
        let mut h = hour;
        if is_pm && h < 12 {
            h += 12;
        } else if is_am && h == 12 {
            h = 0;
        }
        
        let minute = digits.split(':').nth(1)
            .and_then(|m| m.parse::<u32>().ok())
            .unwrap_or(0);
        
        if let Some(time) = NaiveTime::from_hms_opt(h, minute, 0) {
            return Ok(time);
        }
    }
    
    Err(anyhow!("Could not parse time: {}", s))
}
