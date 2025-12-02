use anyhow::Result;
use chrono::{DateTime, Local, NaiveDate, NaiveDateTime, NaiveTime, Duration, Datelike};
use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::storage::Database;

/// An event in the calendar
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: i64,
    pub title: String,
    pub description: Option<String>,
    pub start_time: DateTime<Local>,
    pub end_time: Option<DateTime<Local>>,
    pub location: Option<String>,
    pub attendees: Vec<String>, // Names of people (can be linked to KG)
}

impl Event {
    /// Format event for display
    pub fn display(&self) -> String {
        let mut output = format!(
            "📅 {} - {}",
            self.start_time.format("%a %b %d, %Y at %I:%M %p"),
            self.title
        );
        
        if let Some(end) = &self.end_time {
            output.push_str(&format!(" (until {})", end.format("%I:%M %p")));
        }
        
        if let Some(loc) = &self.location {
            output.push_str(&format!("\n   📍 {}", loc));
        }
        
        if let Some(desc) = &self.description {
            output.push_str(&format!("\n   {}", desc));
        }
        
        if !self.attendees.is_empty() {
            output.push_str(&format!("\n   👥 {}", self.attendees.join(", ")));
        }
        
        output
    }

    /// Short display for lists
    pub fn display_short(&self) -> String {
        format!(
            "{} - {}",
            self.start_time.format("%I:%M %p"),
            self.title
        )
    }
}

/// Calendar manager for event operations
pub struct Calendar<'a> {
    db: &'a Database,
}

impl<'a> Calendar<'a> {
    pub fn new(db: &'a Database) -> Self {
        Calendar { db }
    }

    /// Initialize calendar tables
    pub fn init_tables(db: &Database) -> Result<()> {
        db.connection().execute(
            "CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                location TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        db.connection().execute(
            "CREATE TABLE IF NOT EXISTS event_attendees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                person_name TEXT NOT NULL,
                person_entity_id INTEGER,
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
            )",
            [],
        )?;

        db.connection().execute(
            "CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_time)",
            [],
        )?;

        Ok(())
    }

    /// Add a new event
    pub fn add_event(
        &self,
        title: &str,
        start_time: DateTime<Local>,
        end_time: Option<DateTime<Local>>,
        description: Option<&str>,
        location: Option<&str>,
    ) -> Result<Event> {
        self.db.connection().execute(
            "INSERT INTO events (title, description, start_time, end_time, location) 
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                title,
                description,
                start_time.to_rfc3339(),
                end_time.map(|t| t.to_rfc3339()),
                location
            ],
        )?;

        let id = self.db.connection().last_insert_rowid();
        
        Ok(Event {
            id,
            title: title.to_string(),
            description: description.map(|s| s.to_string()),
            start_time,
            end_time,
            location: location.map(|s| s.to_string()),
            attendees: vec![],
        })
    }

    /// Add an attendee to an event
    pub fn add_attendee(&self, event_id: i64, person_name: &str, entity_id: Option<i64>) -> Result<()> {
        self.db.connection().execute(
            "INSERT INTO event_attendees (event_id, person_name, person_entity_id) 
             VALUES (?1, ?2, ?3)",
            params![event_id, person_name, entity_id],
        )?;
        Ok(())
    }

    /// Get events for a specific date
    pub fn get_events_for_date(&self, date: NaiveDate) -> Result<Vec<Event>> {
        let start_of_day = date.and_hms_opt(0, 0, 0).unwrap();
        let end_of_day = date.and_hms_opt(23, 59, 59).unwrap();
        
        self.get_events_in_range(start_of_day, end_of_day)
    }

    /// Get today's events
    pub fn get_today_events(&self) -> Result<Vec<Event>> {
        self.get_events_for_date(Local::now().date_naive())
    }

    /// Get tomorrow's events
    pub fn get_tomorrow_events(&self) -> Result<Vec<Event>> {
        let tomorrow = Local::now().date_naive() + Duration::days(1);
        self.get_events_for_date(tomorrow)
    }

    /// Get this week's events
    pub fn get_week_events(&self) -> Result<Vec<Event>> {
        let today = Local::now().date_naive();
        let week_end = today + Duration::days(7);
        
        let start = today.and_hms_opt(0, 0, 0).unwrap();
        let end = week_end.and_hms_opt(23, 59, 59).unwrap();
        
        self.get_events_in_range(start, end)
    }

    /// Get events in a datetime range
    fn get_events_in_range(&self, start: NaiveDateTime, end: NaiveDateTime) -> Result<Vec<Event>> {
        let start_str = start.format("%Y-%m-%dT%H:%M:%S").to_string();
        let end_str = end.format("%Y-%m-%dT%H:%M:%S").to_string();

        let mut stmt = self.db.connection().prepare(
            "SELECT id, title, description, start_time, end_time, location 
             FROM events 
             WHERE start_time >= ?1 AND start_time <= ?2
             ORDER BY start_time"
        )?;

        let rows = stmt.query_map(params![start_str, end_str], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        })?;

        let mut events = Vec::new();
        for row in rows {
            let (id, title, description, start_str, end_str, location) = row?;
            
            let start_time = DateTime::parse_from_rfc3339(&start_str)
                .map(|dt| dt.with_timezone(&Local))
                .unwrap_or_else(|_| Local::now());
            
            let end_time = end_str.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .map(|dt| dt.with_timezone(&Local))
                    .ok()
            });

            // Get attendees
            let attendees = self.get_event_attendees(id)?;

            events.push(Event {
                id,
                title,
                description,
                start_time,
                end_time,
                location,
                attendees,
            });
        }

        Ok(events)
    }

    /// Get all upcoming events
    pub fn get_upcoming_events(&self, limit: usize) -> Result<Vec<Event>> {
        let now = Local::now().format("%Y-%m-%dT%H:%M:%S").to_string();

        let mut stmt = self.db.connection().prepare(
            "SELECT id, title, description, start_time, end_time, location 
             FROM events 
             WHERE start_time >= ?1
             ORDER BY start_time
             LIMIT ?2"
        )?;

        let rows = stmt.query_map(params![now, limit as i64], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        })?;

        let mut events = Vec::new();
        for row in rows {
            let (id, title, description, start_str, end_str, location) = row?;
            
            let start_time = DateTime::parse_from_rfc3339(&start_str)
                .map(|dt| dt.with_timezone(&Local))
                .unwrap_or_else(|_| Local::now());
            
            let end_time = end_str.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .map(|dt| dt.with_timezone(&Local))
                    .ok()
            });

            let attendees = self.get_event_attendees(id)?;

            events.push(Event {
                id,
                title,
                description,
                start_time,
                end_time,
                location,
                attendees,
            });
        }

        Ok(events)
    }

    /// Get attendees for an event
    fn get_event_attendees(&self, event_id: i64) -> Result<Vec<String>> {
        let mut stmt = self.db.connection().prepare(
            "SELECT person_name FROM event_attendees WHERE event_id = ?1"
        )?;
        
        let rows = stmt.query_map(params![event_id], |row| row.get(0))?;
        
        let mut attendees = Vec::new();
        for row in rows {
            attendees.push(row?);
        }
        Ok(attendees)
    }

    /// Delete an event
    pub fn delete_event(&self, event_id: i64) -> Result<bool> {
        let rows = self.db.connection().execute(
            "DELETE FROM events WHERE id = ?1",
            params![event_id],
        )?;
        Ok(rows > 0)
    }

    /// Find event by title (partial match)
    pub fn find_event_by_title(&self, title: &str) -> Result<Option<Event>> {
        let mut stmt = self.db.connection().prepare(
            "SELECT id, title, description, start_time, end_time, location 
             FROM events 
             WHERE LOWER(title) LIKE LOWER(?1)
             ORDER BY start_time DESC
             LIMIT 1"
        )?;

        let pattern = format!("%{}%", title);
        let result = stmt.query_row(params![pattern], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        });

        match result {
            Ok((id, title, description, start_str, end_str, location)) => {
                let start_time = DateTime::parse_from_rfc3339(&start_str)
                    .map(|dt| dt.with_timezone(&Local))
                    .unwrap_or_else(|_| Local::now());
                
                let end_time = end_str.and_then(|s| {
                    DateTime::parse_from_rfc3339(&s)
                        .map(|dt| dt.with_timezone(&Local))
                        .ok()
                });

                let attendees = self.get_event_attendees(id)?;

                Ok(Some(Event {
                    id,
                    title,
                    description,
                    start_time,
                    end_time,
                    location,
                    attendees,
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get a summary for LLM context
    pub fn get_summary(&self) -> Result<String> {
        let today = self.get_today_events()?;
        let upcoming = self.get_upcoming_events(5)?;

        let mut summary = String::new();

        if !today.is_empty() {
            summary.push_str("Today's events:\n");
            for event in &today {
                summary.push_str(&format!("- {} at {}\n", 
                    event.title, 
                    event.start_time.format("%I:%M %p")
                ));
            }
        }

        if !upcoming.is_empty() {
            if !summary.is_empty() {
                summary.push('\n');
            }
            summary.push_str("Upcoming events:\n");
            for event in &upcoming {
                summary.push_str(&format!("- {} on {}\n", 
                    event.title, 
                    event.start_time.format("%a %b %d at %I:%M %p")
                ));
            }
        }

        if summary.is_empty() {
            summary = "No upcoming events.".to_string();
        }

        Ok(summary)
    }
}

/// Parse a date/time string into a DateTime
/// Supports formats like:
/// - "tomorrow 2pm"
/// - "dec 5 3:30pm"
/// - "2024-12-05 15:30"
/// - "next monday 10am"
pub fn parse_datetime(input: &str) -> Option<DateTime<Local>> {
    let input = input.trim().to_lowercase();
    let now = Local::now();
    let today = now.date_naive();

    // Try relative dates first
    if input.starts_with("today") {
        let time_part = input.strip_prefix("today").unwrap().trim();
        let time = parse_time(time_part).unwrap_or(NaiveTime::from_hms_opt(9, 0, 0).unwrap());
        return Some(today.and_time(time).and_local_timezone(Local).unwrap());
    }

    if input.starts_with("tomorrow") {
        let time_part = input.strip_prefix("tomorrow").unwrap().trim();
        let time = parse_time(time_part).unwrap_or(NaiveTime::from_hms_opt(9, 0, 0).unwrap());
        let date = today + Duration::days(1);
        return Some(date.and_time(time).and_local_timezone(Local).unwrap());
    }

    // Try "next <weekday>"
    if input.starts_with("next ") {
        let rest = input.strip_prefix("next ").unwrap();
        if let Some((weekday, time_part)) = parse_weekday_and_time(rest) {
            let days_ahead = (weekday as i64 - today.weekday().num_days_from_monday() as i64 + 7) % 7;
            let days_ahead = if days_ahead == 0 { 7 } else { days_ahead };
            let date = today + Duration::days(days_ahead);
            let time = parse_time(time_part).unwrap_or(NaiveTime::from_hms_opt(9, 0, 0).unwrap());
            return Some(date.and_time(time).and_local_timezone(Local).unwrap());
        }
    }

    // Try parsing month day time (e.g., "dec 5 3pm")
    if let Some(dt) = parse_month_day_time(&input, now.year()) {
        return Some(dt);
    }

    // Try ISO format
    if let Ok(dt) = DateTime::parse_from_rfc3339(&input) {
        return Some(dt.with_timezone(&Local));
    }

    // Try simple date format
    if let Ok(date) = NaiveDate::parse_from_str(&input, "%Y-%m-%d") {
        let time = NaiveTime::from_hms_opt(9, 0, 0).unwrap();
        return Some(date.and_time(time).and_local_timezone(Local).unwrap());
    }

    None
}

fn parse_time(input: &str) -> Option<NaiveTime> {
    let input = input.trim().to_lowercase();
    
    if input.is_empty() {
        return None;
    }

    // Handle "3pm", "3:30pm", "15:30"
    let (time_str, is_pm) = if input.ends_with("pm") {
        (input.trim_end_matches("pm").trim(), true)
    } else if input.ends_with("am") {
        (input.trim_end_matches("am").trim(), false)
    } else {
        (input.as_str(), false)
    };

    if time_str.contains(':') {
        let parts: Vec<&str> = time_str.split(':').collect();
        if parts.len() == 2 {
            let hour: u32 = parts[0].parse().ok()?;
            let minute: u32 = parts[1].parse().ok()?;
            let hour = if is_pm && hour < 12 { hour + 12 } else if !is_pm && hour == 12 { 0 } else { hour };
            return NaiveTime::from_hms_opt(hour, minute, 0);
        }
    } else if let Ok(hour) = time_str.parse::<u32>() {
        let hour = if is_pm && hour < 12 { hour + 12 } else if !is_pm && hour == 12 { 0 } else { hour };
        return NaiveTime::from_hms_opt(hour, 0, 0);
    }

    None
}

fn parse_weekday_and_time(input: &str) -> Option<(u32, &str)> {
    let weekdays = [
        ("monday", 0), ("mon", 0),
        ("tuesday", 1), ("tue", 1),
        ("wednesday", 2), ("wed", 2),
        ("thursday", 3), ("thu", 3),
        ("friday", 4), ("fri", 4),
        ("saturday", 5), ("sat", 5),
        ("sunday", 6), ("sun", 6),
    ];

    for (name, day) in weekdays {
        if input.starts_with(name) {
            let rest = input.strip_prefix(name).unwrap().trim();
            return Some((day, rest));
        }
    }
    None
}

fn parse_month_day_time(input: &str, year: i32) -> Option<DateTime<Local>> {
    let months = [
        ("january", 1), ("jan", 1),
        ("february", 2), ("feb", 2),
        ("march", 3), ("mar", 3),
        ("april", 4), ("apr", 4),
        ("may", 5),
        ("june", 6), ("jun", 6),
        ("july", 7), ("jul", 7),
        ("august", 8), ("aug", 8),
        ("september", 9), ("sep", 9),
        ("october", 10), ("oct", 10),
        ("november", 11), ("nov", 11),
        ("december", 12), ("dec", 12),
    ];

    for (name, month) in months {
        if input.starts_with(name) {
            let rest = input.strip_prefix(name).unwrap().trim();
            let parts: Vec<&str> = rest.splitn(2, ' ').collect();
            
            if let Some(day_str) = parts.first() {
                if let Ok(day) = day_str.parse::<u32>() {
                    let time_part = parts.get(1).copied().unwrap_or("");
                    let time = parse_time(time_part).unwrap_or(NaiveTime::from_hms_opt(9, 0, 0).unwrap());
                    
                    if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
                        return Some(date.and_time(time).and_local_timezone(Local).unwrap());
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Timelike;

    #[test]
    fn test_parse_time() {
        assert_eq!(parse_time("3pm"), NaiveTime::from_hms_opt(15, 0, 0));
        assert_eq!(parse_time("10am"), NaiveTime::from_hms_opt(10, 0, 0));
        assert_eq!(parse_time("3:30pm"), NaiveTime::from_hms_opt(15, 30, 0));
        assert_eq!(parse_time("12pm"), NaiveTime::from_hms_opt(12, 0, 0));
    }

    #[test]
    fn test_parse_datetime_tomorrow() {
        let result = parse_datetime("tomorrow 2pm");
        assert!(result.is_some());
        let dt = result.unwrap();
        assert_eq!(dt.hour(), 14);
    }
}
