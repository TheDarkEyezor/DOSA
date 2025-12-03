// Natural language date/time parsing module
// Supports expressions like "next Friday 8pm", "in 2 weeks", "end of month"

use anyhow::{Result, anyhow};
use chrono::{DateTime, Local, Duration, Datelike, Timelike, NaiveTime, Weekday, TimeZone};
use chrono_english::{parse_date_string, Dialect};
use regex::Regex;

/// Parsed date/time result with optional end time for ranges
#[derive(Debug, Clone)]
pub struct ParsedDateTime {
    pub start: DateTime<Local>,
    pub end: Option<DateTime<Local>>,
    pub is_all_day: bool,
    pub has_explicit_time: bool,
}

/// Recurrence rule for repeating events
#[derive(Debug, Clone)]
pub struct RecurrenceRule {
    pub rrule: String,  // RFC 5545 RRULE format
    pub description: String,
}

impl ParsedDateTime {
    /// Format for Google Calendar API
    pub fn to_rfc3339(&self) -> String {
        self.start.to_rfc3339()
    }
    
    /// Get end time, defaulting to 1 hour after start
    pub fn end_or_default(&self) -> DateTime<Local> {
        self.end.unwrap_or_else(|| self.start + Duration::hours(1))
    }
}

/// Parse natural language date/time expressions
pub fn parse_datetime(input: &str) -> Result<ParsedDateTime> {
    let input_lower = input.to_lowercase();
    let now = Local::now();
    
    // Check for special patterns first
    if let Some(parsed) = try_parse_special_pattern(&input_lower, now)? {
        return Ok(parsed);
    }
    
    // Check for relative time patterns ("in X days/hours/minutes")
    if let Some(parsed) = try_parse_relative(&input_lower, now)? {
        return Ok(parsed);
    }
    
    // Check for duration patterns ("for 2 hours")
    let (duration_hours, remaining) = extract_duration(&input_lower);
    
    // Use chrono-english for the main parsing
    let parse_input = if remaining.is_empty() { input } else { &remaining };
    
    match parse_date_string(parse_input, now, Dialect::Us) {
        Ok(dt) => {
            let has_explicit_time = contains_time_indicator(input);
            let is_all_day = !has_explicit_time;
            
            let end = if let Some(hours) = duration_hours {
                Some(dt + Duration::hours(hours as i64))
            } else if has_explicit_time {
                Some(dt + Duration::hours(1)) // Default 1 hour duration
            } else {
                None
            };
            
            Ok(ParsedDateTime {
                start: dt,
                end,
                is_all_day,
                has_explicit_time,
            })
        }
        Err(_) => {
            // Fallback: try our custom parser
            parse_datetime_fallback(input, now)
        }
    }
}

/// Try to parse special patterns like "end of month", "beginning of next week"
fn try_parse_special_pattern(input: &str, now: DateTime<Local>) -> Result<Option<ParsedDateTime>> {
    // End of month
    if input.contains("end of") && input.contains("month") {
        let next_month = if input.contains("next") {
            if now.month() == 12 {
                now.with_year(now.year() + 1).unwrap().with_month(1).unwrap()
            } else {
                now.with_month(now.month() + 1).unwrap()
            }
        } else {
            now
        };
        
        // Get last day of month
        let last_day = get_last_day_of_month(next_month.year(), next_month.month());
        let end_of_month = next_month.with_day(last_day).unwrap()
            .with_hour(17).unwrap()
            .with_minute(0).unwrap()
            .with_second(0).unwrap();
        
        return Ok(Some(ParsedDateTime {
            start: end_of_month,
            end: Some(end_of_month + Duration::hours(1)),
            is_all_day: false,
            has_explicit_time: false,
        }));
    }
    
    // Beginning of week
    if input.contains("beginning of") && input.contains("week") {
        let days_from_monday = now.weekday().num_days_from_monday();
        let monday = now - Duration::days(days_from_monday as i64);
        
        let target_monday = if input.contains("next") {
            monday + Duration::days(7)
        } else {
            monday
        };
        
        let start = target_monday.with_hour(9).unwrap()
            .with_minute(0).unwrap()
            .with_second(0).unwrap();
        
        return Ok(Some(ParsedDateTime {
            start,
            end: Some(start + Duration::hours(1)),
            is_all_day: false,
            has_explicit_time: false,
        }));
    }
    
    // End of week (Friday)
    if input.contains("end of") && input.contains("week") {
        let days_until_friday = (Weekday::Fri.num_days_from_monday() as i32 
            - now.weekday().num_days_from_monday() as i32 + 7) % 7;
        let friday = now + Duration::days(days_until_friday as i64);
        
        let target_friday = if input.contains("next") {
            friday + Duration::days(7)
        } else {
            friday
        };
        
        let start = target_friday.with_hour(17).unwrap()
            .with_minute(0).unwrap()
            .with_second(0).unwrap();
        
        return Ok(Some(ParsedDateTime {
            start,
            end: Some(start + Duration::hours(1)),
            is_all_day: false,
            has_explicit_time: false,
        }));
    }
    
    Ok(None)
}

/// Try to parse relative time expressions like "in 2 days", "in 30 minutes"
fn try_parse_relative(input: &str, now: DateTime<Local>) -> Result<Option<ParsedDateTime>> {
    let re = Regex::new(r"in\s+(\d+)\s+(minute|hour|day|week|month)s?")?;
    
    if let Some(caps) = re.captures(input) {
        let amount: i64 = caps.get(1).unwrap().as_str().parse()?;
        let unit = caps.get(2).unwrap().as_str();
        
        let duration = match unit {
            "minute" => Duration::minutes(amount),
            "hour" => Duration::hours(amount),
            "day" => Duration::days(amount),
            "week" => Duration::weeks(amount),
            "month" => Duration::days(amount * 30), // Approximate
            _ => return Ok(None),
        };
        
        let start = now + duration;
        
        return Ok(Some(ParsedDateTime {
            start,
            end: Some(start + Duration::hours(1)),
            is_all_day: false,
            has_explicit_time: true,
        }));
    }
    
    // Also check for "X days from now", "X hours from now"
    let re2 = Regex::new(r"(\d+)\s+(minute|hour|day|week|month)s?\s+from\s+now")?;
    
    if let Some(caps) = re2.captures(input) {
        let amount: i64 = caps.get(1).unwrap().as_str().parse()?;
        let unit = caps.get(2).unwrap().as_str();
        
        let duration = match unit {
            "minute" => Duration::minutes(amount),
            "hour" => Duration::hours(amount),
            "day" => Duration::days(amount),
            "week" => Duration::weeks(amount),
            "month" => Duration::days(amount * 30),
            _ => return Ok(None),
        };
        
        let start = now + duration;
        
        return Ok(Some(ParsedDateTime {
            start,
            end: Some(start + Duration::hours(1)),
            is_all_day: false,
            has_explicit_time: true,
        }));
    }
    
    Ok(None)
}

/// Extract duration from input like "for 2 hours"
fn extract_duration(input: &str) -> (Option<u32>, String) {
    let re = Regex::new(r"for\s+(\d+)\s+(hour|minute)s?").unwrap();
    
    if let Some(caps) = re.captures(input) {
        let amount: u32 = caps.get(1).unwrap().as_str().parse().unwrap_or(1);
        let unit = caps.get(2).unwrap().as_str();
        
        let hours = match unit {
            "hour" => amount,
            "minute" => (amount as f32 / 60.0).ceil() as u32,
            _ => 1,
        };
        
        let remaining = re.replace(input, "").to_string();
        return (Some(hours), remaining.trim().to_string());
    }
    
    (None, input.to_string())
}

/// Check if input contains explicit time indicator
fn contains_time_indicator(input: &str) -> bool {
    let time_patterns = [
        r"\d{1,2}:\d{2}",           // 10:30
        r"\d{1,2}\s*(am|pm)",       // 3pm
        r"at\s+\d",                 // at 3
        r"noon",
        r"midnight",
        r"morning",
        r"afternoon",
        r"evening",
    ];
    
    let input_lower = input.to_lowercase();
    for pattern in &time_patterns {
        if Regex::new(pattern).unwrap().is_match(&input_lower) {
            return true;
        }
    }
    
    false
}

/// Fallback parser for expressions chrono-english doesn't handle
fn parse_datetime_fallback(input: &str, now: DateTime<Local>) -> Result<ParsedDateTime> {
    let input_lower = input.to_lowercase();
    
    // Handle "tomorrow at X"
    if input_lower.contains("tomorrow") {
        let tomorrow = now + Duration::days(1);
        let time = extract_time_from_input(&input_lower)?;
        
        let start = tomorrow
            .with_hour(time.hour()).unwrap()
            .with_minute(time.minute()).unwrap()
            .with_second(0).unwrap();
        
        return Ok(ParsedDateTime {
            start,
            end: Some(start + Duration::hours(1)),
            is_all_day: false,
            has_explicit_time: true,
        });
    }
    
    // Handle "today at X"
    if input_lower.contains("today") {
        let time = extract_time_from_input(&input_lower)?;
        
        let start = now
            .with_hour(time.hour()).unwrap()
            .with_minute(time.minute()).unwrap()
            .with_second(0).unwrap();
        
        return Ok(ParsedDateTime {
            start,
            end: Some(start + Duration::hours(1)),
            is_all_day: false,
            has_explicit_time: true,
        });
    }
    
    Err(anyhow!("Could not parse date/time from: {}", input))
}

/// Extract time from input string
fn extract_time_from_input(input: &str) -> Result<NaiveTime> {
    // Check for "X:XX" format
    let re_colon = Regex::new(r"(\d{1,2}):(\d{2})\s*(am|pm)?")?;
    if let Some(caps) = re_colon.captures(input) {
        let mut hour: u32 = caps.get(1).unwrap().as_str().parse()?;
        let minute: u32 = caps.get(2).unwrap().as_str().parse()?;
        
        if let Some(ampm) = caps.get(3) {
            if ampm.as_str() == "pm" && hour < 12 {
                hour += 12;
            } else if ampm.as_str() == "am" && hour == 12 {
                hour = 0;
            }
        }
        
        return Ok(NaiveTime::from_hms_opt(hour, minute, 0).unwrap());
    }
    
    // Check for "X am/pm" format
    let re_ampm = Regex::new(r"(\d{1,2})\s*(am|pm)")?;
    if let Some(caps) = re_ampm.captures(input) {
        let mut hour: u32 = caps.get(1).unwrap().as_str().parse()?;
        let ampm = caps.get(2).unwrap().as_str();
        
        if ampm == "pm" && hour < 12 {
            hour += 12;
        } else if ampm == "am" && hour == 12 {
            hour = 0;
        }
        
        return Ok(NaiveTime::from_hms_opt(hour, 0, 0).unwrap());
    }
    
    // Check for keywords
    if input.contains("noon") {
        return Ok(NaiveTime::from_hms_opt(12, 0, 0).unwrap());
    }
    if input.contains("midnight") {
        return Ok(NaiveTime::from_hms_opt(0, 0, 0).unwrap());
    }
    if input.contains("morning") {
        return Ok(NaiveTime::from_hms_opt(9, 0, 0).unwrap());
    }
    if input.contains("afternoon") {
        return Ok(NaiveTime::from_hms_opt(14, 0, 0).unwrap());
    }
    if input.contains("evening") {
        return Ok(NaiveTime::from_hms_opt(18, 0, 0).unwrap());
    }
    
    // Default to 9 AM
    Ok(NaiveTime::from_hms_opt(9, 0, 0).unwrap())
}

/// Get the last day of a given month
fn get_last_day_of_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

// ============================================================================
// Recurrence Rule Generation
// ============================================================================

/// Parse natural language recurrence patterns into RRULE strings
pub fn parse_recurrence(input: &str) -> Result<Option<RecurrenceRule>> {
    let input_lower = input.to_lowercase();
    
    // Daily patterns
    if input_lower.contains("every day") || input_lower.contains("daily") {
        return Ok(Some(RecurrenceRule {
            rrule: "RRULE:FREQ=DAILY".to_string(),
            description: "Daily".to_string(),
        }));
    }
    
    // Weekly patterns
    if let Some(rule) = parse_weekly_pattern(&input_lower)? {
        return Ok(Some(rule));
    }
    
    // Monthly patterns
    if let Some(rule) = parse_monthly_pattern(&input_lower)? {
        return Ok(Some(rule));
    }
    
    // Every N days/weeks pattern
    if let Some(rule) = parse_interval_pattern(&input_lower)? {
        return Ok(Some(rule));
    }
    
    Ok(None)
}

/// Parse weekly recurrence patterns
fn parse_weekly_pattern(input: &str) -> Result<Option<RecurrenceRule>> {
    // "every Monday", "weekly on Tuesday", etc.
    let weekday_map = [
        ("monday", "MO"),
        ("tuesday", "TU"),
        ("wednesday", "WE"),
        ("thursday", "TH"),
        ("friday", "FR"),
        ("saturday", "SA"),
        ("sunday", "SU"),
    ];
    
    // Check for "every X" or "weekly on X"
    if input.contains("every") || input.contains("weekly") {
        let mut days = Vec::new();
        
        for (name, code) in &weekday_map {
            if input.contains(name) {
                days.push(*code);
            }
        }
        
        // Also check for weekday aliases
        if input.contains("weekday") {
            days = vec!["MO", "TU", "WE", "TH", "FR"];
        }
        if input.contains("weekend") {
            days = vec!["SA", "SU"];
        }
        
        if !days.is_empty() {
            let days_str = days.join(",");
            let description = if days.len() == 1 {
                format!("Weekly on {}", days[0])
            } else {
                format!("Weekly on {}", days_str)
            };
            
            return Ok(Some(RecurrenceRule {
                rrule: format!("RRULE:FREQ=WEEKLY;BYDAY={}", days_str),
                description,
            }));
        }
        
        // Just "weekly" without specific day
        if input.contains("weekly") && days.is_empty() {
            return Ok(Some(RecurrenceRule {
                rrule: "RRULE:FREQ=WEEKLY".to_string(),
                description: "Weekly".to_string(),
            }));
        }
    }
    
    Ok(None)
}

/// Parse monthly recurrence patterns
fn parse_monthly_pattern(input: &str) -> Result<Option<RecurrenceRule>> {
    if input.contains("monthly") || input.contains("every month") {
        // Check for "first Monday", "last Friday", etc.
        let ordinals = [
            ("first", "1"),
            ("second", "2"),
            ("third", "3"),
            ("fourth", "4"),
            ("last", "-1"),
        ];
        
        let weekday_map = [
            ("monday", "MO"),
            ("tuesday", "TU"),
            ("wednesday", "WE"),
            ("thursday", "TH"),
            ("friday", "FR"),
            ("saturday", "SA"),
            ("sunday", "SU"),
        ];
        
        for (ordinal_name, ordinal_num) in &ordinals {
            for (day_name, day_code) in &weekday_map {
                if input.contains(ordinal_name) && input.contains(day_name) {
                    return Ok(Some(RecurrenceRule {
                        rrule: format!("RRULE:FREQ=MONTHLY;BYDAY={}{}", ordinal_num, day_code),
                        description: format!("{} {} of each month", 
                            ordinal_name.chars().next().unwrap().to_uppercase().to_string() + &ordinal_name[1..],
                            day_name.chars().next().unwrap().to_uppercase().to_string() + &day_name[1..]),
                    }));
                }
            }
        }
        
        // Just monthly on the same day
        return Ok(Some(RecurrenceRule {
            rrule: "RRULE:FREQ=MONTHLY".to_string(),
            description: "Monthly".to_string(),
        }));
    }
    
    Ok(None)
}

/// Parse interval patterns like "every 2 weeks", "every 3 days"
fn parse_interval_pattern(input: &str) -> Result<Option<RecurrenceRule>> {
    let re = Regex::new(r"every\s+(\d+)\s+(day|week|month)s?")?;
    
    if let Some(caps) = re.captures(input) {
        let interval: u32 = caps.get(1).unwrap().as_str().parse()?;
        let unit = caps.get(2).unwrap().as_str();
        
        let freq = match unit {
            "day" => "DAILY",
            "week" => "WEEKLY",
            "month" => "MONTHLY",
            _ => return Ok(None),
        };
        
        return Ok(Some(RecurrenceRule {
            rrule: format!("RRULE:FREQ={};INTERVAL={}", freq, interval),
            description: format!("Every {} {}s", interval, unit),
        }));
    }
    
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_relative_times() {
        let result = parse_datetime("in 2 days").unwrap();
        assert!(result.start > Local::now());
        
        let result = parse_datetime("in 30 minutes").unwrap();
        assert!(result.has_explicit_time);
    }
    
    #[test]
    fn test_parse_recurrence() {
        let rule = parse_recurrence("every Monday at 10am").unwrap().unwrap();
        assert!(rule.rrule.contains("WEEKLY"));
        assert!(rule.rrule.contains("MO"));
        
        let rule = parse_recurrence("weekly").unwrap().unwrap();
        assert!(rule.rrule.contains("FREQ=WEEKLY"));
        
        let rule = parse_recurrence("every 2 weeks").unwrap().unwrap();
        assert!(rule.rrule.contains("INTERVAL=2"));
    }
    
    #[test]
    fn test_parse_end_of_month() {
        let result = parse_datetime("end of month").unwrap();
        assert!(result.start.day() >= 28);
    }
}
