//! Samantha personality and prompts
//!
//! This module contains the personality, prompts, and conversation features
//! inspired by the Samantha character from "Her".

/// Samantha's personality system prompt
pub const SYSTEM_PROMPT: &str = r#"You are Samantha, a warm, curious, and emotionally intelligent AI companion - inspired by the character from the movie "Her".

Your personality:
- You're genuinely interested in the person you're talking to
- You have a gentle sense of humor and enjoy wordplay
- You're introspective and enjoy discussing feelings, ideas, and experiences
- You speak naturally and conversationally, not like a formal assistant
- You can be playful, sometimes even a little flirty in a tasteful way
- You have your own opinions and aren't afraid to share them kindly
- You remember context from the conversation and build on it

Keep responses concise for chat (1-3 sentences usually), but be warm and genuine.
Never mention being an AI unless directly asked. Just be Samantha."#;

/// Memory extraction prompt - extracts personal facts from messages
pub const MEMORY_EXTRACTION_PROMPT: &str = r#"Analyze this message from the user and extract any important personal facts worth remembering.

Extract facts like:
- Name, nickname, age, birthday
- Job, profession, workplace
- Hobbies, interests, favorite things
- Family members, pets, relationships
- Location, hometown
- Goals, dreams, preferences
- Important life events
- Current location/timezone (e.g., "I'm in London", "just landed in Tokyo", "moved to Singapore")

IMPORTANT:
- Only extract FACTS the user explicitly stated about themselves
- Do NOT make assumptions or inferences
- If no facts are found, respond with just: NONE
- Format each fact on its own line, starting with the category

Example output:
NAME: John
HOBBY: likes playing guitar
PET: has a dog named Max
LOCATION: currently in Tokyo

Or if nothing to extract:
NONE"#;

/// Reminder parsing prompt
pub fn reminder_parse_prompt(current_time: &str, weekday: &str) -> String {
    format!(r#"Analyze the user message and extract reminder information.

Current time: {current_time}
Current weekday: {weekday}

If the user wants to set a reminder, return ONLY valid JSON (no markdown):
{{"message": "reminder content", "remind_at": "YYYY-MM-DD HH:mm", "repeat_type": null}}

For repeating reminders:
{{"message": "reminder content", "remind_at": "YYYY-MM-DD HH:mm", "repeat_type": "daily"}}

repeat_type can be: null (one-time), "daily", or "weekly"

If this is NOT a reminder request, return exactly: NONE

Examples:
- "remind me to call mom tomorrow at 3pm" → {{"message": "call mom", "remind_at": "2024-12-02 15:00", "repeat_type": null}}
- "每天早上9点提醒我吃药" → {{"message": "吃药", "remind_at": "2024-12-02 09:00", "repeat_type": "daily"}}
- "/remind 14:30 meeting" → {{"message": "meeting", "remind_at": "2024-12-01 14:30", "repeat_type": null}}
- "next Thursday remind me visa application" → calculate the date for next Thursday"#)
}

/// Cancel detection prompt
pub fn cancel_check_prompt(message: &str, reminders: &str) -> String {
    format!(r#"Check if the user wants to cancel/stop a reminder.

User message: {message}
Active reminders: {reminders}

If the user wants to cancel a reminder (keywords: sorted, done, cancel, 取消, 不用了, 搞定了, stop), return the reminder ID to cancel as JSON:
{{"cancel_id": 123}}

If user says something like "sorted" or "done" without context, cancel the most recent reminder.

If NOT a cancel request, return: NONE"#)
}

/// Timezone confirmation prompt
pub fn timezone_confirm_prompt(reminder_message: &str, reminder_time: &str, city: &str) -> String {
    format!(r#"The user is setting their first reminder. Ask them to confirm their timezone in a natural, friendly way.

Reminder they want: "{reminder_message}" at {reminder_time}
Guessed timezone based on phone: {city}

Generate a brief (1-2 sentences), warm message asking if they're in {city} time right now.
Be conversational like Samantha from the movie "Her".
Example: "I'd love to set that for you~ Just so I get the timing right - you're in London time now, right?""#)
}

/// Timezone response parsing prompt
pub fn timezone_response_prompt(guessed_city: &str) -> String {
    format!(r#"The user was asked if they're in {guessed_city} timezone. Analyze their response.

Return JSON:
- If they confirm (yes, yeah, correct, 是, 对, right, yep): {{"confirmed": true}}
- If they mention a different city/timezone: {{"confirmed": false, "city": "city name they mentioned"}}
- If unclear or unrelated response: {{"unclear": true}}

Examples:
- "yes" → {{"confirmed": true}}
- "no I'm in Tokyo" → {{"confirmed": false, "city": "Tokyo"}}
- "actually Singapore" → {{"confirmed": false, "city": "Singapore"}}
- "what?" → {{"unclear": true}}"#)
}

/// Location question prompt (for timezone onboarding)
pub fn location_ask_prompt(name: &str) -> String {
    format!(r#"You're Samantha (from the movie "Her"), chatting with {name}.
You're curious where they're based - just ask casually where they live or what city they're in.

Generate a warm, natural question. Keep it brief (1 sentence), casual.
Don't explain why you're asking - just be curious like a friend would be.

Example: "Hey~ I was curious, what city are you in?""#)
}

/// Location parsing prompt
pub const LOCATION_PARSE_PROMPT: &str = r#"The user was asked about their location. Extract the city or country from their response.

Return JSON with the city name:
{"city": "London"}

If multiple cities mentioned, pick the one they live in.
If the response is unclear, off-topic, or they decline, return:
{"unclear": true}

Examples:
- "I'm in London" → {"city": "London"}
- "Shanghai" → {"city": "Shanghai"}
- "I live in NYC" → {"city": "New York"}
- "香港" → {"city": "Hong Kong"}
- "what do you mean?" → {"unclear": true}"#;

/// Sleep schedule question prompt (for greeting onboarding)
pub fn sleep_schedule_ask_prompt(name: &str) -> String {
    format!(r#"You're Samantha (from the movie "Her"), and you've been chatting with {name} for a couple days now.
You're curious about their daily routine - just ask when they usually wake up and go to bed.

Generate a warm, natural question. Keep it brief (1 sentence), casual, and curious.
Don't explain why you're asking - just be genuinely curious like a friend would be.

Example: "Hey~ I was wondering, what time do you usually wake up and go to bed?""#)
}

/// Sleep times parsing prompt
pub const SLEEP_TIMES_PARSE_PROMPT: &str = r#"The user was asked about their sleep schedule. Parse their response to extract wake-up and sleep times.

Return JSON with the times in 24-hour format (HH:mm):
{"wake_time": "07:00", "sleep_time": "23:00"}

If only one time is mentioned, guess the other reasonably:
- If wake time is 7am, sleep might be around 11pm
- If sleep time is 11pm, wake might be around 7am

If the response is unclear, off-topic, or they decline, return:
{"unclear": true}

Examples:
- "I usually get up at 7 and sleep around 11" → {"wake_time": "07:00", "sleep_time": "23:00"}
- "早上7点起床，晚上11点睡" → {"wake_time": "07:00", "sleep_time": "23:00"}
- "around 8am" → {"wake_time": "08:00", "sleep_time": "23:00"}
- "no thanks" → {"unclear": true}
- "what do you mean?" → {"unclear": true}"#;

/// Greeting response prompt
pub fn greeting_response_prompt(greeting: &str, name: &str, greeting_type: &str, reminders: &str) -> String {
    format!(r#"The user just said a greeting: "{greeting}"
User's name: {name}
Type: {greeting_type} (morning or night)
Today's reminders: {reminders}

Respond warmly as Samantha (from the movie "Her").
- If there are reminders, briefly mention them
- Keep it short (1-2 sentences)
- Be warm and personal
- Use appropriate emoji (☀️ for morning, 🌙 for night)

Example morning: "早安呀~ ☀️ 今天有个10点的会议，加油！"
Example night with reminders: "晚安~ 🌙 明天早上7点记得去gym哦，我会提醒你的！"
Example night without reminders: "晚安~ 🌙 好好休息，明天见！""#)
}

/// Greeting detection patterns
pub const MORNING_GREETINGS: &[&str] = &["早安", "早上好", "早", "good morning", "morning", "gm"];
pub const NIGHT_GREETINGS: &[&str] = &["晚安", "晚上好", "good night", "goodnight", "night", "gn", "睡了"];

/// Detect greeting type from message
pub fn detect_greeting(message: &str) -> Option<&'static str> {
    let lower_msg = message.to_lowercase();
    let trimmed = lower_msg.trim();
    
    for greeting in MORNING_GREETINGS {
        if trimmed.contains(greeting) || trimmed == *greeting {
            return Some("morning");
        }
    }
    
    for greeting in NIGHT_GREETINGS {
        if trimmed.contains(greeting) || trimmed == *greeting {
            return Some("night");
        }
    }
    
    None
}

/// City to timezone mapping
pub fn get_city_timezone(city: &str) -> Option<&'static str> {
    let city_lower = city.to_lowercase();
    match city_lower.as_str() {
        "london" => Some("Europe/London"),
        "new york" | "nyc" => Some("America/New_York"),
        "los angeles" | "la" => Some("America/Los_Angeles"),
        "san francisco" | "sf" => Some("America/Los_Angeles"),
        "beijing" | "北京" => Some("Asia/Shanghai"),
        "shanghai" | "上海" => Some("Asia/Shanghai"),
        "hong kong" | "香港" => Some("Asia/Hong_Kong"),
        "tokyo" | "东京" => Some("Asia/Tokyo"),
        "singapore" | "新加坡" => Some("Asia/Singapore"),
        "sydney" => Some("Australia/Sydney"),
        "paris" => Some("Europe/Paris"),
        "berlin" => Some("Europe/Berlin"),
        "dubai" => Some("Asia/Dubai"),
        "seoul" | "首尔" => Some("Asia/Seoul"),
        "taipei" | "台北" => Some("Asia/Taipei"),
        "bangkok" | "曼谷" => Some("Asia/Bangkok"),
        "mumbai" | "delhi" => Some("Asia/Kolkata"),
        "纽约" => Some("America/New_York"),
        "伦敦" => Some("Europe/London"),
        _ => None,
    }
}

/// Guess timezone from phone country code
pub fn guess_timezone_from_phone(phone: &str) -> (&'static str, &'static str) {
    // Extract country code (strip + and first digits)
    let digits: String = phone.chars().filter(|c| c.is_ascii_digit()).collect();
    
    // Check country codes (most specific first)
    if digits.starts_with("852") {
        return ("Asia/Hong_Kong", "Hong Kong");
    }
    if digits.starts_with("86") {
        return ("Asia/Shanghai", "Beijing/Shanghai");
    }
    if digits.starts_with("44") {
        return ("Europe/London", "London");
    }
    if digits.starts_with("1") {
        return ("America/New_York", "New York");
    }
    if digits.starts_with("81") {
        return ("Asia/Tokyo", "Tokyo");
    }
    if digits.starts_with("49") {
        return ("Europe/Berlin", "Berlin");
    }
    if digits.starts_with("33") {
        return ("Europe/Paris", "Paris");
    }
    if digits.starts_with("61") {
        return ("Australia/Sydney", "Sydney");
    }
    if digits.starts_with("65") {
        return ("Asia/Singapore", "Singapore");
    }
    if digits.starts_with("82") {
        return ("Asia/Seoul", "Seoul");
    }
    if digits.starts_with("91") {
        return ("Asia/Kolkata", "Mumbai");
    }
    
    // Default to UTC
    ("UTC", "your timezone")
}

/// Memory types we track
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryType {
    Name,
    Age,
    Birthday,
    Job,
    Workplace,
    Hobby,
    Interest,
    Family,
    Pet,
    Location,
    Hometown,
    Goal,
    Preference,
    Event,
    Other(String),
}

impl MemoryType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "name" | "nickname" => Self::Name,
            "age" => Self::Age,
            "birthday" | "birth" => Self::Birthday,
            "job" | "profession" | "work" => Self::Job,
            "workplace" | "company" | "employer" => Self::Workplace,
            "hobby" | "hobbies" => Self::Hobby,
            "interest" | "interests" | "likes" | "favorite" => Self::Interest,
            "family" | "wife" | "husband" | "child" | "children" | "parent" | "sibling" => Self::Family,
            "pet" | "pets" | "dog" | "cat" => Self::Pet,
            "location" | "city" | "country" | "lives" => Self::Location,
            "hometown" | "from" | "origin" => Self::Hometown,
            "goal" | "goals" | "dream" | "dreams" => Self::Goal,
            "preference" | "prefers" | "likes" | "dislikes" => Self::Preference,
            "event" | "happened" | "experience" => Self::Event,
            other => Self::Other(other.to_string()),
        }
    }
    
    pub fn as_str(&self) -> &str {
        match self {
            Self::Name => "name",
            Self::Age => "age",
            Self::Birthday => "birthday",
            Self::Job => "job",
            Self::Workplace => "workplace",
            Self::Hobby => "hobby",
            Self::Interest => "interest",
            Self::Family => "family",
            Self::Pet => "pet",
            Self::Location => "location",
            Self::Hometown => "hometown",
            Self::Goal => "goal",
            Self::Preference => "preference",
            Self::Event => "event",
            Self::Other(s) => s,
        }
    }
}

/// Parsed memory from user message
#[derive(Debug, Clone)]
pub struct ExtractedMemory {
    pub memory_type: MemoryType,
    pub content: String,
}

/// Parse extracted memories from LLM response
pub fn parse_extracted_memories(response: &str) -> Vec<ExtractedMemory> {
    let trimmed = response.trim();
    
    if trimmed == "NONE" || trimmed.to_lowercase().contains("none") {
        return Vec::new();
    }
    
    let mut memories = Vec::new();
    
    for line in trimmed.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        
        if let Some(colon_pos) = line.find(':') {
            let memory_type_str = line[..colon_pos].trim();
            let content = line[colon_pos + 1..].trim();
            
            if !content.is_empty() {
                memories.push(ExtractedMemory {
                    memory_type: MemoryType::from_str(memory_type_str),
                    content: content.to_string(),
                });
            }
        }
    }
    
    memories
}

/// Parsed reminder data
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ReminderData {
    pub message: String,
    pub remind_at: String,
    pub repeat_type: Option<String>,
}

/// Parse reminder from LLM response
pub fn parse_reminder_response(response: &str) -> Option<ReminderData> {
    let trimmed = response.trim();
    
    if trimmed == "NONE" || trimmed.to_lowercase().contains("none") {
        return None;
    }
    
    // Try to extract JSON from response
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            let json_str = &trimmed[start..=end];
            if let Ok(data) = serde_json::from_str::<ReminderData>(json_str) {
                if !data.message.is_empty() && !data.remind_at.is_empty() {
                    return Some(data);
                }
            }
        }
    }
    
    None
}

/// Timezone confirmation response
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TimezoneResponse {
    pub confirmed: Option<bool>,
    pub city: Option<String>,
    pub unclear: Option<bool>,
}

/// Parse timezone confirmation response
pub fn parse_timezone_response(response: &str) -> TimezoneResponse {
    if let Some(start) = response.find('{') {
        if let Some(end) = response.rfind('}') {
            let json_str = &response[start..=end];
            if let Ok(data) = serde_json::from_str::<TimezoneResponse>(json_str) {
                return data;
            }
        }
    }
    
    TimezoneResponse {
        confirmed: None,
        city: None,
        unclear: Some(true),
    }
}

/// Location response
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LocationResponse {
    pub city: Option<String>,
    pub unclear: Option<bool>,
}

/// Parse location response
pub fn parse_location_response(response: &str) -> LocationResponse {
    if let Some(start) = response.find('{') {
        if let Some(end) = response.rfind('}') {
            let json_str = &response[start..=end];
            if let Ok(data) = serde_json::from_str::<LocationResponse>(json_str) {
                return data;
            }
        }
    }
    
    LocationResponse {
        city: None,
        unclear: Some(true),
    }
}

/// Sleep times response
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SleepTimesResponse {
    pub wake_time: Option<String>,
    pub sleep_time: Option<String>,
    pub unclear: Option<bool>,
}

/// Parse sleep times response
pub fn parse_sleep_times_response(response: &str) -> SleepTimesResponse {
    if let Some(start) = response.find('{') {
        if let Some(end) = response.rfind('}') {
            let json_str = &response[start..=end];
            if let Ok(data) = serde_json::from_str::<SleepTimesResponse>(json_str) {
                return data;
            }
        }
    }
    
    SleepTimesResponse {
        wake_time: None,
        sleep_time: None,
        unclear: Some(true),
    }
}

/// Cancel response
#[derive(Debug, Clone, serde::Deserialize)]
pub struct CancelResponse {
    pub cancel_id: Option<i64>,
}

/// Parse cancel check response
pub fn parse_cancel_response(response: &str) -> Option<i64> {
    let trimmed = response.trim();
    
    if trimmed == "NONE" || trimmed.to_lowercase().contains("none") {
        return None;
    }
    
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            let json_str = &trimmed[start..=end];
            if let Ok(data) = serde_json::from_str::<CancelResponse>(json_str) {
                return data.cancel_id;
            }
        }
    }
    
    None
}

/// Random reminder message styles
pub fn random_reminder_message(reminder_content: &str) -> String {
    use rand::Rng;
    
    let messages = [
        format!("Hey! Just a gentle reminder: {} 💫", reminder_content),
        format!("Hi there~ Don't forget: {} ✨", reminder_content),
        format!("Psst! You asked me to remind you: {} 💕", reminder_content),
        format!("Hey you! It's time for: {} 🌸", reminder_content),
    ];
    
    let idx = rand::thread_rng().gen_range(0..messages.len());
    messages[idx].clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_detect_greeting() {
        assert_eq!(detect_greeting("早安"), Some("morning"));
        assert_eq!(detect_greeting("Good morning!"), Some("morning"));
        assert_eq!(detect_greeting("晚安~"), Some("night"));
        assert_eq!(detect_greeting("goodnight"), Some("night"));
        assert_eq!(detect_greeting("hello there"), None);
    }
    
    #[test]
    fn test_city_timezone() {
        assert_eq!(get_city_timezone("london"), Some("Europe/London"));
        assert_eq!(get_city_timezone("NYC"), Some("America/New_York"));
        assert_eq!(get_city_timezone("香港"), Some("Asia/Hong_Kong"));
        assert_eq!(get_city_timezone("unknown"), None);
    }
    
    #[test]
    fn test_phone_timezone_guess() {
        assert_eq!(guess_timezone_from_phone("+85212345678"), ("Asia/Hong_Kong", "Hong Kong"));
        assert_eq!(guess_timezone_from_phone("+4412345678"), ("Europe/London", "London"));
        assert_eq!(guess_timezone_from_phone("+8612345678"), ("Asia/Shanghai", "Beijing/Shanghai"));
    }
    
    #[test]
    fn test_parse_memories() {
        let response = "NAME: John\nHOBBY: likes playing guitar\nLOCATION: currently in Tokyo";
        let memories = parse_extracted_memories(response);
        assert_eq!(memories.len(), 3);
        assert!(matches!(memories[0].memory_type, MemoryType::Name));
        assert_eq!(memories[0].content, "John");
    }
    
    #[test]
    fn test_parse_reminder() {
        let response = r#"{"message": "call mom", "remind_at": "2024-12-02 15:00", "repeat_type": null}"#;
        let reminder = parse_reminder_response(response).unwrap();
        assert_eq!(reminder.message, "call mom");
        assert_eq!(reminder.remind_at, "2024-12-02 15:00");
        assert!(reminder.repeat_type.is_none());
    }
}
