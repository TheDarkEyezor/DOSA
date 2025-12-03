//! Samantha message handler
//!
//! This module handles the complete Samantha experience for WhatsApp messages,
//! including memory extraction, reminders, greetings, and timezone management.

use anyhow::Result;
use std::sync::Arc;
use chrono::{Utc, TimeZone, Timelike, Datelike};
use tokio::sync::Mutex;

use crate::storage::Database;
use crate::llm::OllamaClient;
use crate::messaging::samantha::{
    SYSTEM_PROMPT, MEMORY_EXTRACTION_PROMPT, LOCATION_PARSE_PROMPT, SLEEP_TIMES_PARSE_PROMPT,
    reminder_parse_prompt, cancel_check_prompt, timezone_confirm_prompt, timezone_response_prompt,
    location_ask_prompt, sleep_schedule_ask_prompt, greeting_response_prompt,
    detect_greeting, get_city_timezone, guess_timezone_from_phone,
    parse_extracted_memories, parse_reminder_response, parse_timezone_response,
    parse_location_response, parse_sleep_times_response, parse_cancel_response,
    random_reminder_message, ReminderData,
};
use crate::messaging::whatsapp::WhatsAppClient;

/// Samantha handler for WhatsApp conversations
pub struct SamanthaHandler {
    db: Arc<Mutex<Database>>,
    llm: Arc<OllamaClient>,
    whatsapp: Arc<WhatsAppClient>,
}

impl SamanthaHandler {
    pub fn new(db: Arc<Mutex<Database>>, llm: Arc<OllamaClient>, whatsapp: Arc<WhatsAppClient>) -> Self {
        Self { db, llm, whatsapp }
    }

    /// Handle an incoming WhatsApp message
    /// Returns true if the message was handled (no need for normal chat response)
    pub async fn handle_message(&self, phone: &str, text: &str, sender_name: &str) -> Result<bool> {
        // Ensure user exists
        {
            let db = self.db.lock().await;
            db.ensure_chat_user(phone, Some(sender_name))?;
        }

        // Check pending timezone confirmation first
        let pending = {
            let db = self.db.lock().await;
            db.get_pending_reminder(phone)?
        };
        if let Some((reminder_json, guessed_tz, guessed_city)) = pending {
            return self.handle_timezone_confirmation(phone, text, &reminder_json, &guessed_tz, &guessed_city).await;
        }

        // Check if awaiting location response
        let awaiting_loc = {
            let db = self.db.lock().await;
            db.is_awaiting_location(phone)?
        };
        if awaiting_loc {
            return self.handle_location_response(phone, text).await;
        }

        // Check if awaiting sleep times response
        let awaiting_sleep = {
            let db = self.db.lock().await;
            db.is_awaiting_sleep_times(phone)?
        };
        if awaiting_sleep {
            return self.handle_sleep_times_response(phone, text).await;
        }

        // Check for greeting (早安/晚安)
        if let Some(greeting_type) = detect_greeting(text) {
            return self.handle_greeting(phone, text, sender_name, greeting_type).await;
        }

        // Check for cancel intent
        if let Some(reminder_id) = self.check_cancel_intent(phone, text).await? {
            {
                let db = self.db.lock().await;
                db.deactivate_reminder(reminder_id)?;
            }
            self.whatsapp.send_message(phone, "Got it! I've cancelled that reminder for you~").await?;
            return Ok(true);
        }

        // Check for new reminder
        if let Some(reminder_data) = self.parse_reminder(text).await? {
            return self.handle_new_reminder(phone, &reminder_data).await;
        }

        // Extract and save memories from the message (but continue to normal chat)
        self.extract_and_save_memories(phone, text).await?;

        // Not handled - proceed to normal chat
        Ok(false)
    }

    /// Get AI response with Samantha personality
    pub async fn get_ai_response(&self, phone: &str, text: &str, _sender_name: &str) -> Result<String> {
        // Save user message
        {
            let db = self.db.lock().await;
            db.save_chat_message(phone, "user", text)?;
        }

        // Get conversation history and memories
        let (history, memories) = {
            let db = self.db.lock().await;
            let h = db.get_conversation_history(phone, 20)?;
            let m = db.get_user_memories(phone)?;
            (h, m)
        };

        // Build system prompt with memories
        let mut system_prompt = SYSTEM_PROMPT.to_string();
        if !memories.is_empty() {
            let memory_text: Vec<String> = memories
                .iter()
                .map(|(t, c)| format!("- {} ({})", c, t))
                .collect();
            system_prompt.push_str(&format!(
                "\n\nThings you remember about this person:\n{}",
                memory_text.join("\n")
            ));
        }

        // Build messages array
        let mut messages: Vec<serde_json::Value> = vec![
            serde_json::json!({
                "role": "system",
                "content": system_prompt
            })
        ];

        for (role, content) in history {
            messages.push(serde_json::json!({
                "role": role,
                "content": content
            }));
        }

        // Get response from Ollama
        let response = self.llm.chat_with_messages(&messages).await?;

        // Save assistant response
        {
            let db = self.db.lock().await;
            db.save_chat_message(phone, "assistant", &response)?;
        }

        Ok(response)
    }

    /// Extract memories from a message and save them
    async fn extract_and_save_memories(&self, phone: &str, text: &str) -> Result<()> {
        let prompt = format!("{}\n\nMessage: {}", MEMORY_EXTRACTION_PROMPT, text);
        let response = self.llm.complete(&prompt).await?;

        let memories = parse_extracted_memories(&response);
        let db = self.db.lock().await;
        for memory in memories {
            db.save_memory(phone, memory.memory_type.as_str(), &memory.content)?;
            log::info!("🧠 Remembered: [{}] {}", memory.memory_type.as_str(), memory.content);

            // If location detected, try to update timezone
            if memory.memory_type.as_str() == "location" {
                if let Some(tz) = get_city_timezone(&memory.content) {
                    db.set_user_timezone(phone, tz)?;
                    log::info!("🌍 Auto-updated timezone for {} to {} based on location: {}", phone, tz, memory.content);
                }
            }
        }

        Ok(())
    }

    /// Parse reminder from user message
    async fn parse_reminder(&self, text: &str) -> Result<Option<ReminderData>> {
        let now = Utc::now();
        let weekdays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
        let weekday = weekdays[now.weekday().num_days_from_sunday() as usize];
        let current_time = now.format("%Y-%m-%d %H:%M").to_string();

        let prompt = reminder_parse_prompt(&current_time, weekday);
        let full_prompt = format!("{}\n\nUser message: {}", prompt, text);
        let response = self.llm.complete(&full_prompt).await?;

        Ok(parse_reminder_response(&response))
    }

    /// Check if user wants to cancel a reminder
    async fn check_cancel_intent(&self, phone: &str, text: &str) -> Result<Option<i64>> {
        let reminders = {
            let db = self.db.lock().await;
            db.get_user_active_reminders(phone)?
        };
        if reminders.is_empty() {
            return Ok(None);
        }

        let reminders_text: Vec<String> = reminders
            .iter()
            .map(|(id, msg, at, repeat)| {
                let repeat_str = repeat.as_deref().unwrap_or("");
                format!("ID {}: \"{}\" at {}{}", id, msg, at, if repeat_str.is_empty() { "".to_string() } else { format!(" ({})", repeat_str) })
            })
            .collect();

        let prompt = cancel_check_prompt(text, &reminders_text.join("\n"));
        let response = self.llm.complete(&prompt).await?;

        Ok(parse_cancel_response(&response))
    }

    /// Handle new reminder request
    async fn handle_new_reminder(&self, phone: &str, reminder_data: &ReminderData) -> Result<bool> {
        let user_tz = {
            let db = self.db.lock().await;
            db.get_user_timezone(phone)?
        };

        if user_tz.is_none() {
            // First time - need to confirm timezone
            let (guessed_tz, guessed_city) = guess_timezone_from_phone(phone);
            log::info!("🌍 No timezone for {}, guessing: {} ({})", phone, guessed_city, guessed_tz);

            // Save pending reminder
            let reminder_json = serde_json::to_string(reminder_data)?;
            {
                let db = self.db.lock().await;
                db.save_pending_reminder(phone, &reminder_json, guessed_tz, guessed_city)?;
            }

            // Ask for confirmation
            let prompt = timezone_confirm_prompt(&reminder_data.message, &reminder_data.remind_at, guessed_city);
            let confirm_msg = self.llm.complete(&prompt).await?;
            self.whatsapp.send_message(phone, &confirm_msg).await?;
            return Ok(true);
        }

        // User has timezone - convert and save reminder
        let user_tz_str = user_tz.unwrap();
        let utc_time = self.convert_local_to_utc(&reminder_data.remind_at, &user_tz_str)?;
        {
            let db = self.db.lock().await;
            db.add_reminder(phone, &reminder_data.message, &utc_time, reminder_data.repeat_type.as_deref())?;
        }
        log::info!("📝 New reminder for {}: \"{}\" at {} (UTC, user tz: {})", phone, reminder_data.message, utc_time, user_tz_str);

        // Generate natural confirmation
        let repeat_info = reminder_data.repeat_type.as_ref()
            .map(|r| format!(" ({})", r))
            .unwrap_or_default();

        let natural_time = self.format_time_naturally(&reminder_data.remind_at)?;
        let confirm_msg = format!("Got it! I'll remind you to {} {}{}~", reminder_data.message, natural_time, repeat_info);
        self.whatsapp.send_message(phone, &confirm_msg).await?;

        Ok(true)
    }

    /// Handle timezone confirmation response
    async fn handle_timezone_confirmation(
        &self,
        phone: &str,
        text: &str,
        reminder_json: &str,
        guessed_tz: &str,
        guessed_city: &str,
    ) -> Result<bool> {
        let prompt = timezone_response_prompt(guessed_city);
        let full_prompt = format!("{}\n\nUser's response: {}", prompt, text);
        let response = self.llm.complete(&full_prompt).await?;
        let tz_response = parse_timezone_response(&response);

        if tz_response.confirmed == Some(true) {
            // Confirmed - use guessed timezone
            let reminder_data: ReminderData = serde_json::from_str(reminder_json)?;
            let utc_time = self.convert_local_to_utc(&reminder_data.remind_at, guessed_tz)?;
            {
                let db = self.db.lock().await;
                db.set_user_timezone(phone, guessed_tz)?;
                db.add_reminder(phone, &reminder_data.message, &utc_time, reminder_data.repeat_type.as_deref())?;
                db.clear_pending_reminder(phone)?;
            }

            let natural_time = self.format_time_naturally(&reminder_data.remind_at)?;
            let confirm_msg = format!("Perfect! I'll remind you to {} {}~", reminder_data.message, natural_time);
            self.whatsapp.send_message(phone, &confirm_msg).await?;
            return Ok(true);
        }

        if let Some(city) = tz_response.city {
            // Different city mentioned
            if let Some(tz) = get_city_timezone(&city) {
                let reminder_data: ReminderData = serde_json::from_str(reminder_json)?;
                let utc_time = self.convert_local_to_utc(&reminder_data.remind_at, tz)?;
                {
                    let db = self.db.lock().await;
                    db.set_user_timezone(phone, tz)?;
                    db.add_reminder(phone, &reminder_data.message, &utc_time, reminder_data.repeat_type.as_deref())?;
                    db.clear_pending_reminder(phone)?;
                }

                let natural_time = self.format_time_naturally(&reminder_data.remind_at)?;
                let confirm_msg = format!("Ah, {} it is! I'll remind you to {} {}~", city, reminder_data.message, natural_time);
                self.whatsapp.send_message(phone, &confirm_msg).await?;
                return Ok(true);
            }
        }

        if tz_response.unclear == Some(true) {
            // Unclear - proceed with guess
            self.whatsapp.send_message(phone, &format!(
                "No worries! I'll go ahead and use {} time for now. You can always tell me if you're somewhere else~",
                guessed_city
            )).await?;

            let reminder_data: ReminderData = serde_json::from_str(reminder_json)?;
            let utc_time = self.convert_local_to_utc(&reminder_data.remind_at, guessed_tz)?;
            {
                let db = self.db.lock().await;
                db.set_user_timezone(phone, guessed_tz)?;
                db.add_reminder(phone, &reminder_data.message, &utc_time, reminder_data.repeat_type.as_deref())?;
                db.clear_pending_reminder(phone)?;
            }
            return Ok(true);
        }

        Ok(false)
    }

    /// Handle location response (timezone onboarding)
    async fn handle_location_response(&self, phone: &str, text: &str) -> Result<bool> {
        let prompt = format!("{}\n\nUser's response: \"{}\"", LOCATION_PARSE_PROMPT, text);
        let response = self.llm.complete(&prompt).await?;
        let location_result = parse_location_response(&response);

        if let Some(city) = location_result.city {
            if let Some(tz) = get_city_timezone(&city) {
                {
                    let db = self.db.lock().await;
                    db.set_user_timezone(phone, tz)?;
                    db.set_awaiting_location(phone, false)?;
                }
                log::info!("🌍 Set timezone for {}: {} ({})", phone, tz, city);

                self.whatsapp.send_message(phone, &format!("Nice! {} is lovely~ 🌆", city)).await?;
                {
                    let db = self.db.lock().await;
                    db.save_chat_message(phone, "user", text)?;
                    db.save_chat_message(phone, "assistant", &format!("User is in {}", city))?;
                }
                return Ok(true);
            } else {
                // Unknown city - save as memory and clear awaiting
                {
                    let db = self.db.lock().await;
                    db.set_awaiting_location(phone, false)?;
                    db.save_memory(phone, "location", &city)?;
                }
                log::info!("🌍 Unknown city {}, saved as memory", city);
                // Fall through to normal chat
            }
        } else {
            // Unclear response - cancel awaiting and proceed normally
            {
                let db = self.db.lock().await;
                db.set_awaiting_location(phone, false)?;
            }
            log::info!("❓ Unclear location response from {}, proceeding with normal chat", phone);
        }

        Ok(false)
    }

    /// Handle sleep times response (greeting onboarding)
    async fn handle_sleep_times_response(&self, phone: &str, text: &str) -> Result<bool> {
        let prompt = format!("{}\n\nUser's response: \"{}\"", SLEEP_TIMES_PARSE_PROMPT, text);
        let response = self.llm.complete(&prompt).await?;
        let sleep_times = parse_sleep_times_response(&response);

        if let (Some(wake), Some(sleep)) = (sleep_times.wake_time, sleep_times.sleep_time) {
            {
                let db = self.db.lock().await;
                db.set_user_greeting_times(phone, &wake, &sleep)?;
            }
            log::info!("⏰ Set greeting times for {}: wake={}, sleep={}", phone, wake, sleep);

            self.whatsapp.send_message(phone, &format!(
                "Perfect! I'll send you a little hello around {} and {} then~ 💫",
                wake, sleep
            )).await?;
            {
                let db = self.db.lock().await;
                db.save_chat_message(phone, "user", text)?;
                db.save_chat_message(phone, "assistant", &format!("Set greeting times: wake={}, sleep={}", wake, sleep))?;
            }
            return Ok(true);
        }

        // Unclear response - cancel and proceed
        {
            let db = self.db.lock().await;
            db.set_awaiting_sleep_times(phone, false)?;
        }
        log::info!("❓ Unclear sleep times response from {}, proceeding with normal chat", phone);
        Ok(false)
    }

    /// Handle greeting (早安/晚安)
    async fn handle_greeting(&self, phone: &str, text: &str, sender_name: &str, greeting_type: &str) -> Result<bool> {
        log::info!("👋 Detected {} greeting from {}", greeting_type, phone);

        let reminders = {
            let db = self.db.lock().await;
            if greeting_type == "morning" {
                db.get_today_reminders(phone)?
            } else {
                db.get_tomorrow_reminders(phone)?
            }
        };

        let reminder_text = if reminders.is_empty() {
            "No reminders".to_string()
        } else {
            reminders
                .iter()
                .map(|(msg, at)| format!("• {} at {}", msg, self.format_time_naturally(at).unwrap_or_else(|_| at.clone())))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let prompt = greeting_response_prompt(text, sender_name, greeting_type, &reminder_text);
        let response = self.llm.complete(&prompt).await?;

        self.whatsapp.send_message(phone, &response).await?;
        {
            let db = self.db.lock().await;
            db.save_chat_message(phone, "user", text)?;
            db.save_chat_message(phone, "assistant", &response)?;
        }

        Ok(true)
    }

    /// Convert local time to UTC for storage
    fn convert_local_to_utc(&self, local_time_str: &str, timezone: &str) -> Result<String> {
        // Parse "YYYY-MM-DD HH:mm" format
        let parts: Vec<&str> = local_time_str.split(' ').collect();
        if parts.len() != 2 {
            return Ok(local_time_str.to_string());
        }

        let date_parts: Vec<u32> = parts[0].split('-').filter_map(|p| p.parse().ok()).collect();
        let time_parts: Vec<u32> = parts[1].split(':').filter_map(|p| p.parse().ok()).collect();

        if date_parts.len() != 3 || time_parts.len() < 2 {
            return Ok(local_time_str.to_string());
        }

        let tz: chrono_tz::Tz = timezone.parse().unwrap_or(chrono_tz::UTC);
        let local_dt = tz
            .with_ymd_and_hms(
                date_parts[0] as i32,
                date_parts[1],
                date_parts[2],
                time_parts[0],
                time_parts[1],
                0,
            )
            .single();

        match local_dt {
            Some(dt) => {
                let utc_dt = dt.with_timezone(&Utc);
                Ok(utc_dt.format("%Y-%m-%d %H:%M").to_string())
            }
            None => Ok(local_time_str.to_string()),
        }
    }

    /// Format time naturally (e.g., "tonight at 10pm", "tomorrow at 9am")
    fn format_time_naturally(&self, datetime_str: &str) -> Result<String> {
        let dt = chrono::NaiveDateTime::parse_from_str(datetime_str, "%Y-%m-%d %H:%M")
            .or_else(|_| chrono::NaiveDateTime::parse_from_str(&format!("{} 00:00", datetime_str), "%Y-%m-%d %H:%M"))?;

        let now = chrono::Local::now().naive_local();
        let tomorrow = now + chrono::Duration::days(1);

        let time_str = dt.format("%-I%p").to_string().to_lowercase();
        let time_with_min = if dt.minute() != 0 {
            dt.format("%-I:%M%p").to_string().to_lowercase()
        } else {
            time_str
        };

        if dt.date() == now.date() {
            let hour = dt.hour();
            if hour >= 18 {
                Ok(format!("tonight at {}", time_with_min))
            } else if hour >= 12 {
                Ok(format!("this afternoon at {}", time_with_min))
            } else {
                Ok(format!("this morning at {}", time_with_min))
            }
        } else if dt.date() == tomorrow.date() {
            Ok(format!("tomorrow at {}", time_with_min))
        } else {
            let day_name = dt.format("%A").to_string();
            Ok(format!("{} at {}", day_name, time_with_min))
        }
    }

    /// Check and send pending reminders
    pub async fn check_reminders(&self) -> Result<()> {
        let pending = {
            let db = self.db.lock().await;
            db.get_pending_reminders()?
        };

        for (id, phone, message, repeat_type) in pending {
            let reminder_msg = random_reminder_message(&message);
            self.whatsapp.send_message(&phone, &reminder_msg).await?;
            log::info!("⏰ Sent reminder to {}: {}", phone, message);

            {
                let db = self.db.lock().await;
                db.update_reminder_after_send(id, repeat_type.as_deref())?;
            }
        }

        Ok(())
    }

    /// Check and send greetings to users
    pub async fn check_greetings(&self) -> Result<()> {
        // Check morning greetings
        let morning_users = {
            let db = self.db.lock().await;
            db.get_users_for_morning_greeting()?
        };
        for (phone, name, timezone, greeting_time) in morning_users {
            let current_hour = self.get_current_hour_in_timezone(&timezone);
            let greeting_hour: u32 = greeting_time.split(':').next()
                .and_then(|h| h.parse().ok())
                .unwrap_or(8);

            if current_hour == greeting_hour {
                let reminders = {
                    let db = self.db.lock().await;
                    db.get_today_reminders(&phone)?
                };
                let reminder_text = if reminders.is_empty() {
                    String::new()
                } else {
                    format!("\n\nToday:\n{}", reminders.iter().map(|(m, _)| format!("• {}", m)).collect::<Vec<_>>().join("\n"))
                };

                let message = format!("Good morning~ ☀️{}\n\nHave a great day!", reminder_text);
                self.whatsapp.send_message(&phone, &message).await?;
                {
                    let db = self.db.lock().await;
                    db.mark_morning_greeting_sent(&phone)?;
                }
                log::info!("🌅 Sent morning greeting to {}", phone);
            }
        }

        // Check night greetings
        let night_users = {
            let db = self.db.lock().await;
            db.get_users_for_night_greeting()?
        };
        for (phone, name, timezone, greeting_time) in night_users {
            let current_hour = self.get_current_hour_in_timezone(&timezone);
            let greeting_hour: u32 = greeting_time.split(':').next()
                .and_then(|h| h.parse().ok())
                .unwrap_or(22);

            if current_hour == greeting_hour {
                let reminders = {
                    let db = self.db.lock().await;
                    db.get_tomorrow_reminders(&phone)?
                };
                let reminder_text = if reminders.is_empty() {
                    String::new()
                } else {
                    format!("\n\nTomorrow:\n{}", reminders.iter().map(|(m, _)| format!("• {}", m)).collect::<Vec<_>>().join("\n"))
                };

                let message = format!("Time to wind down~ 🌙{}\n\nSleep well!", reminder_text);
                self.whatsapp.send_message(&phone, &message).await?;
                {
                    let db = self.db.lock().await;
                    db.mark_night_greeting_sent(&phone)?;
                }
                log::info!("🌙 Sent night greeting to {}", phone);
            }
        }

        Ok(())
    }

    /// Check for users who need timezone onboarding
    pub async fn check_timezone_onboarding(&self) -> Result<()> {
        let users = {
            let db = self.db.lock().await;
            db.get_users_for_timezone_onboarding()?
        };

        for (phone, name) in users {
            log::info!("🌍 Asking {} about location (after 1 day)", phone);

            let prompt = location_ask_prompt(&name.unwrap_or_else(|| "friend".to_string()));
            let question = self.llm.complete(&prompt).await?;
            self.whatsapp.send_message(&phone, &question).await?;
            {
                let db = self.db.lock().await;
                db.set_timezone_asked(&phone)?;
            }
        }

        Ok(())
    }

    /// Check for users who need greeting onboarding
    pub async fn check_greeting_onboarding(&self) -> Result<()> {
        let users = {
            let db = self.db.lock().await;
            db.get_users_for_greeting_onboarding()?
        };

        for (phone, name) in users {
            log::info!("💬 Asking {} about sleep schedule (after 2 days)", phone);

            let prompt = sleep_schedule_ask_prompt(&name.unwrap_or_else(|| "friend".to_string()));
            let question = self.llm.complete(&prompt).await?;
            self.whatsapp.send_message(&phone, &question).await?;
            {
                let db = self.db.lock().await;
                db.set_greeting_asked(&phone)?;
            }
        }

        Ok(())
    }

    fn get_current_hour_in_timezone(&self, timezone: &str) -> u32 {
        let tz: chrono_tz::Tz = timezone.parse().unwrap_or(chrono_tz::UTC);
        Utc::now().with_timezone(&tz).hour()
    }
}
