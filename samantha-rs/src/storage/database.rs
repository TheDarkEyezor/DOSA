use anyhow::Result;
use rusqlite::{Connection, params};
use std::path::Path;

/// Database wrapper for SQLite operations
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Open or create a database at the given path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path)?;
        let db = Database { conn };
        db.initialize_schema()?;
        Ok(db)
    }

    /// Create an in-memory database (for testing)
    #[allow(dead_code)]
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let db = Database { conn };
        db.initialize_schema()?;
        Ok(db)
    }

    /// Initialize the database schema
    fn initialize_schema(&self) -> Result<()> {
        // Entities table - stores all nodes in the knowledge graph
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Entity properties - key-value pairs for entity attributes
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS entity_properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                UNIQUE(entity_id, key)
            )",
            [],
        )?;

        // Relationships table - edges in the knowledge graph
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_entity_id INTEGER NOT NULL,
                to_entity_id INTEGER NOT NULL,
                relationship_type TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (from_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (to_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                UNIQUE(from_entity_id, to_entity_id, relationship_type)
            )",
            [],
        )?;

        // Relationship properties - additional data on relationships
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS relationship_properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                relationship_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                FOREIGN KEY (relationship_id) REFERENCES relationships(id) ON DELETE CASCADE,
                UNIQUE(relationship_id, key)
            )",
            [],
        )?;

        // Create indexes for faster queries
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type)",
            [],
        )?;

        // Settings table - key-value store for app settings (like user identity)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )",
            [],
        )?;

        // ============= Samantha/WhatsApp tables =============
        
        // Chat users table - stores messaging users with preferences
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS chat_users (
                phone TEXT PRIMARY KEY,
                name TEXT,
                timezone TEXT,
                morning_greeting_time TEXT,
                night_greeting_time TEXT,
                morning_greeting_sent_today INTEGER DEFAULT 0,
                night_greeting_sent_today INTEGER DEFAULT 0,
                timezone_asked INTEGER DEFAULT 0,
                greeting_asked INTEGER DEFAULT 0,
                awaiting_location INTEGER DEFAULT 0,
                awaiting_sleep_times INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_active DATETIME
            )",
            [],
        )?;

        // Chat messages - conversation history per user
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_phone TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_phone) REFERENCES chat_users(phone)
            )",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_user ON chat_messages(user_phone)",
            [],
        )?;

        // Chat memories - facts extracted about users
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS chat_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_phone TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_phone) REFERENCES chat_users(phone)
            )",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user ON chat_memories(user_phone)",
            [],
        )?;

        // Reminders table
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_phone TEXT NOT NULL,
                message TEXT NOT NULL,
                remind_at DATETIME NOT NULL,
                repeat_type TEXT,
                is_active INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_phone) REFERENCES chat_users(phone)
            )",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reminders_user ON reminders(user_phone)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reminders_time ON reminders(remind_at)",
            [],
        )?;

        // Pending reminders (awaiting timezone confirmation)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS pending_reminders (
                user_phone TEXT PRIMARY KEY,
                reminder_data TEXT NOT NULL,
                guessed_timezone TEXT NOT NULL,
                guessed_city TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_phone) REFERENCES chat_users(phone)
            )",
            [],
        )?;

        Ok(())
    }

    /// Get a setting value
    pub fn get_setting(&self, key: &str) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare("SELECT value FROM settings WHERE key = ?1")?;
        let result = stmt.query_row(params![key], |row| row.get(0));
        match result {
            Ok(value) => Ok(Some(value)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Set a setting value
    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO settings (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = ?2",
            params![key, value],
        )?;
        Ok(())
    }

    /// Delete a setting
    pub fn delete_setting(&self, key: &str) -> Result<bool> {
        let rows = self.conn.execute("DELETE FROM settings WHERE key = ?1", params![key])?;
        Ok(rows > 0)
    }

    /// List all settings with a given prefix
    pub fn list_settings_by_prefix(&self, prefix: &str) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT key, value FROM settings WHERE key LIKE ?1 ORDER BY key"
        )?;
        let pattern = format!("{}%", prefix);
        let rows = stmt.query_map(params![pattern], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Get a reference to the connection for direct queries
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    /// Insert a new entity and return its ID
    pub fn insert_entity(&self, entity_type: &str, name: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO entities (entity_type, name) VALUES (?1, ?2)",
            params![entity_type, name],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Set a property on an entity (upsert)
    pub fn set_entity_property(&self, entity_id: i64, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO entity_properties (entity_id, key, value) 
             VALUES (?1, ?2, ?3)
             ON CONFLICT(entity_id, key) DO UPDATE SET value = ?3",
            params![entity_id, key, value],
        )?;
        Ok(())
    }

    /// Get an entity property
    pub fn get_entity_property(&self, entity_id: i64, key: &str) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT value FROM entity_properties WHERE entity_id = ?1 AND key = ?2"
        )?;
        let result = stmt.query_row(params![entity_id, key], |row| row.get(0));
        match result {
            Ok(value) => Ok(Some(value)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Create a relationship between two entities
    pub fn create_relationship(
        &self,
        from_id: i64,
        to_id: i64,
        relationship_type: &str,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT OR IGNORE INTO relationships (from_entity_id, to_entity_id, relationship_type) 
             VALUES (?1, ?2, ?3)",
            params![from_id, to_id, relationship_type],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Find entity by name and type
    pub fn find_entity(&self, entity_type: &str, name: &str) -> Result<Option<i64>> {
        // First try exact match (case-insensitive)
        let mut stmt = self.conn.prepare(
            "SELECT id FROM entities WHERE entity_type = ?1 AND LOWER(name) = LOWER(?2)"
        )?;
        let result = stmt.query_row(params![entity_type, name], |row| row.get(0));
        match result {
            Ok(id) => return Ok(Some(id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => {},
            Err(e) => return Err(e.into()),
        }

        // Try partial match (name contains search term, or search term is first/last name)
        let search_lower = name.to_lowercase();
        let mut stmt = self.conn.prepare(
            "SELECT id, name FROM entities WHERE entity_type = ?1"
        )?;
        let mut rows = stmt.query(params![entity_type])?;
        
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let full_name: String = row.get(1)?;
            let full_name_lower = full_name.to_lowercase();
            
            // Check if search term matches first name, last name, or is contained in full name
            let name_parts: Vec<&str> = full_name_lower.split_whitespace().collect();
            
            // Match if:
            // 1. Full name contains the search term
            // 2. First name matches
            // 3. Last name matches
            // 4. Search term starts with first name (e.g., "Amogh's" matches "Amogh")
            let search_clean = search_lower.trim_end_matches("'s").trim_end_matches("'");
            
            if full_name_lower.contains(&search_lower) ||
               name_parts.first().map(|f| *f == search_clean).unwrap_or(false) ||
               name_parts.last().map(|l| *l == search_clean).unwrap_or(false) ||
               name_parts.first().map(|f| search_clean.starts_with(f)).unwrap_or(false) {
                return Ok(Some(id));
            }
        }
        
        Ok(None)
    }

    /// Find entity by name (any type)
    pub fn find_entity_by_name(&self, name: &str) -> Result<Option<(i64, String)>> {
        // First try exact match
        let mut stmt = self.conn.prepare(
            "SELECT id, entity_type FROM entities WHERE LOWER(name) = LOWER(?1)"
        )?;
        let result = stmt.query_row(params![name], |row| {
            Ok((row.get(0)?, row.get(1)?))
        });
        match result {
            Ok(data) => return Ok(Some(data)),
            Err(rusqlite::Error::QueryReturnedNoRows) => {},
            Err(e) => return Err(e.into()),
        }

        // Try partial match for people
        let search_lower = name.to_lowercase();
        let search_clean = search_lower.trim_end_matches("'s").trim_end_matches("'");
        
        let mut stmt = self.conn.prepare(
            "SELECT id, entity_type, name FROM entities"
        )?;
        let mut rows = stmt.query([])?;
        
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let entity_type: String = row.get(1)?;
            let full_name: String = row.get(2)?;
            let full_name_lower = full_name.to_lowercase();
            
            let name_parts: Vec<&str> = full_name_lower.split_whitespace().collect();
            
            if full_name_lower.contains(&search_lower) ||
               name_parts.first().map(|f| *f == search_clean).unwrap_or(false) ||
               name_parts.last().map(|l| *l == search_clean).unwrap_or(false) {
                return Ok(Some((id, entity_type)));
            }
        }
        
        Ok(None)
    }

    /// Find entity by property value (e.g., email)
    pub fn find_entity_by_property(&self, key: &str, value: &str) -> Result<Option<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT entity_id FROM entity_properties WHERE LOWER(key) = LOWER(?1) AND LOWER(value) = LOWER(?2) LIMIT 1"
        )?;
        let result = stmt.query_row(params![key, value], |row| row.get(0));
        match result {
            Ok(id) => Ok(Some(id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// List all entities of a given type
    pub fn list_entities(&self, entity_type: &str) -> Result<Vec<(i64, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name FROM entities WHERE entity_type = ?1 ORDER BY name"
        )?;
        let rows = stmt.query_map(params![entity_type], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// List all entities (id, name, type)
    pub fn list_all_entities(&self) -> Result<Vec<(i64, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, entity_type FROM entities ORDER BY entity_type, name"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Get all properties for an entity
    pub fn get_entity_properties(&self, entity_id: i64) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT key, value FROM entity_properties WHERE entity_id = ?1"
        )?;
        let rows = stmt.query_map(params![entity_id], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Get relationships from an entity
    pub fn get_relationships_from(&self, entity_id: i64) -> Result<Vec<(i64, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT r.to_entity_id, r.relationship_type, e.name 
             FROM relationships r
             JOIN entities e ON r.to_entity_id = e.id
             WHERE r.from_entity_id = ?1"
        )?;
        let rows = stmt.query_map(params![entity_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Get relationships to an entity
    pub fn get_relationships_to(&self, entity_id: i64) -> Result<Vec<(i64, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT r.from_entity_id, r.relationship_type, e.name 
             FROM relationships r
             JOIN entities e ON r.from_entity_id = e.id
             WHERE r.to_entity_id = ?1"
        )?;
        let rows = stmt.query_map(params![entity_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Delete an entity by ID (also deletes related properties and relationships via CASCADE)
    pub fn delete_entity(&self, entity_id: i64) -> Result<bool> {
        let rows = self.conn.execute(
            "DELETE FROM entities WHERE id = ?1",
            params![entity_id],
        )?;
        Ok(rows > 0)
    }

    /// Update an entity's name
    pub fn update_entity_name(&self, entity_id: i64, new_name: &str) -> Result<bool> {
        let rows = self.conn.execute(
            "UPDATE entities SET name = ?1, updated_at = CURRENT_TIMESTAMP WHERE id = ?2",
            params![new_name, entity_id],
        )?;
        Ok(rows > 0)
    }

    /// Delete a specific relationship
    pub fn delete_relationship(&self, from_id: i64, to_id: i64, rel_type: &str) -> Result<bool> {
        let rows = self.conn.execute(
            "DELETE FROM relationships WHERE from_entity_id = ?1 AND to_entity_id = ?2 AND relationship_type = ?3",
            params![from_id, to_id, rel_type],
        )?;
        Ok(rows > 0)
    }

    /// Delete a property from an entity
    pub fn delete_entity_property(&self, entity_id: i64, key: &str) -> Result<bool> {
        let rows = self.conn.execute(
            "DELETE FROM entity_properties WHERE entity_id = ?1 AND key = ?2",
            params![entity_id, key],
        )?;
        Ok(rows > 0)
    }

    /// Get entity name by ID
    pub fn get_entity_name(&self, entity_id: i64) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT name FROM entities WHERE id = ?1"
        )?;
        let result = stmt.query_row(params![entity_id], |row| row.get(0));
        match result {
            Ok(name) => Ok(Some(name)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get entity name and type by ID
    pub fn get_entity_by_id(&self, entity_id: i64) -> Result<Option<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, entity_type FROM entities WHERE id = ?1"
        )?;
        let result = stmt.query_row(params![entity_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        });
        match result {
            Ok(data) => Ok(Some(data)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    // ============= Samantha/Chat User Methods =============

    /// Ensure a chat user exists (create if not)
    pub fn ensure_chat_user(&self, phone: &str, name: Option<&str>) -> Result<()> {
        self.conn.execute(
            "INSERT INTO chat_users (phone, name, last_active) VALUES (?1, ?2, CURRENT_TIMESTAMP)
             ON CONFLICT(phone) DO UPDATE SET last_active = CURRENT_TIMESTAMP, name = COALESCE(?2, name)",
            params![phone, name],
        )?;
        Ok(())
    }

    /// Get chat user timezone
    pub fn get_user_timezone(&self, phone: &str) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare("SELECT timezone FROM chat_users WHERE phone = ?1")?;
        let result = stmt.query_row(params![phone], |row| row.get(0));
        match result {
            Ok(tz) => Ok(tz),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Set chat user timezone
    pub fn set_user_timezone(&self, phone: &str, timezone: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE chat_users SET timezone = ?1 WHERE phone = ?2",
            params![timezone, phone],
        )?;
        Ok(())
    }

    /// Check if user is awaiting location response
    pub fn is_awaiting_location(&self, phone: &str) -> Result<bool> {
        let mut stmt = self.conn.prepare(
            "SELECT awaiting_location FROM chat_users WHERE phone = ?1"
        )?;
        let result = stmt.query_row(params![phone], |row| row.get::<_, i32>(0));
        match result {
            Ok(val) => Ok(val == 1),
            Err(_) => Ok(false),
        }
    }

    /// Set awaiting location flag
    pub fn set_awaiting_location(&self, phone: &str, awaiting: bool) -> Result<()> {
        self.conn.execute(
            "UPDATE chat_users SET awaiting_location = ?1 WHERE phone = ?2",
            params![if awaiting { 1 } else { 0 }, phone],
        )?;
        Ok(())
    }

    /// Check if user is awaiting sleep times response
    pub fn is_awaiting_sleep_times(&self, phone: &str) -> Result<bool> {
        let mut stmt = self.conn.prepare(
            "SELECT awaiting_sleep_times FROM chat_users WHERE phone = ?1"
        )?;
        let result = stmt.query_row(params![phone], |row| row.get::<_, i32>(0));
        match result {
            Ok(val) => Ok(val == 1),
            Err(_) => Ok(false),
        }
    }

    /// Set awaiting sleep times flag
    pub fn set_awaiting_sleep_times(&self, phone: &str, awaiting: bool) -> Result<()> {
        self.conn.execute(
            "UPDATE chat_users SET awaiting_sleep_times = ?1 WHERE phone = ?2",
            params![if awaiting { 1 } else { 0 }, phone],
        )?;
        Ok(())
    }

    /// Set user greeting times
    pub fn set_user_greeting_times(&self, phone: &str, morning: &str, night: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE chat_users SET morning_greeting_time = ?1, night_greeting_time = ?2, awaiting_sleep_times = 0 WHERE phone = ?3",
            params![morning, night, phone],
        )?;
        Ok(())
    }

    /// Mark timezone asked
    pub fn set_timezone_asked(&self, phone: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE chat_users SET timezone_asked = 1, awaiting_location = 1 WHERE phone = ?1",
            params![phone],
        )?;
        Ok(())
    }

    /// Mark greeting asked
    pub fn set_greeting_asked(&self, phone: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE chat_users SET greeting_asked = 1, awaiting_sleep_times = 1 WHERE phone = ?1",
            params![phone],
        )?;
        Ok(())
    }

    /// Get users for timezone onboarding (after 1 day, no timezone set)
    pub fn get_users_for_timezone_onboarding(&self) -> Result<Vec<(String, Option<String>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT phone, name FROM chat_users 
             WHERE timezone IS NULL 
             AND timezone_asked = 0 
             AND datetime(created_at, '+1 day') <= datetime('now')"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Get users for greeting onboarding (after 2 days, no greeting times set)
    pub fn get_users_for_greeting_onboarding(&self) -> Result<Vec<(String, Option<String>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT phone, name FROM chat_users 
             WHERE morning_greeting_time IS NULL 
             AND greeting_asked = 0 
             AND datetime(created_at, '+2 days') <= datetime('now')"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Get users for morning greeting
    pub fn get_users_for_morning_greeting(&self) -> Result<Vec<(String, Option<String>, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT phone, name, timezone, morning_greeting_time FROM chat_users 
             WHERE morning_greeting_time IS NOT NULL 
             AND timezone IS NOT NULL 
             AND morning_greeting_sent_today = 0"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Get users for night greeting
    pub fn get_users_for_night_greeting(&self) -> Result<Vec<(String, Option<String>, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT phone, name, timezone, night_greeting_time FROM chat_users 
             WHERE night_greeting_time IS NOT NULL 
             AND timezone IS NOT NULL 
             AND night_greeting_sent_today = 0"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Mark morning greeting sent
    pub fn mark_morning_greeting_sent(&self, phone: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE chat_users SET morning_greeting_sent_today = 1 WHERE phone = ?1",
            params![phone],
        )?;
        Ok(())
    }

    /// Mark night greeting sent
    pub fn mark_night_greeting_sent(&self, phone: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE chat_users SET night_greeting_sent_today = 1 WHERE phone = ?1",
            params![phone],
        )?;
        Ok(())
    }

    /// Reset daily greeting flags (call at midnight)
    pub fn reset_daily_greetings(&self) -> Result<()> {
        self.conn.execute(
            "UPDATE chat_users SET morning_greeting_sent_today = 0, night_greeting_sent_today = 0",
            [],
        )?;
        Ok(())
    }

    // ============= Chat Message Methods =============

    /// Save a chat message
    pub fn save_chat_message(&self, phone: &str, role: &str, content: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO chat_messages (user_phone, role, content) VALUES (?1, ?2, ?3)",
            params![phone, role, content],
        )?;
        Ok(())
    }

    /// Get conversation history
    pub fn get_conversation_history(&self, phone: &str, limit: usize) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT role, content FROM chat_messages 
             WHERE user_phone = ?1 
             ORDER BY timestamp DESC 
             LIMIT ?2"
        )?;
        let rows = stmt.query_map(params![phone, limit as i64], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        let mut results: Vec<(String, String)> = Vec::new();
        for row in rows {
            results.push(row?);
        }
        results.reverse(); // Chronological order
        Ok(results)
    }

    // ============= Memory Methods =============

    /// Save a memory about a user
    pub fn save_memory(&self, phone: &str, memory_type: &str, content: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO chat_memories (user_phone, memory_type, content) VALUES (?1, ?2, ?3)",
            params![phone, memory_type, content],
        )?;
        Ok(())
    }

    /// Get user memories
    pub fn get_user_memories(&self, phone: &str) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT memory_type, content FROM chat_memories WHERE user_phone = ?1 ORDER BY created_at"
        )?;
        let rows = stmt.query_map(params![phone], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ============= Reminder Methods =============

    /// Add a reminder
    pub fn add_reminder(&self, phone: &str, message: &str, remind_at: &str, repeat_type: Option<&str>) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO reminders (user_phone, message, remind_at, repeat_type) VALUES (?1, ?2, ?3, ?4)",
            params![phone, message, remind_at, repeat_type],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get pending reminders (due now or in the past)
    pub fn get_pending_reminders(&self) -> Result<Vec<(i64, String, String, Option<String>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, user_phone, message, repeat_type FROM reminders 
             WHERE is_active = 1 AND datetime(remind_at) <= datetime('now')"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Get user's active reminders
    pub fn get_user_active_reminders(&self, phone: &str) -> Result<Vec<(i64, String, String, Option<String>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, message, remind_at, repeat_type FROM reminders 
             WHERE user_phone = ?1 AND is_active = 1 ORDER BY remind_at"
        )?;
        let rows = stmt.query_map(params![phone], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Get today's reminders for a user
    pub fn get_today_reminders(&self, phone: &str) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT message, remind_at FROM reminders 
             WHERE user_phone = ?1 AND is_active = 1 
             AND date(remind_at) = date('now') 
             ORDER BY remind_at"
        )?;
        let rows = stmt.query_map(params![phone], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Get tomorrow's reminders for a user
    pub fn get_tomorrow_reminders(&self, phone: &str) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT message, remind_at FROM reminders 
             WHERE user_phone = ?1 AND is_active = 1 
             AND date(remind_at) = date('now', '+1 day') 
             ORDER BY remind_at"
        )?;
        let rows = stmt.query_map(params![phone], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Deactivate a reminder
    pub fn deactivate_reminder(&self, reminder_id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE reminders SET is_active = 0 WHERE id = ?1",
            params![reminder_id],
        )?;
        Ok(())
    }

    /// Update reminder after sending (for repeating reminders)
    pub fn update_reminder_after_send(&self, reminder_id: i64, repeat_type: Option<&str>) -> Result<()> {
        match repeat_type {
            Some("daily") => {
                self.conn.execute(
                    "UPDATE reminders SET remind_at = datetime(remind_at, '+1 day') WHERE id = ?1",
                    params![reminder_id],
                )?;
            }
            Some("weekly") => {
                self.conn.execute(
                    "UPDATE reminders SET remind_at = datetime(remind_at, '+7 days') WHERE id = ?1",
                    params![reminder_id],
                )?;
            }
            _ => {
                self.deactivate_reminder(reminder_id)?;
            }
        }
        Ok(())
    }

    // ============= Pending Reminder Methods =============

    /// Save pending reminder (awaiting timezone confirmation)
    pub fn save_pending_reminder(&self, phone: &str, reminder_data: &str, guessed_tz: &str, guessed_city: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO pending_reminders (user_phone, reminder_data, guessed_timezone, guessed_city) 
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(user_phone) DO UPDATE SET 
                reminder_data = ?2, guessed_timezone = ?3, guessed_city = ?4, created_at = CURRENT_TIMESTAMP",
            params![phone, reminder_data, guessed_tz, guessed_city],
        )?;
        Ok(())
    }

    /// Get pending reminder for user
    pub fn get_pending_reminder(&self, phone: &str) -> Result<Option<(String, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT reminder_data, guessed_timezone, guessed_city FROM pending_reminders WHERE user_phone = ?1"
        )?;
        let result = stmt.query_row(params![phone], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        });
        match result {
            Ok(data) => Ok(Some(data)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Clear pending reminder
    pub fn clear_pending_reminder(&self, phone: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM pending_reminders WHERE user_phone = ?1",
            params![phone],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_find_entity() {
        let db = Database::in_memory().unwrap();
        
        let id = db.insert_entity("person", "John Doe").unwrap();
        assert!(id > 0);
        
        let found = db.find_entity("person", "John Doe").unwrap();
        assert_eq!(found, Some(id));
    }

    #[test]
    fn test_entity_properties() {
        let db = Database::in_memory().unwrap();
        
        let id = db.insert_entity("person", "Jane").unwrap();
        db.set_entity_property(id, "email", "jane@example.com").unwrap();
        
        let email = db.get_entity_property(id, "email").unwrap();
        assert_eq!(email, Some("jane@example.com".to_string()));
    }

    #[test]
    fn test_relationships() {
        let db = Database::in_memory().unwrap();
        
        let person_id = db.insert_entity("person", "Alice").unwrap();
        let org_id = db.insert_entity("organization", "TechCorp").unwrap();
        
        db.create_relationship(person_id, org_id, "WORKS_AT").unwrap();
        
        let rels = db.get_relationships_from(person_id).unwrap();
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].1, "WORKS_AT");
        assert_eq!(rels[0].2, "TechCorp");
    }

    #[test]
    fn test_chat_user_and_memories() {
        let db = Database::in_memory().unwrap();
        
        db.ensure_chat_user("+1234567890", Some("Test User")).unwrap();
        db.save_memory("+1234567890", "hobby", "likes playing guitar").unwrap();
        
        let memories = db.get_user_memories("+1234567890").unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].0, "hobby");
        assert_eq!(memories[0].1, "likes playing guitar");
    }

    #[test]
    fn test_reminders() {
        let db = Database::in_memory().unwrap();
        
        db.ensure_chat_user("+1234567890", Some("Test")).unwrap();
        let id = db.add_reminder("+1234567890", "call mom", "2024-12-02 15:00", None).unwrap();
        assert!(id > 0);
        
        let reminders = db.get_user_active_reminders("+1234567890").unwrap();
        assert_eq!(reminders.len(), 1);
        assert_eq!(reminders[0].1, "call mom");
    }
}
