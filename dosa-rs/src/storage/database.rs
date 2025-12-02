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
}
