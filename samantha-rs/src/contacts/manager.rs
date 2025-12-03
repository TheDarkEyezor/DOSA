use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::knowledge::KnowledgeGraph;

/// A contact with extended information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contact {
    pub entity_id: i64,
    pub name: String,
    pub email: Option<String>,
    pub phone: Option<String>,
    pub organization: Option<String>,
    pub role: Option<String>,
    pub notes: Option<String>,
    pub relationship: Option<String>, // e.g., "friend", "colleague", "family"
}

impl Contact {
    /// Format contact for display
    pub fn display(&self) -> String {
        let mut output = format!("👤 {}", self.name);

        if let Some(org) = &self.organization {
            if let Some(role) = &self.role {
                output.push_str(&format!("\n   💼 {} at {}", role, org));
            } else {
                output.push_str(&format!("\n   🏢 {}", org));
            }
        } else if let Some(role) = &self.role {
            output.push_str(&format!("\n   💼 {}", role));
        }

        if let Some(email) = &self.email {
            output.push_str(&format!("\n   📧 {}", email));
        }

        if let Some(phone) = &self.phone {
            output.push_str(&format!("\n   📱 {}", phone));
        }

        if let Some(rel) = &self.relationship {
            output.push_str(&format!("\n   🔗 {}", rel));
        }

        if let Some(notes) = &self.notes {
            output.push_str(&format!("\n   📝 {}", notes));
        }

        output
    }

    /// Short display for lists
    pub fn display_short(&self) -> String {
        let mut output = self.name.clone();
        
        if let Some(org) = &self.organization {
            output.push_str(&format!(" ({})", org));
        }
        
        output
    }
}

/// Contact manager that integrates with knowledge graph
pub struct ContactManager<'a> {
    graph: &'a KnowledgeGraph,
}

impl<'a> ContactManager<'a> {
    pub fn new(graph: &'a KnowledgeGraph) -> Self {
        ContactManager { graph }
    }

    /// Add a new contact (creates a person in the knowledge graph)
    pub fn add_contact(&self, name: &str) -> Result<Contact> {
        // Check if person already exists
        if let Some(entity) = self.graph.find_person(name)? {
            return self.entity_to_contact(entity.id, name);
        }

        // Create new person
        let entity = self.graph.add_person(name)?;
        
        Ok(Contact {
            entity_id: entity.id,
            name: name.to_string(),
            email: None,
            phone: None,
            organization: None,
            role: None,
            notes: None,
            relationship: None,
        })
    }

    /// Set contact email
    pub fn set_email(&self, entity_id: i64, email: &str) -> Result<()> {
        self.graph.set_property(entity_id, "email", email)
    }

    /// Set contact phone
    pub fn set_phone(&self, entity_id: i64, phone: &str) -> Result<()> {
        self.graph.set_property(entity_id, "phone", phone)
    }

    /// Set contact role/title
    pub fn set_role(&self, entity_id: i64, role: &str) -> Result<()> {
        self.graph.set_property(entity_id, "role", role)
    }

    /// Set relationship to user
    pub fn set_relationship(&self, entity_id: i64, relationship: &str) -> Result<()> {
        self.graph.set_property(entity_id, "relationship", relationship)
    }

    /// Set notes
    pub fn set_notes(&self, entity_id: i64, notes: &str) -> Result<()> {
        self.graph.set_property(entity_id, "notes", notes)
    }

    /// Link contact to an organization
    pub fn link_to_organization(&self, person_id: i64, org_name: &str) -> Result<()> {
        // Find or create organization
        let org = if let Some(org) = self.graph.find_organization(org_name)? {
            org
        } else {
            self.graph.add_organization(org_name)?
        };

        self.graph.link_works_at(person_id, org.id)?;
        Ok(())
    }

    /// Find a contact by name
    pub fn find_contact(&self, name: &str) -> Result<Option<Contact>> {
        if let Some(entity) = self.graph.find_person(name)? {
            Ok(Some(self.entity_to_contact(entity.id, &entity.name)?))
        } else {
            Ok(None)
        }
    }

    /// Get all contacts
    pub fn list_contacts(&self) -> Result<Vec<Contact>> {
        let people = self.graph.list_people()?;
        let mut contacts = Vec::new();

        for person in people {
            contacts.push(self.entity_to_contact(person.id, &person.name)?);
        }

        Ok(contacts)
    }

    /// Get contacts at a specific organization
    pub fn get_contacts_at_org(&self, org_name: &str) -> Result<Vec<Contact>> {
        let org = match self.graph.find_organization(org_name)? {
            Some(o) => o,
            None => return Ok(vec![]),
        };

        // Get all relationships pointing to this org
        let rels = self.graph.database().get_relationships_to(org.id)?;
        let mut contacts = Vec::new();

        for (person_id, rel_type, person_name) in rels {
            if rel_type == "WORKS_AT" {
                contacts.push(self.entity_to_contact(person_id, &person_name)?);
            }
        }

        Ok(contacts)
    }

    /// Convert a knowledge graph entity to a Contact
    fn entity_to_contact(&self, entity_id: i64, name: &str) -> Result<Contact> {
        let props = self.graph.database().get_entity_properties(entity_id)?;
        let rels = self.graph.database().get_relationships_from(entity_id)?;

        let mut contact = Contact {
            entity_id,
            name: name.to_string(),
            email: None,
            phone: None,
            organization: None,
            role: None,
            notes: None,
            relationship: None,
        };

        // Extract properties
        for (key, value) in props {
            match key.as_str() {
                "email" => contact.email = Some(value),
                "phone" => contact.phone = Some(value),
                "role" => contact.role = Some(value),
                "notes" => contact.notes = Some(value),
                "relationship" => contact.relationship = Some(value),
                _ => {}
            }
        }

        // Find organization from relationships
        for (_, rel_type, org_name) in rels {
            if rel_type == "WORKS_AT" {
                contact.organization = Some(org_name);
                break;
            }
        }

        Ok(contact)
    }

    /// Get a summary for LLM context
    pub fn get_summary(&self) -> Result<String> {
        let contacts = self.list_contacts()?;
        
        if contacts.is_empty() {
            return Ok("No contacts saved.".to_string());
        }

        let mut summary = format!("{} contact(s):\n", contacts.len());
        
        for contact in contacts.iter().take(10) {
            summary.push_str(&format!("- {}", contact.name));
            if let Some(org) = &contact.organization {
                summary.push_str(&format!(" ({})", org));
            }
            if let Some(rel) = &contact.relationship {
                summary.push_str(&format!(" - {}", rel));
            }
            summary.push('\n');
        }

        if contacts.len() > 10 {
            summary.push_str(&format!("...and {} more\n", contacts.len() - 10));
        }

        Ok(summary)
    }
}

#[cfg(test)]
mod tests {
    // Contact tests require integration with KnowledgeGraph
    // which is tested in knowledge/graph.rs
    // The ContactManager is a thin wrapper, so we skip unit tests here
}
