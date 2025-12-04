use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Types of entities in the knowledge graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    Organization,
    Project,
    Location,
    Event,
    Task,
    Deadline,   // Temporal node for due dates
    TimePoint,  // Generic temporal marker
    // New entity types for richer knowledge graphs
    Skill,          // Technologies, competencies (e.g., "Rust", "NLP", "market-making")
    Industry,       // Fields/sectors (e.g., "Healthcare", "Finance", "Quant Trading")
    Product,        // Distinct products or services (e.g., "VaNI")
    Goal,           // Aspirational targets (e.g., "impact 100 people")
    Problem,        // Domain problems being addressed
    Idea,           // Abstract concepts or innovations
    // Education & Career types
    Degree,         // Academic qualifications (e.g., "MEng Computing", "BSc Physics")
    University,     // Educational institutions (e.g., "Imperial College London")
    SpokenLanguage, // Human languages (e.g., "English", "French", "Tamil")
    Achievement,    // Awards, recognitions (e.g., "1st place hackathon")
    Technology,     // Specific tech/frameworks (e.g., "React", "Docker", "AWS")
    Custom(String),
}

impl EntityType {
    pub fn as_str(&self) -> &str {
        match self {
            EntityType::Person => "person",
            EntityType::Organization => "organization",
            EntityType::Project => "project",
            EntityType::Location => "location",
            EntityType::Event => "event",
            EntityType::Task => "task",
            EntityType::Deadline => "deadline",
            EntityType::TimePoint => "timepoint",
            EntityType::Skill => "skill",
            EntityType::Industry => "industry",
            EntityType::Product => "product",
            EntityType::Goal => "goal",
            EntityType::Problem => "problem",
            EntityType::Idea => "idea",
            EntityType::Degree => "degree",
            EntityType::University => "university",
            EntityType::SpokenLanguage => "spokenlanguage",
            EntityType::Achievement => "achievement",
            EntityType::Technology => "technology",
            EntityType::Custom(s) => s,
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "person" => EntityType::Person,
            "organization" | "org" | "company" => EntityType::Organization,
            "project" | "proj" => EntityType::Project,
            "location" | "loc" | "place" | "city" => EntityType::Location,
            "event" => EntityType::Event,
            "task" => EntityType::Task,
            "deadline" => EntityType::Deadline,
            "timepoint" | "time" => EntityType::TimePoint,
            "skill" => EntityType::Skill,
            "industry" | "field" | "sector" => EntityType::Industry,
            "product" | "service" => EntityType::Product,
            "goal" | "target" | "objective" => EntityType::Goal,
            "problem" | "challenge" | "issue" => EntityType::Problem,
            "idea" | "concept" | "innovation" => EntityType::Idea,
            "degree" | "qualification" => EntityType::Degree,
            "university" | "school" | "college" | "institution" => EntityType::University,
            "spokenlanguage" | "language" | "lang" => EntityType::SpokenLanguage,
            "achievement" | "award" | "recognition" | "prize" => EntityType::Achievement,
            "technology" | "tech" | "framework" | "tool" => EntityType::Technology,
            other => EntityType::Custom(other.to_string()),
        }
    }
}

/// Relationship types between entities
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RelationshipType {
    // Person → Organization
    WorksAt,    // Currently employed
    WorkedAt,   // Previously employed (past)
    MemberOf,
    
    // Person ↔ Person
    Knows,
    ReportsTo,
    Manages,
    CollaboratesWith,
    
    // Person/Org → Project
    WorksOn,    // Currently working on (present)
    WorkedOn,   // Previously worked on (past)
    Owns,
    
    // Person/Org → Location
    LocatedIn,
    LivesIn,
    BasedIn,
    
    // Task relationships
    AssignedTo,     // Task → Person (who is responsible)
    HasTask,        // Person/Project → Task
    BlockedBy,      // Task → Task (dependency)
    DependsOn,      // Task → Task (soft dependency)
    DueBefore,      // Task → Deadline (temporal)
    DueAfter,       // Task → Deadline (temporal)
    
    // Temporal relationships
    OccursBefore,   // Event/Task → Event/Task
    OccursAfter,    // Event/Task → Event/Task
    During,         // Event → TimePoint
    
    // Generic
    HasDeadline,
    Attending,
    RelatedTo,
    
    // Founding & Investment
    Founded,        // Person → Organization/Project (created)
    CoFounded,      // Person → Organization/Project (co-created)
    Invested,       // Person/Org → Project/Org (financial investment)
    FundedBy,       // Project/Org → Person/Org (received funding)
    
    // Skills & Expertise
    HasSkill,       // Person → Skill
    UsedIn,         // Skill/Technology → Project
    Requires,       // Project/Task → Skill
    
    // Industry & Domain
    InIndustry,     // Person/Project/Org → Industry
    Solves,         // Project/Product → Problem
    Targets,        // Project → Goal
    
    // Mentorship & Partnerships
    Mentors,        // Person → Person
    MentoredBy,     // Person → Person
    PartnersWith,   // Org ↔ Org
    CompetesWith,   // Org ↔ Org
    
    // Education & Learning
    StudiedAt,      // Person → Organization (university/school)
    Studied,        // Person → Skill/Field
    HasDegree,      // Person → Degree
    GraduatedFrom,  // Person → University
    
    // Languages
    Speaks,         // Person → SpokenLanguage
    
    // Achievements & Recognition
    Won,            // Person → Achievement
    Achieved,       // Person → Achievement
    
    // Employment types
    ContractorAt,   // Person → Organization (contract work)
    InternedAt,     // Person → Organization (internship)
    
    // Creation & Building
    Built,          // Person → Project/Product (created)
    Developed,      // Person → Project/Product
    UsesTech,       // Project/Org → Technology
    
    Custom(String),
}

impl RelationshipType {
    pub fn as_str(&self) -> &str {
        match self {
            RelationshipType::WorksAt => "WORKS_AT",
            RelationshipType::WorkedAt => "WORKED_AT",
            RelationshipType::MemberOf => "MEMBER_OF",
            RelationshipType::Knows => "KNOWS",
            RelationshipType::ReportsTo => "REPORTS_TO",
            RelationshipType::Manages => "MANAGES",
            RelationshipType::CollaboratesWith => "COLLABORATES_WITH",
            RelationshipType::WorksOn => "WORKS_ON",
            RelationshipType::WorkedOn => "WORKED_ON",
            RelationshipType::Owns => "OWNS",
            RelationshipType::LocatedIn => "LOCATED_IN",
            RelationshipType::LivesIn => "LIVES_IN",
            RelationshipType::BasedIn => "BASED_IN",
            RelationshipType::AssignedTo => "ASSIGNED_TO",
            RelationshipType::HasTask => "HAS_TASK",
            RelationshipType::BlockedBy => "BLOCKED_BY",
            RelationshipType::DependsOn => "DEPENDS_ON",
            RelationshipType::DueBefore => "DUE_BEFORE",
            RelationshipType::DueAfter => "DUE_AFTER",
            RelationshipType::OccursBefore => "OCCURS_BEFORE",
            RelationshipType::OccursAfter => "OCCURS_AFTER",
            RelationshipType::During => "DURING",
            RelationshipType::HasDeadline => "HAS_DEADLINE",
            RelationshipType::Attending => "ATTENDING",
            RelationshipType::RelatedTo => "RELATED_TO",
            RelationshipType::Founded => "FOUNDED",
            RelationshipType::CoFounded => "CO_FOUNDED",
            RelationshipType::Invested => "INVESTED",
            RelationshipType::FundedBy => "FUNDED_BY",
            RelationshipType::HasSkill => "HAS_SKILL",
            RelationshipType::UsedIn => "USED_IN",
            RelationshipType::Requires => "REQUIRES",
            RelationshipType::InIndustry => "IN_INDUSTRY",
            RelationshipType::Solves => "SOLVES",
            RelationshipType::Targets => "TARGETS",
            RelationshipType::Mentors => "MENTORS",
            RelationshipType::MentoredBy => "MENTORED_BY",
            RelationshipType::PartnersWith => "PARTNERS_WITH",
            RelationshipType::CompetesWith => "COMPETES_WITH",
            RelationshipType::StudiedAt => "STUDIED_AT",
            RelationshipType::Studied => "STUDIED",
            RelationshipType::HasDegree => "HAS_DEGREE",
            RelationshipType::GraduatedFrom => "GRADUATED_FROM",
            RelationshipType::Speaks => "SPEAKS",
            RelationshipType::Won => "WON",
            RelationshipType::Achieved => "ACHIEVED",
            RelationshipType::ContractorAt => "CONTRACTOR_AT",
            RelationshipType::InternedAt => "INTERNED_AT",
            RelationshipType::Built => "BUILT",
            RelationshipType::Developed => "DEVELOPED",
            RelationshipType::UsesTech => "USES_TECH",
            RelationshipType::Custom(s) => s,
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().replace(' ', "_").as_str() {
            "WORKS_AT" => RelationshipType::WorksAt,
            "WORKED_AT" => RelationshipType::WorkedAt,
            "MEMBER_OF" => RelationshipType::MemberOf,
            "KNOWS" => RelationshipType::Knows,
            "REPORTS_TO" => RelationshipType::ReportsTo,
            "MANAGES" => RelationshipType::Manages,
            "COLLABORATES_WITH" => RelationshipType::CollaboratesWith,
            "WORKS_ON" => RelationshipType::WorksOn,
            "WORKED_ON" => RelationshipType::WorkedOn,
            "OWNS" => RelationshipType::Owns,
            "LOCATED_IN" => RelationshipType::LocatedIn,
            "LIVES_IN" => RelationshipType::LivesIn,
            "BASED_IN" => RelationshipType::BasedIn,
            "ASSIGNED_TO" => RelationshipType::AssignedTo,
            "HAS_TASK" => RelationshipType::HasTask,
            "BLOCKED_BY" => RelationshipType::BlockedBy,
            "DEPENDS_ON" => RelationshipType::DependsOn,
            "DUE_BEFORE" => RelationshipType::DueBefore,
            "DUE_AFTER" => RelationshipType::DueAfter,
            "OCCURS_BEFORE" => RelationshipType::OccursBefore,
            "OCCURS_AFTER" => RelationshipType::OccursAfter,
            "DURING" => RelationshipType::During,
            "HAS_DEADLINE" => RelationshipType::HasDeadline,
            "ATTENDING" => RelationshipType::Attending,
            "RELATED_TO" => RelationshipType::RelatedTo,
            "FOUNDED" => RelationshipType::Founded,
            "CO_FOUNDED" | "COFOUNDED" => RelationshipType::CoFounded,
            "INVESTED" | "INVESTED_IN" => RelationshipType::Invested,
            "FUNDED_BY" => RelationshipType::FundedBy,
            "HAS_SKILL" => RelationshipType::HasSkill,
            "USED_IN" => RelationshipType::UsedIn,
            "REQUIRES" => RelationshipType::Requires,
            "IN_INDUSTRY" | "IN_FIELD" => RelationshipType::InIndustry,
            "SOLVES" => RelationshipType::Solves,
            "TARGETS" => RelationshipType::Targets,
            "MENTORS" => RelationshipType::Mentors,
            "MENTORED_BY" => RelationshipType::MentoredBy,
            "PARTNERS_WITH" => RelationshipType::PartnersWith,
            "COMPETES_WITH" => RelationshipType::CompetesWith,
            "STUDIED_AT" => RelationshipType::StudiedAt,
            "STUDIED" => RelationshipType::Studied,
            "HAS_DEGREE" => RelationshipType::HasDegree,
            "GRADUATED_FROM" => RelationshipType::GraduatedFrom,
            "SPEAKS" => RelationshipType::Speaks,
            "WON" => RelationshipType::Won,
            "ACHIEVED" => RelationshipType::Achieved,
            "CONTRACTOR_AT" => RelationshipType::ContractorAt,
            "INTERNED_AT" => RelationshipType::InternedAt,
            "BUILT" => RelationshipType::Built,
            "DEVELOPED" => RelationshipType::Developed,
            "USES_TECH" => RelationshipType::UsesTech,
            other => RelationshipType::Custom(other.to_string()),
        }
    }
    
    /// Get a human-readable description of the relationship
    pub fn description(&self) -> &str {
        match self {
            RelationshipType::WorksAt => "works at",
            RelationshipType::WorkedAt => "worked at",
            RelationshipType::MemberOf => "is a member of",
            RelationshipType::Knows => "knows",
            RelationshipType::ReportsTo => "reports to",
            RelationshipType::Manages => "manages",
            RelationshipType::CollaboratesWith => "collaborates with",
            RelationshipType::WorksOn => "works on",
            RelationshipType::WorkedOn => "worked on",
            RelationshipType::Owns => "owns",
            RelationshipType::LocatedIn => "is located in",
            RelationshipType::LivesIn => "lives in",
            RelationshipType::BasedIn => "is based in",
            RelationshipType::AssignedTo => "is assigned to",
            RelationshipType::HasTask => "has task",
            RelationshipType::BlockedBy => "is blocked by",
            RelationshipType::DependsOn => "depends on",
            RelationshipType::DueBefore => "is due before",
            RelationshipType::DueAfter => "is due after",
            RelationshipType::OccursBefore => "occurs before",
            RelationshipType::OccursAfter => "occurs after",
            RelationshipType::During => "during",
            RelationshipType::HasDeadline => "has deadline",
            RelationshipType::Attending => "is attending",
            RelationshipType::RelatedTo => "is related to",
            RelationshipType::Founded => "founded",
            RelationshipType::CoFounded => "co-founded",
            RelationshipType::Invested => "invested in",
            RelationshipType::FundedBy => "is funded by",
            RelationshipType::HasSkill => "has skill",
            RelationshipType::UsedIn => "is used in",
            RelationshipType::Requires => "requires",
            RelationshipType::InIndustry => "is in industry",
            RelationshipType::Solves => "solves",
            RelationshipType::Targets => "targets",
            RelationshipType::Mentors => "mentors",
            RelationshipType::MentoredBy => "is mentored by",
            RelationshipType::PartnersWith => "partners with",
            RelationshipType::CompetesWith => "competes with",
            RelationshipType::StudiedAt => "studied at",
            RelationshipType::Studied => "studied",
            RelationshipType::HasDegree => "has degree",
            RelationshipType::GraduatedFrom => "graduated from",
            RelationshipType::Speaks => "speaks",
            RelationshipType::Won => "won",
            RelationshipType::Achieved => "achieved",
            RelationshipType::ContractorAt => "was contractor at",
            RelationshipType::InternedAt => "interned at",
            RelationshipType::Built => "built",
            RelationshipType::Developed => "developed",
            RelationshipType::UsesTech => "uses",
            RelationshipType::Custom(_) => "is connected to",
        }
    }
}

/// An entity in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: i64,
    pub entity_type: EntityType,
    pub name: String,
    pub properties: HashMap<String, String>,
}

impl PartialEq for Entity {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Entity {}

impl std::hash::Hash for Entity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Entity {
    pub fn new(id: i64, entity_type: EntityType, name: String) -> Self {
        Entity {
            id,
            entity_type,
            name,
            properties: HashMap::new(),
        }
    }

    pub fn with_property(mut self, key: &str, value: &str) -> Self {
        self.properties.insert(key.to_string(), value.to_string());
        self
    }

    /// Format entity for display
    pub fn display(&self) -> String {
        let mut output = format!("{} ({})", self.name, self.entity_type.as_str());
        if !self.properties.is_empty() {
            output.push_str("\n  Properties:");
            for (key, value) in &self.properties {
                output.push_str(&format!("\n    - {}: {}", key, value));
            }
        }
        output
    }
}

/// A relationship between two entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub id: i64,
    pub from_entity_id: i64,
    pub to_entity_id: i64,
    pub relationship_type: RelationshipType,
    pub properties: HashMap<String, String>,
}

impl Relationship {
    pub fn new(
        id: i64,
        from_entity_id: i64,
        to_entity_id: i64,
        relationship_type: RelationshipType,
    ) -> Self {
        Relationship {
            id,
            from_entity_id,
            to_entity_id,
            relationship_type,
            properties: HashMap::new(),
        }
    }
}
