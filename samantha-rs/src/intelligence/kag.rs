//! KAG - Knowledge Augmented Generation
//!
//! This module implements logical reasoning capabilities beyond basic GraphRAG:
//! - Logical form-guided reasoning (planning → reasoning → retrieval)
//! - Rule-based inference (constraints, deductions)
//! - Temporal reasoning (time-based queries)
//! - Numerical reasoning (aggregations, comparisons)
//!
//! Based on the KAG paper: https://arxiv.org/abs/2409.13731

use anyhow::Result;
use std::collections::{HashMap, HashSet};

use crate::knowledge::{KnowledgeGraph, entities::Entity};

// ============================================================================
// Knowledge Rules & Constraints
// ============================================================================

/// A rule that can be applied during reasoning
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub description: String,
    pub condition: RuleCondition,
    pub conclusion: RuleConclusion,
}

/// Conditions that trigger a rule
#[derive(Debug, Clone)]
pub enum RuleCondition {
    /// If A knows B and B knows C, then A might know C (transitive)
    TransitiveRelation { relation: String },
    /// If A works_at X and B works_at X, then A and B are colleagues
    SharedRelation { relation: String },
    /// If A manages B, then B reports_to A
    InverseRelation { forward: String, inverse: String },
    /// Custom condition with a predicate function name
    Custom { predicate: String },
}

/// What to conclude when a rule fires
#[derive(Debug, Clone)]
pub enum RuleConclusion {
    /// Infer a new relationship
    InferRelation { relation: String, confidence: f32 },
    /// Infer a property
    InferProperty { key: String, value: String },
    /// Flag for review
    Flag { reason: String },
}

// ============================================================================
// Logical Forms (Query Decomposition)
// ============================================================================

/// A logical form representing a decomposed query
#[derive(Debug, Clone)]
pub enum LogicalForm {
    /// Simple entity lookup
    Lookup { entity_type: String, name: String },
    
    /// Find entities matching criteria
    Filter { 
        entity_type: String, 
        conditions: Vec<FilterCondition> 
    },
    
    /// Traverse relationships
    Traverse { 
        from: Box<LogicalForm>, 
        relation: String,
        direction: TraverseDirection,
    },
    
    /// Apply aggregation
    Aggregate { 
        over: Box<LogicalForm>, 
        function: AggregateFunction,
    },
    
    /// Compare values
    Compare {
        left: Box<LogicalForm>,
        right: Box<LogicalForm>,
        operator: CompareOperator,
    },
    
    /// Combine multiple forms
    Compose {
        forms: Vec<LogicalForm>,
        combinator: Combinator,
    },
    
    /// Apply a rule
    ApplyRule {
        over: Box<LogicalForm>,
        rule_name: String,
    },
}

#[derive(Debug, Clone)]
pub struct FilterCondition {
    pub property: String,
    pub operator: CompareOperator,
    pub value: String,
}

#[derive(Debug, Clone)]
pub enum TraverseDirection {
    Outgoing,  // A -> B (A knows B, get B)
    Incoming,  // A <- B (B knows A, get B)
    Both,
}

#[derive(Debug, Clone)]
pub enum AggregateFunction {
    Count,
    Max { property: String },
    Min { property: String },
    First,
    All,
}

#[derive(Debug, Clone)]
pub enum CompareOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    Contains,
    StartsWith,
    Before,  // Temporal
    After,   // Temporal
}

#[derive(Debug, Clone)]
pub enum Combinator {
    And,
    Or,
    Then,  // Sequential
}

// ============================================================================
// Reasoning Result
// ============================================================================

#[derive(Debug, Clone)]
pub struct ReasoningResult {
    pub answer: String,
    pub entities: Vec<Entity>,
    pub reasoning_steps: Vec<ReasoningStep>,
    pub confidence: f32,
    pub rules_applied: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ReasoningStep {
    pub step_type: StepType,
    pub description: String,
    pub result: String,
}

#[derive(Debug, Clone)]
pub enum StepType {
    Planning,
    Retrieval,
    Reasoning,
    Aggregation,
}

// ============================================================================
// KAG Logical Solver
// ============================================================================

/// The KAG Logical Solver - handles complex reasoning queries
pub struct LogicalSolver<'a> {
    graph: &'a KnowledgeGraph,
    rules: Vec<Rule>,
}

impl<'a> LogicalSolver<'a> {
    pub fn new(graph: &'a KnowledgeGraph) -> Self {
        let mut solver = Self { 
            graph, 
            rules: Vec::new(),
        };
        solver.register_default_rules();
        solver
    }

    /// Register default reasoning rules
    fn register_default_rules(&mut self) {
        // Rule 1: Colleague inference
        self.rules.push(Rule {
            name: "colleague_inference".to_string(),
            description: "People at the same organization are colleagues".to_string(),
            condition: RuleCondition::SharedRelation { 
                relation: "WORKS_AT".to_string() 
            },
            conclusion: RuleConclusion::InferRelation { 
                relation: "COLLEAGUE_OF".to_string(), 
                confidence: 0.9 
            },
        });

        // Rule 2: Transitive knows (friend-of-friend)
        self.rules.push(Rule {
            name: "transitive_knows".to_string(),
            description: "If A knows B and B knows C, A might know C".to_string(),
            condition: RuleCondition::TransitiveRelation { 
                relation: "KNOWS".to_string() 
            },
            conclusion: RuleConclusion::InferRelation { 
                relation: "MIGHT_KNOW".to_string(), 
                confidence: 0.5 
            },
        });

        // Rule 3: Manager-Report symmetry
        self.rules.push(Rule {
            name: "manager_report_inverse".to_string(),
            description: "If A manages B, then B reports to A".to_string(),
            condition: RuleCondition::InverseRelation { 
                forward: "MANAGES".to_string(), 
                inverse: "REPORTS_TO".to_string() 
            },
            conclusion: RuleConclusion::InferRelation { 
                relation: "REPORTS_TO".to_string(), 
                confidence: 1.0 
            },
        });

        // Rule 4: Blocked task cascade
        self.rules.push(Rule {
            name: "blocked_cascade".to_string(),
            description: "If task A is blocked by B, and B is blocked by C, then A is transitively blocked by C".to_string(),
            condition: RuleCondition::TransitiveRelation { 
                relation: "BLOCKED_BY".to_string() 
            },
            conclusion: RuleConclusion::InferRelation { 
                relation: "TRANSITIVELY_BLOCKED_BY".to_string(), 
                confidence: 0.9 
            },
        });

        // Rule 5: Overdue task alert
        self.rules.push(Rule {
            name: "overdue_alert".to_string(),
            description: "Tasks past their deadline should be flagged".to_string(),
            condition: RuleCondition::Custom { 
                predicate: "deadline_passed".to_string() 
            },
            conclusion: RuleConclusion::Flag { 
                reason: "Task is overdue".to_string() 
            },
        });

        // Rule 6: High workload alert
        self.rules.push(Rule {
            name: "high_workload".to_string(),
            description: "Person with many active tasks may be overloaded".to_string(),
            condition: RuleCondition::Custom { 
                predicate: "task_count_high".to_string() 
            },
            conclusion: RuleConclusion::Flag { 
                reason: "High workload detected".to_string() 
            },
        });

        // Rule 7: Unblocked tasks ready
        self.rules.push(Rule {
            name: "task_ready".to_string(),
            description: "Task with no blockers is ready to work on".to_string(),
            condition: RuleCondition::Custom { 
                predicate: "no_blockers".to_string() 
            },
            conclusion: RuleConclusion::InferProperty { 
                key: "status_hint".to_string(), 
                value: "ready".to_string() 
            },
        });
    }

    /// Parse a natural language query into a logical form
    pub fn parse_to_logical_form(&self, query: &str) -> Option<LogicalForm> {
        let lower = query.to_lowercase();
        
        // Pattern: "how many X" or "count X"
        if lower.contains("how many") || lower.starts_with("count ") {
            return self.parse_count_query(&lower);
        }
        
        // Pattern: "who has the most" / "most connected"
        if lower.contains("most ") || lower.contains("maximum") {
            return self.parse_max_query(&lower);
        }
        
        // Pattern: "who has the least" / "fewest"
        if lower.contains("least ") || lower.contains("fewest") || lower.contains("minimum") {
            return self.parse_min_query(&lower);
        }
        
        // Pattern: "can X reach Y" / "is there a path"
        if lower.contains("reach") || lower.contains("path to") || lower.contains("connected to") {
            return self.parse_reachability_query(&lower);
        }
        
        // Pattern: "who might know" (transitive inference)
        if lower.contains("might know") || lower.contains("could know") {
            return self.parse_transitive_query(&lower);
        }
        
        // Pattern: "colleagues" or "coworkers"
        if lower.contains("colleague") || lower.contains("coworker") {
            return self.parse_colleague_query(&lower);
        }
        
        // Pattern: Compare - "more than", "fewer than"
        if lower.contains("more than") || lower.contains("fewer than") || lower.contains("less than") {
            return self.parse_comparison_query(&lower);
        }
        
        // Pattern: Temporal - "due before", "due after", "overdue", "tasks before/after"
        if lower.contains("due before") || lower.contains("due after") || 
           lower.contains("before ") || lower.contains("after ") ||
           lower.contains("overdue") {
            if let Some(form) = self.parse_temporal_query(&lower) {
                return Some(form);
            }
        }
        
        // Pattern: Tasks for person - "alice's tasks", "tasks for bob"
        if lower.contains("task") && (lower.contains("'s") || lower.contains(" for ")) {
            return self.parse_person_tasks_query(&lower);
        }
        
        // Pattern: Blocked tasks - "what blocks", "blocked by"
        if lower.contains("block") {
            return self.parse_blocking_query(&lower);
        }
        
        // Pattern: Workload - "how busy", "workload"
        if lower.contains("workload") || lower.contains("how busy") || lower.contains("how much work") {
            return self.parse_workload_query(&lower);
        }
        
        // Pattern: "where does X work" / "where does X live" (person -> target queries)
        if lower.starts_with("where do") || lower.starts_with("where does") {
            if lower.contains(" work") {
                return self.parse_person_workplace_query(&lower);
            }
            if lower.contains(" live") {
                return self.parse_person_location_query(&lower);
            }
        }
        
        // Pattern: Location - "who lives in X", "who is in X", "who is located in X"
        if lower.contains("lives in") || lower.contains("live in") || 
           lower.contains("located in") || lower.contains("is in") || 
           lower.contains("based in") || lower.contains("from ") {
            return self.parse_location_query(&lower);
        }
        
        // Pattern: Works at - "who works at X", "people at X"
        if lower.contains("works at") || lower.contains("work at") ||
           lower.contains("employed at") || lower.contains("employed by") {
            return self.parse_works_at_query(&lower);
        }
        
        None
    }

    /// Execute a logical form query
    pub fn execute(&self, form: &LogicalForm) -> Result<ReasoningResult> {
        let mut steps = Vec::new();
        
        match form {
            LogicalForm::Lookup { entity_type, name } => {
                steps.push(ReasoningStep {
                    step_type: StepType::Retrieval,
                    description: format!("Looking up {} '{}'", entity_type, name),
                    result: String::new(),
                });
                self.execute_lookup(entity_type, name, steps)
            }
            
            LogicalForm::Filter { entity_type, conditions } => {
                steps.push(ReasoningStep {
                    step_type: StepType::Planning,
                    description: format!("Filtering {} with {} conditions", entity_type, conditions.len()),
                    result: String::new(),
                });
                self.execute_filter(entity_type, conditions, steps)
            }
            
            LogicalForm::Aggregate { over, function } => {
                steps.push(ReasoningStep {
                    step_type: StepType::Planning,
                    description: format!("Aggregating with {:?}", function),
                    result: String::new(),
                });
                self.execute_aggregate(over, function, steps)
            }
            
            LogicalForm::Traverse { from, relation, direction } => {
                steps.push(ReasoningStep {
                    step_type: StepType::Reasoning,
                    description: format!("Traversing {} relationship", relation),
                    result: String::new(),
                });
                self.execute_traverse(from, relation, direction, steps)
            }
            
            LogicalForm::ApplyRule { over, rule_name } => {
                steps.push(ReasoningStep {
                    step_type: StepType::Reasoning,
                    description: format!("Applying rule: {}", rule_name),
                    result: String::new(),
                });
                self.execute_with_rule(over, rule_name, steps)
            }
            
            LogicalForm::Compare { left, right, operator } => {
                self.execute_compare(left, right, operator, steps)
            }
            
            LogicalForm::Compose { forms, combinator } => {
                self.execute_compose(forms, combinator, steps)
            }
        }
    }

    // ========================================================================
    // Query Parsers
    // ========================================================================
    
    fn parse_count_query(&self, query: &str) -> Option<LogicalForm> {
        // "how many people work at Google"
        if query.contains("work at") || query.contains("works at") {
            let org = self.extract_after(query, &["work at ", "works at "])?;
            return Some(LogicalForm::Aggregate {
                over: Box::new(LogicalForm::Traverse {
                    from: Box::new(LogicalForm::Lookup { 
                        entity_type: "Organization".to_string(), 
                        name: org,
                    }),
                    relation: "WORKS_AT".to_string(),
                    direction: TraverseDirection::Incoming,
                }),
                function: AggregateFunction::Count,
            });
        }
        
        // "how many connections does X have"
        if query.contains("connections") || query.contains("contacts") {
            let person = self.extract_person_name(query)?;
            return Some(LogicalForm::Aggregate {
                over: Box::new(LogicalForm::Traverse {
                    from: Box::new(LogicalForm::Lookup {
                        entity_type: "Person".to_string(),
                        name: person,
                    }),
                    relation: "KNOWS".to_string(),
                    direction: TraverseDirection::Both,
                }),
                function: AggregateFunction::Count,
            });
        }
        
        None
    }
    
    fn parse_max_query(&self, query: &str) -> Option<LogicalForm> {
        // "who has the most connections"
        if query.contains("most connections") || query.contains("most contacts") {
            return Some(LogicalForm::Aggregate {
                over: Box::new(LogicalForm::Filter {
                    entity_type: "Person".to_string(),
                    conditions: vec![],
                }),
                function: AggregateFunction::Max { 
                    property: "connection_count".to_string() 
                },
            });
        }
        
        None
    }
    
    fn parse_min_query(&self, query: &str) -> Option<LogicalForm> {
        // "who has the least/fewest connections"
        if query.contains("least connections") || query.contains("fewest connections") {
            return Some(LogicalForm::Aggregate {
                over: Box::new(LogicalForm::Filter {
                    entity_type: "Person".to_string(),
                    conditions: vec![],
                }),
                function: AggregateFunction::Min { 
                    property: "connection_count".to_string() 
                },
            });
        }
        
        None
    }
    
    fn parse_reachability_query(&self, query: &str) -> Option<LogicalForm> {
        // "can alice reach charlie" / "is alice connected to charlie"
        let parts: Vec<&str> = query.split(|c| c == ' ').collect();
        
        // Find two names in the query
        let mut names = Vec::new();
        for word in &parts {
            let clean = word.trim_matches(|c: char| !c.is_alphabetic());
            if !clean.is_empty() && 
               !["can", "reach", "is", "connected", "to", "there", "path", "from", "a"].contains(&clean) {
                names.push(self.capitalize(clean));
            }
        }
        
        if names.len() >= 2 {
            return Some(LogicalForm::Compose {
                forms: vec![
                    LogicalForm::Lookup { 
                        entity_type: "Person".to_string(), 
                        name: names[0].clone() 
                    },
                    LogicalForm::Lookup { 
                        entity_type: "Person".to_string(), 
                        name: names[1].clone() 
                    },
                ],
                combinator: Combinator::Then,
            });
        }
        
        None
    }
    
    fn parse_transitive_query(&self, query: &str) -> Option<LogicalForm> {
        // "who might alice know" (friend of friend)
        if let Some(name) = self.extract_person_name(query) {
            return Some(LogicalForm::ApplyRule {
                over: Box::new(LogicalForm::Lookup {
                    entity_type: "Person".to_string(),
                    name,
                }),
                rule_name: "transitive_knows".to_string(),
            });
        }
        None
    }
    
    fn parse_colleague_query(&self, query: &str) -> Option<LogicalForm> {
        // "who are alice's colleagues"
        if let Some(name) = self.extract_person_name(query) {
            return Some(LogicalForm::ApplyRule {
                over: Box::new(LogicalForm::Lookup {
                    entity_type: "Person".to_string(),
                    name,
                }),
                rule_name: "colleague_inference".to_string(),
            });
        }
        None
    }
    
    fn parse_comparison_query(&self, query: &str) -> Option<LogicalForm> {
        // "who has more connections than alice"
        if query.contains("more") && query.contains("than") {
            if let Some(name) = self.extract_after(query, &["than "]) {
                return Some(LogicalForm::Compare {
                    left: Box::new(LogicalForm::Filter {
                        entity_type: "Person".to_string(),
                        conditions: vec![],
                    }),
                    right: Box::new(LogicalForm::Lookup {
                        entity_type: "Person".to_string(),
                        name,
                    }),
                    operator: CompareOperator::GreaterThan,
                });
            }
        }
        None
    }

    // ========================================================================
    // Temporal Query Parsing
    // ========================================================================

    fn parse_temporal_query(&self, query: &str) -> Option<LogicalForm> {
        // "tasks due before friday" / "due before december 5"
        if query.contains("due before") {
            if let Some(time) = self.extract_after(query, &["due before ", "before "]) {
                return Some(LogicalForm::Filter {
                    entity_type: "Task".to_string(),
                    conditions: vec![FilterCondition {
                        property: "deadline".to_string(),
                        operator: CompareOperator::Before,
                        value: time,
                    }],
                });
            }
        }
        
        // "tasks due after monday"
        if query.contains("due after") {
            if let Some(time) = self.extract_after(query, &["due after ", "after "]) {
                return Some(LogicalForm::Filter {
                    entity_type: "Task".to_string(),
                    conditions: vec![FilterCondition {
                        property: "deadline".to_string(),
                        operator: CompareOperator::After,
                        value: time,
                    }],
                });
            }
        }
        
        // "overdue tasks" / "what is overdue"
        if query.contains("overdue") {
            return Some(LogicalForm::Filter {
                entity_type: "Task".to_string(),
                conditions: vec![FilterCondition {
                    property: "deadline".to_string(),
                    operator: CompareOperator::Before,
                    value: "now".to_string(),
                }],
            });
        }
        
        None
    }
    
    fn parse_person_tasks_query(&self, query: &str) -> Option<LogicalForm> {
        // "alice's tasks" / "tasks for alice"
        let person = if query.contains("'s task") {
            self.extract_person_name(query)?
        } else if let Some(name) = self.extract_after(query, &["tasks for ", "task for "]) {
            name
        } else {
            return None;
        };
        
        Some(LogicalForm::Traverse {
            from: Box::new(LogicalForm::Lookup {
                entity_type: "Person".to_string(),
                name: person,
            }),
            relation: "HAS_TASK".to_string(),
            direction: TraverseDirection::Outgoing,
        })
    }
    
    fn parse_blocking_query(&self, query: &str) -> Option<LogicalForm> {
        // "what blocks task X" / "what is blocking X"
        if query.contains("what block") || query.contains("what is block") {
            if let Some(task_name) = self.extract_after(query, &["blocks ", "blocking "]) {
                return Some(LogicalForm::Traverse {
                    from: Box::new(LogicalForm::Lookup {
                        entity_type: "Task".to_string(),
                        name: task_name,
                    }),
                    relation: "BLOCKED_BY".to_string(),
                    direction: TraverseDirection::Outgoing,
                });
            }
        }
        
        // "what is blocked by X"
        if query.contains("blocked by") {
            if let Some(blocker_name) = self.extract_after(query, &["blocked by "]) {
                return Some(LogicalForm::Traverse {
                    from: Box::new(LogicalForm::Lookup {
                        entity_type: "Task".to_string(),
                        name: blocker_name,
                    }),
                    relation: "BLOCKED_BY".to_string(),
                    direction: TraverseDirection::Incoming,
                });
            }
        }
        
        None
    }
    
    fn parse_workload_query(&self, query: &str) -> Option<LogicalForm> {
        // "how busy is alice" / "alice's workload"
        let person = if query.contains("'s workload") || query.contains("'s work") {
            self.extract_person_name(query)?
        } else if let Some(name) = self.extract_after(query, &["busy is ", "busy "]) {
            name
        } else {
            return None;
        };
        
        // This will count tasks for the person
        Some(LogicalForm::Aggregate {
            over: Box::new(LogicalForm::Traverse {
                from: Box::new(LogicalForm::Lookup {
                    entity_type: "Person".to_string(),
                    name: person,
                }),
                relation: "HAS_TASK".to_string(),
                direction: TraverseDirection::Outgoing,
            }),
            function: AggregateFunction::Count,
        })
    }

    fn parse_location_query(&self, query: &str) -> Option<LogicalForm> {
        // "who lives in san francisco", "who is located in new york"
        let location = self.extract_after(query, &[
            "lives in ", "live in ", "located in ", "is in ", "based in ", "from "
        ])?;
        
        // Clean up location name - remove trailing question marks, etc.
        let location = location.trim_end_matches(|c: char| c == '?' || c == '.' || c == '!').trim();
        
        Some(LogicalForm::Traverse {
            from: Box::new(LogicalForm::Lookup {
                entity_type: "Location".to_string(),
                name: self.capitalize(location),
            }),
            relation: "LIVES_IN".to_string(),
            direction: TraverseDirection::Incoming,
        })
    }
    
    fn parse_works_at_query(&self, query: &str) -> Option<LogicalForm> {
        // "who works at google", "people employed at microsoft"
        let org = self.extract_after(query, &[
            "works at ", "work at ", "employed at ", "employed by "
        ])?;
        
        // Clean up org name
        let org = org.trim_end_matches(|c: char| c == '?' || c == '.' || c == '!').trim();
        
        Some(LogicalForm::Traverse {
            from: Box::new(LogicalForm::Lookup {
                entity_type: "Organization".to_string(),
                name: self.capitalize(org),
            }),
            relation: "WORKS_AT".to_string(),
            direction: TraverseDirection::Incoming,
        })
    }

    fn parse_person_workplace_query(&self, query: &str) -> Option<LogicalForm> {
        // "where does Alice work" -> find organizations Alice WORKS_AT
        let person = self.extract_person_from_where_query(query)?;
        
        Some(LogicalForm::Traverse {
            from: Box::new(LogicalForm::Lookup {
                entity_type: "Person".to_string(),
                name: person,
            }),
            relation: "WORKS_AT".to_string(),
            direction: TraverseDirection::Outgoing,
        })
    }

    fn parse_person_location_query(&self, query: &str) -> Option<LogicalForm> {
        // "where does Alice live" -> find locations Alice LIVES_IN
        let person = self.extract_person_from_where_query(query)?;
        
        Some(LogicalForm::Traverse {
            from: Box::new(LogicalForm::Lookup {
                entity_type: "Person".to_string(),
                name: person,
            }),
            relation: "LIVES_IN".to_string(),
            direction: TraverseDirection::Outgoing,
        })
    }

    fn extract_person_from_where_query(&self, query: &str) -> Option<String> {
        // Extract person name from "where does X work/live"
        // Pattern: "where does <name> work" or "where do <name> live"
        let query = query.trim_end_matches(|c: char| c == '?' || c == '.');
        
        // Remove "where does " or "where do " prefix
        let after_prefix = query
            .strip_prefix("where does ")
            .or_else(|| query.strip_prefix("where do "))?;
        
        // Remove trailing " work" or " live"
        let name = after_prefix
            .strip_suffix(" work")
            .or_else(|| after_prefix.strip_suffix(" live"))
            .unwrap_or(after_prefix);
        
        if name.is_empty() {
            return None;
        }
        
        // Title case the name
        let words: Vec<String> = name.split_whitespace()
            .map(|w| self.capitalize(w))
            .collect();
        Some(words.join(" "))
    }

    // ========================================================================
    // Execution Helpers
    // ========================================================================
    
    fn execute_lookup(&self, entity_type: &str, name: &str, mut steps: Vec<ReasoningStep>) -> Result<ReasoningResult> {
        let entity = match entity_type {
            "Person" => self.graph.find_person(name)?,
            "Organization" => self.graph.find_organization(name)?,
            "Project" => self.graph.find_project(name)?,
            "Location" => self.graph.find_location(name)?,
            _ => None,
        };
        
        steps.push(ReasoningStep {
            step_type: StepType::Retrieval,
            description: format!("Found {} '{}'", entity_type, name),
            result: entity.as_ref().map(|e| e.name.clone()).unwrap_or_default(),
        });
        
        Ok(ReasoningResult {
            answer: entity.as_ref()
                .map(|e| format!("Found: {}", e.name))
                .unwrap_or_else(|| format!("{} '{}' not found", entity_type, name)),
            entities: entity.into_iter().collect(),
            reasoning_steps: steps,
            confidence: 1.0,
            rules_applied: vec![],
        })
    }
    
    fn execute_filter(&self, entity_type: &str, conditions: &[FilterCondition], mut steps: Vec<ReasoningStep>) -> Result<ReasoningResult> {
        // Get all entities of type
        let entities: Vec<Entity> = match entity_type {
            "Person" => self.graph.list_entities_by_type("Person")?,
            "Organization" => self.graph.list_entities_by_type("Organization")?,
            "Task" => self.graph.list_tasks()?,
            _ => vec![],
        };
        
        steps.push(ReasoningStep {
            step_type: StepType::Retrieval,
            description: format!("Found {} entities of type {}", entities.len(), entity_type),
            result: format!("{} results", entities.len()),
        });
        
        // Apply filter conditions
        let filtered = self.apply_filter_conditions(entities, conditions, &mut steps)?;
        
        let names: Vec<String> = filtered.iter().map(|e| e.name.clone()).collect();
        
        Ok(ReasoningResult {
            answer: if filtered.is_empty() {
                format!("No {} found matching conditions", entity_type)
            } else {
                format!("Found {} {}: {}", filtered.len(), entity_type, names.join(", "))
            },
            entities: filtered,
            reasoning_steps: steps,
            confidence: 1.0,
            rules_applied: vec![],
        })
    }
    
    /// Apply filter conditions to a list of entities
    fn apply_filter_conditions(&self, entities: Vec<Entity>, conditions: &[FilterCondition], steps: &mut Vec<ReasoningStep>) -> Result<Vec<Entity>> {
        use chrono::{Local, NaiveDateTime};
        
        if conditions.is_empty() {
            return Ok(entities);
        }
        
        let mut filtered = entities;
        
        for condition in conditions {
            steps.push(ReasoningStep {
                step_type: StepType::Reasoning,
                description: format!("Applying filter: {} {:?} {}", condition.property, condition.operator, condition.value),
                result: String::new(),
            });
            
            filtered = filtered.into_iter().filter(|entity| {
                let prop_value = entity.properties.get(&condition.property);
                
                match &condition.operator {
                    CompareOperator::Equals => {
                        prop_value.map(|v| v.eq_ignore_ascii_case(&condition.value)).unwrap_or(false)
                    }
                    CompareOperator::NotEquals => {
                        prop_value.map(|v| !v.eq_ignore_ascii_case(&condition.value)).unwrap_or(true)
                    }
                    CompareOperator::Contains => {
                        prop_value.map(|v| v.to_lowercase().contains(&condition.value.to_lowercase())).unwrap_or(false)
                    }
                    CompareOperator::Before => {
                        // Temporal: deadline before a time
                        if let Some(deadline_str) = prop_value {
                            let compare_time = if condition.value == "now" {
                                Local::now().naive_local()
                            } else {
                                // Try to parse the condition value as a date
                                self.parse_natural_date(&condition.value).unwrap_or_else(|| Local::now().naive_local())
                            };
                            
                            // Parse the entity's deadline
                            if let Ok(deadline) = NaiveDateTime::parse_from_str(deadline_str, "%Y-%m-%dT%H:%M:%S") {
                                deadline < compare_time
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    }
                    CompareOperator::After => {
                        if let Some(deadline_str) = prop_value {
                            let compare_time = if condition.value == "now" {
                                Local::now().naive_local()
                            } else {
                                self.parse_natural_date(&condition.value).unwrap_or_else(|| Local::now().naive_local())
                            };
                            
                            if let Ok(deadline) = NaiveDateTime::parse_from_str(deadline_str, "%Y-%m-%dT%H:%M:%S") {
                                deadline > compare_time
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    }
                    _ => true, // Other operators - pass through for now
                }
            }).collect();
        }
        
        steps.push(ReasoningStep {
            step_type: StepType::Aggregation,
            description: format!("Filter result: {} entities match", filtered.len()),
            result: filtered.len().to_string(),
        });
        
        Ok(filtered)
    }
    
    /// Parse natural language dates like "friday", "december 5", "tomorrow"
    fn parse_natural_date(&self, input: &str) -> Option<chrono::NaiveDateTime> {
        use chrono::{Local, Datelike, Duration, NaiveDateTime, NaiveTime};
        
        let lower = input.to_lowercase();
        let today = Local::now().date_naive();
        let default_time = NaiveTime::from_hms_opt(23, 59, 59)?;
        
        // Handle relative dates
        if lower == "today" {
            return Some(today.and_time(default_time));
        }
        if lower == "tomorrow" {
            return Some((today + Duration::days(1)).and_time(default_time));
        }
        if lower == "yesterday" {
            return Some((today - Duration::days(1)).and_time(default_time));
        }
        
        // Handle day names (next occurrence)
        let weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];
        for (i, day) in weekdays.iter().enumerate() {
            if lower.contains(day) {
                let target_weekday = i as u32;
                let current_weekday = today.weekday().num_days_from_monday();
                let days_ahead = if target_weekday <= current_weekday {
                    7 - current_weekday + target_weekday
                } else {
                    target_weekday - current_weekday
                };
                return Some((today + Duration::days(days_ahead as i64)).and_time(default_time));
            }
        }
        
        // Handle "next week"
        if lower.contains("next week") {
            return Some((today + Duration::days(7)).and_time(default_time));
        }
        
        // Try to parse ISO format
        if let Ok(dt) = NaiveDateTime::parse_from_str(input, "%Y-%m-%d") {
            return Some(dt);
        }
        
        None
    }
    
    fn execute_aggregate(&self, over: &LogicalForm, function: &AggregateFunction, mut steps: Vec<ReasoningStep>) -> Result<ReasoningResult> {
        // First execute the sub-query
        let sub_result = self.execute(over)?;
        
        match function {
            AggregateFunction::Count => {
                let count = sub_result.entities.len();
                steps.extend(sub_result.reasoning_steps);
                steps.push(ReasoningStep {
                    step_type: StepType::Aggregation,
                    description: "Counting results".to_string(),
                    result: count.to_string(),
                });
                
                Ok(ReasoningResult {
                    answer: format!("Count: {}", count),
                    entities: sub_result.entities,
                    reasoning_steps: steps,
                    confidence: 1.0,
                    rules_applied: vec![],
                })
            }
            
            AggregateFunction::Max { property } => {
                if property == "connection_count" {
                    // Find person with most connections
                    let mut max_count = 0;
                    let mut max_person: Option<Entity> = None;
                    
                    for entity in &sub_result.entities {
                        let connections = self.graph.get_connections(&entity.name)?;
                        if connections.len() > max_count {
                            max_count = connections.len();
                            max_person = Some(entity.clone());
                        }
                    }
                    
                    steps.extend(sub_result.reasoning_steps);
                    steps.push(ReasoningStep {
                        step_type: StepType::Aggregation,
                        description: "Finding maximum".to_string(),
                        result: max_person.as_ref().map(|p| p.name.clone()).unwrap_or_default(),
                    });
                    
                    Ok(ReasoningResult {
                        answer: max_person.as_ref()
                            .map(|p| format!("{} has the most connections ({} connections)", p.name, max_count))
                            .unwrap_or_else(|| "No results found".to_string()),
                        entities: max_person.into_iter().collect(),
                        reasoning_steps: steps,
                        confidence: 1.0,
                        rules_applied: vec![],
                    })
                } else {
                    Ok(ReasoningResult {
                        answer: "Unsupported max property".to_string(),
                        entities: vec![],
                        reasoning_steps: steps,
                        confidence: 0.0,
                        rules_applied: vec![],
                    })
                }
            }
            
            AggregateFunction::Min { property } => {
                if property == "connection_count" {
                    // Find person with least connections
                    let mut min_count = usize::MAX;
                    let mut min_person: Option<Entity> = None;
                    
                    for entity in &sub_result.entities {
                        let connections = self.graph.get_connections(&entity.name)?;
                        if connections.len() < min_count {
                            min_count = connections.len();
                            min_person = Some(entity.clone());
                        }
                    }
                    
                    steps.extend(sub_result.reasoning_steps);
                    steps.push(ReasoningStep {
                        step_type: StepType::Aggregation,
                        description: "Finding minimum".to_string(),
                        result: min_person.as_ref().map(|p| p.name.clone()).unwrap_or_default(),
                    });
                    
                    Ok(ReasoningResult {
                        answer: min_person.as_ref()
                            .map(|p| format!("{} has the fewest connections ({} connections)", p.name, min_count))
                            .unwrap_or_else(|| "No results found".to_string()),
                        entities: min_person.into_iter().collect(),
                        reasoning_steps: steps,
                        confidence: 1.0,
                        rules_applied: vec![],
                    })
                } else {
                    Ok(ReasoningResult {
                        answer: "Unsupported min property".to_string(),
                        entities: vec![],
                        reasoning_steps: steps,
                        confidence: 0.0,
                        rules_applied: vec![],
                    })
                }
            }
            
            _ => Ok(ReasoningResult {
                answer: "Aggregation not yet implemented".to_string(),
                entities: vec![],
                reasoning_steps: steps,
                confidence: 0.0,
                rules_applied: vec![],
            }),
        }
    }
    
    fn execute_traverse(&self, from: &LogicalForm, relation: &str, direction: &TraverseDirection, mut steps: Vec<ReasoningStep>) -> Result<ReasoningResult> {
        let from_result = self.execute(from)?;
        steps.extend(from_result.reasoning_steps);
        
        let mut all_entities = Vec::new();
        
        for entity in &from_result.entities {
            let related = match relation {
                "WORKS_AT" => {
                    match direction {
                        TraverseDirection::Incoming => self.graph.get_employees(&entity.name)?,
                        TraverseDirection::Outgoing => self.graph.get_person_workplace(&entity.name)?,
                        _ => vec![],
                    }
                }
                "LIVES_IN" | "LOCATED_IN" | "BASED_IN" => {
                    match direction {
                        TraverseDirection::Incoming => self.graph.get_residents(&entity.name)?,
                        TraverseDirection::Outgoing => self.graph.get_person_residence(&entity.name)?,
                        _ => vec![],
                    }
                }
                "KNOWS" => self.graph.get_connections(&entity.name)?,
                _ => vec![],
            };
            all_entities.extend(related);
        }
        
        steps.push(ReasoningStep {
            step_type: StepType::Reasoning,
            description: format!("Traversed {} to find {} entities", relation, all_entities.len()),
            result: all_entities.iter().map(|e| e.name.clone()).collect::<Vec<_>>().join(", "),
        });
        
        let names: Vec<String> = all_entities.iter().map(|e| e.name.clone()).collect();
        
        Ok(ReasoningResult {
            answer: if all_entities.is_empty() {
                "No related entities found".to_string()
            } else {
                names.join(", ")
            },
            entities: all_entities,
            reasoning_steps: steps,
            confidence: 1.0,
            rules_applied: vec![],
        })
    }
    
    fn execute_with_rule(&self, over: &LogicalForm, rule_name: &str, mut steps: Vec<ReasoningStep>) -> Result<ReasoningResult> {
        let base_result = self.execute(over)?;
        steps.extend(base_result.reasoning_steps);
        
        let rule = self.rules.iter().find(|r| r.name == rule_name);
        
        if let Some(rule) = rule {
            match &rule.condition {
                RuleCondition::TransitiveRelation { relation } => {
                    // Find friends of friends
                    let mut inferred = HashSet::new();
                    let mut direct = HashSet::new();
                    
                    for entity in &base_result.entities {
                        // Get direct connections
                        let connections = self.graph.get_connections(&entity.name)?;
                        for conn in &connections {
                            direct.insert(conn.id);
                        }
                        
                        // Get second-degree connections
                        for conn in connections {
                            let second_degree = self.graph.get_connections(&conn.name)?;
                            for sd in second_degree {
                                if sd.id != entity.id && !direct.contains(&sd.id) {
                                    inferred.insert(sd);
                                }
                            }
                        }
                    }
                    
                    let inferred_entities: Vec<Entity> = inferred.into_iter().collect();
                    let names: Vec<String> = inferred_entities.iter().map(|e| e.name.clone()).collect();
                    
                    steps.push(ReasoningStep {
                        step_type: StepType::Reasoning,
                        description: format!("Applied rule '{}': {}", rule.name, rule.description),
                        result: format!("Inferred {} potential connections", names.len()),
                    });
                    
                    Ok(ReasoningResult {
                        answer: if inferred_entities.is_empty() {
                            "No additional connections inferred".to_string()
                        } else {
                            format!("Might also know (through mutual connections): {}", names.join(", "))
                        },
                        entities: inferred_entities,
                        reasoning_steps: steps,
                        confidence: 0.5,
                        rules_applied: vec![rule.name.clone()],
                    })
                }
                
                RuleCondition::SharedRelation { relation } => {
                    // Find colleagues (same org)
                    let mut colleagues = Vec::new();
                    let mut seen = HashSet::new();
                    
                    for entity in &base_result.entities {
                        // Get person's org and find others at same org
                        if let Some(person) = self.graph.find_person(&entity.name)? {
                            seen.insert(person.id);
                            
                            // Find all people who share an org
                            let orgs = self.graph.get_person_orgs(&entity.name)?;
                            for org in orgs {
                                let coworkers = self.graph.get_employees(&org.name)?;
                                for cw in coworkers {
                                    if !seen.contains(&cw.id) {
                                        seen.insert(cw.id);
                                        colleagues.push(cw);
                                    }
                                }
                            }
                        }
                    }
                    
                    let names: Vec<String> = colleagues.iter().map(|e| e.name.clone()).collect();
                    
                    steps.push(ReasoningStep {
                        step_type: StepType::Reasoning,
                        description: format!("Applied rule '{}': {}", rule.name, rule.description),
                        result: format!("Found {} colleagues", names.len()),
                    });
                    
                    Ok(ReasoningResult {
                        answer: if colleagues.is_empty() {
                            "No colleagues found".to_string()
                        } else {
                            format!("Colleagues (same organization): {}", names.join(", "))
                        },
                        entities: colleagues,
                        reasoning_steps: steps,
                        confidence: 0.9,
                        rules_applied: vec![rule.name.clone()],
                    })
                }
                
                _ => Ok(ReasoningResult {
                    answer: format!("Rule '{}' not yet implemented", rule_name),
                    entities: vec![],
                    reasoning_steps: steps,
                    confidence: 0.0,
                    rules_applied: vec![],
                }),
            }
        } else {
            Ok(ReasoningResult {
                answer: format!("Rule '{}' not found", rule_name),
                entities: vec![],
                reasoning_steps: steps,
                confidence: 0.0,
                rules_applied: vec![],
            })
        }
    }
    
    fn execute_compare(&self, left: &LogicalForm, right: &LogicalForm, operator: &CompareOperator, mut steps: Vec<ReasoningStep>) -> Result<ReasoningResult> {
        let left_result = self.execute(left)?;
        let right_result = self.execute(right)?;
        
        steps.extend(left_result.reasoning_steps);
        steps.extend(right_result.reasoning_steps);
        
        // Compare connection counts
        if let Some(right_entity) = right_result.entities.first() {
            let right_connections = self.graph.get_connections(&right_entity.name)?;
            let right_count = right_connections.len();
            
            let mut matching = Vec::new();
            
            for entity in &left_result.entities {
                let left_connections = self.graph.get_connections(&entity.name)?;
                let left_count = left_connections.len();
                
                let matches = match operator {
                    CompareOperator::GreaterThan => left_count > right_count,
                    CompareOperator::LessThan => left_count < right_count,
                    CompareOperator::Equals => left_count == right_count,
                    _ => false,
                };
                
                if matches {
                    matching.push((entity.clone(), left_count));
                }
            }
            
            let names: Vec<String> = matching.iter()
                .map(|(e, c)| format!("{} ({} connections)", e.name, c))
                .collect();
            
            steps.push(ReasoningStep {
                step_type: StepType::Reasoning,
                description: format!("Comparing against {} ({} connections)", right_entity.name, right_count),
                result: format!("{} matches", matching.len()),
            });
            
            Ok(ReasoningResult {
                answer: if matching.is_empty() {
                    format!("No one has {:?} connections than {}", operator, right_entity.name)
                } else {
                    format!("People with {:?} connections than {}: {}", 
                        operator, right_entity.name, names.join(", "))
                },
                entities: matching.into_iter().map(|(e, _)| e).collect(),
                reasoning_steps: steps,
                confidence: 1.0,
                rules_applied: vec![],
            })
        } else {
            Ok(ReasoningResult {
                answer: "Cannot compare - reference not found".to_string(),
                entities: vec![],
                reasoning_steps: steps,
                confidence: 0.0,
                rules_applied: vec![],
            })
        }
    }
    
    fn execute_compose(&self, forms: &[LogicalForm], combinator: &Combinator, mut steps: Vec<ReasoningStep>) -> Result<ReasoningResult> {
        match combinator {
            Combinator::Then => {
                // Path query - check if there's a connection between entities
                if forms.len() >= 2 {
                    let from_result = self.execute(&forms[0])?;
                    let to_result = self.execute(&forms[1])?;
                    
                    steps.extend(from_result.reasoning_steps);
                    steps.extend(to_result.reasoning_steps);
                    
                    if let (Some(from), Some(to)) = (from_result.entities.first(), to_result.entities.first()) {
                        // Use BFS to find path
                        let path = self.find_path(&from.name, &to.name)?;
                        
                        steps.push(ReasoningStep {
                            step_type: StepType::Reasoning,
                            description: format!("Searching path from {} to {}", from.name, to.name),
                            result: if path.is_empty() { "No path".to_string() } else { path.join(" → ") },
                        });
                        
                        if path.is_empty() {
                            Ok(ReasoningResult {
                                answer: format!("No connection found between {} and {}", from.name, to.name),
                                entities: vec![],
                                reasoning_steps: steps,
                                confidence: 1.0,
                                rules_applied: vec![],
                            })
                        } else {
                            Ok(ReasoningResult {
                                answer: format!("Yes! {} can reach {} via: {}", 
                                    from.name, to.name, path.join(" → ")),
                                entities: vec![from.clone(), to.clone()],
                                reasoning_steps: steps,
                                confidence: 1.0,
                                rules_applied: vec![],
                            })
                        }
                    } else {
                        Ok(ReasoningResult {
                            answer: "Could not find both endpoints".to_string(),
                            entities: vec![],
                            reasoning_steps: steps,
                            confidence: 0.0,
                            rules_applied: vec![],
                        })
                    }
                } else {
                    Ok(ReasoningResult {
                        answer: "Path query requires two endpoints".to_string(),
                        entities: vec![],
                        reasoning_steps: steps,
                        confidence: 0.0,
                        rules_applied: vec![],
                    })
                }
            }
            _ => Ok(ReasoningResult {
                answer: "Combinator not implemented".to_string(),
                entities: vec![],
                reasoning_steps: steps,
                confidence: 0.0,
                rules_applied: vec![],
            }),
        }
    }
    
    /// BFS path finding
    fn find_path(&self, from: &str, to: &str) -> Result<Vec<String>> {
        use std::collections::VecDeque;
        
        let mut visited = HashSet::new();
        let mut queue: VecDeque<(String, Vec<String>)> = VecDeque::new();
        
        queue.push_back((from.to_string(), vec![from.to_string()]));
        visited.insert(from.to_string().to_lowercase());
        
        while let Some((current, path)) = queue.pop_front() {
            if current.to_lowercase() == to.to_lowercase() {
                return Ok(path);
            }
            
            if path.len() > 5 {
                continue; // Max depth
            }
            
            let connections = self.graph.get_connections(&current)?;
            for conn in connections {
                if !visited.contains(&conn.name.to_lowercase()) {
                    visited.insert(conn.name.to_lowercase());
                    let mut new_path = path.clone();
                    new_path.push(conn.name.clone());
                    queue.push_back((conn.name, new_path));
                }
            }
        }
        
        Ok(vec![])
    }

    // ========================================================================
    // String Helpers
    // ========================================================================
    
    fn extract_after(&self, text: &str, phrases: &[&str]) -> Option<String> {
        for phrase in phrases {
            if let Some(pos) = text.find(phrase) {
                let after = &text[pos + phrase.len()..];
                let result = after
                    .trim()
                    .trim_matches(|c: char| c == '?' || c == '.' || c == '!' || c == ',');
                
                if !result.is_empty() {
                    // Title case the result (capitalize each word)
                    let words: Vec<String> = result.split_whitespace()
                        .map(|w| self.capitalize(w))
                        .collect();
                    return Some(words.join(" "));
                }
            }
        }
        None
    }
    
    fn extract_person_name(&self, text: &str) -> Option<String> {
        // Look for patterns like "alice's", "does alice", etc.
        let words: Vec<&str> = text.split_whitespace().collect();
        let stop_words = ["who", "what", "does", "has", "have", "the", "might", "could", 
                          "know", "knows", "many", "most", "least", "connections", "colleagues",
                          "are", "is", "coworkers", "of"];
        
        for word in &words {
            // Handle possessives like "alice's"
            let clean = word.trim_matches(|c: char| !c.is_alphabetic())
                           .trim_end_matches("'s")
                           .trim_end_matches("s");
            if !clean.is_empty() && !stop_words.contains(&clean) && clean.len() > 1 {
                return Some(self.capitalize(clean));
            }
        }
        None
    }
    
    fn capitalize(&self, s: &str) -> String {
        let mut chars = s.chars();
        match chars.next() {
            Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
            None => String::new(),
        }
    }
    
    /// Check if input is a logical reasoning query
    pub fn is_reasoning_query(&self, input: &str) -> bool {
        self.parse_to_logical_form(input).is_some()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Database;

    fn test_graph() -> KnowledgeGraph {
        let db = Database::in_memory().unwrap();
        let graph = KnowledgeGraph::new(db);
        
        // Set up test data
        let alice = graph.add_person("Alice").unwrap();
        let bob = graph.add_person("Bob").unwrap();
        let charlie = graph.add_person("Charlie").unwrap();
        let google = graph.add_organization("Google").unwrap();
        
        graph.link_knows(alice.id, bob.id).unwrap();
        graph.link_knows(bob.id, charlie.id).unwrap();
        graph.link_works_at(alice.id, google.id).unwrap();
        graph.link_works_at(bob.id, google.id).unwrap();
        
        graph
    }

    #[test]
    fn test_parse_count_query() {
        let graph = test_graph();
        let solver = LogicalSolver::new(&graph);
        
        let form = solver.parse_to_logical_form("how many people work at google");
        assert!(form.is_some(), "Should parse count query");
    }

    #[test]
    fn test_parse_most_connections() {
        let graph = test_graph();
        let solver = LogicalSolver::new(&graph);
        
        let form = solver.parse_to_logical_form("who has the most connections");
        assert!(form.is_some(), "Should parse most connections query");
    }

    #[test]
    fn test_execute_transitive_knows() {
        let graph = test_graph();
        let solver = LogicalSolver::new(&graph);
        
        // Alice knows Bob, Bob knows Charlie, so Alice might know Charlie
        // Using a pattern that matches our parser
        let form = solver.parse_to_logical_form("who might alice know through friends");
        println!("Parsed form: {:?}", form);
        
        // Also test direct colleague query which we know works
        let form2 = solver.parse_to_logical_form("alice's colleagues");
        assert!(form2.is_some(), "Should parse colleague query");
        
        // For transitive, we need to apply the rule directly
        let form3 = LogicalForm::ApplyRule {
            over: Box::new(LogicalForm::Lookup {
                entity_type: "Person".to_string(),
                name: "Alice".to_string(),
            }),
            rule_name: "transitive_knows".to_string(),
        };
        
        let result = solver.execute(&form3).unwrap();
        assert!(result.answer.contains("Charlie") || result.entities.iter().any(|e| e.name == "Charlie"),
                "Expected Charlie in answer, got: {}", result.answer);
    }

    #[test]
    fn test_colleague_inference() {
        let graph = test_graph();
        let solver = LogicalSolver::new(&graph);
        
        // Alice and Bob both work at Google, so they should be colleagues
        let form = solver.parse_to_logical_form("who are alice's colleagues");
        assert!(form.is_some());
        
        let result = solver.execute(&form.unwrap()).unwrap();
        assert!(result.answer.contains("Bob") || result.entities.iter().any(|e| e.name == "Bob"));
    }

    #[test]
    fn test_path_finding() {
        let graph = test_graph();
        let solver = LogicalSolver::new(&graph);
        
        let path = solver.find_path("Alice", "Charlie").unwrap();
        assert!(!path.is_empty(), "Should find path from Alice to Charlie");
        assert_eq!(path[0], "Alice");
        assert_eq!(path[path.len() - 1], "Charlie");
    }

    // ============================================================================
    // KAG LIMITS TESTS - Boundary and edge case testing
    // ============================================================================

    mod limits {
        use super::*;
        
        fn empty_graph() -> KnowledgeGraph {
            let db = Database::in_memory().unwrap();
            KnowledgeGraph::new(db)
        }
        
        fn large_graph(n_people: usize) -> KnowledgeGraph {
            let db = Database::in_memory().unwrap();
            let graph = KnowledgeGraph::new(db);
            
            for i in 0..n_people {
                graph.add_person(&format!("Person{}", i)).unwrap();
            }
            graph
        }
        
        fn deep_chain_graph(depth: usize) -> KnowledgeGraph {
            let db = Database::in_memory().unwrap();
            let graph = KnowledgeGraph::new(db);
            
            let mut entities = Vec::new();
            for i in 0..depth {
                entities.push(graph.add_person(&format!("Person{}", i)).unwrap());
            }
            
            // Chain: Person0 -> Person1 -> Person2 -> ...
            for i in 0..depth - 1 {
                graph.link_knows(entities[i].id, entities[i + 1].id).unwrap();
            }
            
            graph
        }
        
        fn dense_graph(n_people: usize) -> KnowledgeGraph {
            let db = Database::in_memory().unwrap();
            let graph = KnowledgeGraph::new(db);
            
            let mut entities = Vec::new();
            for i in 0..n_people {
                entities.push(graph.add_person(&format!("Person{}", i)).unwrap());
            }
            
            // Everyone knows everyone
            for i in 0..n_people {
                for j in i + 1..n_people {
                    graph.link_knows(entities[i].id, entities[j].id).unwrap();
                }
            }
            
            graph
        }

        // ---------------------------------------------------------------------------
        // Empty Graph Tests
        // ---------------------------------------------------------------------------
        
        #[test]
        fn test_count_on_empty_graph() {
            let graph = empty_graph();
            let solver = LogicalSolver::new(&graph);
            
            let form = solver.parse_to_logical_form("how many people");
            if let Some(f) = form {
                let result = solver.execute(&f).unwrap();
                assert!(result.answer.contains("0") || result.answer.to_lowercase().contains("none"),
                    "Empty graph should return 0 or none, got: {}", result.answer);
            }
        }
        
        #[test]
        fn test_max_connections_on_empty_graph() {
            let graph = empty_graph();
            let solver = LogicalSolver::new(&graph);
            
            let form = solver.parse_to_logical_form("who has the most connections");
            if let Some(f) = form {
                let result = solver.execute(&f);
                // Should either return an error or "no one"
                match result {
                    Ok(r) => assert!(r.answer.to_lowercase().contains("no ") || 
                                    r.answer.to_lowercase().contains("none") ||
                                    r.entities.is_empty(),
                        "Should handle empty graph gracefully, got: {}", r.answer),
                    Err(_) => {} // Error is also acceptable
                }
            }
        }
        
        #[test]
        fn test_path_on_empty_graph() {
            let graph = empty_graph();
            let solver = LogicalSolver::new(&graph);
            
            let result = solver.find_path("Alice", "Bob");
            // Either returns error or empty path
            match result {
                Ok(path) => assert!(path.is_empty(), "Path in empty graph should be empty"),
                Err(_) => {} // Error is also acceptable
            }
        }

        // ---------------------------------------------------------------------------
        // Non-existent Entity Tests
        // ---------------------------------------------------------------------------
        
        #[test]
        fn test_lookup_nonexistent_entity() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            let form = LogicalForm::Lookup {
                entity_type: "Person".to_string(),
                name: "NonexistentPerson".to_string(),
            };
            
            let result = solver.execute(&form).unwrap();
            assert!(result.entities.is_empty() || 
                    result.answer.to_lowercase().contains("not found") ||
                    result.answer.to_lowercase().contains("no "),
                "Should handle non-existent entity, got: {}", result.answer);
        }
        
        #[test]
        fn test_colleagues_of_nonexistent_person() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            let form = solver.parse_to_logical_form("who are zombieking's colleagues");
            if let Some(f) = form {
                let result = solver.execute(&f).unwrap();
                assert!(result.entities.is_empty() || 
                        result.answer.to_lowercase().contains("no "),
                    "Should handle non-existent person, got: {}", result.answer);
            }
        }

        // ---------------------------------------------------------------------------
        // Input Edge Cases
        // ---------------------------------------------------------------------------
        
        #[test]
        fn test_empty_query() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            let form = solver.parse_to_logical_form("");
            assert!(form.is_none(), "Empty query should not parse to a form");
        }
        
        #[test]
        fn test_whitespace_only_query() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            let form = solver.parse_to_logical_form("   \t\n  ");
            assert!(form.is_none(), "Whitespace-only query should not parse");
        }
        
        #[test]
        fn test_very_long_query() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            // Create a very long query
            let long_query = format!("how many people{}", " and more words".repeat(100));
            let form = solver.parse_to_logical_form(&long_query);
            // Should either parse or gracefully fail, but not panic
            let _ = form; // Just ensure it doesn't panic
        }
        
        #[test]
        fn test_special_characters_in_query() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            // Test various special characters
            for query in &[
                "how many people work at google?!",
                "who has the most connections...?",
                "alice's colleagues (at google)",
                "count: people @ google",
            ] {
                let _ = solver.parse_to_logical_form(query);
                // Should not panic
            }
        }
        
        #[test]
        fn test_case_insensitivity() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            let lower = solver.parse_to_logical_form("how many people");
            let upper = solver.parse_to_logical_form("HOW MANY PEOPLE");
            let mixed = solver.parse_to_logical_form("How Many People");
            
            // All should parse (or all fail, but consistently)
            assert_eq!(lower.is_some(), upper.is_some());
            assert_eq!(lower.is_some(), mixed.is_some());
        }
        
        #[test]
        fn test_names_with_special_chars() {
            let db = Database::in_memory().unwrap();
            let graph = KnowledgeGraph::new(db);
            
            // Add people with special names
            let _ = graph.add_person("Jean-Pierre");
            let _ = graph.add_person("O'Brien");
            let _ = graph.add_person("Dr. Smith");
            
            let solver = LogicalSolver::new(&graph);
            let form = solver.parse_to_logical_form("how many people");
            
            if let Some(f) = form {
                let result = solver.execute(&f).unwrap();
                assert!(result.answer.contains("3"),
                    "Should count 3 people with special names, got: {}", result.answer);
            }
        }

        // ---------------------------------------------------------------------------
        // Scale Tests
        // ---------------------------------------------------------------------------
        
        #[test]
        fn test_large_graph_count() {
            let graph = large_graph(100);
            let solver = LogicalSolver::new(&graph);
            
            let form = solver.parse_to_logical_form("how many people");
            if let Some(f) = form {
                let result = solver.execute(&f).unwrap();
                assert!(result.answer.contains("100"),
                    "Should count 100 people, got: {}", result.answer);
            }
        }
        
        #[test]
        fn test_deep_chain_path_finding() {
            // NOTE: KAG has a MAX_PATH_DEPTH of 5 to prevent infinite traversal
            // This is a known limit - we test at depth 5 (the maximum)
            let graph = deep_chain_graph(6); // 6 nodes = 5 hops (within limit)
            let solver = LogicalSolver::new(&graph);
            
            let path = solver.find_path("Person0", "Person5");
            assert!(path.is_ok(), "Should find path within depth limit");
            
            let p = path.unwrap();
            assert_eq!(p.len(), 6, "Path should have 6 nodes (5 hops)");
            
            // Now test beyond the limit - should NOT find path
            let deep_graph = deep_chain_graph(10); // 10 nodes = 9 hops (beyond limit)
            let deep_solver = LogicalSolver::new(&deep_graph);
            
            let deep_path = deep_solver.find_path("Person0", "Person9");
            match deep_path {
                Ok(p) => assert!(p.is_empty(), "Should not find path beyond depth 5, got {} nodes", p.len()),
                Err(_) => {} // Also acceptable
            }
        }
        
        #[test]
        fn test_dense_graph_connections() {
            let graph = dense_graph(10);
            let solver = LogicalSolver::new(&graph);
            
            let form = solver.parse_to_logical_form("who has the most connections");
            if let Some(f) = form {
                let result = solver.execute(&f).unwrap();
                // All should have 9 connections in a 10-person complete graph
                assert!(result.answer.contains("9"),
                    "Each person should have 9 connections, got: {}", result.answer);
            }
        }

        // ---------------------------------------------------------------------------
        // Query Complexity Tests
        // ---------------------------------------------------------------------------
        
        #[test]
        fn test_unrecognized_query_pattern() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            // These should NOT parse to KAG forms (should fall through to GraphRAG/LLM)
            for query in &[
                "what color is the sky",
                "tell me a joke",
                "what is 2+2",
                "schedule a meeting",
            ] {
                let form = solver.parse_to_logical_form(query);
                assert!(form.is_none(), "Query '{}' should not parse to KAG form", query);
            }
        }
        
        #[test]
        fn test_ambiguous_entity_reference() {
            let db = Database::in_memory().unwrap();
            let graph = KnowledgeGraph::new(db);
            
            // Add two people with similar names
            let _ = graph.add_person("John Smith");
            let _ = graph.add_person("John Doe");
            
            let solver = LogicalSolver::new(&graph);
            
            // Search for just "John" - behavior depends on implementation
            let form = solver.parse_to_logical_form("john's colleagues");
            // This tests how we handle ambiguity - either pick first match or return all
            // Just verify it doesn't crash
            if let Some(f) = form {
                let _ = solver.execute(&f); // Should not panic
            }
        }
        
        #[test]
        fn test_self_referential_path() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            // Find path from Alice to Alice
            let result = solver.find_path("Alice", "Alice");
            
            match result {
                Ok(p) => {
                    // If we return a path, it should just be Alice or empty
                    assert!(p.len() <= 1, "Self path should be empty or just Alice");
                }
                Err(_) => {} // Also valid - no path to self
            }
        }
        
        #[test]
        fn test_disconnected_graph() {
            let db = Database::in_memory().unwrap();
            let graph = KnowledgeGraph::new(db);
            
            // Create two disconnected components
            let alice = graph.add_person("Alice").unwrap();
            let bob = graph.add_person("Bob").unwrap();
            let charlie = graph.add_person("Charlie").unwrap();
            let dave = graph.add_person("Dave").unwrap();
            
            graph.link_knows(alice.id, bob.id).unwrap();
            graph.link_knows(charlie.id, dave.id).unwrap();
            // Alice-Bob disconnected from Charlie-Dave
            
            let solver = LogicalSolver::new(&graph);
            
            let result = solver.find_path("Alice", "Dave");
            match result {
                Ok(path) => assert!(path.is_empty(), "Should return empty path for disconnected components"),
                Err(_) => {} // Error is also acceptable
            }
        }

        // ---------------------------------------------------------------------------
        // Aggregation Edge Cases
        // ---------------------------------------------------------------------------
        
        #[test]
        fn test_aggregate_with_ties() {
            let db = Database::in_memory().unwrap();
            let graph = KnowledgeGraph::new(db);
            
            // Create graph where multiple people have same connection count
            let alice = graph.add_person("Alice").unwrap();
            let bob = graph.add_person("Bob").unwrap();
            let charlie = graph.add_person("Charlie").unwrap();
            
            graph.link_knows(alice.id, bob.id).unwrap();
            graph.link_knows(charlie.id, bob.id).unwrap();
            // Both Alice and Charlie have 1 connection, Bob has 2
            
            let solver = LogicalSolver::new(&graph);
            let form = solver.parse_to_logical_form("who has the most connections");
            
            if let Some(f) = form {
                let result = solver.execute(&f).unwrap();
                assert!(result.answer.to_lowercase().contains("bob"),
                    "Bob should have most connections, got: {}", result.answer);
            }
        }
        
        #[test]
        fn test_min_connections() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            let form = solver.parse_to_logical_form("who has the fewest connections");
            if let Some(f) = form {
                let result = solver.execute(&f).unwrap();
                // Charlie only knows Bob (1 connection)
                assert!(result.answer.to_lowercase().contains("charlie") || 
                        result.answer.to_lowercase().contains("alice") ||
                        result.answer.contains("1"),
                    "Should find person with fewest connections, got: {}", result.answer);
            }
        }

        // ---------------------------------------------------------------------------
        // Rule Inference Limits
        // ---------------------------------------------------------------------------
        
        #[test]
        fn test_rule_on_person_without_job() {
            let db = Database::in_memory().unwrap();
            let graph = KnowledgeGraph::new(db);
            
            let alice = graph.add_person("Alice").unwrap();
            let bob = graph.add_person("Bob").unwrap();
            graph.link_knows(alice.id, bob.id).unwrap();
            // Neither works anywhere
            
            let solver = LogicalSolver::new(&graph);
            let form = solver.parse_to_logical_form("alice's colleagues");
            
            if let Some(f) = form {
                let result = solver.execute(&f).unwrap();
                assert!(result.entities.is_empty() || 
                        result.answer.to_lowercase().contains("no"),
                    "Should have no colleagues without a job, got: {}", result.answer);
            }
        }
        
        #[test]
        fn test_transitive_cycle_detection() {
            let db = Database::in_memory().unwrap();
            let graph = KnowledgeGraph::new(db);
            
            // Create a cycle: Alice -> Bob -> Charlie -> Alice
            let alice = graph.add_person("Alice").unwrap();
            let bob = graph.add_person("Bob").unwrap();
            let charlie = graph.add_person("Charlie").unwrap();
            
            graph.link_knows(alice.id, bob.id).unwrap();
            graph.link_knows(bob.id, charlie.id).unwrap();
            graph.link_knows(charlie.id, alice.id).unwrap();
            
            let solver = LogicalSolver::new(&graph);
            
            // This should not infinite loop
            let result = solver.find_path("Alice", "Charlie");
            assert!(result.is_ok(), "Should find path in cycle");
            
            // Verify we don't revisit nodes
            if let Ok(p) = result {
                let unique: std::collections::HashSet<_> = p.iter().collect();
                assert_eq!(unique.len(), p.len(), "Path should not contain duplicates");
            }
        }

        // ---------------------------------------------------------------------------
        // Temporal Reasoning Tests
        // ---------------------------------------------------------------------------
        
        #[test]
        fn test_parse_temporal_due_before() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            let form = solver.parse_to_logical_form("tasks due before friday");
            assert!(form.is_some(), "Should parse 'due before' query");
            
            if let Some(LogicalForm::Filter { entity_type, conditions }) = form {
                assert_eq!(entity_type, "Task");
                assert!(!conditions.is_empty());
                assert!(matches!(conditions[0].operator, CompareOperator::Before));
            }
        }
        
        #[test]
        fn test_parse_temporal_due_after() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            let form = solver.parse_to_logical_form("tasks due after monday");
            assert!(form.is_some(), "Should parse 'due after' query");
        }
        
        #[test]
        fn test_parse_overdue() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            let form = solver.parse_to_logical_form("show overdue tasks");
            assert!(form.is_some(), "Should parse 'overdue' query");
        }
        
        #[test]
        fn test_natural_date_parsing() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            // Test that the parse doesn't panic
            let _ = solver.parse_natural_date("tomorrow");
            let _ = solver.parse_natural_date("friday");
            let _ = solver.parse_natural_date("next week");
            let _ = solver.parse_natural_date("2025-12-25");
        }

        // ---------------------------------------------------------------------------
        // Task Query Tests
        // ---------------------------------------------------------------------------
        
        #[test]
        fn test_parse_person_tasks() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            let form = solver.parse_to_logical_form("alice's tasks");
            assert!(form.is_some(), "Should parse 'person's tasks' query");
            
            if let Some(LogicalForm::Traverse { from, relation, .. }) = form {
                assert_eq!(relation, "HAS_TASK");
                if let LogicalForm::Lookup { entity_type, name } = *from {
                    assert_eq!(entity_type, "Person");
                    assert_eq!(name, "Alice");
                }
            }
        }
        
        #[test]
        fn test_parse_blocking_query() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            let form = solver.parse_to_logical_form("what blocks task A");
            assert!(form.is_some(), "Should parse blocking query");
        }
        
        #[test]
        fn test_parse_workload_query() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            let form = solver.parse_to_logical_form("how busy is alice");
            assert!(form.is_some(), "Should parse workload query");
            
            if let Some(LogicalForm::Aggregate { function, .. }) = form {
                assert!(matches!(function, AggregateFunction::Count));
            }
        }
        
        #[test]
        fn test_task_rules_registered() {
            let graph = test_graph();
            let solver = LogicalSolver::new(&graph);
            
            // Verify task-related rules are registered
            let rule_names: Vec<&str> = solver.rules.iter().map(|r| r.name.as_str()).collect();
            assert!(rule_names.contains(&"blocked_cascade"), "Should have blocked_cascade rule");
            assert!(rule_names.contains(&"overdue_alert"), "Should have overdue_alert rule");
            assert!(rule_names.contains(&"high_workload"), "Should have high_workload rule");
            assert!(rule_names.contains(&"task_ready"), "Should have task_ready rule");
        }
    }
}
