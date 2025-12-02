# DOSA Development Plan

## Overview
Building a CLI-based personal assistant with:
- Local LLM inference (Ollama)
- Knowledge graph for user information
- Extensible command routing system
- Future: Calendar, Reminders, Contacts integration

## Tech Stack
- **Language**: Rust
- **LLM**: Ollama (REST API)
- **Database**: SQLite (via rusqlite)
- **HTTP Client**: reqwest
- **Serialization**: serde + serde_json

---

## Phase 1: Foundation ✅ COMPLETE

### Tasks
- [x] Initialize Rust project with Cargo
- [x] Set up Ollama client wrapper
- [x] Create SQLite database schema for knowledge graph
- [x] Implement basic knowledge graph (Person, Organization entities)
- [x] Build extensible command router
- [x] Basic CLI natural language parsing via LLM
- [x] Test: Verify LLM responds and knowledge graph persists data

### Implemented Commands
| Command | Description |
|---------|-------------|
| `/add person <name>` | Add a person to the knowledge graph |
| `/add org <name>` | Add an organization |
| `/link <A> works_at <B>` | Link person to organization |
| `/link <A> knows <B>` | Link two people |
| `/set <name> <key> <value>` | Set a property on an entity |
| `/info <name>` | Get information about an entity |
| `/list [people\|orgs]` | List entities |
| `/summary` | Get knowledge graph summary |
| `/help` | Show available commands |
| `/exit` | Exit the assistant |

### Project Structure
```
dosa-rs/
├── Cargo.toml
├── src/
│   ├── main.rs                  # Entry point, CLI loop
│   ├── llm/
│   │   ├── mod.rs
│   │   └── ollama.rs            # Ollama API client
│   ├── knowledge/
│   │   ├── mod.rs
│   │   ├── graph.rs             # Knowledge graph operations
│   │   └── entities.rs          # Entity/Relationship types
│   ├── calendar/
│   │   ├── mod.rs
│   │   └── events.rs            # Calendar & date parsing
│   ├── reminders/
│   │   ├── mod.rs
│   │   └── tasks.rs             # Tasks & reminders
│   ├── contacts/
│   │   ├── mod.rs
│   │   └── manager.rs           # Contact management
│   ├── router/
│   │   ├── mod.rs
│   │   ├── commands.rs          # Core commands
│   │   ├── phase2_commands.rs   # Calendar/Task/Contact commands
│   │   └── phase3_commands.rs   # Intelligence commands
│   ├── intelligence/
│   │   ├── mod.rs
│   │   ├── alerts.rs            # Proactive notifications
│   │   ├── query.rs             # Natural language KG queries
│   │   └── extraction.rs        # Entity extraction
│   └── storage/
│       ├── mod.rs
│       └── database.rs          # SQLite wrapper
└── data/
    └── dosa.db                  # SQLite database (created at runtime)
```

---

## Phase 2: Core Functionality ✅ COMPLETE

### Tasks
- [x] Calendar events module (CRUD)
- [x] Reminders module with scheduling
- [x] Contact management module
- [x] Link contacts to knowledge graph entities
- [x] Test: All 13 unit tests passing

### New Commands Added
| Command | Description |
|---------|-------------|
| `/event add <title> at <time>` | Add a calendar event |
| `/event list [today\|week]` | List events |
| `/today` | Show today's schedule + tasks |
| `/remind <task> [by <time>]` | Add a quick reminder |
| `/task add <title> [priority:X] [by <time>]` | Add task with priority |
| `/tasks [all\|overdue]` | List tasks |
| `/done <id or title>` | Mark task as done |
| `/contact add <name>` | Add a new contact |
| `/contacts [at <org>]` | List contacts |
| `/contact <name>` | View contact details |

### Date/Time Parsing
Supports natural language:
- `tomorrow 2pm`
- `dec 5 3:30pm`
- `next monday 10am`
- `today`, `tomorrow`, `next week`

---

## Phase 3: Intelligence ✅ COMPLETE

### Tasks
- [x] Conversation context/memory (keeps last 10 exchanges)
- [x] Proactive reminder notifications on startup
- [x] Natural language queries over knowledge graph
- [x] Pattern-based entity extraction from conversations
- [x] Test: Query relationships ("Who works at X?")

### New Commands Added
| Command | Description |
|---------|-------------|
| `/briefing` | Get daily briefing with events and tasks |
| `/alerts` | Show current notifications |
| `/query <question>` | Query knowledge graph naturally |
| `/who <query>` | Find people (e.g., "works at Google") |
| `/search <term>` | Search knowledge graph |

### Natural Language Features
- Ask "Who works at X?" without commands
- Ask "What do you know about Y?"
- Say "Alice works at Google" and it auto-extracts entities

---

## Phase 4: Enhanced GraphRAG & Schema ⏳ NEXT

### Goals
Enable natural language queries like "How can I contact Charlie?" or "Who works on Project X?" 
using GraphRAG-style intent detection and multi-hop graph traversal.

### Enhanced Schema
```
ENTITIES (Nodes)
├── Person: name, email, phone, role
├── Organization: name, industry
├── Project: name, status, deadline, description
└── Location: city, country, timezone

RELATIONSHIPS (Edges)
├── WORKS_AT: Person → Organization
├── KNOWS: Person ↔ Person
├── WORKS_ON: Person → Project
├── OWNS: Person/Org → Project
├── COLLABORATES_WITH: Person ↔ Person (context: project)
├── LOCATED_IN: Person/Org → Location
├── MANAGES: Person → Project/Person
└── REPORTS_TO: Person → Person
```

### Phase 4A: Schema Enhancement
- [ ] Add `Project` entity type with CRUD
- [ ] Add `Location` entity type
- [ ] Add new relationship types
- [ ] Update database schema with migrations

### Phase 4B: Intent-Based Query Router
- [ ] Create intent classifier (contact, location, project, relationship)
- [ ] Build query templates for common intents
- [ ] LLM-based entity extraction from queries

### Phase 4C: GraphRAG Query Engine
- [ ] Multi-hop graph traversal
- [ ] Context gathering from related entities
- [ ] Natural response formatting

### Phase 4D: Action Preparation
- [ ] "Send email to X" → returns email + action structure
- [ ] "Call Y" → returns phone + action structure
- [ ] Prepare for future agent actions

### Query Examples
| User Query | Intent | Graph Operation |
|------------|--------|-----------------|
| "How can I contact Charlie?" | contact_info | Get person's email, phone |
| "Who works on Project X?" | project_members | Traverse WORKS_ON edges |
| "Who is in London?" | location_query | Traverse LOCATED_IN edges |
| "Who can help with Project X?" | multi_hop | Project → members → their skills |

---

## Phase 5: macOS Integration (Future)

### Tasks
- [ ] Apple Calendar sync
- [ ] Apple Contacts sync
- [ ] System notifications
- [ ] Test: Bidirectional sync works

---

## Phase 6: External Service Integrations ⏳ PLANNED

> Full details in `dosa-rs/docs/INTEGRATIONS_PLAN.md`

### Overview
Integrate with external services for a unified personal assistant experience:
- **Google Calendar**: Two-way sync, event creation, invitations
- **Gmail**: Email retrieval, importance scoring, send/reply
- **WhatsApp**: Message retrieval via Business API, summaries, sending

### Phase 6A: OAuth Foundation
- [ ] OAuth2 manager with macOS Keychain token storage
- [ ] Google OAuth flow with browser-based auth
- [ ] WhatsApp Business API token configuration
- [ ] `/auth` commands (google, whatsapp, status, revoke)

### Phase 6B: Google Calendar
- [ ] GoogleCalendarClient with full CRUD operations
- [ ] Two-way sync with conflict resolution
- [ ] Event invitations and RSVP handling
- [ ] `/sync calendar`, `/gcal`, `/invite` commands

### Phase 6C: Gmail
- [ ] GmailClient for read/send/reply operations
- [ ] Email ↔ Knowledge graph contact linking
- [ ] Email storage with threading
- [ ] `/email` commands (list, read, summary, compose, send)

### Phase 6D: WhatsApp
- [ ] WhatsAppClient for Business API
- [ ] Webhook receiver for real-time messages
- [ ] Message ↔ Contact linking
- [ ] `/wa` commands (list, read, summary, send)

### Phase 6E: AI-Powered Importance Scoring
- [ ] ImportanceScorer using local LLM
- [ ] Personalized scoring based on knowledge graph
- [ ] Factors: known contact, VIP, action required, time-sensitive
- [ ] Batch summarization for daily briefings

### Phase 6F: Unified Inbox
- [ ] Combined view of all channels
- [ ] Enhanced `/briefing` with emails, messages, events
- [ ] `/inbox` command with filtering
- [ ] Natural language summaries across all sources

### New Commands Summary
| Command | Description |
|---------|-------------|
| `/auth <service>` | Authenticate with external service |
| `/sync calendar` | Sync with Google Calendar |
| `/gcal list/create` | Google Calendar operations |
| `/invite <event> <person>` | Invite contact to event |
| `/email list/read/send` | Gmail operations |
| `/email summary` | AI summary of important emails |
| `/wa list/send` | WhatsApp operations |
| `/wa summary` | AI summary of messages |
| `/inbox` | Unified important items across channels |

### Dependencies to Add
```toml
oauth2 = "4.4"
jsonwebtoken = "9"
keyring = "2"
```

---

## Knowledge Graph Schema

### Entities
| Entity | Fields |
|--------|--------|
| Person | id, name, + any custom properties |
| Organization | id, name, + any custom properties |
| Event | id, title, datetime, location, notes |
| Task | id, title, deadline, priority, status |

### Relationships
| Relationship | From | To |
|--------------|------|-----|
| WORKS_AT | Person | Organization |
| KNOWS | Person | Person |
| HAS_DEADLINE | Person | Task |
| ATTENDING | Person | Event |
| MEMBER_OF | Person | Organization |

---

## How to Run

```bash
# Make sure Ollama is running
ollama serve

# Build and run
cd dosa-rs
cargo run --release
```

---

## Progress Log

### 2025-12-01 (Phase 3)
- ✅ Completed Phase 3: Intelligence
  - Proactive alerts system on startup (overdue tasks, upcoming events)
  - Natural language queries over knowledge graph ("Who works at X?")
  - Pattern-based entity extraction from conversations
  - Daily briefing command
  - 5 new intelligence commands
  - All 21 unit tests passing

### 2025-12-01 (Phase 2)
- ✅ Completed Phase 2 implementation
  - Calendar module with events and natural language date parsing
  - Reminders/Tasks module with priorities and deadlines
  - Contacts module integrated with knowledge graph
  - 10 new commands added
  - LLM now gets context from calendar, tasks, and contacts
  - All 13 unit tests passing

### 2025-11-30
- ✅ Created project plan
- ✅ Completed Phase 1 implementation
  - Rust project initialized with all dependencies
  - Ollama client wrapper working
  - SQLite database with knowledge graph schema
  - Extensible command router with 10 built-in commands
  - LLM integration with conversation context
  - All 9 unit tests passing


