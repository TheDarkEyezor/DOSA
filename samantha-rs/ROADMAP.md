# DOSA Roadmap & Feature Extensions

## Current Architecture

### Intent Detection Flow
```
User Input
    ↓
Command Router (/help, /gcal, /email, etc.)
    ↓ (if not handled)
Knowledge Graph Query Detection
    ↓ (if not handled)
Email Intent Detection (compose, attendees, query)
    ↓ (if not handled)
Calendar Intent Detection (create, update, delete, query)
    ↓ (if not handled)
LLM Fallback (general conversation)
```

### Key Components
- **OllamaClient**: LLM interface with JSON mode support
- **KnowledgeGraph**: Entity/relationship storage (SQLite)
- **OAuthManager**: Google OAuth2 with Keychain storage
- **IntentDetection**: Keyword-based + LLM classification

---

## Optimizations

### 1. Intent Disambiguation (Priority: High)

**Current Issues:**
- Keyword matching can conflict (e.g., "change" triggers update intent even when emailing about changes)
- Order-dependent: first match wins
- No confidence scoring

**Proposed Solution: Unified Intent Classifier**

```rust
pub enum UnifiedIntent {
    Command(String),           // /help, /gcal, etc.
    CalendarCreate(String),    // "schedule lunch with John"
    CalendarQuery(DateRange),  // "what's on tomorrow"
    CalendarUpdate(String),    // "move the meeting to 3pm"
    CalendarDelete(String),    // "cancel the dinner"
    EmailCompose(String),      // "email John about the project"
    EmailAttendees(String),    // "email dinner attendees"
    EmailQuery(EmailFilter),   // "show unread emails"
    KnowledgeQuery(String),    // "who works at Google"
    Conversation(String),      // General chat
}

pub struct IntentResult {
    pub intent: UnifiedIntent,
    pub confidence: f32,       // 0.0 - 1.0
    pub alternatives: Vec<(UnifiedIntent, f32)>,
}
```

**Implementation Options:**

| Approach | Speed | Accuracy | Complexity |
|----------|-------|----------|------------|
| **Keyword-first, LLM-fallback** | Fast | Medium | Low |
| **LLM-only with JSON mode** | Slow | High | Low |
| **Hybrid: Keywords + LLM confirm** | Medium | High | Medium |
| **Fine-tuned classifier** | Fast | Highest | High |

**Recommended: Hybrid Approach**
1. Fast keyword scan to get candidate intents
2. If ambiguous (multiple candidates or low confidence), use LLM to disambiguate
3. Cache common patterns for speed

### 2. Code Deduplication (Priority: Medium)

**Current State:** main.rs has ~400 lines of repetitive intent handling

**Proposed Refactor:**

```rust
// src/handlers/mod.rs
pub trait IntentHandler {
    fn can_handle(&self, intent: &UnifiedIntent) -> bool;
    async fn handle(&self, input: &str, ctx: &HandlerContext) -> Result<HandlerResult>;
}

// src/handlers/email.rs
pub struct EmailHandler;
impl IntentHandler for EmailHandler { ... }

// src/handlers/calendar.rs  
pub struct CalendarHandler;
impl IntentHandler for CalendarHandler { ... }
```

### 3. Async Optimization (Priority: Low)

**Current:** `tokio::task::block_in_place` everywhere

**Better:** True async with proper error handling
- Use `tokio::select!` for concurrent operations
- Add timeouts for API calls
- Implement retry logic with exponential backoff

---

## Feature Extensions

### Phase 1: Core Improvements (Next Sprint)

#### 1.1 Smart Reply
```
you> reply to John's last email saying I'll be there at 3pm
```
- Find recent email from John
- Generate contextual reply
- Preview and confirm

#### 1.2 Email Threading
```
you> summarize my conversation with Sarah about the project
```
- Group emails by thread
- Extract key points
- Show timeline

#### 1.3 Recurring Events
```
you> schedule weekly standup every Monday at 10am
```
- Support RRULE generation
- Handle exceptions
- "Skip next week's standup"

#### 1.4 Natural Date Parsing
```
you> schedule something for end of next month
you> what do I have the week after Christmas
```
- Relative dates: "in 2 weeks", "next quarter"
- Holidays awareness
- Fiscal periods for work context

### Phase 2: Advanced Features

#### 2.1 Multi-hop Reasoning
```
you> email everyone I'm meeting with this week about the venue change
```
Steps:
1. Query calendar for this week's events
2. Extract unique attendees
3. Compose batch email
4. Preview and confirm

#### 2.2 Context Memory
```
you> schedule lunch with John tomorrow
you> make it at noon
you> add Sarah too
```
- Track conversation context
- Resolve pronouns ("it", "that", "them")
- Support incremental modifications

#### 2.3 Conflict Detection
```
you> schedule call with Bob at 3pm tomorrow
dosa> ⚠️ You have "Team Standup" at 3pm. Suggest 4pm instead?
```
- Check for overlaps before creating
- Suggest alternatives
- Consider travel time between locations

#### 2.4 Smart Suggestions
```
dosa> 💡 You have 2 hours free tomorrow afternoon. 
      Schedule the follow-up call with Sarah?
```
- Proactive reminders
- Task suggestions based on context
- Meeting prep summaries

### Phase 3: Integrations

#### 3.1 Slack Integration
```
you> message #engineering about the deploy delay
you> dm Bob asking about the PR review
```

#### 3.2 Notion/Tasks Integration
```
you> create a task to follow up on John's email
you> what tasks are due this week
```

#### 3.3 Meeting Links
```
you> add a zoom link to tomorrow's meeting
you> create a google meet for the interview
```

#### 3.4 Document Context
```
you> what did we discuss in last week's meeting notes
you> summarize the Q4 planning doc
```

**Implementation (RAG Pipeline):**
```
Document Ingestion:
  PDF/Markdown/Text → Semantic Chunking → Embeddings → SQLite Vector Store

Query Pipeline:
  User Query → Embedding → Hybrid Search (Semantic + Keyword) → Rerank → LLM
```

**Best Practices:**
| Component | Approach | Rationale |
|-----------|----------|-----------|
| Chunking | 1000 chars, 200 overlap | Balances context vs retrieval precision |
| Embeddings | Ollama `nomic-embed-text` | Local, fast, good quality |
| Vector Store | SQLite + cosine similarity | No external dependencies |
| Retrieval | Hybrid (semantic + BM25) | Handles both meaning and keywords |
| Grounding | Source citations | Builds trust, enables verification |

**Rust Implementation:**
- `src/intelligence/rag.rs` - RAG pipeline
- `src/intelligence/embeddings.rs` - Ollama embeddings client
- `src/intelligence/chunking.rs` - Semantic text chunker
- Document store in existing SQLite database

### Phase 4: Advanced AI

#### 4.1 Voice Interface
- Whisper for speech-to-text
- Edge TTS for responses
- Wake word detection

**Implementation:**
```
Microphone → cpal (audio capture) → whisper-rs (STT) → Intent Handler → TTS
```

**Model Options:**
| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| tiny.en | 39M | ~10x | Quick commands |
| base.en | 74M | ~7x | Balanced (recommended) |
| small.en | 244M | ~4x | High accuracy |

**Rust Libraries:**
- `whisper-rs` - Rust bindings for whisper.cpp
- `cpal` - Cross-platform audio I/O
- System TTS (macOS `say`, Windows SAPI)

#### 4.2 Proactive Assistant
- "You have a meeting in 30 minutes - here's the context"
- "John hasn't replied in 3 days - want to follow up?"
- "Your calendar is empty Friday - good day for deep work"

**Implementation (Event-Driven Triggers):**
```
Background Service:
  Calendar Monitor ──┐
  Email Monitor ─────┼──→ Trigger Engine ──→ Proactive Alerts
  Task Monitor ──────┘
```

**Trigger Types:**
1. Time-based: Meeting reminders with context
2. Event-based: New email from important contact
3. Pattern-based: "You usually email Sarah on Mondays"
4. Gap-based: "No response from Bob in 3 days"

#### 4.3 Learning & Personalization
- Learn user preferences (meeting duration, times, contacts)
- Adapt language style
- Priority inbox based on user behavior

**Implementation:**
- Store patterns in SQLite `user_preferences` table
- Track: meeting durations, preferred times, frequent contacts
- Use for suggestions: "You usually schedule 30-min meetings with Bob"

---

## Technical Debt

### Current Issues
1. [ ] `block_in_place` throughout main.rs - should be async
2. [ ] Duplicate email handling code in main.rs
3. [ ] No request caching (same calendar fetched multiple times)
4. [ ] Error messages leak internal details
5. [ ] No rate limiting for API calls
6. [ ] Tests are minimal

### Proposed Fixes
1. [ ] Create `HandlerContext` with cached OAuth/clients
2. [ ] Implement `IntentHandler` trait pattern
3. [ ] Add Redis/in-memory cache with TTL
4. [ ] User-friendly error messages with debug logging
5. [ ] Token bucket rate limiter
6. [ ] Integration tests with mock APIs

---

## Performance Targets

| Operation | Current | Target |
|-----------|---------|--------|
| Intent detection | ~500ms (LLM) | <100ms (keywords) |
| Calendar query | ~800ms | <500ms (cached) |
| Email compose | ~2s | <1.5s |
| Event creation | ~1.5s | <1s |

---

## Known Issues & Research

### PDF/Document Ingestion Bug (Priority: High)

**Problem:** When ingesting a PDF (e.g., "Amogh's resume"), all extracted entities (skills, universities, organizations) are incorrectly linked to the **logged-in user** instead of the **document subject**.

**Root Cause:** In `src/intelligence/document.rs`:
- Line 181: `extract_patterns_static()` calls `get_user_identity_name()` 
- Lines 225-250: All extracted relationships use `user.clone()` as the subject
- This assumes the document is about the user, not someone else

**Fix Required:**
1. Extract document subject (person name) from content:
   - For resumes: Look for name at top/header
   - For emails: Extract sender/subject
   - For general docs: Ask user or extract mentioned person
   
2. Modify `DocumentRelationship` creation:
   ```rust
   // Instead of:
   subject: user_name.clone()
   // Use:
   subject: document_subject.unwrap_or(user_name.clone())
   ```

3. Add CLI option to specify subject:
   ```
   /ingest /path/to/resume.pdf --for "Amogh Atreya"
   ```

---

### NLU Dataset Research (Priority: Medium)

**Current State:** Using custom training data with 4,765 examples across 13 intents. Need to expand for better intent disambiguation.

**Evaluated Datasets:**

| Dataset | Intents | Examples | License | Useful For DOSA? |
|---------|---------|----------|---------|------------------|
| **SNIPS** (sonos/nlu-benchmark) | 7 | ~2000/intent | CC0 | ❌ No - weather, music, restaurants only |
| **CLINC150** (clinc/oos-eval) | 150 | ~100/intent | MIT | ⚠️ Partial - has calendar, reminder, but no KG vs web search |

**CLINC150 Relevant Intents:**
- ✅ `calendar` - "what appointments do I have tomorrow"
- ✅ `calendar_update` - "remove the wedding from my calendar"
- ✅ `meeting_schedule` - "do I have any meetings today"
- ✅ `reminder` - "what reminders do I have"
- ✅ `reminder_update` - "remind me to take out the trash"
- ✅ `todo_list` - "what do I have to do today"
- ✅ `weather` - "what's the weather today"
- ✅ `balance` - "what's my bank balance" (could map to knowledge_query)
- ❌ No explicit `knowledge_query` vs `web_search` distinction
- ❌ No email-specific intents beyond `todo_list`

**Gap Analysis - DOSA-Specific Intents Not Covered:**
1. `knowledge_query` - "who works at Google", "what skills does John have"
2. `web_search` - "search for latest AI news", "look up restaurants nearby"
3. `email_compose` - "email John about the meeting"
4. `email_query` - "show unread emails from Sarah"
5. `email_reply` - "reply to John's last email"
6. `graphrag_query` - "what documents mention the Q4 project"

**Recommended Approach:**
1. Use CLINC150 for calendar/reminder intents (extract ~2000 examples)
2. Generate synthetic data for DOSA-specific intents:
   - Use LLM to generate variations
   - Template-based expansion with entity substitution
   - Crowdsource from beta users

**Data Generation Plan:**
```python
# Templates for knowledge_query
templates = [
    "who works at {company}",
    "what skills does {person} have",
    "tell me about {person}",
    "what do you know about {entity}",
    "where did {person} go to school",
    ...
]
# Expand with entity variations
```

---

### Questions to Resolve

1. **PDF Subject Extraction:** What heuristics work best for extracting the subject of a document?
   - Resumes: Name usually in first few lines, larger font
   - Emails: Sender info
   - Meeting notes: Title/header
   
2. **Intent Disambiguation:** When should we ask for clarification vs. make a best guess?
   - High confidence (>0.9): Execute directly
   - Medium confidence (0.6-0.9): Execute with confirmation
   - Low confidence (<0.6): Ask for clarification

3. **Knowledge vs Web Search:** How do we decide?
   - If entity exists in KG → knowledge_query
   - If query mentions "search", "look up", "find online" → web_search
   - If entity not in KG but could be → hybrid (check KG, fallback to web)

4. **Conversation Continuity:** How to maintain context across turns?
   - Store last N turns in session
   - Extract entities and intents from history
   - Resolve pronouns using context

---

## Next Steps

1. **Immediate:** Fix PDF extraction bug - extract document subject before entity linking
2. **This Week:** Implement hybrid intent detection
3. **This Month:** 
   - Refactor main.rs with handler pattern
   - Generate synthetic training data for knowledge_query vs web_search
4. **Q1 2026:** Phase 1 features (reply, recurring, natural dates)
