#!/usr/bin/env python3
"""
Generate training data for DOSA NLU model.

This script generates synthetic training examples for:
- Intent classification
- Slot/Entity extraction (BIO tagging)

Output format is JSONL with fields:
- text: The input text
- intent: The intent label
- slots: List of {start, end, label, text} for entities
"""

import json
import random
import os
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Slot:
    start: int
    end: int
    label: str
    text: str

@dataclass 
class Example:
    text: str
    intent: str
    slots: List[Slot]
    
    def to_dict(self):
        return {
            "text": self.text,
            "intent": self.intent,
            "slots": [asdict(s) for s in self.slots]
        }

# =============================================================================
# Name/Entity Lists for Generation
# =============================================================================

FIRST_NAMES = [
    "John", "Sarah", "Mike", "Alice", "Bob", "Emma", "David", "Lisa",
    "James", "Emily", "Michael", "Jessica", "William", "Ashley", "Daniel",
    "Amogh", "Rohan", "Priya", "Raj", "Anita", "Vikram", "Neha", "Arjun",
    "Chen", "Wei", "Yuki", "Hiroshi", "Maria", "Carlos", "Ahmed", "Fatima"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Atreya", "Mathew", "Patel", "Shah", "Kumar", "Singh", "Gupta",
    "Wang", "Li", "Zhang", "Tanaka", "Yamamoto", "Silva", "Santos"
]

ORGANIZATIONS = [
    "Google", "Microsoft", "Apple", "Amazon", "Meta", "Netflix", "Tesla",
    "Acme Corp", "TechStart Inc", "GlobalTech", "Innovate Labs",
    "Imperial College London", "Stanford University", "MIT", "Oxford",
    "Harvard Business School", "McKinsey", "Goldman Sachs", "JPMorgan"
]

PROJECTS = [
    "Project Alpha", "the website redesign", "the mobile app", "Q4 planning",
    "the marketing campaign", "the database migration", "DOSA", "the API",
    "the quarterly report", "the product launch", "the integration project"
]

LOCATIONS = [
    "conference room A", "the office", "downtown", "Central Park",
    "Starbucks on Main Street", "the meeting room", "building 5",
    "room 301", "the lobby", "my office", "their office"
]

EVENT_TYPES = [
    "meeting", "call", "standup", "sync", "1:1", "one-on-one", "interview",
    "presentation", "demo", "review", "retrospective", "planning session",
    "lunch", "coffee", "dinner", "catch-up", "brainstorm", "workshop"
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
RELATIVE_DAYS = ["today", "tomorrow", "next week", "this week", "next Monday", 
                 "this Friday", "the day after tomorrow"]
TIMES = ["9am", "10am", "11am", "noon", "1pm", "2pm", "3pm", "4pm", "5pm",
         "9:30am", "10:30am", "2:30pm", "3:30pm", "in the morning", "in the afternoon"]

# =============================================================================
# Template-based Generation
# =============================================================================

def random_name() -> Tuple[str, str]:
    """Return (full_name, first_name)"""
    first = random.choice(FIRST_NAMES)
    if random.random() > 0.3:
        return f"{first} {random.choice(LAST_NAMES)}", first
    return first, first

def random_org() -> str:
    return random.choice(ORGANIZATIONS)

def random_time() -> str:
    return random.choice(TIMES)

def random_day() -> str:
    if random.random() > 0.5:
        return random.choice(RELATIVE_DAYS)
    return random.choice(DAYS)

def random_event() -> str:
    return random.choice(EVENT_TYPES)

def random_location() -> str:
    return random.choice(LOCATIONS)

def random_project() -> str:
    return random.choice(PROJECTS)

# =============================================================================
# Intent: calendar_create
# =============================================================================

def generate_calendar_create() -> List[Example]:
    examples = []
    
    templates = [
        # Basic scheduling
        ("Schedule a {event} with {person} {day}", ["EVENT", "PER", "DATE"]),
        ("Set up a {event} with {person} at {time}", ["EVENT", "PER", "TIME"]),
        ("Book a {event} for {day} at {time}", ["EVENT", "DATE", "TIME"]),
        ("Create a {event} with {person} {day} at {time}", ["EVENT", "PER", "DATE", "TIME"]),
        ("Add {event} to my calendar for {day}", ["EVENT", "DATE"]),
        ("I need to schedule a {event} with {person}", ["EVENT", "PER"]),
        ("Can you set up a {event} with {person} {day}?", ["EVENT", "PER", "DATE"]),
        ("Let's have a {event} with {person} at {time}", ["EVENT", "PER", "TIME"]),
        ("Plan a {event} for {day} with {person}", ["EVENT", "DATE", "PER"]),
        ("Arrange a {event} at {location} {day}", ["EVENT", "LOC", "DATE"]),
        
        # With location
        ("Schedule {event} with {person} at {location} {day}", ["EVENT", "PER", "LOC", "DATE"]),
        ("Book {location} for a {event} {day} at {time}", ["LOC", "EVENT", "DATE", "TIME"]),
        
        # Informal
        ("Meeting with {person} {day}", ["PER", "DATE"]),
        ("Lunch with {person} {day} at {time}", ["PER", "DATE", "TIME"]),
        ("{event} {day} at {time}", ["EVENT", "DATE", "TIME"]),
        ("I'm having {event} with {person}", ["EVENT", "PER"]),
    ]
    
    for _ in range(200):
        template, slot_types = random.choice(templates)
        
        person_full, person_first = random_name()
        values = {
            "event": random_event(),
            "person": person_full if random.random() > 0.3 else person_first,
            "day": random_day(),
            "time": random_time(),
            "location": random_location(),
        }
        
        text = template
        slots = []
        
        for slot_type in slot_types:
            key = {"EVENT": "event", "PER": "person", "DATE": "day", 
                   "TIME": "time", "LOC": "location"}[slot_type]
            value = values[key]
            
            # Find position in text
            placeholder = "{" + key + "}"
            if placeholder in text:
                start = text.index(placeholder)
                text = text.replace(placeholder, value, 1)
                end = start + len(value)
                slots.append(Slot(start=start, end=end, label=slot_type, text=value))
        
        examples.append(Example(text=text, intent="calendar_create", slots=slots))
    
    return examples

# =============================================================================
# Intent: calendar_query
# =============================================================================

def generate_calendar_query() -> List[Example]:
    examples = []
    
    queries = [
        ("What's on my calendar today?", []),
        ("What do I have scheduled today?", []),
        ("Show me today's events", []),
        ("What's happening today?", []),
        ("What meetings do I have today?", []),
        ("Any events today?", []),
        ("What's on for today?", []),
        
        ("What's on my calendar tomorrow?", []),
        ("What do I have tomorrow?", []),
        ("Show me tomorrow's schedule", []),
        ("Anything scheduled for tomorrow?", []),
        
        ("What's my schedule this week?", []),
        ("Show me this week's calendar", []),
        ("What events do I have this week?", []),
        ("What's coming up this week?", []),
        
        ("What's my schedule for Monday?", [Slot(25, 31, "DATE", "Monday")]),
        ("Show me Friday's events", [Slot(8, 14, "DATE", "Friday")]),
        ("Any meetings on Tuesday?", [Slot(16, 23, "DATE", "Tuesday")]),
        
        ("When is my next meeting?", []),
        ("What's next on my calendar?", []),
        ("Do I have any meetings coming up?", []),
        ("What's my schedule look like?", []),
        ("Am I free tomorrow afternoon?", []),
        ("When am I meeting with John?", [Slot(21, 25, "PER", "John")]),
    ]
    
    for text, slots in queries:
        examples.append(Example(text=text, intent="calendar_query", slots=slots))
    
    # Generate variations
    for day in DAYS + RELATIVE_DAYS:
        text = f"What's on my calendar {day}?"
        start = text.index(day)
        examples.append(Example(
            text=text,
            intent="calendar_query", 
            slots=[Slot(start, start + len(day), "DATE", day)]
        ))
    
    return examples

# =============================================================================
# Intent: calendar_update
# =============================================================================

def generate_calendar_update() -> List[Example]:
    examples = []
    
    templates = [
        "Reschedule the {event} to {day}",
        "Move the {event} to {time}",
        "Change the {event} to {day} at {time}",
        "Push the {event} back to {time}",
        "Delay the {event} by an hour",
        "Move my meeting with {person} to {day}",
        "Reschedule {event} with {person} to {time}",
        "Change the time of the {event} to {time}",
        "Update the {event} location to {location}",
        "The {event} needs to move to {day}",
    ]
    
    for _ in range(80):
        template = random.choice(templates)
        person_full, _ = random_name()
        
        values = {
            "event": random_event(),
            "person": person_full,
            "day": random_day(),
            "time": random_time(),
            "location": random_location(),
        }
        
        text = template
        slots = []
        
        for key, slot_type in [("event", "EVENT"), ("person", "PER"), 
                                ("day", "DATE"), ("time", "TIME"), ("location", "LOC")]:
            placeholder = "{" + key + "}"
            if placeholder in text:
                value = values[key]
                start = text.index(placeholder)
                text = text.replace(placeholder, value, 1)
                slots.append(Slot(start, start + len(value), slot_type, value))
        
        examples.append(Example(text=text, intent="calendar_update", slots=slots))
    
    return examples

# =============================================================================
# Intent: calendar_delete
# =============================================================================

def generate_calendar_delete() -> List[Example]:
    examples = []
    
    templates = [
        "Cancel the {event} {day}",
        "Delete the {event}",
        "Remove the {event} from my calendar",
        "Cancel my meeting with {person}",
        "I need to cancel the {event}",
        "Please delete the {event} {day}",
        "Remove {day}'s {event}",
        "Cancel {event} with {person}",
    ]
    
    for _ in range(50):
        template = random.choice(templates)
        person_full, _ = random_name()
        
        values = {
            "event": random_event(),
            "person": person_full,
            "day": random_day(),
        }
        
        text = template
        slots = []
        
        for key, slot_type in [("event", "EVENT"), ("person", "PER"), ("day", "DATE")]:
            placeholder = "{" + key + "}"
            if placeholder in text:
                value = values[key]
                start = text.index(placeholder)
                text = text.replace(placeholder, value, 1)
                slots.append(Slot(start, start + len(value), slot_type, value))
        
        examples.append(Example(text=text, intent="calendar_delete", slots=slots))
    
    return examples

# =============================================================================
# Intent: email_compose
# =============================================================================

def generate_email_compose() -> List[Example]:
    examples = []
    
    templates = [
        "Send an email to {person} about {project}",
        "Email {person} about the {event}",
        "Write an email to {person}",
        "Draft an email to {person} regarding {project}",
        "Compose an email to {person}",
        "Send {person} an email about {project}",
        "I need to email {person}",
        "Can you help me write an email to {person}?",
        "Message {person} about {project}",
    ]
    
    for _ in range(80):
        template = random.choice(templates)
        person_full, _ = random_name()
        
        values = {
            "person": person_full,
            "project": random_project(),
            "event": random_event(),
        }
        
        text = template
        slots = []
        
        for key, slot_type in [("person", "PER"), ("project", "EVENT"), ("event", "EVENT")]:
            placeholder = "{" + key + "}"
            if placeholder in text:
                value = values[key]
                start = text.index(placeholder)
                text = text.replace(placeholder, value, 1)
                slots.append(Slot(start, start + len(value), slot_type, value))
        
        examples.append(Example(text=text, intent="email_compose", slots=slots))
    
    return examples

# =============================================================================
# Intent: email_reply
# =============================================================================

def generate_email_reply() -> List[Example]:
    examples = []
    
    templates = [
        "Reply to {person}'s email",
        "Respond to {person}'s message",
        "Reply to the email from {person}",
        "Get back to {person}'s email",
        "Answer {person}'s email",
        "Reply to {person} saying I'll be there",
        "Respond to the email about {project}",
        "Reply to that email from {person}",
    ]
    
    for _ in range(60):
        template = random.choice(templates)
        person_full, person_first = random_name()
        
        values = {
            "person": person_first if random.random() > 0.5 else person_full,
            "project": random_project(),
        }
        
        text = template
        slots = []
        
        for key, slot_type in [("person", "PER"), ("project", "EVENT")]:
            placeholder = "{" + key + "}"
            if placeholder in text:
                value = values[key]
                start = text.index(placeholder)
                text = text.replace(placeholder, value, 1)
                slots.append(Slot(start, start + len(value), slot_type, value))
        
        examples.append(Example(text=text, intent="email_reply", slots=slots))
    
    return examples

# =============================================================================
# Intent: email_attendees
# =============================================================================

def generate_email_attendees() -> List[Example]:
    examples = []
    
    templates = [
        "Email the attendees of the {event}",
        "Let the attendees of {day}'s {event} know it's postponed",
        "Notify the {event} attendees about the change",
        "Send an email to everyone in the {event}",
        "Message the attendees about the delay",
        "Let them know the {event} is cancelled",
        "Email everyone about the {event} change",
        "Tell the attendees the {event} is rescheduled",
    ]
    
    for _ in range(50):
        template = random.choice(templates)
        
        values = {
            "event": random_event(),
            "day": random_day(),
        }
        
        text = template
        slots = []
        
        for key, slot_type in [("event", "EVENT"), ("day", "DATE")]:
            placeholder = "{" + key + "}"
            if placeholder in text:
                value = values[key]
                start = text.index(placeholder)
                text = text.replace(placeholder, value, 1)
                slots.append(Slot(start, start + len(value), slot_type, value))
        
        examples.append(Example(text=text, intent="email_attendees", slots=slots))
    
    return examples

# =============================================================================
# Intent: email_query
# =============================================================================

def generate_email_query() -> List[Example]:
    examples = []
    
    queries = [
        ("Show me my unread emails", []),
        ("Check my inbox", []),
        ("Any new emails?", []),
        ("What emails do I have?", []),
        ("Show my unread messages", []),
        ("Do I have any unread emails?", []),
        ("Check for new emails", []),
        ("What's in my inbox?", []),
        ("Show me my emails", []),
        ("List my unread emails", []),
        ("Summarize my emails", []),
        ("Give me an email summary", []),
    ]
    
    for text, slots in queries:
        examples.append(Example(text=text, intent="email_query", slots=slots))
    
    # With person
    for _ in range(30):
        person_full, person_first = random_name()
        person = person_first if random.random() > 0.5 else person_full
        
        templates = [
            f"Show emails from {person}",
            f"Any emails from {person}?",
            f"Check for emails from {person}",
            f"Show me {person}'s emails",
        ]
        
        text = random.choice(templates)
        start = text.lower().find(person.lower())
        if start >= 0:
            examples.append(Example(
                text=text,
                intent="email_query",
                slots=[Slot(start, start + len(person), "PER", person)]
            ))
    
    return examples

# =============================================================================
# Intent: contact_update
# =============================================================================

def generate_contact_update() -> List[Example]:
    examples = []
    
    # Add person patterns
    for _ in range(80):
        person_full, person_first = random_name()
        person = person_full if random.random() > 0.3 else person_first
        org = random_org()
        
        templates = [
            (f"Add {person} as a person I know", [("PER", person)]),
            (f"Add {person} as a contact", [("PER", person)]),
            (f"Add {person}, he works at {org}", [("PER", person), ("ORG", org)]),
            (f"Add {person}, she works at {org}", [("PER", person), ("ORG", org)]),
            (f"Add {person}, they work at {org}", [("PER", person), ("ORG", org)]),
            (f"Add {person}, he studies at {org}", [("PER", person), ("ORG", org)]),
            (f"Add {person}, she studies at {org}", [("PER", person), ("ORG", org)]),
            (f"{person} works at {org}", [("PER", person), ("ORG", org)]),
            (f"{person} is a colleague at {org}", [("PER", person), ("ORG", org)]),
            (f"Remember that {person} works at {org}", [("PER", person), ("ORG", org)]),
        ]
        
        text, slot_info = random.choice(templates)
        slots = []
        for label, value in slot_info:
            start = text.find(value)
            if start >= 0:
                slots.append(Slot(start, start + len(value), label, value))
        
        examples.append(Example(text=text, intent="contact_update", slots=slots))
    
    # Email setting patterns
    for _ in range(60):
        person_full, person_first = random_name()
        person = person_first
        email = f"{person.lower()}@example.com"
        
        templates = [
            f"{person}'s email is {email}",
            f"{person}'s email address is {email}",
            f"Set {person}'s email to {email}",
            f"Remember that {person}'s email is {email}",
            f"The email for {person} is {email}",
        ]
        
        text = random.choice(templates)
        slots = []
        
        person_start = text.find(person)
        if person_start >= 0:
            slots.append(Slot(person_start, person_start + len(person), "PER", person))
        
        email_start = text.find(email)
        if email_start >= 0:
            slots.append(Slot(email_start, email_start + len(email), "EMAIL", email))
        
        examples.append(Example(text=text, intent="contact_update", slots=slots))
    
    return examples

# =============================================================================
# Intent: knowledge_query
# =============================================================================

def generate_knowledge_query() -> List[Example]:
    examples = []
    
    # Who is queries
    for _ in range(40):
        person_full, _ = random_name()
        
        templates = [
            f"Who is {person_full}?",
            f"Tell me about {person_full}",
            f"What do you know about {person_full}?",
            f"Who is {person_full} again?",
        ]
        
        text = random.choice(templates)
        start = text.find(person_full)
        examples.append(Example(
            text=text,
            intent="knowledge_query",
            slots=[Slot(start, start + len(person_full), "PER", person_full)]
        ))
    
    # Who works at queries
    for _ in range(30):
        org = random_org()
        
        templates = [
            f"Who works at {org}?",
            f"Who do I know at {org}?",
            f"Show me people at {org}",
            f"List contacts at {org}",
        ]
        
        text = random.choice(templates)
        start = text.find(org)
        examples.append(Example(
            text=text,
            intent="knowledge_query",
            slots=[Slot(start, start + len(org), "ORG", org)]
        ))
    
    # General queries
    general = [
        "Who do I know?",
        "Show me my contacts",
        "List all people",
        "Who are my contacts?",
        "Show everyone in my network",
    ]
    
    for text in general:
        examples.append(Example(text=text, intent="knowledge_query", slots=[]))
    
    return examples

# =============================================================================
# Intent: conversation
# =============================================================================

def generate_conversation() -> List[Example]:
    examples = []
    
    conversations = [
        "Hello",
        "Hi there",
        "Hey",
        "Good morning",
        "Good afternoon", 
        "How are you?",
        "What's up?",
        "Thanks",
        "Thank you",
        "That's helpful",
        "Great, thanks!",
        "Okay",
        "Sure",
        "Got it",
        "Makes sense",
        "I see",
        "Interesting",
        "What can you do?",
        "Help me",
        "I need help",
        "What's the weather like?",
        "Tell me a joke",
        "That's funny",
        "Never mind",
        "Forget it",
        "Actually, no",
        "Wait",
        "Hold on",
        "Let me think",
        "Hmm",
        "I'm not sure",
        "Maybe later",
        "Goodbye",
        "Bye",
        "See you",
        "Talk later",
    ]
    
    for text in conversations:
        examples.append(Example(text=text, intent="conversation", slots=[]))
    
    return examples

# =============================================================================
# Intent: command
# =============================================================================

def generate_command() -> List[Example]:
    examples = []
    
    commands = [
        "/help",
        "/add person John",
        "/add organization Acme",
        "/set John email john@example.com",
        "/link John works_at Acme",
        "/gcal list",
        "/gcal today",
        "/email list",
        "/email unread",
        "/auth google",
        "/auth status",
        "/briefing",
        "/sync calendar",
        "/exit",
        "/quit",
    ]
    
    for text in commands:
        examples.append(Example(text=text, intent="command", slots=[]))
    
    return examples

# =============================================================================
# Main
# =============================================================================

def main():
    random.seed(42)
    
    # Generate all examples
    all_examples = []
    
    generators = [
        ("calendar_create", generate_calendar_create),
        ("calendar_query", generate_calendar_query),
        ("calendar_update", generate_calendar_update),
        ("calendar_delete", generate_calendar_delete),
        ("email_compose", generate_email_compose),
        ("email_reply", generate_email_reply),
        ("email_attendees", generate_email_attendees),
        ("email_query", generate_email_query),
        ("contact_update", generate_contact_update),
        ("knowledge_query", generate_knowledge_query),
        ("conversation", generate_conversation),
        ("command", generate_command),
    ]
    
    for name, generator in generators:
        examples = generator()
        print(f"Generated {len(examples)} examples for {name}")
        all_examples.extend(examples)
    
    # Shuffle
    random.shuffle(all_examples)
    
    # Split into train/val/test (80/10/10)
    n = len(all_examples)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    train = all_examples[:train_end]
    val = all_examples[train_end:val_end]
    test = all_examples[val_end:]
    
    print(f"\nTotal: {n} examples")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Create data directory
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Write files
    for name, examples in [("train", train), ("val", val), ("test", test)]:
        path = data_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict()) + "\n")
        print(f"Wrote {path}")
    
    # Write label files
    intents = sorted(set(ex.intent for ex in all_examples))
    with open(data_dir / "intents.txt", "w") as f:
        for intent in intents:
            f.write(intent + "\n")
    
    slot_labels = ["O"]  # Outside
    for label in ["PER", "ORG", "DATE", "TIME", "LOC", "EMAIL", "EVENT", "REL"]:
        slot_labels.extend([f"B-{label}", f"I-{label}"])
    
    with open(data_dir / "slot_labels.txt", "w") as f:
        for label in slot_labels:
            f.write(label + "\n")
    
    print(f"\nIntents: {intents}")
    print(f"Slot labels: {slot_labels}")

if __name__ == "__main__":
    main()
