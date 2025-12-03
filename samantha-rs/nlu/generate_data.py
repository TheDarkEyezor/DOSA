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
    
    # Generate variations with days
    for day in DAYS + RELATIVE_DAYS:
        text = f"What's on my calendar {day}?"
        start = text.index(day)
        examples.append(Example(
            text=text,
            intent="calendar_query", 
            slots=[Slot(start, start + len(day), "DATE", day)]
        ))
    
    # Person-based calendar queries
    person_templates = [
        "meetings with {name}",
        "show meetings with {name}",
        "what meetings do I have with {name}",
        "do I have any meetings with {name}",
        "when am I meeting {name}",
        "when is my next meeting with {name}",
        "any events with {name}",
        "what's scheduled with {name}",
        "am I meeting with {name} this week",
        "show calendar events with {name}",
        "meetings involving {name}",
        "calls with {name}",
        "when do I see {name}",
        "my meetings with {name}",
    ]
    
    for template in person_templates:
        for _ in range(3):  # Generate 3 variations per template
            full_name, first_name = random_name()
            name = random.choice([full_name, first_name])
            text = template.format(name=name)
            start = text.index(name)
            examples.append(Example(
                text=text,
                intent="calendar_query",
                slots=[Slot(start, start + len(name), "PER", name)]
            ))
    
    # Topic/project-based calendar queries
    topic_templates = [
        "meetings about {topic}",
        "events related to {topic}",
        "what meetings are about {topic}",
        "calendar events for {topic}",
        "show me meetings regarding {topic}",
        "any events about {topic}",
        "meetings concerning {topic}",
        "what's scheduled for {topic}",
        "do I have meetings about {topic}",
        "calendar for {topic}",
    ]
    
    topics = PROJECTS + ["AI", "machine learning", "budgets", "hiring", "sales", 
                         "product", "engineering", "design", "marketing", "finance",
                         "strategy", "roadmap", "planning", "review", "demo"]
    
    for template in topic_templates:
        for topic in random.sample(topics, min(5, len(topics))):
            text = template.format(topic=topic)
            examples.append(Example(
                text=text,
                intent="calendar_query",
                slots=[]  # Topics aren't entity-tagged for now
            ))
    
    # Organization-based calendar queries  
    org_templates = [
        "meetings with {org}",
        "do I have any calls with {org}",
        "when am I meeting someone from {org}",
        "show meetings with people from {org}",
        "events with {org}",
        "calendar with {org}",
    ]
    
    for template in org_templates:
        for org in random.sample(ORGANIZATIONS, min(5, len(ORGANIZATIONS))):
            text = template.format(org=org)
            start = text.index(org)
            examples.append(Example(
                text=text,
                intent="calendar_query",
                slots=[Slot(start, start + len(org), "ORG", org)]
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
    
    # Topic-based email queries
    topic_templates = [
        "emails about {topic}",
        "what emails did I get about {topic}",
        "show me emails related to {topic}",
        "any emails regarding {topic}",
        "emails concerning {topic}",
        "show emails about {topic}",
        "find emails related to {topic}",
        "what emails are about {topic}",
        "do I have emails about {topic}",
        "messages about {topic}",
        "emails mentioning {topic}",
        "search emails for {topic}",
        "emails on the topic of {topic}",
        "show me {topic} related emails",
        "filter emails by {topic}",
    ]
    
    email_topics = [
        "startups", "AI", "machine learning", "work", "project", "budget",
        "sales", "hiring", "interviews", "meetings", "deadlines", "reports",
        "marketing", "design", "engineering", "finance", "strategy",
        "the product launch", "quarterly review", "performance", "feedback",
        "travel", "expenses", "invoices", "contracts", "partnership",
        "investment", "funding", "clients", "customers", "support",
        "shipping", "delivery", "orders", "subscriptions", "renewals",
    ]
    
    for template in topic_templates:
        for topic in random.sample(email_topics, min(6, len(email_topics))):
            text = template.format(topic=topic)
            examples.append(Example(
                text=text,
                intent="email_query",
                slots=[]
            ))
    
    # Organization-based email queries
    org_email_templates = [
        "emails from {org}",
        "show emails from {org}",
        "any emails from {org}",
        "messages from {org}",
        "emails involving {org}",
        "show me {org} emails",
        "emails about {org}",
    ]
    
    for template in org_email_templates:
        for org in random.sample(ORGANIZATIONS, min(5, len(ORGANIZATIONS))):
            text = template.format(org=org)
            start = text.find(org)
            if start >= 0:
                examples.append(Example(
                    text=text,
                    intent="email_query",
                    slots=[Slot(start, start + len(org), "ORG", org)]
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

# Skills and attributes for KG queries
SKILLS = [
    "Python", "Rust", "JavaScript", "TypeScript", "React", "machine learning",
    "data science", "project management", "design", "marketing", "sales",
    "finance", "accounting", "legal", "HR", "recruiting", "engineering",
    "product management", "UX", "UI design", "DevOps", "cloud computing",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "SQL", "databases"
]

UNIVERSITIES = [
    "Stanford", "MIT", "Harvard", "Oxford", "Cambridge", "Imperial College",
    "Berkeley", "Yale", "Princeton", "Columbia", "UCLA", "NYU", "Cornell"
]

INDUSTRIES = [
    "tech", "finance", "healthcare", "education", "retail", "manufacturing",
    "consulting", "legal", "media", "entertainment", "real estate", "energy"
]

RELATIONSHIPS = [
    "manager", "colleague", "teammate", "mentor", "friend", "classmate",
    "coworker", "boss", "direct report", "collaborator"
]

def generate_knowledge_query() -> List[Example]:
    examples = []
    
    # =========================================================================
    # Person queries - "Who is X?" style (200 examples)
    # =========================================================================
    person_templates = [
        "Who is {person}?",
        "Tell me about {person}",
        "What do you know about {person}?",
        "Who is {person} again?",
        "Give me info on {person}",
        "What's {person}'s background?",
        "Show me {person}'s profile",
        "Look up {person}",
        "Find {person}",
        "What can you tell me about {person}?",
        "I need information about {person}",
        "Do you have info on {person}?",
        "Pull up {person}",
        "{person} - who are they?",
        "Remind me who {person} is",
        "What's the deal with {person}?",
        "Give me details on {person}",
        "Who exactly is {person}?",
    ]
    
    for _ in range(200):
        person_full, _ = random_name()
        template = random.choice(person_templates)
        text = template.format(person=person_full)
        start = text.find(person_full)
        if start >= 0:
            examples.append(Example(
                text=text,
                intent="knowledge_query",
                slots=[Slot(start, start + len(person_full), "PER", person_full)]
            ))
        else:
            examples.append(Example(text=text, intent="knowledge_query", slots=[]))
    
    # =========================================================================
    # Organization queries - "Who works at X?" style (150 examples)
    # =========================================================================
    org_templates = [
        "Who works at {org}?",
        "Who do I know at {org}?",
        "Show me people at {org}",
        "List contacts at {org}",
        "Anyone at {org}?",
        "Do I know anyone at {org}?",
        "Who's at {org}?",
        "My contacts at {org}",
        "People from {org}",
        "Show {org} connections",
        "Find people at {org}",
        "Who do I have at {org}?",
        "Connections at {org}",
        "Anyone working at {org}?",
        "List {org} contacts",
    ]
    
    for _ in range(150):
        org = random_org()
        template = random.choice(org_templates)
        text = template.format(org=org)
        start = text.find(org)
        if start >= 0:
            examples.append(Example(
                text=text,
                intent="knowledge_query",
                slots=[Slot(start, start + len(org), "ORG", org)]
            ))
        else:
            examples.append(Example(text=text, intent="knowledge_query", slots=[]))
    
    # =========================================================================
    # Skill-based queries (100 examples)
    # =========================================================================
    skill_templates = [
        "Who knows {skill}?",
        "Find people who know {skill}",
        "Who has experience with {skill}?",
        "Anyone with {skill} skills?",
        "Show me {skill} experts",
        "Who can help with {skill}?",
        "People skilled in {skill}",
        "List contacts who know {skill}",
        "Who's good at {skill}?",
        "Find someone who knows {skill}",
        "Anyone familiar with {skill}?",
        "Who has {skill} experience?",
    ]
    
    for _ in range(100):
        skill = random.choice(SKILLS)
        template = random.choice(skill_templates)
        text = template.format(skill=skill)
        examples.append(Example(text=text, intent="knowledge_query", slots=[]))
    
    # =========================================================================
    # Relationship queries (80 examples)
    # =========================================================================
    relationship_templates = [
        "Who is {person}'s {relationship}?",
        "Who does {person} work with?",
        "Show me {person}'s team",
        "Who manages {person}?",
        "Who reports to {person}?",
        "List {person}'s colleagues",
        "{person}'s connections",
        "Who knows {person}?",
        "How do I know {person}?",
        "What's my connection to {person}?",
        "Show relationships for {person}",
    ]
    
    for _ in range(80):
        person_full, _ = random_name()
        relationship = random.choice(RELATIONSHIPS)
        template = random.choice(relationship_templates)
        text = template.format(person=person_full, relationship=relationship)
        start = text.find(person_full)
        if start >= 0:
            examples.append(Example(
                text=text,
                intent="knowledge_query",
                slots=[Slot(start, start + len(person_full), "PER", person_full)]
            ))
        else:
            examples.append(Example(text=text, intent="knowledge_query", slots=[]))
    
    # =========================================================================
    # Education-based queries (60 examples)
    # =========================================================================
    edu_templates = [
        "Who went to {university}?",
        "Anyone from {university}?",
        "Show contacts from {university}",
        "Find people who studied at {university}",
        "Who studied at {university}?",
        "{university} alumni in my network",
        "Contacts from {university}",
        "Who do I know from {university}?",
    ]
    
    for _ in range(60):
        uni = random.choice(UNIVERSITIES)
        template = random.choice(edu_templates)
        text = template.format(university=uni)
        examples.append(Example(text=text, intent="knowledge_query", slots=[]))
    
    # =========================================================================
    # Industry queries (50 examples)
    # =========================================================================
    industry_templates = [
        "Who works in {industry}?",
        "Anyone in {industry}?",
        "Contacts in {industry}",
        "People working in {industry}",
        "Show me {industry} contacts",
        "Find people in {industry}",
        "Who do I know in {industry}?",
    ]
    
    for _ in range(50):
        industry = random.choice(INDUSTRIES)
        template = random.choice(industry_templates)
        text = template.format(industry=industry)
        examples.append(Example(text=text, intent="knowledge_query", slots=[]))
    
    # =========================================================================
    # General contact queries (100 examples)
    # =========================================================================
    general_queries = [
        "Who do I know?",
        "Show me my contacts",
        "List all people",
        "Who are my contacts?",
        "Show everyone in my network",
        "My network",
        "List contacts",
        "Show contacts",
        "All my people",
        "Everyone I know",
        "Display my contacts",
        "Show all contacts",
        "Who's in my network?",
        "List everyone",
        "My connections",
        "Show my connections",
        "People I know",
        "Display contacts",
        "Contact list",
        "All contacts",
        "List my network",
        "Network contacts",
        "Show network",
        "My people",
        "Who's saved?",
        "Saved contacts",
        "Known people",
        "All known contacts",
        "Everyone saved",
        "My address book",
    ]
    
    for text in general_queries:
        for _ in range(3):  # Add each 3 times with slight variations
            examples.append(Example(text=text, intent="knowledge_query", slots=[]))
    
    # =========================================================================
    # Property queries - "What is X's Y?" (80 examples)
    # =========================================================================
    property_templates = [
        "What is {person}'s email?",
        "What's {person}'s phone number?",
        "{person}'s email address",
        "Get {person}'s contact info",
        "Where does {person} work?",
        "What company is {person} at?",
        "{person}'s company",
        "What does {person} do?",
        "{person}'s job title",
        "What's {person}'s role?",
        "{person}'s position",
        "Where is {person} located?",
        "{person}'s location",
        "What team is {person} on?",
        "What skills does {person} have?",
        "{person}'s skills",
        "What's {person}'s background?",
    ]
    
    for _ in range(80):
        person_full, _ = random_name()
        template = random.choice(property_templates)
        text = template.format(person=person_full)
        start = text.find(person_full)
        if start >= 0:
            examples.append(Example(
                text=text,
                intent="knowledge_query",
                slots=[Slot(start, start + len(person_full), "PER", person_full)]
            ))
        else:
            examples.append(Example(text=text, intent="knowledge_query", slots=[]))
    
    # =========================================================================
    # Recent/context queries (40 examples)
    # =========================================================================
    context_queries = [
        "Who did I add recently?",
        "Recent contacts",
        "New people added",
        "Latest additions to contacts",
        "Recently added people",
        "Who's new in my network?",
        "New contacts",
        "Show recent additions",
        "People I recently met",
        "Contacts added this week",
        "New connections",
        "Latest contacts",
        "Most recent contacts",
        "Who did I just add?",
    ]
    
    for text in context_queries:
        for _ in range(3):
            examples.append(Example(text=text, intent="knowledge_query", slots=[]))
    
    # =========================================================================
    # Natural language variations (100 examples)
    # =========================================================================
    natural_templates = [
        "I need to find {person}",
        "Can you find {person} for me?",
        "Looking for {person}",
        "Help me find {person}",
        "Where can I find info on {person}?",
        "I'm looking for info on {person}",
        "Do we have anything on {person}?",
        "What have we got on {person}?",
        "Any information about {person}?",
        "I want to know about {person}",
        "Tell me everything about {person}",
        "Quick lookup on {person}",
        "Search for {person}",
        "Find information on {person}",
        "I need to check on {person}",
    ]
    
    for _ in range(100):
        person_full, _ = random_name()
        template = random.choice(natural_templates)
        text = template.format(person=person_full)
        start = text.find(person_full)
        if start >= 0:
            examples.append(Example(
                text=text,
                intent="knowledge_query",
                slots=[Slot(start, start + len(person_full), "PER", person_full)]
            ))
        else:
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
        "You're welcome",
        "No problem",
        "That makes sense",
        "Perfect",
        "Awesome",
        "Cool",
        "Nice",
        "I understand",
        "Clear",
        "Noted",
    ]
    
    for text in conversations:
        examples.append(Example(text=text, intent="conversation", slots=[]))
    
    return examples

# =============================================================================
# Intent: web_search (EXPANDED)
# =============================================================================

CITIES = [
    "London", "New York", "San Francisco", "Tokyo", "Paris", "Berlin",
    "Singapore", "Sydney", "Mumbai", "Toronto", "Los Angeles", "Seattle",
    "Chicago", "Boston", "Austin", "Denver", "Miami", "Atlanta", "Phoenix",
    "Dubai", "Hong Kong", "Shanghai", "Beijing", "Seoul", "Bangkok",
    "Rome", "Madrid", "Amsterdam", "Stockholm", "Vienna", "Zurich",
    "Cape Town", "Cairo", "Lagos", "Nairobi", "Mexico City", "São Paulo",
    "Buenos Aires", "Vancouver", "Montreal", "Melbourne", "Auckland",
    "Canada", "Japan", "France", "Germany", "Italy", "Spain", "Australia"
]

COMPANIES = [
    "Apple", "Google", "Tesla", "Amazon", "Microsoft", "Meta", "Netflix",
    "Nvidia", "AMD", "Intel", "OpenAI", "Anthropic", "SpaceX", "Twitter",
    "Uber", "Airbnb", "Stripe", "Shopify", "Salesforce", "Adobe", "Oracle",
    "IBM", "Cisco", "Samsung", "Sony", "Nintendo", "Disney", "Warner Bros",
    "Nike", "Coca-Cola", "McDonald's", "Starbucks", "Walmart", "Target",
    "Goldman Sachs", "JPMorgan", "Bank of America", "Visa", "Mastercard",
    "Boeing", "Lockheed Martin", "General Motors", "Ford", "Toyota", "Honda"
]

TOPICS = [
    "Rust programming", "machine learning", "quantum computing", "AI", 
    "blockchain", "climate change", "electric vehicles", "space exploration",
    "cryptocurrency", "self-driving cars", "renewable energy", "5G technology",
    "Python", "JavaScript", "web development", "data science", "cybersecurity",
    "cloud computing", "kubernetes", "docker", "microservices", "DevOps",
    "artificial intelligence", "deep learning", "natural language processing",
    "computer vision", "robotics", "IoT", "augmented reality", "virtual reality",
    "solar energy", "wind power", "nuclear fusion", "battery technology"
]

FAMOUS_PEOPLE = [
    "Elon Musk", "Tim Cook", "Satya Nadella", "Jeff Bezos", "Mark Zuckerberg",
    "Sundar Pichai", "Sam Altman", "Jensen Huang", "Dario Amodei",
    "Bill Gates", "Warren Buffett", "Larry Page", "Sergey Brin", "Steve Jobs",
    "Barack Obama", "Joe Biden", "Donald Trump", "Kamala Harris",
    "Taylor Swift", "Beyoncé", "Drake", "Ed Sheeran", "Adele",
    "Leonardo DiCaprio", "Tom Hanks", "Meryl Streep", "Denzel Washington",
    "LeBron James", "Cristiano Ronaldo", "Lionel Messi", "Serena Williams",
    "Stephen Hawking", "Albert Einstein", "Neil deGrasse Tyson", "Elon Musk",
    "Oprah Winfrey", "Ellen DeGeneres", "Jimmy Fallon", "Trevor Noah"
]

CEO_QUERIES = [
    # CEO queries - these should ALWAYS be web search
    ("Who is the CEO of {company}?", "ORG"),
    ("Who runs {company}?", "ORG"),
    ("Who is {company}'s CEO?", "ORG"),
    ("Who founded {company}?", "ORG"),
    ("Who started {company}?", "ORG"),
    ("Who leads {company}?", "ORG"),
    ("Who is the founder of {company}?", "ORG"),
    ("Who is in charge of {company}?", "ORG"),
    ("Who is the president of {company}?", "ORG"),
    ("{company} CEO", "ORG"),
    ("{company} founder", "ORG"),
]

def generate_web_search() -> List[Example]:
    examples = []
    
    # Weather queries - basic
    weather_templates = [
        "What's the weather like?",
        "What's the weather today?",
        "Is it going to rain today?",
        "Is it going to rain tomorrow?",
        "Will it rain today?",
        "What's the forecast for today?",
        "What's the temperature outside?",
        "How's the weather?",
        "Do I need an umbrella today?",
        "Is it cold outside?",
        "Is it hot today?",
        "What's the weather forecast?",
        "Should I bring a jacket?",
        "Is it sunny today?",
        "Will it snow?",
        "What's the humidity?",
        "Is it windy outside?",
        "What's the forecast for this week?",
        "Weather forecast",
        "Today's weather",
        "Tomorrow's weather",
        "Weather report",
        "Current temperature",
        "What's it like outside?",
    ]
    
    for text in weather_templates:
        examples.append(Example(text=text, intent="web_search", slots=[]))
    
    # Weather with location - MANY variations
    for city in CITIES:
        templates = [
            f"What's the weather in {city}?",
            f"What's the weather like in {city}?",
            f"Weather in {city}",
            f"Temperature in {city}",
            f"Is it raining in {city}?",
            f"What's the forecast for {city}?",
            f"How's the weather in {city}?",
            f"{city} weather",
            f"{city} forecast",
            f"Will it rain in {city}?",
            f"Is it cold in {city}?",
            f"Is it hot in {city}?",
            f"Current weather in {city}",
            f"Weather forecast for {city}",
            f"What's the temperature in {city}?",
            f"How hot is it in {city}?",
            f"How cold is it in {city}?",
        ]
        for text in templates:
            start = text.find(city)
            examples.append(Example(
                text=text,
                intent="web_search",
                slots=[Slot(start, start + len(city), "LOC", city)]
            ))
    
    # Stock price queries - EXPANDED
    for company in COMPANIES:
        templates = [
            f"What's the stock price of {company}?",
            f"How is {company} stock doing?",
            f"What's {company}'s stock price?",
            f"Check {company} stock",
            f"How's {company} stock today?",
            f"{company} stock price",
            f"How much is {company} stock?",
            f"{company} stock",
            f"What is {company} trading at?",
            f"{company} share price",
            f"Stock price {company}",
            f"Is {company} stock up or down?",
            f"How did {company} stock do today?",
            f"{company} market cap",
            f"What's {company} worth?",
            f"{company} stock news",
            f"Should I buy {company} stock?",
            f"Is {company} a good investment?",
            f"{company} earnings",
            f"{company} quarterly results",
        ]
        for text in templates:
            start = text.find(company)
            examples.append(Example(
                text=text,
                intent="web_search",
                slots=[Slot(start, start + len(company), "ORG", company)]
            ))
    
    # CEO/Company leadership queries - CRITICAL for routing
    for company in COMPANIES:
        for template, slot_type in CEO_QUERIES:
            text = template.replace("{company}", company)
            start = text.find(company)
            examples.append(Example(
                text=text,
                intent="web_search",
                slots=[Slot(start, start + len(company), slot_type, company)]
            ))
    
    # General search queries
    search_templates = [
        ("Search for {topic}", "topic"),
        ("Look up {topic}", "topic"),
        ("Google {topic}", "topic"),
        ("Find information about {topic}", "topic"),
        ("What is {topic}?", "topic"),
        ("Tell me about {topic}", "topic"),
        ("How does {topic} work?", "topic"),
        ("Latest news on {topic}", "topic"),
        ("News about {topic}", "topic"),
        ("What's happening with {topic}?", "topic"),
        ("Research {topic}", "topic"),
        ("Info on {topic}", "topic"),
        ("Learn about {topic}", "topic"),
        ("Explain {topic}", "topic"),
        ("I want to know about {topic}", "topic"),
        ("Can you tell me about {topic}?", "topic"),
    ]
    
    for template, key in search_templates:
        for topic in TOPICS:
            text = template.replace(f"{{{key}}}", topic)
            start = text.find(topic)
            examples.append(Example(
                text=text,
                intent="web_search",
                slots=[Slot(start, start + len(topic), "EVENT", topic)]
            ))
    
    # Who is queries for famous people (web search, not knowledge graph)
    for person in FAMOUS_PEOPLE:
        templates = [
            f"Who is {person}?",
            f"Tell me about {person}",
            f"Search for {person}",
            f"What does {person} do?",
            f"Look up {person}",
            f"What is {person} known for?",
            f"Biography of {person}",
            f"{person} biography",
            f"Where is {person} from?",
            f"How old is {person}?",
            f"What did {person} do?",
            f"Is {person} famous?",
            f"Why is {person} famous?",
            f"{person} net worth",
            f"What company does {person} run?",
        ]
        for text in templates:
            start = text.find(person)
            examples.append(Example(
                text=text,
                intent="web_search",
                slots=[Slot(start, start + len(person), "PER", person)]
            ))
    
    # Time/date queries
    time_queries = [
        "What time is it?",
        "What's the current time?",
        "What's today's date?",
        "What day is it?",
        "What's the date today?",
        "Current time",
        "What year is it?",
        "What month is it?",
        "What day of the week is it?",
        "Time now",
    ]
    for text in time_queries:
        examples.append(Example(text=text, intent="web_search", slots=[]))
    
    # Sports/entertainment queries - EXPANDED
    sports_queries = [
        "What's the score of the game?",
        "Who won the match?",
        "Latest sports news",
        "Premier League results",
        "NBA scores",
        "World Cup standings",
        "Olympics results",
        "NFL scores",
        "MLB results",
        "Champions League scores",
        "Tennis results",
        "Golf scores",
        "UFC results",
        "Boxing results",
        "F1 standings",
        "NASCAR results",
        "Who won last night's game?",
        "Sports scores",
        "Live sports scores",
        "Football scores",
        "Basketball scores",
        "Baseball scores",
        "Soccer scores",
        "Hockey scores",
    ]
    for text in sports_queries:
        examples.append(Example(text=text, intent="web_search", slots=[]))
    
    # News queries - EXPANDED
    news_queries = [
        "What's in the news today?",
        "Latest headlines",
        "Top news stories",
        "Breaking news",
        "What's happening in the world?",
        "Current events",
        "Today's news",
        "News update",
        "What's new?",
        "Any news?",
        "World news",
        "Tech news",
        "Business news",
        "Politics news",
        "Entertainment news",
        "Science news",
        "Health news",
        "What's trending?",
        "Trending topics",
        "What's viral today?",
        "Latest updates",
        "News briefing",
        "Morning news",
        "Evening news",
    ]
    for text in news_queries:
        examples.append(Example(text=text, intent="web_search", slots=[]))
    
    # Definition queries - EXPANDED
    definition_templates = [
        "Define {word}",
        "What does {word} mean?",
        "Definition of {word}",
        "Meaning of {word}",
        "What is {word}?",
        "Explain {word}",
        "What's {word}?",
    ]
    words = [
        "AI", "machine learning", "blockchain", "cryptocurrency", "neural network", "API",
        "algorithm", "database", "encryption", "firewall", "malware", "phishing",
        "quantum", "robotics", "software", "hardware", "firmware", "protocol",
        "GDP", "inflation", "recession", "interest rate", "stock market", "bonds",
        "democracy", "republic", "socialism", "capitalism", "communism",
    ]
    for template in definition_templates:
        for word in words:
            text = template.replace("{word}", word)
            examples.append(Example(text=text, intent="web_search", slots=[]))
    
    # How to queries - EXPANDED significantly
    how_to_queries = [
        "How to make pasta?",
        "How to learn programming?",
        "How to start a business?",
        "How to invest in stocks?",
        "How to cook rice?",
        "How to tie a tie?",
        "How do I reset my password?",
        "How can I improve my sleep?",
        "How to learn Python?",
        "How to build a website?",
        "How to lose weight?",
        "How to save money?",
        "How to get a job?",
        "How to write a resume?",
        "How to negotiate salary?",
        "How to buy a house?",
        "How to rent an apartment?",
        "How to fix a flat tire?",
        "How to change oil?",
        "How to meditate?",
        "How to exercise at home?",
        "How to bake a cake?",
        "How to make coffee?",
        "How to speak Spanish?",
        "How to play guitar?",
        "How to draw?",
        "How to paint?",
        "How to budget?",
        "How to file taxes?",
        "How to get a passport?",
        "How to apply for a visa?",
        "How do you make bread?",
        "How can I learn faster?",
        "What's the best way to study?",
        "Tips for cooking",
        "Guide to investing",
        "Tutorial on programming",
        "Steps to start a company",
        "How to become a developer?",
        "How to become rich?",
        "How to be successful?",
        "How to be happy?",
        "How to reduce stress?",
        "How to deal with anxiety?",
    ]
    for text in how_to_queries:
        examples.append(Example(text=text, intent="web_search", slots=[]))
    
    # Comparison queries
    comparison_queries = [
        "What's the difference between Python and JavaScript?",
        "Compare iPhone and Android",
        "Mac vs Windows",
        "AWS vs Azure",
        "React vs Angular",
        "Which is better, Uber or Lyft?",
        "Tesla vs Rivian",
        "Netflix vs Disney Plus",
        "Spotify vs Apple Music",
        "What's better, coffee or tea?",
    ]
    for text in comparison_queries:
        examples.append(Example(text=text, intent="web_search", slots=[]))
    
    # Recipe queries
    recipe_queries = [
        "Recipe for chocolate cake",
        "How to make spaghetti?",
        "Best pizza recipe",
        "Chicken curry recipe",
        "Vegetarian recipes",
        "Quick dinner ideas",
        "Healthy breakfast recipes",
        "Dessert recipes",
        "What should I cook tonight?",
        "Easy recipes for beginners",
    ]
    for text in recipe_queries:
        examples.append(Example(text=text, intent="web_search", slots=[]))
    
    # Travel queries
    for city in CITIES[:20]:  # Use subset to avoid too many
        templates = [
            f"Things to do in {city}",
            f"Best restaurants in {city}",
            f"Hotels in {city}",
            f"Flights to {city}",
            f"Places to visit in {city}",
            f"Tourist attractions in {city}",
            f"{city} travel guide",
        ]
        for text in templates:
            start = text.find(city)
            examples.append(Example(
                text=text,
                intent="web_search",
                slots=[Slot(start, start + len(city), "LOC", city)]
            ))
    
    # Product/review queries
    product_queries = [
        "Best laptops 2024",
        "Best phones 2024",
        "Best headphones",
        "Best TV to buy",
        "iPhone 15 review",
        "MacBook Pro review",
        "Samsung Galaxy review",
        "PlayStation 5 review",
        "Best electric cars",
        "Best budget phone",
        "Top rated restaurants",
        "Best movies to watch",
        "Best books to read",
        "Best Netflix shows",
        "Best podcasts",
    ]
    for text in product_queries:
        examples.append(Example(text=text, intent="web_search", slots=[]))
    
    # Fact queries - things you'd google
    fact_queries = [
        "Population of China",
        "How tall is the Eiffel Tower?",
        "When did World War 2 end?",
        "Who invented the telephone?",
        "What is the capital of France?",
        "How far is the moon?",
        "Largest country in the world",
        "Richest person in the world",
        "Tallest building in the world",
        "Oldest person alive",
        "Speed of light",
        "Distance to Mars",
        "How many countries are there?",
        "What's the biggest animal?",
        "When was the internet invented?",
        "Who discovered electricity?",
        "How many bones in the human body?",
        "What causes earthquakes?",
        "Why is the sky blue?",
        "How do airplanes fly?",
    ]
    for text in fact_queries:
        examples.append(Example(text=text, intent="web_search", slots=[]))
    
    # Price/cost queries
    price_queries = [
        "How much does an iPhone cost?",
        "Price of Tesla Model 3",
        "Cost of living in New York",
        "Average rent in San Francisco",
        "Gas prices today",
        "Gold price",
        "Bitcoin price",
        "Oil prices",
        "Silver price",
        "Ethereum price",
    ]
    for text in price_queries:
        examples.append(Example(text=text, intent="web_search", slots=[]))
    
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
        ("web_search", generate_web_search),
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
