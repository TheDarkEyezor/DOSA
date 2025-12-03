# DOSA NLU - DistilBERT Intent & Slot Filling

This directory contains the training pipeline for a fine-tuned DistilBERT model
that performs joint intent classification and slot (entity) filling.

## Architecture

```
Input: "Add Amogh Atreya, he studies at Imperial College"
        ↓ DistilBERT Encoder ↓
[CLS] Add  Amo  ##gh  Atr  ##eya  ,   he  studi  ##es  at  Imper  ##ial  College
  ↓    ↓    ↓    ↓    ↓    ↓    ↓   ↓    ↓     ↓   ↓    ↓      ↓      ↓
Intent: contact_update (from [CLS] token)
Slots:  O   B-PER I-PER I-PER I-PER I-PER O   O    O     O   O   B-ORG  I-ORG  I-ORG
```

## Intent Types

| Intent | Description | Example |
|--------|-------------|---------|
| `calendar_create` | Create a new event | "Schedule meeting with John tomorrow" |
| `calendar_query` | Query calendar | "What's on my schedule today?" |
| `calendar_update` | Modify an event | "Reschedule the standup to 3pm" |
| `calendar_delete` | Delete an event | "Cancel tomorrow's meeting" |
| `email_compose` | Write new email | "Send an email to John about the project" |
| `email_reply` | Reply to email | "Reply to Sarah's email" |
| `email_attendees` | Email event attendees | "Email the attendees about the delay" |
| `email_query` | Query emails | "Show my unread emails" |
| `contact_update` | Add/update contacts | "Add John Smith, he works at Google" |
| `knowledge_query` | Query knowledge graph | "Who works at Acme Corp?" |
| `conversation` | General chat | "Hello, how are you?" |
| `command` | Explicit command | "/help" |

## Slot/Entity Types (BIO format)

| Tag | Description | Example |
|-----|-------------|---------|
| `B-PER` / `I-PER` | Person name | "John Smith" |
| `B-ORG` / `I-ORG` | Organization | "Imperial College London" |
| `B-DATE` / `I-DATE` | Date expression | "tomorrow", "next Monday" |
| `B-TIME` / `I-TIME` | Time expression | "at 3pm", "in the afternoon" |
| `B-LOC` / `I-LOC` | Location | "conference room A" |
| `B-EMAIL` / `I-EMAIL` | Email address | "john@example.com" |
| `B-EVENT` / `I-EVENT` | Event name | "team standup" |
| `B-REL` / `I-REL` | Relationship type | "works at", "studies at" |
| `O` | Outside (no entity) | - |

## Training Data Format

```json
{
  "text": "Add Amogh Atreya, he studies at Imperial College",
  "intent": "contact_update",
  "slots": [
    {"start": 4, "end": 16, "label": "PER", "text": "Amogh Atreya"},
    {"start": 21, "end": 31, "label": "REL", "text": "studies at"},
    {"start": 32, "end": 48, "label": "ORG", "text": "Imperial College"}
  ]
}
```

## Setup

```bash
cd nlu
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
# Generate training data from examples
python generate_data.py

# Train the model
python train.py --epochs 10 --batch_size 16

# Export to ONNX for Rust
python export_onnx.py --model_dir ./output --output ./model.onnx
```

## Integration with Rust

The exported ONNX model is loaded via the `ort` crate:

```rust
use ort::{Environment, Session, Value};

let model = Session::builder()?
    .with_model_from_file("nlu/model.onnx")?;

let (intent, slots) = model.run(...)?;
```

## Files

- `requirements.txt` - Python dependencies
- `generate_data.py` - Generate training data from patterns
- `train.py` - Training script
- `export_onnx.py` - Export model to ONNX format
- `data/` - Training/validation/test splits
- `output/` - Trained model checkpoints
