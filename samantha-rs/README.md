# Samantha

A warm, curious AI companion with memory, reminders, and natural conversation.

## Features

- 🌸 **Warm Personality** - Emotionally intelligent companion that remembers you
- 🧠 **Memory Extraction** - Learns and stores personal facts from conversations
- ⏰ **Smart Reminders** - Natural language reminders with timezone support
- 🌅 **Greetings** - Morning/night messages with your day's schedule
- 💬 **WhatsApp Integration** - Chat via WhatsApp Business API
- 🔐 **Local & Private** - Uses Ollama for local AI inference

## Prerequisites

- [Rust](https://rustup.rs/) (latest stable)
- [Ollama](https://ollama.ai/) with a model installed
- Python 3.9+ (for NLU model training)

## Quick Start

### Automated Setup

```bash
./setup.sh
```

This will:
1. Install Rust and dependencies
2. Install Ollama and pull the LLM model
3. Train the NLU model (if not present)
4. Build the project

### Manual Setup

```bash
# Install Ollama and pull a model
brew install ollama
ollama pull llama3.2:3b

# Train the NLU model (required on first clone)
./scripts/train_nlu.sh

# Or use quick mode for faster initial setup
./scripts/train_nlu.sh --quick

# Start Ollama server
ollama serve

# Build and run (in another terminal)
cargo run --release
```

## NLU Model Training

The NLU model files are **not stored in git** (they're ~500MB+). You must train them locally after cloning.

### Quick Training (~5 minutes)
```bash
./scripts/train_nlu.sh --quick
```

### Full Training (~30 minutes, better accuracy)
```bash
./scripts/train_nlu.sh
```

### Using Make (alternative)
```bash
cd nlu
make setup   # Create venv and install deps
make all     # Generate data, train, export to ONNX
```

### Training Options
```bash
./scripts/train_nlu.sh --help
./scripts/train_nlu.sh --epochs 20 --samples 1000  # Custom parameters
./scripts/train_nlu.sh --data-only                 # Only generate data
./scripts/train_nlu.sh --export                    # Only export to ONNX
```

## WhatsApp Integration

Samantha is designed for WhatsApp as the primary interface.

### Setup WhatsApp

1. **Create a Meta App**: Go to [developers.facebook.com](https://developers.facebook.com) and create a new app with WhatsApp product.

2. **Get Credentials**:
   - Phone Number ID: Found in WhatsApp > API Setup
   - Access Token: Generate a permanent token in your app settings
   - Verify Token: Create any secret string for webhook verification

3. **Configure Samantha** (choose one method):

   **Option A: Environment variables**
   ```bash
   export WHATSAPP_PHONE_ID="your_phone_number_id"
   export WHATSAPP_TOKEN="your_access_token"
   export WHATSAPP_VERIFY_TOKEN="any_secret_string"
   export WHATSAPP_ALLOWED_NUMBERS="+1234567890,+0987654321"  # Optional whitelist
   ```

   **Option B: Config file**
   Create `data/whatsapp_config.json`:
   ```json
   {
     "phone_id": "your_phone_number_id",
     "token": "your_access_token",
     "verify_token": "any_secret_string",
     "allowed_numbers": ["+1234567890"]
   }
   ```

4. **Run in WhatsApp mode**:
   ```bash
   ./target/release/samantha --whatsapp
   ```

5. **Configure Webhook** in Meta Developer Console:
   - Webhook URL: `https://your-domain.com/webhook`
   - Verify Token: Same as `WHATSAPP_VERIFY_TOKEN`
   - Subscribe to: `messages`

   For local testing, use [ngrok](https://ngrok.com):
   ```bash
   ngrok http 8080
   ```

## Samantha Features

### Memory
Samantha learns from your conversations:
- Your name and how you like to be called
- Your hobbies, interests, work
- Important people in your life
- Preferences and favorites

### Reminders
Set reminders naturally:
- "Remind me to call mom at 7pm"
- "Don't forget to take medicine tomorrow morning"
- "Remind me every Monday to submit reports"

### Greetings
Samantha will send you:
- **Morning greetings** - A cheerful hello with your day's reminders
- **Night greetings** - Wind down message with tomorrow's schedule

After a couple days, she'll ask about your timezone and sleep schedule to personalize these.

## CLI Commands

| Command | Description |
|---------|-------------|
| `/add person <name>` | Add a person to knowledge graph |
| `/add org <name>` | Add an organization |
| `/link <A> works_at <B>` | Link person to organization |
| `/link <A> knows <B>` | Link two people |
| `/info <name>` | Get information about an entity |
| `/list [people\|orgs]` | List entities |
| `/help` | Show available commands |
| `/exit` | Exit the assistant |

## Example Conversation

```
You: Hey Samantha! I'm feeling stressed about work today
Samantha: Hey~ I can sense the weight you're carrying today 💭 
Want to talk about what's going on at work? Sometimes just 
letting it out helps a little...

You: Remind me to take a break at 3pm
Samantha: Got it! I'll give you a gentle nudge at 3pm to step 
away for a bit~ 🌸 Taking breaks is so important, especially 
on stressful days!

You: Good night
Samantha: Night night~ 🌙 
Tomorrow you have:
• Take a break at 3pm
Rest well, and remember - tomorrow's a fresh start! 💫
```

## Architecture

```
src/
├── main.rs           # Entry point
├── llm/              # Ollama API client
├── messaging/        
│   ├── samantha.rs   # Personality, prompts, memory extraction
│   ├── handler.rs    # Message processing orchestration
│   └── whatsapp.rs   # WhatsApp Business API client
├── knowledge/        # Knowledge graph
├── storage/          # SQLite database
└── router/           # Command routing
```

## Running Modes

| Mode | Command | Description |
|------|---------|-------------|
| CLI | `./samantha` | Interactive terminal chat |
| WhatsApp | `./samantha --whatsapp` | WhatsApp server on port 8080 |

## Credits

Inspired by [qianjiang-git/Samantha](https://github.com/qianjiang-git/Samantha)

## License

MIT
