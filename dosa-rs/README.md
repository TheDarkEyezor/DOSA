# DOSA (Rust)

Digitally Optimized Smart Assistant - A personal AI assistant with knowledge graph.

## Features

- 🤖 **Local LLM Integration** - Uses Ollama for private, local AI inference
- 🧠 **Knowledge Graph** - Stores information about people, organizations, and their relationships
- 💬 **Natural Language** - Chat naturally or use commands
- 📝 **Persistent Storage** - SQLite database for all your data
- 🔧 **Extensible** - Easy to add new commands

## Prerequisites

- [Rust](https://rustup.rs/) (latest stable)
- [Ollama](https://ollama.ai/) with a model installed

## Quick Start

```bash
# Install Ollama and pull a model
brew install ollama
ollama pull llama3.2:3b

# Start Ollama server
ollama serve

# Build and run DOSA (in another terminal)
cd dosa-rs
cargo run --release
```

## Commands

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

## Example Session

```
you> /add person John Doe
Added person: John Doe (id: 1)

you> /add org TechCorp
Added organization: TechCorp (id: 2)

you> /link John Doe works_at TechCorp
Linked: John Doe works at TechCorp

you> /set John Doe email john@techcorp.com
Set John Doe.email = john@techcorp.com

you> Who works at TechCorp?
Based on my knowledge graph, John Doe works at TechCorp...
```

## Architecture

```
src/
├── main.rs           # CLI entry point
├── llm/              # Ollama API client
├── knowledge/        # Knowledge graph (entities, relationships)
├── router/           # Command routing system
└── storage/          # SQLite database layer
```

## License

MIT
