# Samantha Setup & Run Guide

This guide covers everything you need to get Samantha running on a new machine.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/TheDarkEyezor/DOSA.git
cd DOSA/samantha-rs

# Run automated setup
./setup.sh

# Run Samantha
./run.sh
```

## Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| macOS / Linux | - | Windows WSL2 also works |
| Python | 3.9+ | For NLU model training |
| ~4GB disk | - | For models and dependencies |
| ~8GB RAM | - | 16GB recommended for training |

The setup script will automatically install:
- Rust (via rustup)
- Ollama (local LLM)
- System dependencies (OpenSSL, SQLite, etc.)

## Step-by-Step Setup

### 1. Clone and Setup

```bash
git clone https://github.com/TheDarkEyezor/DOSA.git
cd DOSA/samantha-rs
./setup.sh
```

This takes ~10-15 minutes and will:
- Install Rust toolchain
- Install system dependencies
- Pull the Ollama LLM model (~2GB download)
- Train the NLU model (quick mode)
- Build the Rust binary

### 2. Train NLU Model

The NLU (Natural Language Understanding) model is **not included in the repo** due to its size (~500MB). You must train it locally.

**Quick training (~5 minutes):**
```bash
./scripts/train_nlu.sh --quick
```

**Full training (~30 minutes, better accuracy):**
```bash
./scripts/train_nlu.sh
```

**Custom training:**
```bash
./scripts/train_nlu.sh --epochs 20 --samples 1000
```

### 3. Google Calendar/Email Integration (Optional)

To enable Google Calendar and Gmail features, you need OAuth credentials.

#### Option A: Credentials File (Recommended)

1. Get your `google_credentials.json` from the project maintainer, or create your own in [Google Cloud Console](https://console.cloud.google.com/)

2. Place it in the config directory:
```bash
mkdir -p ~/.config/samantha
cp /path/to/google_credentials.json ~/.config/samantha/
```

#### Option B: Environment Variables

```bash
export GOOGLE_CLIENT_ID="your-client-id.apps.googleusercontent.com"
export GOOGLE_CLIENT_SECRET="your-client-secret"
```

#### Credential Search Order

Samantha looks for credentials in this order:
1. Environment variables (`GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`)
2. `~/.config/samantha/google_credentials.json`
3. `./data/google_credentials.json`
4. `client_secret*.json` in current directory

### 4. Run Samantha

```bash
./run.sh
```

Or manually:
```bash
# Start Ollama (if not running)
ollama serve &

# Run Samantha
./target/release/samantha
```

## WhatsApp Integration

To use Samantha via WhatsApp:

### 1. Setup Meta Business API

1. Create a Meta App at [developers.facebook.com](https://developers.facebook.com)
2. Add WhatsApp product to your app
3. Get your Phone Number ID and Access Token

### 2. Configure Credentials

```bash
# Run setup with WhatsApp flag
./setup.sh --whatsapp

# Edit the generated config
nano target/release/data/whatsapp_config.json
```

Config format:
```json
{
    "phone_number_id": "YOUR_PHONE_NUMBER_ID",
    "access_token": "YOUR_ACCESS_TOKEN",
    "allowed_numbers": ["+1234567890"],
    "verify_token": "your_custom_verify_token"
}
```

### 3. Run with WhatsApp

```bash
./run_whatsapp.sh
```

### 4. Configure Webhook

In Meta Developer Console, set:
- **Webhook URL:** `https://your-domain.com/webhook`
- **Verify Token:** Same as in config
- **Subscribe to:** `messages`

For local testing, use ngrok:
```bash
ngrok http 8080
```

## Troubleshooting

### NLU Model Not Found

```
Error: NLU model not found
```

**Solution:** Train the model:
```bash
./scripts/train_nlu.sh --quick
```

### Ollama Not Running

```
Error: Connection refused (os error 61)
```

**Solution:** Start Ollama:
```bash
ollama serve
```

### Google Auth Failed

```
Error: Google credentials not found
```

**Solution:** Add credentials to `~/.config/samantha/google_credentials.json`

### Build Errors

```
error: linker `cc` not found
```

**Solution:** Install build tools:
```bash
# macOS
xcode-select --install

# Ubuntu/Debian
sudo apt-get install build-essential

# Fedora
sudo dnf install gcc
```

### Python Dependencies Failed

```
ERROR: Could not build wheels for tokenizers
```

**Solution:** Install Rust (required for tokenizers):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

## Directory Structure

After setup, your directory will look like:

```
samantha-rs/
├── data/
│   └── nlu/              # NLU model files (generated)
│       ├── model.onnx
│       ├── config.json
│       └── vocab.txt
├── nlu/
│   ├── output/           # Training output (generated, gitignored)
│   └── data/             # Training data
├── scripts/
│   └── train_nlu.sh      # NLU training script
├── target/
│   └── release/
│       ├── samantha      # Main binary
│       └── data/         # Runtime data directory
├── run.sh                # Run script (generated)
├── setup.sh              # Setup script
└── Cargo.toml
```

## Development

### Rebuild After Changes

```bash
cargo build --release
```

### Run Tests

```bash
cargo test
```

### Check for Errors

```bash
cargo check
```

### Run with Debug Output

```bash
RUST_LOG=debug ./target/release/samantha
```

## Updating

```bash
git pull
cargo build --release

# Re-train NLU if training data changed
./scripts/train_nlu.sh
```

## Uninstalling

```bash
# Remove the project
rm -rf /path/to/DOSA

# Remove config (optional)
rm -rf ~/.config/samantha

# Remove Ollama models (optional)
ollama rm llama3.2:3b
```
