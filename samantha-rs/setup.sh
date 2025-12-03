#!/bin/bash
#
# DOSA Setup Script
# Installs all prerequisites and builds the project
#
# Usage: ./setup.sh [--whatsapp]
#
# Options:
#   --whatsapp    Enable WhatsApp integration features
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Parse arguments
ENABLE_WHATSAPP=false
for arg in "$@"; do
    case $arg in
        --whatsapp)
            ENABLE_WHATSAPP=true
            shift
            ;;
    esac
done

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  DOSA - Digitally Optimized Smart Assistant                ║"
echo "║  Setup Script                                              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"
print_step "Detected: $OS ($ARCH)"

# =============================================================================
# Install Rust (if not present)
# =============================================================================
print_step "Checking Rust installation..."

if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version | cut -d' ' -f2)
    print_success "Rust is installed (version $RUST_VERSION)"
else
    print_warning "Rust not found. Installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    print_success "Rust installed successfully"
fi

# Ensure cargo is in PATH
if ! command -v cargo &> /dev/null; then
    source "$HOME/.cargo/env"
fi

# =============================================================================
# Install system dependencies
# =============================================================================
print_step "Installing system dependencies..."

case $OS in
    Darwin)
        # macOS
        if ! command -v brew &> /dev/null; then
            print_warning "Homebrew not found. Installing..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Install dependencies
        brew install openssl pkg-config sqlite3 || true
        print_success "macOS dependencies installed"
        ;;
    
    Linux)
        # Linux (Debian/Ubuntu)
        if command -v apt-get &> /dev/null; then
            print_step "Installing via apt-get..."
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                pkg-config \
                libssl-dev \
                libsqlite3-dev \
                curl \
                git
            print_success "Linux dependencies installed"
        # Linux (Fedora/RHEL)
        elif command -v dnf &> /dev/null; then
            print_step "Installing via dnf..."
            sudo dnf install -y \
                gcc \
                pkg-config \
                openssl-devel \
                sqlite-devel \
                curl \
                git
            print_success "Linux dependencies installed"
        # Linux (Arch)
        elif command -v pacman &> /dev/null; then
            print_step "Installing via pacman..."
            sudo pacman -Syu --noconfirm \
                base-devel \
                openssl \
                sqlite \
                curl \
                git
            print_success "Linux dependencies installed"
        else
            print_error "Unknown Linux distribution. Please install manually:"
            echo "  - OpenSSL development headers"
            echo "  - SQLite development headers"
            echo "  - pkg-config"
            echo "  - C compiler (gcc/clang)"
        fi
        ;;
    
    *)
        print_error "Unsupported OS: $OS"
        exit 1
        ;;
esac

# =============================================================================
# Install Ollama (for LLM)
# =============================================================================
print_step "Checking Ollama installation..."

if command -v ollama &> /dev/null; then
    print_success "Ollama is installed"
else
    print_warning "Ollama not found. Installing..."
    
    case $OS in
        Darwin)
            brew install ollama || {
                print_step "Trying direct download..."
                curl -fsSL https://ollama.ai/install.sh | sh
            }
            ;;
        Linux)
            curl -fsSL https://ollama.ai/install.sh | sh
            ;;
    esac
    print_success "Ollama installed"
fi

# Pull the required model
print_step "Pulling llama3.2:3b model (this may take a while)..."
ollama pull llama3.2:3b || print_warning "Failed to pull model. Make sure Ollama is running."

# =============================================================================
# Build DOSA
# =============================================================================
print_step "Building DOSA..."

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build with appropriate features
if [ "$ENABLE_WHATSAPP" = true ]; then
    print_step "Building with WhatsApp support..."
    cargo build --release --features whatsapp
else
    cargo build --release
fi

print_success "DOSA built successfully"

# =============================================================================
# Train NLU Model (if not present)
# =============================================================================
print_step "Checking NLU model..."

NLU_MODEL="$SCRIPT_DIR/data/nlu/model.onnx"

if [ -f "$NLU_MODEL" ]; then
    print_success "NLU model already exists"
else
    print_warning "NLU model not found. Training now..."
    echo ""
    echo "This may take 10-30 minutes depending on your hardware."
    echo "For faster initial setup, you can Ctrl+C and run later with:"
    echo "  ./scripts/train_nlu.sh --quick"
    echo ""
    
    if [ -f "$SCRIPT_DIR/scripts/train_nlu.sh" ]; then
        # Use quick mode for initial setup
        "$SCRIPT_DIR/scripts/train_nlu.sh" --quick
    else
        print_warning "Training script not found. You'll need to train the NLU model manually."
        print_warning "Run: cd nlu && make all"
    fi
fi

# =============================================================================
# Create data directory and copy NLU model
# =============================================================================
print_step "Setting up data directory..."

DATA_DIR="$SCRIPT_DIR/target/release/data"
mkdir -p "$DATA_DIR"

# Copy NLU model if it exists
if [ -d "$SCRIPT_DIR/data/nlu" ] && [ -f "$SCRIPT_DIR/data/nlu/model.onnx" ]; then
    print_step "Copying NLU model to release directory..."
    mkdir -p "$DATA_DIR/nlu"
    cp "$SCRIPT_DIR/data/nlu/"* "$DATA_DIR/nlu/" 2>/dev/null || true
    print_success "NLU model copied"
elif [ -d "$SCRIPT_DIR/nlu/output" ]; then
    print_step "Copying NLU model from training output..."
    rm -rf "$DATA_DIR/nlu"
    cp -r "$SCRIPT_DIR/nlu/output" "$DATA_DIR/nlu"
    print_success "NLU model copied"
else
    print_warning "NLU model not found. Run: ./scripts/train_nlu.sh"
fi

# =============================================================================
# WhatsApp Configuration (if enabled)
# =============================================================================
if [ "$ENABLE_WHATSAPP" = true ]; then
    print_step "Setting up WhatsApp configuration..."
    
    WHATSAPP_CONFIG="$DATA_DIR/whatsapp_config.json"
    
    if [ ! -f "$WHATSAPP_CONFIG" ]; then
        cat > "$WHATSAPP_CONFIG" << 'EOF'
{
    "phone_number_id": "YOUR_PHONE_NUMBER_ID",
    "access_token": "YOUR_ACCESS_TOKEN",
    "allowed_numbers": ["+1234567890"],
    "verify_token": "your_custom_verify_token"
}
EOF
        print_warning "WhatsApp config created at: $WHATSAPP_CONFIG"
        print_warning "Please edit with your Meta Business API credentials."
    else
        print_success "WhatsApp config already exists"
    fi
fi

# =============================================================================
# Create run script
# =============================================================================
print_step "Creating run script..."

RUN_SCRIPT="$SCRIPT_DIR/run.sh"
cat > "$RUN_SCRIPT" << 'EOF'
#!/bin/bash
#
# Run DOSA
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 2
fi

# Run DOSA
cd "$SCRIPT_DIR"
./target/release/dosa "$@"
EOF

chmod +x "$RUN_SCRIPT"
print_success "Run script created: $RUN_SCRIPT"

# =============================================================================
# Create WhatsApp run script (if enabled)
# =============================================================================
if [ "$ENABLE_WHATSAPP" = true ]; then
    WA_RUN_SCRIPT="$SCRIPT_DIR/run_whatsapp.sh"
    cat > "$WA_RUN_SCRIPT" << 'EOF'
#!/bin/bash
#
# Run DOSA with WhatsApp integration
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/target/release/data"

# Check for WhatsApp config
if [ ! -f "$DATA_DIR/whatsapp_config.json" ]; then
    echo "Error: WhatsApp config not found at $DATA_DIR/whatsapp_config.json"
    echo "Please create it with your Meta Business API credentials."
    exit 1
fi

# Load environment from config (optional: you can also set these directly)
export WHATSAPP_PHONE_ID=$(jq -r '.phone_number_id' "$DATA_DIR/whatsapp_config.json")
export WHATSAPP_TOKEN=$(jq -r '.access_token' "$DATA_DIR/whatsapp_config.json")
export WHATSAPP_VERIFY_TOKEN=$(jq -r '.verify_token' "$DATA_DIR/whatsapp_config.json")
export WHATSAPP_ALLOWED_NUMBERS=$(jq -r '.allowed_numbers | join(",")' "$DATA_DIR/whatsapp_config.json")
export WEBHOOK_PORT=${WEBHOOK_PORT:-8080}

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 2
fi

echo "Starting DOSA with WhatsApp integration..."
echo "Webhook endpoint: http://your-server:$WEBHOOK_PORT/webhook"
echo ""

# Run DOSA with WhatsApp mode
cd "$SCRIPT_DIR"
./target/release/dosa --whatsapp "$@"
EOF

    chmod +x "$WA_RUN_SCRIPT"
    print_success "WhatsApp run script created: $WA_RUN_SCRIPT"
fi

# =============================================================================
# Print summary
# =============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Setup Complete!                                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "To run DOSA:"
echo "  ${GREEN}./run.sh${NC}"
echo ""
if [ "$ENABLE_WHATSAPP" = true ]; then
    echo "To run with WhatsApp:"
    echo "  1. Edit ${YELLOW}target/release/data/whatsapp_config.json${NC}"
    echo "  2. Set up webhook URL in Meta Business Portal"
    echo "  3. Run: ${GREEN}./run_whatsapp.sh${NC}"
    echo ""
fi
echo "For help:"
echo "  ${GREEN}./target/release/dosa --help${NC}"
echo ""
print_success "Setup complete!"
