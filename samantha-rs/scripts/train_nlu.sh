#!/bin/bash
#
# NLU Model Training Script
# 
# This script trains the NLU model from scratch and exports it to ONNX format.
# Run this after cloning the repo to generate the model files.
#
# Usage:
#   ./scripts/train_nlu.sh              # Full training pipeline
#   ./scripts/train_nlu.sh --quick      # Quick training (fewer samples/epochs)
#   ./scripts/train_nlu.sh --data-only  # Only generate training data
#   ./scripts/train_nlu.sh --export     # Only export existing model to ONNX
#
# Requirements:
#   - Python 3.9+
#   - ~4GB disk space for training
#   - ~8GB RAM (16GB recommended)
#   - GPU optional but recommended
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

# Default parameters
EPOCHS=10
BATCH_SIZE=16
TRAIN_SAMPLES=500
QUICK_MODE=false
DATA_ONLY=false
EXPORT_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            EPOCHS=3
            TRAIN_SAMPLES=100
            shift
            ;;
        --data-only)
            DATA_ONLY=true
            shift
            ;;
        --export)
            EXPORT_ONLY=true
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --samples)
            TRAIN_SAMPLES="$2"
            shift 2
            ;;
        --help|-h)
            echo "NLU Model Training Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick        Quick training mode (3 epochs, 100 samples)"
            echo "  --data-only    Only generate training data"
            echo "  --export       Only export existing model to ONNX"
            echo "  --epochs N     Set number of training epochs (default: 10)"
            echo "  --samples N    Set number of training samples (default: 500)"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NLU_DIR="$PROJECT_DIR/nlu"
DATA_DIR="$PROJECT_DIR/data/nlu"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Samantha NLU Model Training                               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Project directory: $PROJECT_DIR"
echo "NLU directory: $NLU_DIR"
echo ""

# Check Python
print_step "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 not found. Please install Python 3.9 or later."
    exit 1
fi

# Check if in nlu directory
if [ ! -f "$NLU_DIR/train.py" ]; then
    print_error "Cannot find NLU training scripts in $NLU_DIR"
    print_error "Make sure you're in the samantha-rs directory"
    exit 1
fi

cd "$NLU_DIR"

# =============================================================================
# Setup Virtual Environment
# =============================================================================
print_step "Setting up Python virtual environment..."

VENV_DIR="$NLU_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created"
else
    print_success "Virtual environment exists"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install dependencies
print_step "Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
print_success "Dependencies installed"

# =============================================================================
# Generate Training Data
# =============================================================================
print_step "Generating training data..."

if [ "$EXPORT_ONLY" = false ]; then
    python generate_data.py \
        --output_dir ./data \
        --num_samples "$TRAIN_SAMPLES"
    print_success "Training data generated ($TRAIN_SAMPLES samples)"
fi

if [ "$DATA_ONLY" = true ]; then
    print_success "Data generation complete!"
    echo ""
    echo "Training data saved to: $NLU_DIR/data/"
    echo "  - train.jsonl"
    echo "  - val.jsonl"
    echo "  - test.jsonl"
    exit 0
fi

# =============================================================================
# Train Model
# =============================================================================
if [ "$EXPORT_ONLY" = false ]; then
    print_step "Training NLU model..."
    echo "  Epochs: $EPOCHS"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Training samples: $TRAIN_SAMPLES"
    if [ "$QUICK_MODE" = true ]; then
        echo -e "  ${YELLOW}Quick mode enabled${NC}"
    fi
    echo ""

    python train.py \
        --data_dir ./data \
        --model_dir ./output \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate 5e-5

    print_success "Model training complete"
fi

# =============================================================================
# Export to ONNX
# =============================================================================
print_step "Exporting model to ONNX format..."

python export_onnx.py \
    --model_dir ./output/best_model \
    --output ./output/model.onnx

print_success "ONNX model exported"

# =============================================================================
# Copy to data directory
# =============================================================================
print_step "Setting up model for runtime..."

mkdir -p "$DATA_DIR"

# Copy required files
cp ./output/model.onnx "$DATA_DIR/" 2>/dev/null || true
cp ./output/config.json "$DATA_DIR/" 2>/dev/null || true
cp ./output/vocab.txt "$DATA_DIR/" 2>/dev/null || true
cp ./vocab.txt "$DATA_DIR/" 2>/dev/null || true

# Also copy from best_model if available
if [ -d "./output/best_model" ]; then
    cp ./output/best_model/vocab.txt "$DATA_DIR/" 2>/dev/null || true
    cp ./output/best_model/tokenizer_config.json "$DATA_DIR/" 2>/dev/null || true
fi

print_success "Model files copied to $DATA_DIR"

# =============================================================================
# Run Quick Test
# =============================================================================
print_step "Running inference test..."

python inference.py \
    --model ./output/model.onnx \
    --config ./output/config.json \
    --test

print_success "Inference test passed"

# Deactivate virtual environment
deactivate

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Training Complete!                                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Model files saved to:"
echo "  ${CYAN}$DATA_DIR/model.onnx${NC}"
echo "  ${CYAN}$DATA_DIR/config.json${NC}"
echo "  ${CYAN}$DATA_DIR/vocab.txt${NC}"
echo ""
echo "Next steps:"
echo "  1. Build the Rust project: ${GREEN}cargo build --release${NC}"
echo "  2. Run Samantha: ${GREEN}./run.sh${NC}"
echo ""
print_success "NLU model ready!"
