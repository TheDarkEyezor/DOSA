# NLU Training Pipeline Makefile
#
# Usage:
#   make setup      - Create virtual environment and install dependencies
#   make data       - Generate training data
#   make train      - Train the model
#   make export     - Export to ONNX
#   make test       - Run inference tests
#   make all        - Run full pipeline
#   make clean      - Remove generated files

PYTHON = python3
VENV = .venv
VENV_PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

DATA_DIR = ./data
OUTPUT_DIR = ./output
MODEL_DIR = $(OUTPUT_DIR)/best_model

# Training parameters
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
TRAIN_SAMPLES = 500

.PHONY: all setup data train export test benchmark clean

all: setup data train export test

# ============================================================================
# Setup
# ============================================================================

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

setup: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt
	@echo "✓ Setup complete. Activate with: source $(VENV)/bin/activate"

# ============================================================================
# Data Generation
# ============================================================================

$(DATA_DIR)/train.jsonl: generate_data.py
	@mkdir -p $(DATA_DIR)
	$(VENV_PYTHON) generate_data.py \
		--output_dir $(DATA_DIR) \
		--num_samples $(TRAIN_SAMPLES)

data: setup $(DATA_DIR)/train.jsonl
	@echo "✓ Training data generated in $(DATA_DIR)"

# ============================================================================
# Training
# ============================================================================

$(MODEL_DIR)/config.json: $(DATA_DIR)/train.jsonl train.py
	$(VENV_PYTHON) train.py \
		--data_dir $(DATA_DIR) \
		--model_dir $(OUTPUT_DIR) \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--learning_rate $(LEARNING_RATE)

train: data $(MODEL_DIR)/config.json
	@echo "✓ Model trained and saved to $(MODEL_DIR)"

# ============================================================================
# ONNX Export
# ============================================================================

$(OUTPUT_DIR)/model.onnx: $(MODEL_DIR)/config.json export_onnx.py
	$(VENV_PYTHON) export_onnx.py \
		--model_dir $(MODEL_DIR) \
		--output $(OUTPUT_DIR)/model.onnx

$(OUTPUT_DIR)/model.quantized.onnx: $(OUTPUT_DIR)/model.onnx
	$(VENV_PYTHON) export_onnx.py \
		--model_dir $(MODEL_DIR) \
		--output $(OUTPUT_DIR)/model.onnx \
		--quantize

export: train $(OUTPUT_DIR)/model.onnx
	@echo "✓ Model exported to $(OUTPUT_DIR)/model.onnx"

export-quantized: train $(OUTPUT_DIR)/model.quantized.onnx
	@echo "✓ Quantized model exported to $(OUTPUT_DIR)/model.quantized.onnx"

# ============================================================================
# Testing
# ============================================================================

test: export
	$(VENV_PYTHON) inference.py \
		--model $(OUTPUT_DIR)/model.onnx \
		--config $(OUTPUT_DIR)/config.json

test-interactive: export
	$(VENV_PYTHON) inference.py \
		--model $(OUTPUT_DIR)/model.onnx \
		--config $(OUTPUT_DIR)/config.json \
		--interactive

benchmark: export
	$(VENV_PYTHON) inference.py \
		--model $(OUTPUT_DIR)/model.onnx \
		--config $(OUTPUT_DIR)/config.json \
		--benchmark

# ============================================================================
# Cleanup
# ============================================================================

clean:
	rm -rf $(DATA_DIR)
	rm -rf $(OUTPUT_DIR)
	rm -rf __pycache__
	rm -rf *.egg-info

clean-all: clean
	rm -rf $(VENV)

# ============================================================================
# Development
# ============================================================================

lint:
	$(VENV_PYTHON) -m ruff check .

format:
	$(VENV_PYTHON) -m ruff format .

# ============================================================================
# Help
# ============================================================================

help:
	@echo "NLU Training Pipeline"
	@echo ""
	@echo "Usage:"
	@echo "  make setup            Create virtual environment and install dependencies"
	@echo "  make data             Generate training data"
	@echo "  make train            Train the model"
	@echo "  make export           Export to ONNX format"
	@echo "  make export-quantized Export quantized ONNX model (smaller, faster)"
	@echo "  make test             Run inference tests"
	@echo "  make test-interactive Interactive testing mode"
	@echo "  make benchmark        Run inference benchmarks"
	@echo "  make all              Run full pipeline (setup → data → train → export → test)"
	@echo "  make clean            Remove generated files"
	@echo "  make clean-all        Remove all files including virtual environment"
	@echo ""
	@echo "Parameters (can be overridden):"
	@echo "  EPOCHS=$(EPOCHS)"
	@echo "  BATCH_SIZE=$(BATCH_SIZE)"
	@echo "  LEARNING_RATE=$(LEARNING_RATE)"
	@echo "  TRAIN_SAMPLES=$(TRAIN_SAMPLES)"
	@echo ""
	@echo "Example with custom parameters:"
	@echo "  make train EPOCHS=20 BATCH_SIZE=32"
