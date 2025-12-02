#!/usr/bin/env python3
"""
Export trained JointDistilBERT model to ONNX format for Rust inference.

The exported model has:
- Inputs: input_ids, attention_mask
- Outputs: intent_logits, slot_logits

Usage:
    python export_onnx.py --model_dir ./output/best_model --output ./output/model.onnx
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel, DistilBertConfig

# Import the model class from train.py
from train import JointDistilBERT

def export_to_onnx(
    model_dir: Path,
    output_path: Path,
    max_length: int = 128,
    opset_version: int = 14,
):
    """Export the model to ONNX format."""
    
    print(f"Loading model from {model_dir}...")
    
    # Load config
    config_path = model_dir.parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    num_intents = len(config["intent2id"])
    num_slots = len(config["slot2id"])
    
    # Load model
    model = JointDistilBERT.from_pretrained(
        model_dir,
        num_intents=num_intents,
        num_slots=num_slots,
    )
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    dummy_input_ids = torch.ones(batch_size, max_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, max_length, dtype=torch.long)
    
    # Export
    print(f"Exporting to ONNX (opset {opset_version})...")
    
    # Wrapper to return tuple instead of dict for ONNX export
    class ONNXWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs["intent_logits"], outputs["slot_logits"]
    
    wrapper = ONNXWrapper(model)
    wrapper.eval()
    
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["intent_logits", "slot_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "intent_logits": {0: "batch_size"},
            "slot_logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    print(f"Model exported to {output_path}")
    
    # Verify the model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Test with ONNX Runtime
    print("Testing with ONNX Runtime...")
    session = ort.InferenceSession(str(output_path))
    
    # Run inference
    outputs = session.run(
        None,
        {
            "input_ids": dummy_input_ids.numpy(),
            "attention_mask": dummy_attention_mask.numpy(),
        }
    )
    
    intent_logits, slot_logits = outputs
    print(f"✓ Intent logits shape: {intent_logits.shape}")
    print(f"✓ Slot logits shape: {slot_logits.shape}")
    
    # Compare with PyTorch output
    print("Comparing PyTorch and ONNX outputs...")
    with torch.no_grad():
        pt_intent, pt_slot = wrapper(dummy_input_ids, dummy_attention_mask)
    
    intent_diff = np.abs(intent_logits - pt_intent.numpy()).max()
    slot_diff = np.abs(slot_logits - pt_slot.numpy()).max()
    
    print(f"  Max intent diff: {intent_diff:.6f}")
    print(f"  Max slot diff: {slot_diff:.6f}")
    
    if intent_diff < 1e-4 and slot_diff < 1e-4:
        print("✓ PyTorch and ONNX outputs match!")
    else:
        print("⚠ Warning: Some numerical differences detected")
    
    return output_path

def export_tokenizer_vocab(model_dir: Path, output_dir: Path):
    """Export tokenizer vocabulary for Rust tokenization."""
    
    print(f"Exporting tokenizer vocabulary...")
    
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    
    # Save vocab.txt
    vocab_path = output_dir / "vocab.txt"
    with open(vocab_path, "w") as f:
        for token, idx in sorted(tokenizer.vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\n")
    
    print(f"✓ Vocabulary saved to {vocab_path} ({len(tokenizer.vocab)} tokens)")
    
    # Save special tokens
    special_tokens_path = output_dir / "special_tokens.json"
    special_tokens = {
        "unk_token": tokenizer.unk_token,
        "sep_token": tokenizer.sep_token,
        "pad_token": tokenizer.pad_token,
        "cls_token": tokenizer.cls_token,
        "mask_token": tokenizer.mask_token,
        "unk_token_id": tokenizer.unk_token_id,
        "sep_token_id": tokenizer.sep_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "cls_token_id": tokenizer.cls_token_id,
        "mask_token_id": tokenizer.mask_token_id,
    }
    
    with open(special_tokens_path, "w") as f:
        json.dump(special_tokens, f, indent=2)
    
    print(f"✓ Special tokens saved to {special_tokens_path}")

def optimize_onnx(input_path: Path, output_path: Path):
    """Apply ONNX optimizations for faster inference."""
    
    try:
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions
        
        print("Applying ONNX optimizations...")
        
        optimized_model = optimizer.optimize_model(
            str(input_path),
            model_type="bert",
            num_heads=12,
            hidden_size=768,
        )
        
        optimized_model.save_model_to_file(str(output_path))
        print(f"✓ Optimized model saved to {output_path}")
        
        # Compare sizes
        original_size = input_path.stat().st_size / 1024 / 1024
        optimized_size = output_path.stat().st_size / 1024 / 1024
        print(f"  Original: {original_size:.2f} MB")
        print(f"  Optimized: {optimized_size:.2f} MB")
        
    except ImportError:
        print("⚠ onnxruntime.transformers not available, skipping optimization")
        return input_path
    
    return output_path

def quantize_onnx(input_path: Path, output_path: Path):
    """Apply dynamic quantization for smaller model size."""
    
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        print("Applying dynamic quantization...")
        
        quantize_dynamic(
            str(input_path),
            str(output_path),
            weight_type=QuantType.QUInt8,
        )
        
        print(f"✓ Quantized model saved to {output_path}")
        
        # Compare sizes
        original_size = input_path.stat().st_size / 1024 / 1024
        quantized_size = output_path.stat().st_size / 1024 / 1024
        print(f"  Original: {original_size:.2f} MB")
        print(f"  Quantized: {quantized_size:.2f} MB")
        print(f"  Reduction: {(1 - quantized_size/original_size) * 100:.1f}%")
        
    except ImportError:
        print("⚠ onnxruntime.quantization not available, skipping quantization")
        return input_path
    
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./output/best_model")
    parser.add_argument("--output", type=str, default="./output/model.onnx")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--optimize", action="store_true", help="Apply ONNX optimizations")
    parser.add_argument("--quantize", action="store_true", help="Apply dynamic quantization")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Export base model
    export_to_onnx(model_dir, output_path, args.max_length, args.opset)
    
    # Export tokenizer
    export_tokenizer_vocab(model_dir, output_dir)
    
    # Optional optimizations
    if args.optimize:
        optimized_path = output_path.with_suffix(".optimized.onnx")
        output_path = optimize_onnx(output_path, optimized_path)
    
    if args.quantize:
        quantized_path = output_path.with_suffix(".quantized.onnx")
        output_path = quantize_onnx(output_path, quantized_path)
    
    print("\n" + "="*60)
    print("Export complete!")
    print("="*60)
    print(f"\nFiles created in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.2f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.2f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {f.name}: {size_str}")

if __name__ == "__main__":
    main()
