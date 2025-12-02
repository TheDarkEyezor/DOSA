#!/usr/bin/env python3
"""
Test inference with the exported ONNX model.

This script provides a reference implementation for the Rust inference code.

Usage:
    python inference.py --model ./output/model.onnx --text "Schedule a meeting with John tomorrow"
    python inference.py --model ./output/model.onnx --interactive
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizerFast

@dataclass
class NLUResult:
    """Result from NLU inference."""
    text: str
    intent: str
    intent_confidence: float
    slots: List[Dict]
    
    def __str__(self):
        slots_str = ", ".join(
            f"{s['label']}='{s['text']}'" for s in self.slots
        ) if self.slots else "none"
        return f"Intent: {self.intent} ({self.intent_confidence:.2%})\nSlots: {slots_str}"

class NLUInference:
    """ONNX-based NLU inference."""
    
    def __init__(self, model_path: Path, config_path: Path, tokenizer_path: Optional[Path] = None):
        self.model_path = model_path
        
        # Load config
        with open(config_path) as f:
            config = json.load(f)
        
        self.id2intent = {int(k): v for k, v in config["id2intent"].items()}
        self.id2slot = {int(k): v for k, v in config["id2slot"].items()}
        self.max_length = config.get("max_length", 128)
        
        # Load tokenizer
        tokenizer_path = tokenizer_path or model_path.parent / "best_model"
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
        
        # Load ONNX model
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )
        
        print(f"Loaded model from {model_path}")
        print(f"  Intents: {len(self.id2intent)}")
        print(f"  Slot labels: {len(self.id2slot)}")
    
    def predict(self, text: str) -> NLUResult:
        """Run inference on a single text."""
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="np"
        )
        
        offset_mapping = encoding.pop("offset_mapping")[0]
        
        # Run inference
        outputs = self.session.run(
            None,
            {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
            }
        )
        
        intent_logits, slot_logits = outputs
        
        # Get intent
        intent_probs = softmax(intent_logits[0])
        intent_id = int(np.argmax(intent_probs))
        intent = self.id2intent[intent_id]
        intent_confidence = float(intent_probs[intent_id])
        
        # Get slots
        slot_preds = np.argmax(slot_logits[0], axis=-1)
        slots = self._extract_slots(text, slot_preds, offset_mapping, encoding["attention_mask"][0])
        
        return NLUResult(
            text=text,
            intent=intent,
            intent_confidence=intent_confidence,
            slots=slots
        )
    
    def _extract_slots(
        self,
        text: str,
        slot_preds: np.ndarray,
        offset_mapping: np.ndarray,
        attention_mask: np.ndarray
    ) -> List[Dict]:
        """Extract slot entities from token predictions."""
        
        slots = []
        current_slot = None
        current_start = None
        current_end = None
        
        for i, (pred_id, (start, end), mask) in enumerate(
            zip(slot_preds, offset_mapping, attention_mask)
        ):
            if mask == 0 or (start == 0 and end == 0):
                # End current slot if any
                if current_slot:
                    slots.append({
                        "label": current_slot,
                        "start": current_start,
                        "end": current_end,
                        "text": text[current_start:current_end]
                    })
                    current_slot = None
                continue
            
            label = self.id2slot[int(pred_id)]
            
            if label.startswith("B-"):
                # End current slot if any
                if current_slot:
                    slots.append({
                        "label": current_slot,
                        "start": current_start,
                        "end": current_end,
                        "text": text[current_start:current_end]
                    })
                
                # Start new slot
                current_slot = label[2:]  # Remove B- prefix
                current_start = int(start)
                current_end = int(end)
            
            elif label.startswith("I-") and current_slot == label[2:]:
                # Continue current slot
                current_end = int(end)
            
            else:
                # O label or mismatched I-label
                if current_slot:
                    slots.append({
                        "label": current_slot,
                        "start": current_start,
                        "end": current_end,
                        "text": text[current_start:current_end]
                    })
                    current_slot = None
        
        # Don't forget the last slot
        if current_slot:
            slots.append({
                "label": current_slot,
                "start": current_start,
                "end": current_end,
                "text": text[current_start:current_end]
            })
        
        return slots
    
    def batch_predict(self, texts: List[str]) -> List[NLUResult]:
        """Run inference on multiple texts."""
        
        # Tokenize all texts
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="np"
        )
        
        offset_mappings = encodings.pop("offset_mapping")
        
        # Run inference
        outputs = self.session.run(
            None,
            {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
            }
        )
        
        intent_logits, slot_logits = outputs
        
        # Process each result
        results = []
        for i, text in enumerate(texts):
            intent_probs = softmax(intent_logits[i])
            intent_id = int(np.argmax(intent_probs))
            intent = self.id2intent[intent_id]
            intent_confidence = float(intent_probs[intent_id])
            
            slot_preds = np.argmax(slot_logits[i], axis=-1)
            slots = self._extract_slots(
                text, slot_preds, offset_mappings[i], encodings["attention_mask"][i]
            )
            
            results.append(NLUResult(
                text=text,
                intent=intent,
                intent_confidence=intent_confidence,
                slots=slots
            ))
        
        return results

def softmax(x):
    """Compute softmax values."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def interactive_mode(nlu: NLUInference):
    """Interactive testing mode."""
    
    print("\n" + "="*60)
    print("Interactive NLU Testing")
    print("="*60)
    print("Enter text to analyze. Type 'quit' to exit.\n")
    
    while True:
        try:
            text = input(">>> ").strip()
            if not text:
                continue
            if text.lower() in ("quit", "exit", "q"):
                break
            
            result = nlu.predict(text)
            print(f"\n{result}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("Goodbye!")

def benchmark(nlu: NLUInference, num_iterations: int = 100):
    """Benchmark inference speed."""
    
    import time
    
    test_texts = [
        "Schedule a meeting with John tomorrow at 3pm",
        "What's on my calendar this week?",
        "Send an email to Sarah about the project",
        "Add Mike as a contact at Acme Corp",
        "Who works at TechCo?",
    ]
    
    print(f"\nBenchmarking {num_iterations} iterations...")
    
    # Warmup
    for text in test_texts:
        nlu.predict(text)
    
    # Benchmark single inference
    start = time.perf_counter()
    for _ in range(num_iterations):
        for text in test_texts:
            nlu.predict(text)
    single_time = time.perf_counter() - start
    
    total_inferences = num_iterations * len(test_texts)
    print(f"Single inference:")
    print(f"  Total: {single_time*1000:.2f} ms for {total_inferences} inferences")
    print(f"  Per inference: {single_time*1000/total_inferences:.2f} ms")
    
    # Benchmark batch inference
    start = time.perf_counter()
    for _ in range(num_iterations):
        nlu.batch_predict(test_texts)
    batch_time = time.perf_counter() - start
    
    print(f"Batch inference ({len(test_texts)} texts/batch):")
    print(f"  Total: {batch_time*1000:.2f} ms for {total_inferences} inferences")
    print(f"  Per inference: {batch_time*1000/total_inferences:.2f} ms")
    print(f"  Speedup: {single_time/batch_time:.2f}x")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./output/model.onnx")
    parser.add_argument("--config", type=str, default="./output/config.json")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    
    # Load model
    nlu = NLUInference(
        model_path=Path(args.model),
        config_path=Path(args.config),
        tokenizer_path=Path(args.tokenizer) if args.tokenizer else None
    )
    
    if args.benchmark:
        benchmark(nlu)
    elif args.interactive:
        interactive_mode(nlu)
    elif args.text:
        result = nlu.predict(args.text)
        print(f"\nInput: {args.text}")
        print(result)
    else:
        # Run some test examples
        test_examples = [
            "Schedule a meeting with John tomorrow at 3pm in the conference room",
            "What meetings do I have next week?",
            "Cancel my 2pm appointment on Friday",
            "Send an email to Sarah about the quarterly report",
            "Reply to Mike's email with thanks for the update",
            "Who was in my last meeting?",
            "Add Bob as a colleague at Acme Corp",
            "What do I know about TechCo?",
            "How are you doing today?",
            "Set a reminder for 5pm",
        ]
        
        print("\n" + "="*60)
        print("Test Examples")
        print("="*60)
        
        for text in test_examples:
            result = nlu.predict(text)
            print(f"\nInput: {text}")
            print(result)

if __name__ == "__main__":
    main()
