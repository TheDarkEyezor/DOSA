#!/usr/bin/env python3
"""
Train DistilBERT for joint intent classification and slot filling.

This implements the JointBERT architecture:
- Shared DistilBERT encoder
- Intent classification head on [CLS] token
- Slot filling head (token classification) on all tokens

Usage:
    python train.py --epochs 10 --batch_size 16 --model_dir ./output
"""

import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    DistilBertPreTrainedModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import TokenClassifierOutput
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Model Architecture
# =============================================================================

class JointDistilBERT(DistilBertPreTrainedModel):
    """
    Joint Intent Classification and Slot Filling model based on DistilBERT.
    
    Architecture:
        Input -> DistilBERT -> [CLS] token -> Intent Classifier
                            -> All tokens -> Slot Classifier (per token)
    """
    
    def __init__(self, config, num_intents: int, num_slots: int, slot_loss_coef: float = 1.0):
        super().__init__(config)
        
        self.num_intents = num_intents
        self.num_slots = num_slots
        self.slot_loss_coef = slot_loss_coef
        
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        
        # Intent classifier (from [CLS] token)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intents)
        
        # Slot classifier (token-level)
        self.slot_classifier = nn.Linear(config.hidden_size, num_slots)
        
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        intent_labels: Optional[torch.Tensor] = None,
        slot_labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        pooled_output = sequence_output[:, 0, :]     # [CLS] token
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        # Intent prediction
        intent_logits = self.intent_classifier(pooled_output)  # (batch, num_intents)
        
        # Slot prediction
        slot_logits = self.slot_classifier(sequence_output)    # (batch, seq_len, num_slots)
        
        # Calculate loss if labels provided
        loss = None
        if intent_labels is not None and slot_labels is not None:
            intent_loss_fn = nn.CrossEntropyLoss()
            slot_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            
            intent_loss = intent_loss_fn(intent_logits, intent_labels)
            slot_loss = slot_loss_fn(
                slot_logits.view(-1, self.num_slots),
                slot_labels.view(-1)
            )
            
            loss = intent_loss + self.slot_loss_coef * slot_loss
        
        return {
            "loss": loss,
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
        }

# =============================================================================
# Dataset
# =============================================================================

@dataclass
class NLUExample:
    text: str
    intent: str
    slots: List[Dict]  # [{start, end, label, text}, ...]

class NLUDataset(Dataset):
    def __init__(
        self,
        examples: List[NLUExample],
        tokenizer: DistilBertTokenizerFast,
        intent2id: Dict[str, int],
        slot2id: Dict[str, int],
        max_length: int = 128
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.intent2id = intent2id
        self.slot2id = slot2id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            example.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        # Get offset mapping for slot alignment
        offset_mapping = encoding.pop("offset_mapping")[0]
        
        # Create slot labels aligned to tokens
        slot_labels = self._align_slots_to_tokens(
            example.slots,
            offset_mapping,
            encoding["attention_mask"][0]
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "intent_labels": torch.tensor(self.intent2id[example.intent]),
            "slot_labels": slot_labels,
        }
    
    def _align_slots_to_tokens(
        self,
        slots: List[Dict],
        offset_mapping: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Align character-level slot annotations to tokenized positions."""
        
        labels = torch.full((len(offset_mapping),), -100, dtype=torch.long)
        
        for i, (start, end) in enumerate(offset_mapping.tolist()):
            if attention_mask[i] == 0:
                continue
            if start == 0 and end == 0:
                # Special token
                labels[i] = -100
                continue
            
            # Default to O (outside)
            labels[i] = self.slot2id["O"]
            
            # Check if this token overlaps with any slot
            for slot in slots:
                slot_start, slot_end = slot["start"], slot["end"]
                
                # Check overlap
                if start >= slot_start and end <= slot_end:
                    # Token is inside this slot
                    if start == slot_start:
                        # Beginning of slot
                        labels[i] = self.slot2id[f"B-{slot['label']}"]
                    else:
                        # Inside slot
                        labels[i] = self.slot2id[f"I-{slot['label']}"]
                    break
        
        return labels

# =============================================================================
# Data Loading
# =============================================================================

def load_examples(path: Path) -> List[NLUExample]:
    examples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            examples.append(NLUExample(
                text=data["text"],
                intent=data["intent"],
                slots=data["slots"]
            ))
    return examples

def load_labels(path: Path) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(eval_pred, id2intent, id2slot):
    """Compute intent accuracy and slot F1."""
    
    predictions, labels = eval_pred
    intent_logits, slot_logits = predictions
    intent_labels, slot_labels = labels
    
    # Convert to numpy if tensors
    if hasattr(intent_logits, 'numpy'):
        intent_logits = intent_logits.cpu().numpy()
    if hasattr(slot_logits, 'numpy'):
        slot_logits = slot_logits.cpu().numpy()
    if hasattr(intent_labels, 'numpy'):
        intent_labels = intent_labels.cpu().numpy()
    if hasattr(slot_labels, 'numpy'):
        slot_labels = slot_labels.cpu().numpy()
    
    # Intent accuracy
    intent_preds = np.argmax(intent_logits, axis=1)
    intent_acc = accuracy_score(intent_labels, intent_preds)
    
    # Slot F1 (using seqeval)
    slot_preds = np.argmax(slot_logits, axis=2)
    
    true_labels = []
    pred_labels = []
    
    for i in range(len(slot_labels)):
        true_seq = []
        pred_seq = []
        for j in range(len(slot_labels[i])):
            if slot_labels[i][j] != -100:
                true_seq.append(id2slot[slot_labels[i][j]])
                pred_seq.append(id2slot[slot_preds[i][j]])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)
    
    slot_f1 = f1_score(true_labels, pred_labels)
    slot_precision = precision_score(true_labels, pred_labels)
    slot_recall = recall_score(true_labels, pred_labels)
    
    # Semantic frame accuracy (both intent and all slots correct)
    frame_acc = 0
    for i in range(len(intent_labels)):
        if intent_preds[i] == intent_labels[i]:
            if true_labels[i] == pred_labels[i]:
                frame_acc += 1
    frame_acc /= len(intent_labels)
    
    return {
        "intent_accuracy": intent_acc,
        "slot_f1": slot_f1,
        "slot_precision": slot_precision,
        "slot_recall": slot_recall,
        "semantic_frame_accuracy": frame_acc,
    }

# =============================================================================
# Custom Trainer
# =============================================================================

class NLUTrainer(Trainer):
    """Custom trainer that handles our joint model output format."""
    
    def __init__(self, *args, id2intent=None, id2slot=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.id2intent = id2intent
        self.id2slot = id2slot
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs["loss"]
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Keep as tensors (not numpy) to avoid issues with pad_across_processes
        intent_logits = outputs["intent_logits"].detach()
        slot_logits = outputs["slot_logits"].detach()
        
        intent_labels = inputs["intent_labels"].detach()
        slot_labels = inputs["slot_labels"].detach()
        
        return (
            loss,
            (intent_logits, slot_logits),
            (intent_labels, slot_labels)
        )

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model_dir", type=str, default="./output")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--slot_loss_coef", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    data_dir = Path(args.data_dir)
    train_examples = load_examples(data_dir / "train.jsonl")
    val_examples = load_examples(data_dir / "val.jsonl")
    test_examples = load_examples(data_dir / "test.jsonl")
    
    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")
    
    # Load labels
    intents = load_labels(data_dir / "intents.txt")
    slot_labels = load_labels(data_dir / "slot_labels.txt")
    
    intent2id = {intent: i for i, intent in enumerate(intents)}
    id2intent = {i: intent for intent, i in intent2id.items()}
    slot2id = {label: i for i, label in enumerate(slot_labels)}
    id2slot = {i: label for label, i in slot2id.items()}
    
    print(f"Intents: {len(intents)}, Slot labels: {len(slot_labels)}")
    
    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = NLUDataset(train_examples, tokenizer, intent2id, slot2id, args.max_length)
    val_dataset = NLUDataset(val_examples, tokenizer, intent2id, slot2id, args.max_length)
    test_dataset = NLUDataset(test_examples, tokenizer, intent2id, slot2id, args.max_length)
    
    # Create model
    model = JointDistilBERT.from_pretrained(
        args.model_name,
        num_intents=len(intents),
        num_slots=len(slot_labels),
        slot_loss_coef=args.slot_loss_coef,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="semantic_frame_accuracy",
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Create trainer
    trainer = NLUTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        id2intent=id2intent,
        id2slot=id2slot,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Override compute_metrics
    def metrics_fn(eval_pred):
        return compute_metrics(eval_pred, id2intent, id2slot)
    trainer.compute_metrics = metrics_fn
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60 + "\n")
    
    test_results = trainer.evaluate(test_dataset)
    print("Test results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save model and config
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    
    model.save_pretrained(model_dir / "best_model")
    tokenizer.save_pretrained(model_dir / "best_model")
    
    # Save label mappings
    with open(model_dir / "config.json", "w") as f:
        json.dump({
            "intent2id": intent2id,
            "id2intent": {str(k): v for k, v in id2intent.items()},
            "slot2id": slot2id,
            "id2slot": {str(k): v for k, v in id2slot.items()},
            "max_length": args.max_length,
        }, f, indent=2)
    
    print(f"\nModel saved to {model_dir / 'best_model'}")
    print(f"Config saved to {model_dir / 'config.json'}")

if __name__ == "__main__":
    main()
