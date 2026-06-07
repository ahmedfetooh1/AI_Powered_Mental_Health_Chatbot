import os
import json
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)
from datasets import Dataset

# --- 1. Update Labels ---
MODEL_NAME = "aubmindlab/bert-base-arabertv2"  # Supports Arabic

# New list of labels (6 classifications)
TARGET_LABELS = ['normal', 'depression', 'suicidal', 'anxiety', 'bipolar', 'stress']
NUM_LABELS = len(TARGET_LABELS) # 6

# Assign IDs to labels
LABEL2ID = {label: i for i, label in enumerate(TARGET_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(TARGET_LABELS)}

@dataclass
class Example:
    text: str
    label: str

def load_jsonl(path: str) -> List[Example]:
    items: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["label"] in LABEL2ID:
                items.append(Example(text=obj["text"], label=obj["label"]))
    return items

def main():
    # Load the tokenizer and model
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load the model with the new labels
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Load data from the new split files
    # Note: 'train.jsonl' is used for training, 'eval.jsonl' is used for evaluation
    train_items = load_jsonl("train.jsonl")
    eval_items = load_jsonl("eval.jsonl")

    # Convert to datasets.Dataset
    train_data = [{"text": item.text, "label": LABEL2ID[item.label]} for item in train_items]
    eval_data = [{"text": item.text, "label": LABEL2ID[item.label]} for item in eval_items]
    
    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)

    # Tokenization function
    def tokenize_function(examples):
        return tok(examples["text"], truncation=True, max_length=512)

    # Apply tokenization
    train_ds = train_ds.map(tokenize_function, batched=True)
    eval_ds = eval_ds.map(tokenize_function, batched=True)

    # Data collator for padding
    data_collator = DataCollatorWithPadding(tokenizer=tok)

    # Training arguments
    args = TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        num_train_epochs=2,
        logging_steps=50,
    )

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        target_names = list(LABEL2ID.keys())
        
        report = classification_report(
            p.label_ids, 
            preds, 
            target_names=target_names, 
            output_dict=True, 
            zero_division=0
        )
        
        return {
            "accuracy": accuracy_score(p.label_ids, preds),
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"]
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save the final model
    model.save_pretrained("./final_model")
    tok.save_pretrained("./final_model")
    
    # Save a file containing the new labels for 'app.py' to use
    with open("./final_model/labels.json", "w", encoding="utf-8") as f:
        json.dump(ID2LABEL, f, ensure_ascii=False)

if __name__ == "__main__":
    main()