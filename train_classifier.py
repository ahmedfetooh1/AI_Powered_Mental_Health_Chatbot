import os
import json
from typing import List, Dict
from dataclasses import dataclass
from sklearn.metrics import classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,Trainer, TrainingArguments, DataCollatorWithPadding)
from datasets import Dataset

MODEL_NAME = "aubmindlab/bert-base-arabertv2"  # يدعم العربية
LABEL2ID = {"normal": 0, "concern": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

@dataclass
class Example:
    text: str
    label: str

def load_jsonl(path: str) -> List[Example]:
    items: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            items.append(Example(text=obj["text"], label=obj["label"]))
    return items

def to_hf_ds(examples: List[Example], tok):
    def gen():
        for ex in examples:
            yield {"text": ex.text, "label": LABEL2ID[ex.label]}
    ds = Dataset.from_generator(gen)
    def tokenize(batch: Dict[str, List[str]]):
        return tok(batch["text"], truncation=True)
    return ds.map(tokenize, batched=True)

def main():
    # ضع ملفاتك: train.jsonl / eval.jsonl ، كل سطر: {"text": "...", "label": "normal|concern"}
    train = load_jsonl("train.jsonl")
    eval_ = load_jsonl("eval.jsonl")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = to_hf_ds(train, tok)
    eval_ds = to_hf_ds(eval_, tok)

    data_collator = DataCollatorWithPadding(tokenizer=tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    args = TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        num_train_epochs=2,
        logging_steps=50
    )

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        report = classification_report(p.label_ids, preds, target_names=["normal","concern"], output_dict=True, zero_division=0)
        return {
            "precision_concern": report["concern"]["precision"],
            "recall_concern": report["concern"]["recall"],
            "f1_concern": report["concern"]["f1-score"],
            "macro_f1": report["macro avg"]["f1-score"]
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
    trainer.evaluate()
    trainer.save_model("model")
    tok.save_pretrained("model")

if __name__ == "__main__":
    main()
