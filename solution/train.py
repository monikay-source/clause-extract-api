#!/usr/bin/env python3
"""
Fine-tune the base model at /opt/base_model with LoRA on
/app/data/train_clauses.jsonl and save the adapter to /app/model/adapter/.

Run with no arguments:  python3 /app/train.py
"""
import json
import os
import random

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

BASE_MODEL_PATH = "/opt/base_model"
DATA_PATH = "/app/data/train_clauses.jsonl"
ADAPTER_DIR = "/app/model/adapter"
SEED = 42

# The exact prompt/target format used both here and at serve time. Persisted to
# training_config.json so serve.py never has to guess or duplicate this string.
PROMPT_TEMPLATE = "### Contract Clause:\n{text}\n### Extraction:\n"
TARGET_TEMPLATE = '{{"clause_type": "{clause_type}", "extracted_value": "{extracted_value}"}}'
STOP_STRING = "\n### Contract Clause:"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_records(path: str):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_examples(records, tokenizer, max_length: int = 256):
    input_ids_list, labels_list, attn_list = [], [], []
    for r in records:
        prompt = PROMPT_TEMPLATE.format(text=r["text"])
        target = TARGET_TEMPLATE.format(
            clause_type=r["clause_type"], extracted_value=r["extracted_value"]
        )
        full = prompt + target + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            full, add_special_tokens=False, truncation=True, max_length=max_length
        )["input_ids"]

        labels = list(full_ids)
        # Mask the prompt portion so loss is only computed on the target JSON.
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        attn = [1] * len(full_ids)
        input_ids_list.append(full_ids)
        labels_list.append(labels)
        attn_list.append(attn)

    return Dataset.from_dict(
        {"input_ids": input_ids_list, "attention_mask": attn_list, "labels": labels_list}
    )


class PadCollator:
    """Pads input_ids/attention_mask/labels to the longest sequence in the batch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main():
    set_seed(SEED)
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_records(DATA_PATH)
    print(f"Loaded {len(records)} training records")

    dataset = build_examples(records, tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],  # GPT-2-style attention projection
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="/tmp/train_out",
        num_train_epochs=6,
        per_device_train_batch_size=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    trainer.train()

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    with open(os.path.join(ADAPTER_DIR, "training_config.json"), "w") as f:
        json.dump(
            {
                "prompt_template": PROMPT_TEMPLATE,
                "target_template": TARGET_TEMPLATE,
                "stop_string": STOP_STRING,
                "base_model_path": BASE_MODEL_PATH,
            },
            f,
            indent=2,
        )

    print(f"Saved adapter + tokenizer + training_config.json to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
