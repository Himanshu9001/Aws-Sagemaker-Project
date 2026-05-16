# Fine-tune Flan-T5-Large on telecom retention Q&A dataset
# Uses HuggingFace Transformers + SageMaker Training Job
# Input: telecom Q&A pairs
# Output: fine-tuned model saved to /opt/ml/model/

import os
import json
import torch
import argparse
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from torch.utils.data import Dataset

# SageMaker paths
INPUT_DIR = "/opt/ml/input/data/train"
MODEL_DIR = "/opt/ml/model"


class RetentionDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len=128, max_target_len=32):
        self.data         = data
        self.tokenizer    = tokenizer
        self.max_input    = max_input_len
        self.max_target   = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        inputs = self.tokenizer(
            item["input"],
            max_length=self.max_input,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer(
            item["output"],
            max_length=self.max_target,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = targets["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels":         labels,
        }


def train(args):
    print(f"Loading Flan-T5-Large...")
    model_name = "google/flan-t5-large"
    tokenizer  = T5Tokenizer.from_pretrained(model_name)
    model      = T5ForConditionalGeneration.from_pretrained(model_name)

    # Load dataset
    data_path = os.path.join(INPUT_DIR, "train_data.json")
    with open(data_path) as f:
        data = json.load(f)

    print(f"Dataset size: {len(data)} examples")

    # Split 80/20
    split      = int(len(data) * 0.8)
    train_data = data[:split]
    eval_data  = data[split:]

    train_dataset = RetentionDataset(train_data, tokenizer)
    eval_dataset  = RetentionDataset(eval_data, tokenizer)

    # Training arguments — optimized for small dataset
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=10,
        weight_decay=0.01,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=5,
        fp16=False,  # CPU training — no fp16
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print(f"Starting fine-tuning...")
    trainer.train()

    # Save model + tokenizer
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Fine-tuned model saved to {MODEL_DIR}")

    # Quick inference test
    test_input = "What is the best offer to prevent churn for a telecom customer on month-to-month contract paying $85 monthly with 3 months tenure and 68% churn risk?"
    inputs  = tokenizer(test_input, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=30)
    result  = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nTest inference after fine-tuning:")
    print(f"Input:  {test_input}")
    print(f"Output: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=5)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    args = parser.parse_args()
    train(args)
