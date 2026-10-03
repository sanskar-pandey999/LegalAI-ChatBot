#!/usr/bin/env python3
"""
Fine-tune openai/gpt-oss-120b using QLoRA + bitsandbytes + PEFT.

Dataset format: JSONL or JSON
Each record must have:
    {"prompt": "...", "response": "..."}

Save your dataset at: data/train.jsonl
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

# -------------------
# Config
# -------------------
MODEL_NAME = "tiiuae/falcon-7b-instruct"   # base model
DATA_PATH = "train.json"       # placeholder dataset path
OUTPUT_DIR = "outputs/qlora-gptoss"  # save dir
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LEN = 2048

# -------------------
# Load dataset
# -------------------
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH)

# We expect fields: "prompt" and "response"
def format_example(example):
    return {"text": f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}"}

dataset = dataset.map(format_example)

# -------------------
# Load model & tokenizer
# -------------------
print("Loading base model + tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length"
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

# -------------------
# PEFT + QLoRA config
# -------------------
print("Configuring QLoRA...")
bnb_config = dict(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    **bnb_config
)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # key attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------
# Trainer setup
# -------------------
print("Preparing Trainer...")
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=NUM_EPOCHS,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="no",
    save_total_limit=2,
    bf16=True,
    optim="paged_adamw_32bit",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# -------------------
# Train
# -------------------
print("Starting training...")
trainer.train()

# -------------------
# Save
# -------------------
print("Saving final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved at {OUTPUT_DIR}")
