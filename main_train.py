#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-efficient SFT for LiquidAI/LFM2-VL on realGUI-800K.
Static configuration version: no argparse, configs below as variables.
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# --------------------------
# Static configuration
# --------------------------
MODEL_ID     = "LiquidAI/LFM2-VL-450M"
DATASET_ID   = "maharshpatelx/realGUI-800K"
OUT_DIR      = "lfm2-vl-gui"

EPOCHS       = 1
PER_DEVICE_BS = 1
GRAD_ACCUM    = 16
LR            = 5e-4
IMAGE_SIZE    = 512

WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01
LOGGING_STEPS = 10
SAVE_STEPS    = 1000
EVAL_STEPS    = 1000
SAVE_TOTAL_LIMIT = 2
SEED          = 42
USE_BF16      = False    # Set True if GPU supports BF16 (e.g., A100, H100)
DISABLE_WANDB = True
PUSH_TO_HUB   = False
HUB_REPO      = ""       # e.g., "yourname/lfm2-vl-gui"

# --------------------------
# Environment setup
# --------------------------
if DISABLE_WANDB:
    os.environ["WANDB_DISABLED"] = "true"

torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if USE_BF16 and torch.cuda.is_bf16_supported() else torch.float16

print(f"▶️  Using device: {device}, dtype: {dtype}")
print(f"▶️  Loading processor: {MODEL_ID}")

# --------------------------
# Load model & processor
# --------------------------
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    max_image_tokens=256,
)

print("▶️  Loading model…")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    trust_remote_code=True,
    device_map="auto",
)

# --------------------------
# Dataset loading (Arrow, not Python list)
# --------------------------
print(f"▶️  Loading dataset: {DATASET_ID}")
raw = load_dataset(DATASET_ID)
ds = raw["train"]
split = ds.train_test_split(test_size=0.20, seed=SEED)
train_ds = split["train"]
eval_ds  = split["test"]

print(f"✅ Dataset loaded: train={len(train_ds)}, eval={len(eval_ds)}")

# --------------------------
# Lazy message formatting
# --------------------------
system_message = (
    "You are a GUI automation assistant specialized in understanding user "
    "interfaces and providing guidance on GUI interactions. Analyze the "
    "screenshot and provide accurate responses about GUI elements, actions, "
    "or navigation tasks."
)

def to_messages(example):
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text",  "text": example["question"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]},
        ],
        "image": example["image"],
    }

train_ds = train_ds.map(to_messages, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(to_messages,  remove_columns=eval_ds.column_names)

# --------------------------
# Collate: decode images only per batch
# --------------------------
def collate_fn(batch):
    texts  = [ex["messages"] for ex in batch]
    images = [ex["image"] for ex in batch]

    enc = processor.apply_chat_template(
        texts,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    labels = enc["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    enc["labels"] = labels

    vision = processor(
        images,
        return_tensors="pt",
        size=IMAGE_SIZE,
    )
    enc["pixel_values"] = vision.get("pixel_values")

    return enc

# --------------------------
# PEFT: LoRA
# --------------------------
target_modules = [
    "q_proj", "v_proj", "fc1", "fc2", "linear",
    "gate_proj", "up_proj", "down_proj",
]

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=target_modules,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# --------------------------
# TRL SFT config
# --------------------------
sft_cfg = SFTConfig(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BS,
    per_device_eval_batch_size=PER_DEVICE_BS,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    logging_steps=LOGGING_STEPS,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    optim="adamw_torch_8bit",
    gradient_checkpointing=True,
    fp16=(dtype == torch.float16),
    bf16=(dtype == torch.bfloat16),
    max_length=5000,
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    report_to=None if DISABLE_WANDB else "wandb",
    seed=SEED,
)

# --------------------------
# Trainer
# --------------------------
print("🏗️  Building SFTTrainer…")
trainer = SFTTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collate_fn,
    processing_class=processor.tokenizer,
)

print("🚀 Starting training…")
trainer.train()
print("🎉 Training done.")

# --------------------------
# Save merged model
# --------------------------
if hasattr(model, "peft_config"):
    print("🔄 Merging LoRA weights…")
    model = model.merge_and_unload()

os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR)
processor.save_pretrained(OUT_DIR)
print(f"💾 Saved model to: {OUT_DIR}")

if PUSH_TO_HUB:
    repo = HUB_REPO.strip() or os.path.basename(os.path.abspath(OUT_DIR))
    model.push_to_hub(repo)
    processor.push_to_hub(repo)
    print(f"☁️  Pushed to Hub: {repo}")

print("✅ All done.")
