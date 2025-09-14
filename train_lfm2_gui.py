#!/usr/bin/env python3
"""
LFM2-VL GUI Automation Fine-tuning Script

This script fine-tunes the LiquidAI LFM2-VL model for GUI automation tasks
using the realGUI-800K dataset.
"""

import os
import torch
import transformers
import trl
from typing import List, Tuple, Dict, Any
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from huggingface_hub import login


def setup_environment():
    """Set up environment variables and disable wandb."""
    os.environ["WANDB_DISABLED"] = "true"
    
    print(f"📦 PyTorch version: {torch.__version__}")
    print(f"🤗 Transformers version: {transformers.__version__}")
    print(f"📊 TRL version: {trl.__version__}")


def load_model_and_processor(model_id: str = "LiquidAI/LFM2-VL-450M"):
    """Load the LFM2-VL model and processor.
    
    Args:
        model_id: The model ID to load from Hugging Face Hub
        
    Returns:
        Tuple of (model, processor)
    """
    print("📚 Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        max_image_tokens=256,
    )

    print("🧠 Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype="bfloat16",
        trust_remote_code=True,
        device_map="auto",
    )

    print("\n✅ Local model loaded successfully!")
    print(f"📖 Vocab size: {len(processor.tokenizer)}")
    print(f"🖼️ Image processed in up to {processor.max_tiles} patches of size {processor.tile_size}")
    print(f"🔢 Parameters: {model.num_parameters():,}")
    print(f"💾 Model size: ~{model.num_parameters() * 2 / 1e9:.1f} GB (bfloat16)")
    
    return model, processor


def load_and_prepare_dataset(dataset_name: str = "maharshpatelx/realGUI-800K"):
    """Load and split the dataset for training.
    
    Args:
        dataset_name: The dataset name on Hugging Face Hub
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    print(f"📥 Loading dataset: {dataset_name}")
    raw_ds = load_dataset(dataset_name)
    full_dataset = raw_ds["train"]
    
    # Split the dataset
    split = full_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print("✅ Dataset loaded:")
    print(f"   📚 Train samples: {len(train_dataset)}")
    print(f"   🧪 Eval samples: {len(eval_dataset)}")
    print(f"   📝 Dataset columns: {train_dataset.column_names}")
    
    return train_dataset, eval_dataset


def format_gui_sample(sample: Dict[str, Any]) -> Tuple[List[Dict], Any]:
    """Format a single GUI sample for training.
    
    Args:
        sample: A single sample from the dataset
        
    Returns:
        Tuple of (formatted_conversation, image)
    """
    system_message = (
        "You are a GUI automation assistant specialized in understanding user interfaces and providing guidance on GUI interactions. "
        "Analyze the screenshot and provide accurate responses about GUI elements, actions, or navigation tasks."
    )
    
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["question"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
    ]
    
    return conversation, sample["image"]


def prepare_datasets(train_dataset, eval_dataset):
    """Apply formatting to both training and evaluation datasets.
    
    Args:
        train_dataset: Raw training dataset
        eval_dataset: Raw evaluation dataset
        
    Returns:
        Tuple of formatted datasets
    """
    print("🔄 Formatting datasets...")
    train_formatted = [format_gui_sample(sample) for sample in train_dataset]
    eval_formatted = [format_gui_sample(sample) for sample in eval_dataset]
    
    print("✅ Datasets formatted:")
    print(f"   📚 Train samples: {len(train_formatted)}")
    print(f"   🧪 Eval samples: {len(eval_formatted)}")
    
    return train_formatted, eval_formatted


def create_collate_fn(processor):
    """Create a collate function for batch processing.
    
    Args:
        processor: The model processor
        
    Returns:
        Collate function for DataLoader
    """
    def collate_fn(samples):
        texts, images = zip(*samples)
        batch = processor.apply_chat_template(
            texts, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch
    
    return collate_fn


def setup_lora_config():
    """Set up LoRA configuration for efficient fine-tuning.
    
    Returns:
        LoRA configuration object
    """
    target_modules = [
        "q_proj", "v_proj", "fc1", "fc2", "linear",
        "gate_proj", "up_proj", "down_proj",
    ]

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    
    return peft_config


def create_training_config(output_dir: str = "lfm2-vl-gui"):
    """Create training configuration for SFT.
    
    Args:
        output_dir: Directory to save the model
        
    Returns:
        SFT configuration object
    """
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        optim="adamw_torch_8bit",
        gradient_checkpointing=True,
        max_length=5000,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=None
    )


def train_model(model, processor, train_dataset, eval_dataset, output_dir: str = "lfm2-vl-gui"):
    """Train the model using SFT.
    
    Args:
        model: The model to train
        processor: The model processor
        train_dataset: Formatted training dataset
        eval_dataset: Formatted evaluation dataset
        output_dir: Directory to save the model
    """
    # Setup LoRA
    print("🔧 Setting up LoRA...")
    peft_config = setup_lora_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Create collate function and training config
    collate_fn = create_collate_fn(processor)
    sft_config = create_training_config(output_dir)
    
    print("🏗️  Creating SFT trainer...")
    sft_trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
    )

    print("\n🚀 Starting SFT training...")
    sft_trainer.train()
    print("🎉 SFT training completed!")

    # Save the model
    sft_trainer.save_model()
    print(f"💾 Training artifacts saved to: {sft_config.output_dir}")
    
    return model


def save_and_push_model(model, processor, output_dir: str = "./lfm2-vl-gui", 
                       hub_model_name: str = "maharshpatelx/lfm2-vl-gui"):
    """Save the model locally and optionally push to Hugging Face Hub.
    
    Args:
        model: The trained model
        processor: The model processor
        output_dir: Local directory to save the model
        hub_model_name: Name for the model on Hugging Face Hub
    """
    # Merge LoRA weights if applicable
    if hasattr(model, 'peft_config'):
        print("🔄 Merging LoRA weights...")
        model = model.merge_and_unload()
    
    # Save model and processor locally
    print(f"💾 Saving model to: {output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    # Ask user if they want to push to hub
    push_to_hub = input("\n🤗 Push to Hugging Face Hub? (y/N): ").lower().strip()
    
    if push_to_hub == 'y':
        try:
            # Note: User needs to run `huggingface-cli login` before this
            print("📤 Pushing to Hugging Face Hub...")
            model.push_to_hub(hub_model_name)
            processor.push_to_hub(hub_model_name)
            print(f"✅ Model pushed to: {hub_model_name}")
        except Exception as e:
            print(f"❌ Failed to push to hub: {e}")
            print("💡 Make sure you're logged in with: huggingface-cli login")


def main():
    """Main training pipeline."""
    try:
        # Setup
        setup_environment()
        
        # Load model and processor
        model, processor = load_model_and_processor()
        
        # Load and prepare dataset
        train_raw, eval_raw = load_and_prepare_dataset()
        train_formatted, eval_formatted = prepare_datasets(train_raw, eval_raw)
        
        # Train the model
        trained_model = train_model(model, processor, train_formatted, eval_formatted)
        
        # Save and optionally push to hub
        save_and_push_model(trained_model, processor)
        
        print("\n🎉 Training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()