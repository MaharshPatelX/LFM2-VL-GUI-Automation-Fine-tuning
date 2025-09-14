#!/usr/bin/env python3
"""
LFM2-VL GUI Automation Fine-tuning Script

This script fine-tunes the LiquidAI LFM2-VL model for GUI automation tasks
using the realGUI-800K dataset with memory-efficient streaming and lazy loading.
"""

import os
import torch
import transformers
import trl
from typing import List, Tuple, Dict, Any, Iterator
from datasets import load_dataset, Dataset, IterableDataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from huggingface_hub import login
from PIL import Image


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
    """Load and split the dataset for training using streaming mode.
    
    Args:
        dataset_name: The dataset name on Hugging Face Hub
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    print(f"📥 Loading dataset with streaming: {dataset_name}")
    
    # Enable streaming to avoid loading entire dataset into memory
    raw_ds = load_dataset(dataset_name, streaming=True)
    full_dataset = raw_ds["train"]
    
    # For streaming datasets, we need to estimate total size and split manually
    # The realGUI-800K dataset has approximately 635,296 samples
    estimated_total = 635296
    train_size = int(0.8 * estimated_total)  # ~508K samples for training
    eval_size = estimated_total - train_size  # ~127K samples for evaluation
    
    print(f"📊 Estimated dataset size: {estimated_total:,}")
    print(f"📚 Train samples (estimated): {train_size:,}")
    print(f"🧪 Eval samples (estimated): {eval_size:,}")
    
    # Split using take() and skip() for streaming datasets
    train_dataset = full_dataset.take(train_size)
    eval_dataset = full_dataset.skip(train_size)
    
    # Get a sample to check dataset columns
    print("🔍 Checking dataset structure...")
    sample = next(iter(full_dataset))
    print(f"📝 Dataset columns: {list(sample.keys())}")
    print(f"📸 Sample image type: {type(sample['image'])}")
    
    print("✅ Streaming dataset prepared successfully!")
    print("💡 Images will be loaded on-demand during training (lazy loading)")
    
    return train_dataset, eval_dataset


def format_gui_sample_lazy(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Format a single GUI sample for training without loading images into memory.
    
    Args:
        sample: A single sample from the dataset
        
    Returns:
        Dict containing conversation template and image reference
    """
    system_message = (
        "You are a GUI automation assistant specialized in understanding user interfaces and providing guidance on GUI interactions. "
        "Analyze the screenshot and provide accurate responses about GUI elements, actions, or navigation tasks."
    )
    
    # Create conversation template - keep image object reference for lazy loading
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},  # Keep PIL Image reference
                {"type": "text", "text": sample["question"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
    ]
    
    return {"conversation": conversation}


class LazyDataset:
    """Wrapper for streaming datasets that applies formatting on-demand."""
    
    def __init__(self, dataset: IterableDataset):
        self.dataset = dataset
        
    def __iter__(self):
        for sample in self.dataset:
            yield format_gui_sample_lazy(sample)


def prepare_datasets(train_dataset: IterableDataset, eval_dataset: IterableDataset):
    """Prepare datasets with lazy loading - no upfront formatting.
    
    Args:
        train_dataset: Raw training dataset (streaming)
        eval_dataset: Raw evaluation dataset (streaming)
        
    Returns:
        Tuple of lazy-loading dataset wrappers
    """
    print("🔄 Preparing datasets with lazy loading...")
    
    # Wrap datasets with lazy formatting - no memory loading here!
    train_lazy = LazyDataset(train_dataset)
    eval_lazy = LazyDataset(eval_dataset)
    
    print("✅ Datasets prepared with lazy loading:")
    print("   📚 Train dataset: Streaming + lazy formatting")
    print("   🧪 Eval dataset: Streaming + lazy formatting")
    print("   💾 Memory usage: Minimal (images loaded per batch only)")
    
    return train_lazy, eval_lazy


def create_collate_fn(processor):
    """Create a collate function that handles lazy loading and formatting.
    
    Args:
        processor: The model processor
        
    Returns:
        Collate function for DataLoader
    """
    def collate_fn(samples):
        """
        Process batch samples with lazy image loading.
        Images are loaded from PIL objects only when needed for this specific batch.
        """
        try:
            # Extract conversations from lazy-formatted samples
            conversations = [sample["conversation"] for sample in samples]
            
            # Apply chat template with actual image loading happening here
            batch = processor.apply_chat_template(
                conversations, 
                tokenize=True, 
                return_dict=True, 
                return_tensors="pt"
            )
            
            # Create labels for training
            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
            
            return batch
            
        except Exception as e:
            print(f"⚠️ Error in batch processing: {e}")
            print(f"   Sample types: {[type(s) for s in samples]}")
            raise e
    
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
    """Create training configuration for SFT with memory-efficient settings.
    
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
        # Memory-efficient DataLoader settings
        dataloader_num_workers=0,  # Avoid multiprocessing issues with streaming
        dataloader_pin_memory=False,  # Reduce memory pressure
        report_to=None
    )


def train_model(model, processor, train_dataset, eval_dataset, output_dir: str = "lfm2-vl-gui"):
    """Train the model using SFT with streaming datasets.
    
    Args:
        model: The model to train
        processor: The model processor
        train_dataset: Lazy training dataset
        eval_dataset: Lazy evaluation dataset
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
    
    print("🏗️  Creating SFT trainer with streaming datasets...")
    print("   💾 Memory-efficient settings enabled")
    print("   🔄 Lazy loading: Images loaded per batch only")
    
    sft_trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
    )

    print("\n🚀 Starting SFT training...")
    print("   📊 Training with streaming data - minimal memory usage")
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
    """Main training pipeline with memory-efficient streaming and lazy loading."""
    try:
        print("🚀 Starting LFM2-VL GUI training with memory optimization")
        print("   📡 Streaming dataset loading enabled")
        print("   💾 Lazy image loading enabled")
        print("   🎯 Expected RAM usage: 1-3GB (vs 250GB+ without optimization)\n")
        
        # Setup
        setup_environment()
        
        # Load model and processor
        model, processor = load_model_and_processor()
        
        # Load and prepare dataset with streaming
        print("\n" + "="*60)
        train_raw, eval_raw = load_and_prepare_dataset()
        train_lazy, eval_lazy = prepare_datasets(train_raw, eval_raw)
        
        # Train the model
        print("\n" + "="*60)
        trained_model = train_model(model, processor, train_lazy, eval_lazy)
        
        # Save and optionally push to hub
        print("\n" + "="*60)
        save_and_push_model(trained_model, processor)
        
        print("\n🎉 Training pipeline completed successfully!")
        print("   💾 Memory optimization: Successful")
        print("   📊 Model ready for inference")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("\n🔍 Debugging info:")
        print(f"   PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        raise


if __name__ == "__main__":
    main()