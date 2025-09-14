#!/usr/bin/env python3
"""
Configuration-based training script for LFM2-VL GUI Automation

This script loads configuration from config.yaml and runs the training pipeline.
"""

import yaml
import argparse
from pathlib import Path
from train_lfm2_gui import (
    setup_environment, load_model_and_processor, load_and_prepare_dataset,
    prepare_datasets, train_model, save_and_push_model
)


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """Main training pipeline using configuration file."""
    parser = argparse.ArgumentParser(description="Train LFM2-VL for GUI automation")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--model-id", help="Override model ID from config")
    parser.add_argument("--output-dir", help="Override output directory from config")
    parser.add_argument("--epochs", type=int, help="Override number of epochs from config")
    parser.add_argument("--batch-size", type=int, help="Override batch size from config")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate from config")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print(f"📋 Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Override with command line arguments if provided
        if args.model_id:
            config['model']['model_id'] = args.model_id
        if args.output_dir:
            config['training']['output_dir'] = args.output_dir
            config['output']['local_dir'] = args.output_dir
        if args.epochs:
            config['training']['num_train_epochs'] = args.epochs
        if args.batch_size:
            config['training']['per_device_train_batch_size'] = args.batch_size
        if args.learning_rate:
            config['training']['learning_rate'] = args.learning_rate
        
        print("🔧 Configuration loaded successfully!")
        print(f"   📦 Model: {config['model']['model_id']}")
        print(f"   📊 Dataset: {config['dataset']['name']}")
        print(f"   🏋️ Epochs: {config['training']['num_train_epochs']}")
        print(f"   📁 Output: {config['training']['output_dir']}")
        
        # Setup environment
        setup_environment()
        
        # Load model and processor
        model, processor = load_model_and_processor(config['model']['model_id'])
        
        # Load and prepare dataset
        train_raw, eval_raw = load_and_prepare_dataset(config['dataset']['name'])
        train_formatted, eval_formatted = prepare_datasets(train_raw, eval_raw)
        
        # Train the model with config parameters
        trained_model = train_model(
            model=model,
            processor=processor,
            train_dataset=train_formatted,
            eval_dataset=eval_formatted,
            output_dir=config['training']['output_dir']
        )
        
        # Save and optionally push to hub
        save_and_push_model(
            model=trained_model,
            processor=processor,
            output_dir=config['output']['local_dir'],
            hub_model_name=config['output']['hub_model_name']
        )
        
        print("\n🎉 Training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()