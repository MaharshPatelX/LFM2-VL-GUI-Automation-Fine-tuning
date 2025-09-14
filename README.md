# LFM2-VL GUI Automation Fine-tuning

This project fine-tunes the LiquidAI LFM2-VL vision-language model for GUI automation tasks using the realGUI-800K dataset.

## Features

- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation)
- **Vision-Language Model**: Works with the LFM2-VL model for understanding GUI screenshots
- **Dataset Support**: Uses the realGUI-800K dataset for training GUI automation tasks
- **Automatic Splitting**: Automatically splits data into training/validation sets
- **Hugging Face Integration**: Easy model saving and sharing via Hugging Face Hub

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- At least 16GB GPU memory for the 1.6B model, 8GB for the 450M model

## Installation

### Option 1: Using pip

```bash
# Clone the repository
git clone <your-repo-url>
cd lfm2-vl-gui-training

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
# Create a new conda environment
conda create -n lfm2-gui python=3.10
conda activate lfm2-gui

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Option 3: Development installation

```bash
pip install -e .
```

## Usage

### Basic Usage

```bash
python train_lfm2_gui.py
```

### Customization

The script can be customized by modifying the parameters in the `main()` function or by creating a configuration file.

#### Model Selection

You can choose between different model sizes:

```python
# In the script, modify:
model, processor = load_model_and_processor("LiquidAI/LFM2-VL-450M")  # Smaller, faster
# or
model, processor = load_model_and_processor("LiquidAI/LFM2-VL-1.6B")  # Larger, more capable
```

#### Training Configuration

Key training parameters can be adjusted in the `create_training_config()` function:

- `num_train_epochs`: Number of training epochs (default: 1)
- `per_device_train_batch_size`: Batch size per device (default: 1)
- `gradient_accumulation_steps`: Steps to accumulate gradients (default: 16)
- `learning_rate`: Learning rate (default: 5e-4)

#### LoRA Configuration

LoRA parameters can be modified in `setup_lora_config()`:

- `r`: LoRA rank (default: 8)
- `lora_alpha`: LoRA scaling parameter (default: 16)
- `lora_dropout`: Dropout rate (default: 0.05)

## Dataset

The script uses the `maharshpatelx/realGUI-800K` dataset by default. This dataset contains:

- GUI screenshots
- Natural language questions about the GUI
- Corresponding answers for automation tasks

The dataset is automatically downloaded and split into 80% training and 20% evaluation.

## Model Architecture

The training uses:

- **Base Model**: LiquidAI LFM2-VL (450M or 1.6B parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Task Type**: Vision-language understanding for GUI automation
- **Training Method**: Supervised Fine-Tuning (SFT)

## Output

After training, the script will:

1. Save the trained model locally to `./lfm2-vl-gui/`
2. Optionally push the model to Hugging Face Hub
3. Save training logs and checkpoints

## Hugging Face Hub Integration

To push your trained model to Hugging Face Hub:

1. First, login to Hugging Face:
   ```bash
   huggingface-cli login
   ```

2. The script will ask if you want to push to the hub after training completes.

## Memory Requirements

| Model Size | GPU Memory (Training) | GPU Memory (Inference) |
|------------|----------------------|------------------------|
| 450M       | ~8-12 GB            | ~2-4 GB               |
| 1.6B       | ~16-24 GB           | ~6-8 GB               |

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:

1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Enable `gradient_checkpointing=True` (already enabled by default)
4. Use a smaller model (450M instead of 1.6B)

### Slow Training

1. Ensure you're using a CUDA-capable GPU
2. Use mixed precision training (already enabled with `bfloat16`)
3. Increase batch size if memory allows
4. Use multiple GPUs with `device_map="auto"`

### Dataset Loading Issues

1. Ensure you have internet connection for first-time dataset download
2. Check if you have enough disk space (dataset is several GB)
3. Clear Hugging Face cache if needed: `rm -rf ~/.cache/huggingface/`

## File Structure

```
lfm2-vl-gui-training/
├── train_lfm2_gui.py      # Main training script
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
├── README.md              # This file
└── lfm2-vl-gui/          # Output directory (created after training)
    ├── config.json
    ├── pytorch_model.bin
    ├── preprocessor_config.json
    └── tokenizer files...
```

## Contributing

Feel free to submit issues and pull requests to improve the training script.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LiquidAI for the LFM2-VL model
- Hugging Face for the transformers library and model hub
- TRL for the training utilities
- The creators of the realGUI-800K dataset