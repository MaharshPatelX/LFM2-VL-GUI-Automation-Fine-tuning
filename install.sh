#!/bin/bash

# LFM2-VL GUI Training Setup Script

echo "🚀 Setting up LFM2-VL GUI Training Environment"
echo "=============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $python_version"

if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
    echo "⚠️  Python 3.8+ is recommended. Current version: $python_version"
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv lfm2_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source lfm2_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (adjust for your CUDA version)
echo "🔥 Installing PyTorch with CUDA support..."
if command -v nvidia-smi &> /dev/null; then
    # CUDA is available
    cuda_version=$(nvidia-smi | grep -Po "CUDA Version: \K[\d.]+")
    echo "   🎮 CUDA Version detected: $cuda_version"
    
    if [[ $cuda_version == 11.* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [[ $cuda_version == 12.* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo "⚠️  Unsupported CUDA version. Installing CPU version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "⚠️  CUDA not detected. Installing CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "📚 Installing other dependencies..."
pip install -r requirements.txt

# Install development version if setup.py exists
if [ -f "setup.py" ]; then
    echo "🛠️  Installing package in development mode..."
    pip install -e .
fi

# Verify installation
echo "✅ Verifying installation..."
python3 -c "
import torch
import transformers
import trl
import peft
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'TRL: {trl.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.get_device_name()}')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To activate the environment in future sessions:"
echo "   source lfm2_env/bin/activate"
echo ""
echo "To run training:"
echo "   python train_lfm2_gui.py"
echo "   # or with config file:"
echo "   python run_training.py --config config.yaml"
echo ""
echo "To login to Hugging Face (for pushing models):"
echo "   huggingface-cli login"
echo ""

# Keep environment activated for user
echo "🔓 Virtual environment is now activated!"
exec "$SHELL"