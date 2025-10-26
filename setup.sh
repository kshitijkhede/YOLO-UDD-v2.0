#!/bin/bash

# YOLO-UDD v2.0 Setup Script
# Automated setup for the project environment

echo "========================================="
echo "YOLO-UDD v2.0 Setup Script"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python 3.8+ is required"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/trashcan/{images,annotations}/{train,val,test}
mkdir -p runs/{train,eval,detect}
mkdir -p weights

# Download pretrained weights (optional)
echo ""
read -p "Download YOLOv9c pretrained weights? (y/n): " download_weights
if [ "$download_weights" = "y" ]; then
    echo "Downloading YOLOv9c weights..."
    # Add download command here
    # wget -P weights/ https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9c.pt
    echo "Note: Please manually download weights from YOLOv9 repository"
fi

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python -c "import torch; print(f'GPU device: {torch.cuda.get_device_name(0)}')"
fi

# Test model import
echo ""
echo "Testing model import..."
python -c "from models import build_yolo_udd; print('Model import successful!')"

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Download TrashCan 1.0 dataset to data/trashcan/"
echo "3. (Optional) Download pretrained weights to weights/"
echo "4. Start training: python scripts/train.py --config configs/train_config.yaml"
echo ""
echo "For more information, see README.md and DOCUMENTATION.md"
