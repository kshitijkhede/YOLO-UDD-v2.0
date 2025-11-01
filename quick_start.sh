#!/bin/bash
# Quick Start Script for YOLO-UDD v2.0
# This script will help you get started with training

echo "=========================================="
echo "🚀 YOLO-UDD v2.0 Quick Start"
echo "=========================================="
echo ""

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

echo ""
echo "=========================================="
echo "🔍 System Check"
echo "=========================================="

# Check Python version
echo "Python version:"
python3 --version

# Check PyTorch and CUDA
echo ""
echo "PyTorch & CUDA status:"
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU detected - training will be slower on CPU")
EOF

echo ""
echo "=========================================="
echo "📊 Dataset Status"
echo "=========================================="
python3 scripts/verify_dataset.py --dataset-dir data/trashcan | tail -20

echo ""
echo "=========================================="
echo "🏗️  Model Architecture Test"
echo "=========================================="
python3 << EOF
try:
    from models import build_yolo_udd
    import torch
    
    print("Building YOLO-UDD model...")
    model = build_yolo_udd(num_classes=22)
    
    # Test forward pass
    print("Testing forward pass...")
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ Model successfully built!")
    print(f"✅ Forward pass successful!")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shapes: {[o.shape for o in output]}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
EOF

echo ""
echo "=========================================="
echo "📋 What to do next:"
echo "=========================================="
echo ""
echo "1️⃣  START TRAINING (GPU recommended):"
echo "   python3 scripts/train.py --config configs/train_config.yaml"
echo ""
echo "2️⃣  START TRAINING (CPU fallback):"
echo "   python3 scripts/train.py --config configs/train_config_cpu.yaml"
echo ""
echo "3️⃣  MONITOR TRAINING (in new terminal):"
echo "   source venv/bin/activate"
echo "   tensorboard --logdir runs/"
echo "   # Then open: http://localhost:6006"
echo ""
echo "4️⃣  TEST DATASET LOADING:"
echo "   python3 data/dataset.py"
echo ""
echo "=========================================="
echo "✨ Ready to start training!"
echo "=========================================="
