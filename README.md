# YOLO-UDD v2.0: A Turbidity-Adaptive Architecture for High-Fidelity Underwater Debris Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](DEV_STATUS.md)

> **🚧 Development Status:** This project is in active development. The architecture is complete and tested, but the training pipeline uses placeholder loss functions. See [DEV_STATUS.md](DEV_STATUS.md) for details.

## 📋 Project Overview

YOLO-UDD v2.0 is an advanced deep learning-based object detection model specifically engineered for underwater debris detection. This architecture addresses the unique challenges of the underwater environment through novel adaptive mechanisms and specialized feature processing modules.

### Key Innovations

1. **Turbidity-Adaptive Fusion Module (TAFM)** - ⭐ **Novel contribution** that dynamically adjusts feature fusion based on real-time water turbidity conditions
2. **Partial Semantic Encoding Module (PSEM)** - Enhanced multi-scale feature fusion for detecting objects of varying sizes
3. **Split Dimension Weighting Head (SDWH)** - Attention-based detection head that suppresses background noise

### Architecture

```
Input (640×640) → Backbone (YOLOv9c) → Neck (PSEM + TAFM) → Head (SDWH) → Detections
```

## 🎯 Target Performance

| Model | mAP@50:95 | Improvement | Innovation |
|-------|-----------|-------------|------------|
| YOLOv9c (Baseline) | 75.9% | - | Baseline |
| +PSEM/SDWH | ~78.7% | +2.8% | From Li et al. |
| +TAFM (YOLO-UDD v2.0) | **>82%** | **+6-7%** | **Novel turbidity adaptation** |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOLO-UDD-v2.0.git
cd YOLO-UDD-v2.0

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

1. Download the TrashCan 1.0 dataset from [University of Minnesota](https://conservancy.umn.edu/handle/11299/214865)

2. Create symbolic links (or copy data):
```bash
mkdir -p data/trashcan
ln -s /path/to/trashcan/dataset/instance_version data/trashcan/
```

### Test the Architecture

```bash
# Test model forward pass
python -c "
import torch
from models.yolo_udd import build_yolo_udd

print('Building YOLO-UDD model...')
model = build_yolo_udd(num_classes=3)
model.eval()

print('Testing forward pass...')
x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    predictions, turb_score = model(x)

print(f'✓ Forward pass successful!')
print(f'Turbidity Score: {turb_score.item():.4f}')
print(f'Detection scales: {len(predictions)}')
"
```

### Training (Development Mode)

> **Note:** Current training uses placeholder loss functions. The pipeline works but the model won't learn effectively until proper target assignment is implemented. See [DEV_STATUS.md](DEV_STATUS.md).

```bash
# Run training test
python scripts/train.py \
    --config configs/train_config.yaml \
    --data-dir data/trashcan \
    --batch-size 4 \
    --epochs 10 \
    --save-dir runs/train
```

## 📊 Project Structure

```
YOLO-UDD-v2.0/
├── models/
│   ├── yolo_udd.py         # Main YOLO-UDD architecture ✓
│   ├── tafm.py             # Turbidity-Adaptive Fusion Module ✓
│   ├── psem.py             # Partial Semantic Encoding Module ✓
│   └── sdwh.py             # Split Dimension Weighting Head ✓
├── data/
│   ├── dataset.py          # TrashCan dataset loader ✓
│   └── augmentations.py    # Underwater-specific augmentations ✓
├── utils/
│   ├── loss.py             # Loss functions (⚠️ placeholder)
│   ├── metrics.py          # Evaluation metrics (⚠️ placeholder)
│   └── visualization.py    # Visualization tools ✓
├── scripts/
│   ├── train.py            # Training script ✓
│   ├── evaluate.py         # Evaluation script
│   └── detect.py           # Inference script
├── configs/
│   └── train_config.yaml   # Training configuration ✓
├── requirements.txt        # Dependencies ✓
├── DEV_STATUS.md          # Development status & roadmap
└── README.md              # This file
```

## 🔬 Model Architecture Details

### 1. Backbone: YOLOv9c

- **Feature Extractor**: GELAN (Generalized Efficient Layer Aggregation Network)
- **Multi-scale Features**: P3, P4, P5 (80×80, 40×40, 20×20)
- **Pretrained**: COCO dataset (optional)

### 2. Neck: PSEM-enhanced PANet + TAFM

- **Top-down Pathway**: Enhanced feature fusion with PSEM
- **Bottom-up Pathway**: Additional feature refinement
- **TAFM Integration**: Dynamic turbidity-adaptive weighting

### 3. Head: SDWH

- **Level-wise Attention**: Scale-specific weighting
- **Spatial-wise Attention**: Location-specific focus
- **Channel-wise Attention**: Semantic task weighting

## 📈 Training Configuration

Based on Section 5.2 of the project plan:

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Initial LR | 0.01 |
| LR Schedule | Cosine Annealing |
| Batch Size | 16 |
| Epochs | 100 |
| Image Size | 640×640 |
| Weight Decay | 0.0005 |
| Early Stopping | 20 epochs |

## 🌊 Underwater-Specific Augmentations

1. **Color Jitter** - Simulates color casting at depth
2. **Gaussian/Motion Blur** - Simulates water turbidity
3. **Brightness/Contrast Adjustment** - Simulates lighting variations
4. **Noise Injection** - Simulates sensor noise
5. **RGB Shift** - Additional underwater color effects

## 🎓 Dataset Information

**TrashCan 1.0 Dataset**
- **Total Images**: 7,212
- **Classes (3-Class Configuration)**:
  - Trash (85% of marine debris)
  - Animal
  - ROV
- **Split**: 70% train, 15% validation, 15% test
- **Training Set**: 6,065 images
- **Validation Set**: 1,147 images

## 🔧 Development Status

### ✅ What's Working
- Complete model architecture (Backbone, PSEM, TAFM, SDWH)
- Data loading and augmentation pipeline
- Training loop infrastructure
- Model checkpointing and logging

### ⚠️ In Progress
- Proper target assignment algorithm
- Real loss calculations with GT matching
- Evaluation metrics (mAP, Precision, Recall)

### 📅 Roadmap
See [DEV_STATUS.md](DEV_STATUS.md) for detailed development roadmap and milestones.

## 📖 Citation

If you use YOLO-UDD v2.0 in your research, please cite:

```bibtex
@article{yoloud2025,
  title={YOLO-UDD v2.0: A Turbidity-Adaptive Architecture for High-Fidelity Underwater Debris Detection},
  author={Your Name},
  year={2025}
}
```

## 🔗 References

1. Samanth, K., et al. (2025). "A Comprehensive Study On Underwater Object Detection Using Deep Neural Networks." IEEE Access.
2. Li, X., et al. (2025). "Efficient underwater object detection based on feature enhancement and attention detection head." Scientific Reports.
3. Wang, C.-Y., et al. (2024). "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information." arXiv.

## 🤝 Contributing

Contributions are welcome! Priority areas:

1. **High Priority**: Target assignment algorithm, proper loss implementation
2. **Medium Priority**: Evaluation metrics, NMS post-processing
3. **Nice to Have**: Visualization tools, Docker support

Please read [DEV_STATUS.md](DEV_STATUS.md) before contributing.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TrashCan dataset creators at University of Minnesota
- YOLOv9 developers
- PyTorch team
- Marine conservation organizations

## 📧 Contact

For questions or collaborations:
- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/YOLO-UDD-v2.0/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/YOLO-UDD-v2.0/discussions)

---

**Note**: This project is aimed at environmental conservation by enabling automated cleanup of marine ecosystems. The architecture is complete and novel (TAFM module), but training requires proper loss implementation for effective learning. See [DEV_STATUS.md](DEV_STATUS.md) for current status.
