# YOLO-UDD v2.0: A Turbidity-Adaptive Architecture for High-Fidelity Underwater Debris Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Project Overview

YOLO-UDD v2.0 is an advanced deep learning-based object detection model specifically engineered for underwater debris detection. This architecture addresses the unique challenges of the underwater environment through novel adaptive mechanisms and specialized feature processing modules.

### Key Innovations

1. **Turbidity-Adaptive Fusion Module (TAFM)** - Novel contribution that dynamically adjusts feature fusion based on real-time water turbidity conditions
2. **Partial Semantic Encoding Module (PSEM)** - Enhanced multi-scale feature fusion for detecting objects of varying sizes
3. **Split Dimension Weighting Head (SDWH)** - Attention-based detection head that suppresses background noise

### Architecture

```
Input (640×640) → Backbone (YOLOv9c) → Neck (PSEM + TAFM) → Head (SDWH) → Detections
```

## 🎯 Target Performance

- **Baseline (YOLOv9c)**: 75.9% mAP@50:95
- **Target (YOLO-UDD v2.0)**: >82% mAP@50:95
- **Expected Improvement**: +6-7%

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/YOLO-UDD-v2.0.git
cd YOLO-UDD-v2.0

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

Download the TrashCan 1.0 dataset:

```bash
# Create data directory
mkdir -p data/trashcan

# Download and extract TrashCan 1.0 dataset
# Follow instructions at: https://conservancy.umn.edu/handle/11299/214865
```

### Training

```bash
# Train with default configuration
python scripts/train.py --config configs/train_config.yaml

# Train with custom parameters
python scripts/train.py \
    --data-dir data/trashcan \
    --batch-size 16 \
    --epochs 300 \
    --lr 0.01 \
    --save-dir runs/train
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py \
    --weights runs/train/checkpoints/best.pt \
    --data-dir data/trashcan \
    --save-dir runs/eval
```

### Inference

```bash
# Run inference on images
python scripts/detect.py \
    --weights runs/train/checkpoints/best.pt \
    --source path/to/images \
    --save-dir runs/detect
```

## 📊 Project Structure

```
YOLO-UDD-v2.0/
├── models/
│   ├── __init__.py
│   ├── yolo_udd.py         # Main YOLO-UDD architecture
│   ├── tafm.py             # Turbidity-Adaptive Fusion Module
│   ├── psem.py             # Partial Semantic Encoding Module
│   └── sdwh.py             # Split Dimension Weighting Head
├── data/
│   ├── dataset.py          # TrashCan dataset loader
│   └── augmentations.py    # Underwater-specific augmentations
├── utils/
│   ├── loss.py             # Loss functions (EIoU, Varifocal, BCE)
│   ├── metrics.py          # Evaluation metrics
│   └── visualization.py    # Visualization tools
├── scripts/
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── detect.py           # Inference script
├── configs/
│   └── train_config.yaml   # Training configuration
├── requirements.txt
├── README.md
└── LICENSE
```

## 🔬 Model Architecture Details

### 1. Backbone: YOLOv9c

- **Feature Extractor**: GELAN (Generalized Efficient Layer Aggregation Network)
- **Multi-scale Features**: P3, P4, P5 (80×80, 40×40, 20×20)
- **Pretrained**: COCO dataset

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
| Epochs | 300 |
| Image Size | 640×640 |
| Weight Decay | 0.0005 |
| Early Stopping | 20 epochs |

## 🌊 Underwater-Specific Augmentations

1. **Color Jitter** - Simulates color casting at depth
2. **Gaussian/Motion Blur** - Simulates water turbidity
3. **Brightness/Contrast Adjustment** - Simulates lighting variations
4. **Noise Injection** - Simulates sensor noise
5. **RGB Shift** - Additional underwater color effects

## 📊 Evaluation Metrics

- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **mAP@50**: Mean Average Precision at IoU=0.50
- **mAP@50:95**: Mean Average Precision at IoU=0.50:0.95
- **FPS**: Frames per second for real-time feasibility

## 🎓 Dataset Information

**TrashCan 1.0 Dataset**
- **Total Images**: 7,212
- **Classes (3-Class Configuration)**:
  - Trash (85% of marine debris)
  - Animal
  - ROV
- **Split**: 70% train, 15% validation, 15% test

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

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TrashCan dataset creators at University of Minnesota
- YOLOv9 developers
- PyTorch team
- Marine conservation organizations

## 📧 Contact

For questions or collaborations:
- Email: your.email@example.com
- GitHub Issues: [Create an issue](https://github.com/yourusername/YOLO-UDD-v2.0/issues)

---

**Note**: This project is aimed at environmental conservation by enabling automated cleanup of marine ecosystems. Please use this technology responsibly for environmental applications.
