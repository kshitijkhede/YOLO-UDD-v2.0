# YOLO-UDD v2.0 Quick Start Guide

## 5-Minute Setup

### 1. Clone and Setup (2 minutes)

```bash
cd /home/student/MIR/Project/YOLO-UDD-v2.0
chmod +x setup.sh
./setup.sh
```

### 2. Activate Environment

```bash
source venv/bin/activate
```

### 3. Verify Installation

```bash
python -c "from models import build_yolo_udd; model = build_yolo_udd(); print(model.get_model_info())"
```

## Quick Commands

### Train Model

```bash
# Basic training
python scripts/train.py --config configs/train_config.yaml --data-dir data/trashcan

# Custom training
python scripts/train.py --batch-size 8 --epochs 100 --lr 0.001
```

### Evaluate Model

```bash
python scripts/evaluate.py --weights runs/train/checkpoints/best.pt --data-dir data/trashcan
```

### Run Inference

```bash
# Single image
python scripts/detect.py --weights runs/train/checkpoints/best.pt --source image.jpg

# Folder of images
python scripts/detect.py --weights runs/train/checkpoints/best.pt --source images/
```

## Project Structure at a Glance

```
YOLO-UDD-v2.0/
├── models/          # TAFM, PSEM, SDWH, YOLO-UDD
├── data/            # Dataset loader & augmentations
├── utils/           # Loss functions & metrics
├── scripts/         # train.py, evaluate.py, detect.py
├── configs/         # Configuration files
└── runs/            # Training/evaluation outputs
```

## Key Features

✅ **TAFM**: Turbidity-adaptive fusion  
✅ **PSEM**: Multi-scale feature enhancement  
✅ **SDWH**: Attention-based detection head  
✅ **Target**: >82% mAP@50:95 (vs 75.9% baseline)  

## Underwater Augmentations

The model uses specialized augmentations:
- Color jitter (depth simulation)
- Blur effects (turbidity)
- Brightness/contrast (lighting)
- Noise injection (sensor effects)

## Model Architecture

```
Input → YOLOv9c Backbone → PSEM Neck → TAFM → SDWH Head → Detections
                                         ↓
                                   Turbidity Score
```

## Performance Expectations

| Model | mAP@50:95 | Status |
|-------|-----------|--------|
| YOLOv9c | 75.9% | Baseline |
| +PSEM/SDWH | ~78.7% | Intermediate |
| **YOLO-UDD v2.0** | **>82%** | **Target** |

## Troubleshooting

**Out of memory?**
```bash
python scripts/train.py --batch-size 4
```

**Slow training?**
```bash
# Edit configs/train_config.yaml
# Set: use_amp: true
```

**Need pretrained weights?**
Download YOLOv9c weights and use:
```bash
python scripts/train.py --pretrained weights/yolov9c.pt
```

## Resources

- 📖 Full Documentation: `DOCUMENTATION.md`
- 📊 Project Plan: See original PDF
- 🔬 Paper References: See README.md

## Support

- GitHub Issues
- Email: your.email@example.com

---

**Environmental Impact**: This project aims to help clean our oceans 🌊
