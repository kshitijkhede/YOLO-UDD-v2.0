# 🚀 YOLO-UDD v2.0 Quick Start Guide

## ✅ Training Successfully Tested!

Verified working on:
- **Windows 11** (CPU, AMD Ryzen 5 5600H, 8GB RAM) ✅  
- **Kaggle** (GPU T4 x2) ✅

---

## 🎯 Windows Local Training (CPU)

**Perfect for**: Testing, small datasets, no cloud access

```powershell
# Step 1: Install dependencies (one-time)
C:\ProgramData\miniconda3\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
C:\ProgramData\miniconda3\python.exe -m pip install opencv-python albumentations tensorboard tqdm pyyaml scikit-learn matplotlib seaborn pandas scipy pycocotools

# Step 2: Create test dataset
C:\ProgramData\miniconda3\python.exe scripts/create_dummy_dataset.py --output_dir data/trashcan --num_train 50 --num_val 20

# Step 3: Run training
C:\ProgramData\miniconda3\python.exe scripts/train.py --config configs/train_config_cpu.yaml --epochs 5 --batch-size 1 --img-size 320 --num-workers 0 --device cpu
```

⏱️ **Time**: ~2-3 min (5 epochs) | ~10-20 hours (full dataset)

---

## ☁️ Kaggle Training (GPU - Recommended)

**Perfect for**: Full training, large datasets

```python
# Cell 1: Setup
!git clone https://github.com/kshitijkhede/YOLO-UDD-v2.0.git
%cd YOLO-UDD-v2.0

# Cell 2: Download/setup dataset
!pip install -q gdown
!gdown --id YOUR_FILE_ID -O /kaggle/working/trashcan.zip
!unzip -q /kaggle/working/trashcan.zip -d /kaggle/working/

# Cell 3: Train (auto-setup)
!python scripts/run_kaggle_training.py --data-dir /kaggle/working/trashcan --epochs 100 --batch-size 8
```

⏱️ **Time**: ~10 hours (100 epochs) | **Expected mAP**: 70-72%

---

## 🐧 Linux Setup

```bash
cd /home/student/MIR/Project/YOLO-UDD-v2.0
chmod +x setup.sh
./setup.sh
source venv/bin/activate
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
