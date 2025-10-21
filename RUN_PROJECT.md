# 🚀 YOLO-UDD v2.0 - Complete Running Guide

## Step-by-Step Execution

### STEP 1: Open Terminal and Navigate
```bash
cd /home/student/MIR/Project/YOLO-UDD-v2.0
```

### STEP 2: Activate Virtual Environment
```bash
source venv/bin/activate
```
You should see `(venv)` appear in your terminal prompt.

### STEP 3: Verify Setup
```bash
# Check Python
python --version

# Check PyTorch
python -c "import torch; print('PyTorch:', torch.__version__)"

# Check dataset
ls data/trashcan/images/train/ | wc -l
ls data/trashcan/images/val/ | wc -l
```

Expected output:
- Python 3.8+
- PyTorch 2.0+
- 6065 train images
- 1147 val images

---

## 🎯 TRAINING THE MODEL (Main Task)

### Option 1: Quick Start Training (Reduced settings for CPU)
```bash
python scripts/train.py \
    --config configs/train_config.yaml \
    --data-dir data/trashcan \
    --batch-size 4 \
    --epochs 50 \
    --save-dir runs/train
```

### Option 2: Full Training (If you have GPU)
```bash
python scripts/train.py \
    --config configs/train_config.yaml \
    --data-dir data/trashcan \
    --batch-size 16 \
    --epochs 300 \
    --save-dir runs/train
```

### Option 3: Custom Training
```bash
python scripts/train.py \
    --batch-size 8 \
    --epochs 100 \
    --lr 0.001 \
    --data-dir data/trashcan \
    --save-dir runs/train
```

### During Training:
- Progress will be shown with tqdm progress bars
- Checkpoints saved to: `runs/train/checkpoints/`
- Best model: `runs/train/checkpoints/best.pt`
- Last model: `runs/train/checkpoints/last.pt`
- TensorBoard logs: `runs/train/logs/`

### Monitor Training with TensorBoard:
```bash
# In a new terminal (while training is running)
cd /home/student/MIR/Project/YOLO-UDD-v2.0
source venv/bin/activate
tensorboard --logdir runs/train/logs --port 6006
```
Then open browser: http://localhost:6006

---

## 📊 EVALUATING THE MODEL

After training completes:

```bash
python scripts/evaluate.py \
    --weights runs/train/checkpoints/best.pt \
    --data-dir data/trashcan \
    --save-dir runs/eval
```

Results will be saved to:
- `runs/eval/metrics.json` - Numerical results
- `runs/eval/confusion_matrix.png` - Visual confusion matrix
- `runs/eval/pr_curve.png` - Precision-Recall curve

---

## 🔍 RUNNING INFERENCE (DETECTION)

### Detect on a single image:
```bash
python scripts/detect.py \
    --weights runs/train/checkpoints/best.pt \
    --source path/to/image.jpg \
    --save-dir runs/detect
```

### Detect on validation set:
```bash
python scripts/detect.py \
    --weights runs/train/checkpoints/best.pt \
    --source data/trashcan/images/val/ \
    --save-dir runs/detect
```

### Detect with confidence threshold:
```bash
python scripts/detect.py \
    --weights runs/train/checkpoints/best.pt \
    --source data/trashcan/images/val/ \
    --conf-threshold 0.5 \
    --save-dir runs/detect
```

Detection results saved to:
- `runs/detect/` - Images with bounding boxes
- `runs/detect/labels/` - Detection coordinates

---

## 📁 Output Structure

After running training, evaluation, and detection:

```
runs/
├── train/
│   ├── checkpoints/
│   │   ├── best.pt      # Best model weights
│   │   └── last.pt      # Last epoch weights
│   └── logs/            # TensorBoard logs
├── eval/
│   ├── metrics.json     # mAP, precision, recall
│   └── confusion_matrix.png
└── detect/
    ├── image1.jpg       # Annotated images
    ├── image2.jpg
    └── labels/          # Detection coordinates
```

---

## 🔧 Troubleshooting

### Out of Memory Error:
```bash
# Reduce batch size
python scripts/train.py --batch-size 2 --epochs 50
```

### CUDA not available:
Training will automatically use CPU (slower but works).
To use GPU, ensure NVIDIA drivers and CUDA toolkit are installed.

### Import errors:
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Dataset not found:
```bash
# Verify symbolic links
ls -la data/trashcan/images/
ls -la data/trashcan/annotations/
```

---

## ⚡ Quick Commands Reference

```bash
# Setup (first time only)
./setup.sh

# Activate environment (every session)
source venv/bin/activate

# Train
python scripts/train.py --config configs/train_config.yaml --data-dir data/trashcan --batch-size 4 --epochs 50

# Evaluate
python scripts/evaluate.py --weights runs/train/checkpoints/best.pt --data-dir data/trashcan

# Detect
python scripts/detect.py --weights runs/train/checkpoints/best.pt --source data/trashcan/images/val/

# Monitor with TensorBoard
tensorboard --logdir runs/train/logs --port 6006
```

---

## 📈 Expected Performance

| Model | mAP@50:95 | Target |
|-------|-----------|--------|
| YOLOv9c (Baseline) | 75.9% | Baseline |
| YOLO-UDD v2.0 | >82% | Goal |

---

## 💡 Tips

1. **Start with small epochs** (50-100) to test the pipeline
2. **Monitor TensorBoard** to watch training progress
3. **Use validation set** for early stopping
4. **Save best model** for inference
5. **Adjust batch size** based on available memory

---

## 📞 Common Issues

**Q: Training is slow?**
A: Use GPU or reduce image size/batch size

**Q: No detections appear?**
A: Lower confidence threshold (--conf-threshold 0.3)

**Q: Poor accuracy?**
A: Train for more epochs (200-300)

**Q: Want to resume training?**
A: Use `--resume runs/train/checkpoints/last.pt`

