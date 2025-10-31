# 🚀 YOLO-UDD v2.0 - Copy-Paste Commands

## ✅ Everything Is Ready - Just Copy & Paste!

---

## 🖥️ WINDOWS LOCAL TRAINING

### One-Time Setup (Run Once)
```powershell
# Install PyTorch CPU
C:\ProgramData\miniconda3\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# Install other dependencies
C:\ProgramData\miniconda3\python.exe -m pip install opencv-python pillow albumentations tensorboard tqdm pyyaml scikit-learn matplotlib seaborn pandas scipy pycocotools
```

### Quick Test (2 epochs, ~1 minute)
```powershell
# Create test dataset
C:\ProgramData\miniconda3\python.exe scripts/create_dummy_dataset.py --output_dir data/trashcan --num_train 10 --num_val 3

# Run training
C:\ProgramData\miniconda3\python.exe scripts/train.py --config configs/train_config_cpu.yaml --epochs 2 --batch-size 1 --img-size 320 --num-workers 0 --device cpu
```

### Full Training (Your Dataset)
```powershell
# Replace 'path/to/your/trashcan' with actual path
C:\ProgramData\miniconda3\python.exe scripts/train.py --config configs/train_config_cpu.yaml --data-dir path/to/your/trashcan --epochs 50 --batch-size 1 --img-size 320 --num-workers 0 --device cpu --lr 0.001
```

---

## ☁️ KAGGLE TRAINING (RECOMMENDED)

### Setup in Kaggle Notebook

**Cell 1: Clone Repository**
```python
!git clone https://github.com/kshitijkhede/YOLO-UDD-v2.0.git
%cd YOLO-UDD-v2.0
import sys
sys.path.insert(0, '/kaggle/working/YOLO-UDD-v2.0')
```

**Cell 2: Download Dataset (Option A: Google Drive)**
```python
!pip install -q gdown
!gdown --id YOUR_GOOGLE_DRIVE_FILE_ID -O /kaggle/working/trashcan.zip
!unzip -q /kaggle/working/trashcan.zip -d /kaggle/working/
```

**Cell 2: Dataset (Option B: Kaggle Dataset)**
```python
# If you uploaded dataset to Kaggle datasets
import shutil
shutil.copytree('/kaggle/input/your-dataset-name', '/kaggle/working/trashcan')
```

**Cell 3: Train with Auto-Setup**
```python
# Enable GPU first: Settings → GPU T4 x2
!python scripts/run_kaggle_training.py --data-dir /kaggle/working/trashcan --epochs 100 --batch-size 8 --lr 0.01
```

**If NumPy error appears**: Restart kernel and re-run Cell 3

---

## 📊 Monitor Training

### Windows (TensorBoard)
```powershell
C:\ProgramData\miniconda3\python.exe -m tensorboard.main --logdir runs/train_cpu/logs
```
Then open: http://localhost:6006

### Kaggle
TensorBoard files saved to `/kaggle/working/runs/train/logs/`

---

## 🎯 After Training

### Evaluate Model
```powershell
# Windows
C:\ProgramData\miniconda3\python.exe scripts/evaluate.py --weights runs/train_cpu/checkpoints/best.pt --data-dir data/trashcan
```

```python
# Kaggle
!python scripts/evaluate.py --weights /kaggle/working/runs/train/checkpoints/best.pt --data-dir /kaggle/working/trashcan
```

### Run Inference
```powershell
# Windows - single image
C:\ProgramData\miniconda3\python.exe scripts/detect.py --weights runs/train_cpu/checkpoints/best.pt --source image.jpg

# Windows - folder
C:\ProgramData\miniconda3\python.exe scripts/detect.py --weights runs/train_cpu/checkpoints/best.pt --source images/
```

---

## 🛠️ Quick Fixes

### Out of Memory?
```powershell
# Reduce image size and batch size
--img-size 256 --batch-size 1
```

### Training Too Slow?
Use Kaggle instead of local CPU (100x faster)

### Dataset Not Found?
Check structure:
```
data/trashcan/
├── instances_train_trashcan.json
├── instances_val_trashcan.json
└── images/
    ├── train/
    └── val/
```

---

## 💡 Pro Tips

1. **Always test locally first** (2 epochs) to catch errors early
2. **Use Kaggle for full training** - it's free and 100x faster
3. **Monitor tensorboard** to track progress
4. **Save checkpoints** - training auto-saves best.pt
5. **Start with fewer epochs** (20-30) to validate before long runs

---

## ⏱️ Expected Times

| Platform | Epochs | Dataset Size | Time |
|----------|--------|--------------|------|
| Windows CPU | 2 | 10 images | ~1 min |
| Windows CPU | 50 | 5000 images | ~15 hours |
| Kaggle GPU | 100 | 5000 images | ~10 hours |

---

## 📁 Output Files

After training, find:
- **Best model**: `runs/train_cpu/checkpoints/best.pt`
- **Latest model**: `runs/train_cpu/checkpoints/latest.pt`
- **Logs**: `runs/train_cpu/logs/` (TensorBoard)
- **Metrics**: Printed to console during training

---

## 🎉 You're Ready!

1. Copy commands from sections above
2. Paste into PowerShell (Windows) or Kaggle cell
3. Press Enter and watch it train!

**Need help?** See `KAGGLE_TRAINING_DEBUG.md` for troubleshooting.

---

**Status**: ✅ Tested & Working  
**Date**: October 30, 2025
