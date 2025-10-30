# 🎯 Quick Reference: Google Drive → Kaggle Training

## 🚀 3-Step Process

### Step 1: Prepare Dataset (Windows)
```powershell
# Verify dataset structure
C:\ProgramData\miniconda3\python.exe scripts/verify_dataset.py --dataset-dir path\to\trashcan

# Create ZIP file
Compress-Archive -Path "path\to\trashcan" -DestinationPath "trashcan.zip" -CompressionLevel Optimal
```

### Step 2: Upload to Google Drive
1. Go to https://drive.google.com
2. Upload `trashcan.zip`
3. Right-click → Share → **"Anyone with the link"**
4. Copy link and extract FILE_ID

**Example link:**
```
https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view
                                  ^^^^^^^^^^^^^^^^^^^^^^^^
                                  This is your FILE_ID
```

### Step 3: Train in Kaggle

**Kaggle Notebook (Copy-Paste):**

```python
# Setup
!git clone https://github.com/kshitijkhede/YOLO-UDD-v2.0.git
%cd YOLO-UDD-v2.0
import sys; sys.path.insert(0, '/kaggle/working/YOLO-UDD-v2.0')

# Download dataset
!pip install -q gdown
FILE_ID = "YOUR_FILE_ID_HERE"  # ← CHANGE THIS
!gdown --id {FILE_ID} -O /kaggle/working/trashcan.zip
!unzip -q /kaggle/working/trashcan.zip -d /kaggle/working/

# Install dependencies
!pip install -q torch torchvision albumentations opencv-python-headless pycocotools tensorboard tqdm pyyaml scikit-learn matplotlib seaborn

# Train
!python scripts/run_kaggle_training.py \
    --data-dir /kaggle/working/trashcan \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.01

# Check results
!ls -lh /kaggle/working/runs/train/checkpoints/
```

---

## ⚡ Super Quick Version

**For experienced users - complete notebook in one cell:**

```python
# Enable GPU first: Settings → GPU T4 x2
!git clone https://github.com/kshitijkhede/YOLO-UDD-v2.0.git && cd YOLO-UDD-v2.0
!pip install -q gdown torch torchvision albumentations opencv-python-headless pycocotools tensorboard tqdm pyyaml scikit-learn
!gdown --id YOUR_FILE_ID -O /kaggle/working/trashcan.zip && unzip -q /kaggle/working/trashcan.zip -d /kaggle/working/
!python scripts/run_kaggle_training.py --data-dir /kaggle/working/trashcan --epochs 100 --batch-size 8 --lr 0.01
!ls -lh runs/train/checkpoints/
```

Replace `YOUR_FILE_ID` and run!

---

## 📋 Checklist

- [ ] Dataset has correct structure (run `verify_dataset.py`)
- [ ] Dataset zipped as `trashcan.zip`
- [ ] Uploaded to Google Drive
- [ ] Share link set to "Anyone with the link"
- [ ] FILE_ID extracted from share link
- [ ] Kaggle notebook created
- [ ] GPU enabled (Settings → GPU T4 x2)
- [ ] FILE_ID replaced in notebook code
- [ ] All cells executed
- [ ] Training started successfully
- [ ] Checkpoints downloading after training

---

## 🐛 Quick Fixes

**Download fails?**
```python
# Check file is publicly accessible
# Try manual download to verify link works
```

**Wrong structure after extraction?**
```python
# Check ZIP contents
!unzip -l /kaggle/working/trashcan.zip | head -20
```

**NumPy error?**
```python
!pip uninstall -y numpy && pip install numpy==1.26.4
# Restart kernel and re-run
```

**No GPU?**
```
Settings → Accelerator → GPU T4 x2 → Save
```

---

## 📊 Timeline

| Step | Time |
|------|------|
| Verify dataset | 1 min |
| Create ZIP | 2-5 min |
| Upload to Drive | 5-20 min |
| Setup Kaggle | 5 min |
| Training | ~10 hours |

**Total**: ~10 hours (mostly unattended)

---

## 🎯 Expected Output

```
Training completed!
Best mAP@50:95: 0.7134
Saved: /kaggle/working/runs/train/checkpoints/best.pt (289 MB)
```

**Download `best.pt` from Output tab!**

---

## 📚 Full Guides

- **Detailed guide**: `GDRIVE_KAGGLE_GUIDE.md`
- **All commands**: `COMMANDS.md`
- **Troubleshooting**: `KAGGLE_TRAINING_DEBUG.md`
- **Quick start**: `QUICKSTART.md`

---

## ✨ Summary

1. Verify dataset → ZIP it
2. Upload to Google Drive → Get FILE_ID
3. Copy Kaggle code → Replace FILE_ID → Run
4. Wait ~10 hours → Download best.pt

**That's it!** 🚀
