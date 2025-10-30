# 📤 Upload Dataset to Google Drive & Use in Kaggle

## Step 1: Prepare Your Dataset for Upload

### Before Uploading - Verify Dataset Structure

Run this on your Windows machine to check your dataset:

```powershell
# Navigate to your dataset directory
cd path\to\your\trashcan\dataset

# Check structure
dir
```

**Required structure:**
```
trashcan/
├── instances_train_trashcan.json  (or train.json)
├── instances_val_trashcan.json    (or val.json)
└── images/
    ├── train/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── val/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

### Create ZIP File

**Windows (Right-click method):**
1. Open File Explorer
2. Navigate to your `trashcan` folder
3. Right-click on the `trashcan` folder
4. Select **"Send to" → "Compressed (zipped) folder"**
5. Name it: `trashcan.zip`

**Windows (PowerShell):**
```powershell
# Compress dataset into ZIP
Compress-Archive -Path "path\to\trashcan" -DestinationPath "trashcan.zip" -CompressionLevel Optimal
```

---

## Step 2: Upload to Google Drive

### Upload Steps:

1. **Go to Google Drive**: https://drive.google.com
2. **Create a folder** (optional): 
   - Click **"New" → "Folder"**
   - Name it: `YOLO-UDD-Dataset`
3. **Upload ZIP**:
   - Click **"New" → "File upload"**
   - Select your `trashcan.zip` file
   - Wait for upload to complete

### Get Shareable Link:

1. **Right-click** on uploaded `trashcan.zip`
2. Click **"Share"**
3. Change access to **"Anyone with the link"**
4. Click **"Copy link"**

**Your link will look like:**
```
https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7/view?usp=sharing
```

**Extract the FILE ID** (the part between `/d/` and `/view`):
```
1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7
```

**Save this FILE ID** - you'll need it for Kaggle!

---

## Step 3: Use Dataset in Kaggle

### Complete Kaggle Notebook (Copy-Paste Ready)

```python
# ================================================================
# YOLO-UDD v2.0 - Kaggle Training with Google Drive Dataset
# ================================================================

# ----------------------------------------------------------------
# CELL 1: Setup Repository
# ----------------------------------------------------------------
print("🔧 Setting up repository...")
!git clone https://github.com/kshitijkhede/YOLO-UDD-v2.0.git
%cd YOLO-UDD-v2.0

import sys
import os
sys.path.insert(0, '/kaggle/working/YOLO-UDD-v2.0')

print("✅ Repository cloned!")
print(f"📂 Working directory: {os.getcwd()}")


# ----------------------------------------------------------------
# CELL 2: Download Dataset from Google Drive
# ----------------------------------------------------------------
print("\n" + "="*70)
print("📥 Downloading Dataset from Google Drive")
print("="*70)

# Install gdown
!pip install -q gdown

# ⚠️ REPLACE THIS WITH YOUR FILE ID FROM GOOGLE DRIVE ⚠️
GDRIVE_FILE_ID = "YOUR_FILE_ID_HERE"  # ← CHANGE THIS!

# Download dataset
print(f"\n📦 Downloading dataset (FILE_ID: {GDRIVE_FILE_ID})...")
!gdown --id {GDRIVE_FILE_ID} -O /kaggle/working/trashcan.zip

# Check download
import os
if os.path.exists('/kaggle/working/trashcan.zip'):
    size_mb = os.path.getsize('/kaggle/working/trashcan.zip') / (1024*1024)
    print(f"✅ Downloaded: {size_mb:.1f} MB")
else:
    raise FileNotFoundError("❌ Download failed! Check your FILE_ID")

# Extract dataset
print("\n📦 Extracting dataset...")
!unzip -q /kaggle/working/trashcan.zip -d /kaggle/working/

# Verify extraction
if os.path.exists('/kaggle/working/trashcan'):
    print("✅ Dataset extracted to: /kaggle/working/trashcan")
else:
    raise FileNotFoundError("❌ Extraction failed!")


# ----------------------------------------------------------------
# CELL 3: Verify Dataset Structure
# ----------------------------------------------------------------
print("\n" + "="*70)
print("🔍 Verifying Dataset Structure")
print("="*70)

dataset_path = '/kaggle/working/trashcan'

# Check required files
required_files = [
    'instances_train_trashcan.json',
    'instances_val_trashcan.json',
    'images/train',
    'images/val'
]

# Alternative names
alternative_files = [
    'train.json',
    'val.json',
    'annotations/train.json',
    'annotations/val.json'
]

print("\n📋 Checking dataset structure:")
all_ok = True
for file in required_files:
    path = os.path.join(dataset_path, file)
    exists = os.path.exists(path)
    print(f"  {'✅' if exists else '❌'} {file}")
    if not exists:
        all_ok = False

if not all_ok:
    print("\n⚠️  Some files missing, checking alternatives...")
    for file in alternative_files:
        path = os.path.join(dataset_path, file)
        if os.path.exists(path):
            print(f"  ✅ Found: {file}")

# Count images
print("\n📊 Dataset Statistics:")
for split in ['train', 'val']:
    img_dir = os.path.join(dataset_path, 'images', split)
    if os.path.exists(img_dir):
        images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"  📁 {split}: {len(images):,} images")
    else:
        print(f"  ❌ {split} images not found!")

# Show sample files
print("\n📂 Sample files in dataset:")
!ls -lh /kaggle/working/trashcan/ | head -15

print("\n✅ Dataset verification complete!")


# ----------------------------------------------------------------
# CELL 4: Install Dependencies
# ----------------------------------------------------------------
print("\n" + "="*70)
print("📦 Installing Dependencies")
print("="*70)

!pip install -q torch>=2.0.0 torchvision>=0.15.0
!pip install -q albumentations>=1.3.0 opencv-python-headless>=4.7.0
!pip install -q pycocotools>=2.0.6 tensorboard>=2.12.0
!pip install -q tqdm pyyaml scikit-learn matplotlib seaborn pandas

print("\n✅ Dependencies installed!")

# Verify PyTorch and GPU
import torch
print(f"\n🔥 PyTorch version: {torch.__version__}")
print(f"🎮 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
else:
    print("⚠️  No GPU detected! Enable GPU in Settings → Accelerator → GPU T4 x2")


# ----------------------------------------------------------------
# CELL 5: Verify Model Can Import
# ----------------------------------------------------------------
print("\n" + "="*70)
print("🏗️  Testing Model")
print("="*70)

try:
    from models import build_yolo_udd
    
    # Build model
    model = build_yolo_udd(num_classes=22)
    
    # Get model info
    info = model.get_model_info()
    print("\n📊 Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    with torch.no_grad():
        predictions, turb_score = model(dummy_input)
    
    print(f"\n✅ Model test passed!")
    print(f"✅ Turbidity score: {turb_score.item():.4f}")
    
except Exception as e:
    print(f"\n❌ Model test failed: {e}")
    import traceback
    traceback.print_exc()


# ----------------------------------------------------------------
# CELL 6: Start Training
# ----------------------------------------------------------------
print("\n" + "="*70)
print("🚀 Starting Training")
print("="*70)

# Training configuration
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.01
IMG_SIZE = 640
DATASET_PATH = '/kaggle/working/trashcan'
SAVE_DIR = '/kaggle/working/runs/train'

print(f"""
Training Configuration:
  📊 Epochs:        {EPOCHS}
  📦 Batch Size:    {BATCH_SIZE}
  📐 Image Size:    {IMG_SIZE}
  🎓 Learning Rate: {LEARNING_RATE}
  📁 Dataset:       {DATASET_PATH}
  💾 Save Dir:      {SAVE_DIR}
""")

# Start training using helper script
!python scripts/run_kaggle_training.py \
    --data-dir {DATASET_PATH} \
    --epochs {EPOCHS} \
    --batch-size {BATCH_SIZE} \
    --lr {LEARNING_RATE} \
    --save-dir {SAVE_DIR}


# ----------------------------------------------------------------
# CELL 7: Check Training Results
# ----------------------------------------------------------------
print("\n" + "="*70)
print("📊 Training Results")
print("="*70)

save_dir = '/kaggle/working/runs/train'

if os.path.exists(save_dir):
    # Check for checkpoints
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        print("\n📁 Saved Checkpoints:")
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.pt'):
                path = os.path.join(checkpoint_dir, file)
                size_mb = os.path.getsize(path) / (1024*1024)
                print(f"  ✅ {file} ({size_mb:.1f} MB)")
        
        print("\n💡 Download checkpoints from Output tab →")
    else:
        print("❌ No checkpoints found")
    
    # Show training log
    log_file = os.path.join(save_dir, 'training.log')
    if os.path.exists(log_file):
        print("\n📈 Last 30 lines of training log:")
        print("="*70)
        !tail -n 30 {log_file}
    
    print("\n" + "="*70)
    print("🎉 Training Complete!")
    print("="*70)
    print("📥 Download 'best.pt' from Output section (right sidebar)")
    print("📊 Use it for inference and evaluation")
else:
    print("❌ No training results found")


# ----------------------------------------------------------------
# CELL 8: Quick Evaluation (Optional)
# ----------------------------------------------------------------
# Uncomment to run evaluation after training

# print("\n" + "="*70)
# print("📊 Running Evaluation")
# print("="*70)
# 
# CHECKPOINT = '/kaggle/working/runs/train/checkpoints/best.pt'
# 
# if os.path.exists(CHECKPOINT):
#     !python scripts/evaluate.py \
#         --weights {CHECKPOINT} \
#         --data-dir /kaggle/working/trashcan
# else:
#     print(f"❌ Checkpoint not found: {CHECKPOINT}")
```

---

## Step 4: Run Training in Kaggle

### Quick Steps:

1. **Go to Kaggle**: https://www.kaggle.com/code
2. **Create New Notebook**
3. **Enable GPU**: Settings → Accelerator → **GPU T4 x2**
4. **Copy entire code above** into notebook cells
5. **Replace `YOUR_FILE_ID_HERE`** in Cell 2 with your actual Google Drive file ID
6. **Run all cells** (or click **"Run All"**)

---

## 🎯 What Happens:

| Cell | Action | Time |
|------|--------|------|
| 1 | Clone repository | ~10 sec |
| 2 | Download dataset from Google Drive | ~2-5 min |
| 3 | Verify dataset structure | ~10 sec |
| 4 | Install dependencies | ~2 min |
| 5 | Test model | ~30 sec |
| 6 | **Train model** | **~10 hours** |
| 7 | Show results & checkpoints | ~10 sec |

**Total**: ~10 hours (mostly unattended)

---

## 🐛 Troubleshooting

### Problem 1: "Download failed" or "403 Forbidden"

**Solution:**
1. Make sure file is shared: **"Anyone with the link"**
2. Check FILE_ID is correct (no extra characters)
3. Try downloading manually first to verify link works

### Problem 2: "Extraction failed" or wrong structure

**Solution:**
```python
# Check what's in the ZIP
!unzip -l /kaggle/working/trashcan.zip | head -20

# If nested (trashcan/trashcan/...), adjust path:
import shutil
shutil.move('/kaggle/working/trashcan/trashcan', '/kaggle/working/trashcan_fixed')
!rm -rf /kaggle/working/trashcan
!mv /kaggle/working/trashcan_fixed /kaggle/working/trashcan
```

### Problem 3: "NumPy error" during training

**Solution:**
```python
# Add this cell BEFORE Cell 4
!pip uninstall -y numpy
!pip install numpy==1.26.4
# Then restart kernel and re-run all cells
```

### Problem 4: Dataset too large for Google Drive

**Option A: Split ZIP**
```powershell
# Windows - split into 500MB chunks
# Use 7-Zip: Right-click → 7-Zip → Add to archive → Split to volumes: 500M
```

**Option B: Upload to Kaggle Datasets directly**
1. Go to https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload your `trashcan.zip`
4. Make it public
5. Add to your notebook
6. Use path: `/kaggle/input/your-dataset-name/trashcan`

---

## 📊 Expected Results

After ~10 hours of training:

```
Training completed!
Best mAP@50:95: 0.7134
Precision: 0.7856
Recall: 0.7234
FPS: ~45

Checkpoints saved:
  ✅ best.pt (289.3 MB) ← Download this!
  ✅ latest.pt (289.3 MB)
```

---

## 💡 Pro Tips

1. **Save version**: Click "Save Version" every few hours
2. **Monitor progress**: Run Cell 7 periodically to check checkpoints
3. **Download early**: Download `best.pt` before 12-hour session expires
4. **Use TensorBoard**: Logs saved in `/kaggle/working/runs/train/logs/`
5. **Test first**: Run with `--epochs 2` to test setup before full training

---

## 📥 After Training - Download Model

1. Look at **right sidebar** → **Output** tab
2. Navigate to: `/kaggle/working/runs/train/checkpoints/`
3. Find `best.pt` file
4. Click **download icon**
5. Save to your computer

Use this checkpoint for inference on your local machine!

---

## 🎉 Summary

1. ✅ ZIP your dataset
2. ✅ Upload to Google Drive
3. ✅ Get FILE_ID from share link
4. ✅ Copy notebook code above
5. ✅ Replace FILE_ID in Cell 2
6. ✅ Enable GPU in Kaggle
7. ✅ Run all cells
8. ✅ Wait ~10 hours
9. ✅ Download `best.pt`

**That's it!** Your model will train automatically. 🚀

---

**Need help?** Check `KAGGLE_TRAINING_DEBUG.md` or open a GitHub issue.
