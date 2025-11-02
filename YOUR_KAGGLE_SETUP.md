# 🎯 Your Kaggle Setup - Personalized for k2001hede

**Your Datasets:**
- ✅ Annotations: https://www.kaggle.com/datasets/k2001hede/trashcan-annotations-coco-format
- ✅ Images: https://www.kaggle.com/datasets/k2001hede/trashcan

---

## 🚀 Step-by-Step: Start Training NOW

### Step 1: Create Kaggle Notebook

1. Go to: **https://www.kaggle.com/code**
2. Click **"New Notebook"** (blue button)
3. Upload `YOLO_UDD_Kaggle_Training_Fixed.ipynb` from your local repo
   - File location: `/home/student/MIR/Project/YOLO-UDD-v2.0/YOLO_UDD_Kaggle_Training_Fixed.ipynb`
   - Or create new notebook and copy-paste cells

---

### Step 2: Configure Notebook Settings

Click the **⚙️ Settings** icon (gear icon in right sidebar)

Set these:
```
⚡ Accelerator:  GPU T4 x2  (or GPU P100)
🌐 Internet:    ON
💾 Persistence: ON (optional but recommended)
```

Click **"Save"**

---

### Step 3: Add Your Datasets

Look at the **RIGHT SIDEBAR** in Kaggle:

1. Click **"+ Add Data"** button
2. Click **"Your Datasets"** tab
3. Search for: `trashcan-annotations-coco-format`
4. Click **"Add"** ✓
5. Search for: `trashcan`
6. Click **"Add"** ✓

Your sidebar should now show:
```
📊 Data
├─ 📁 trashcan-annotations-coco-format
└─ 📁 trashcan
```

---

### Step 4: Update Paths in Notebook (CRITICAL!)

Find the cell in **Step 2** that looks like this:

```python
# === MODIFY THESE PATHS TO MATCH YOUR KAGGLE DATASETS ===
ANNOTATIONS_PATH = '/kaggle/input/trashcan-annotations-coco-format/annotations'
IMAGES_PATH = '/kaggle/input/trashcan/images'
```

**✅ FOR YOUR DATASETS, USE THESE EXACT PATHS:**

```python
# === YOUR PATHS ===
ANNOTATIONS_PATH = '/kaggle/input/trashcan-annotations-coco-format'
IMAGES_PATH = '/kaggle/input/trashcan/images'
```

> **Note:** The annotations path might need adjustment. Run Step 5 to verify!

---

### Step 5: Verify Paths (Run This First!)

**BEFORE** running the full training, run this verification cell:

```python
import os

print("=" * 70)
print("🔍 VERIFYING YOUR DATASET PATHS")
print("=" * 70)

# List all datasets
print("\n📂 Available datasets:")
for dataset in os.listdir('/kaggle/input'):
    print(f"  ✓ {dataset}")

# Check annotations structure
ann_base = '/kaggle/input/trashcan-annotations-coco-format'
print(f"\n📋 Contents of {ann_base}:")
for item in os.listdir(ann_base):
    item_path = os.path.join(ann_base, item)
    if os.path.isdir(item_path):
        count = len(os.listdir(item_path))
        print(f"  📁 {item}/ ({count} items)")
    else:
        size = os.path.getsize(item_path) / (1024*1024)
        print(f"  📄 {item} ({size:.1f} MB)")

# Check images structure
img_base = '/kaggle/input/trashcan'
print(f"\n🖼️  Contents of {img_base}:")
for item in os.listdir(img_base)[:5]:  # First 5 items
    item_path = os.path.join(img_base, item)
    if os.path.isdir(item_path):
        count = len(os.listdir(item_path))
        print(f"  📁 {item}/ ({count} items)")
    else:
        print(f"  📄 {item}")

print("\n" + "=" * 70)
```

**Expected Output Pattern:**

```
📂 Available datasets:
  ✓ trashcan-annotations-coco-format
  ✓ trashcan

📋 Contents of /kaggle/input/trashcan-annotations-coco-format:
  📄 train.json (22.0 MB)
  📄 val.json (5.6 MB)

🖼️  Contents of /kaggle/input/trashcan:
  📁 images/ (7212 items)
```

---

### Step 6: Adjust Paths Based on Output

**Scenario A:** If annotations are directly in the folder (most likely):
```python
ANNOTATIONS_PATH = '/kaggle/input/trashcan-annotations-coco-format'
IMAGES_PATH = '/kaggle/input/trashcan/images'
```

**Scenario B:** If annotations are in a subfolder:
```python
ANNOTATIONS_PATH = '/kaggle/input/trashcan-annotations-coco-format/annotations'
IMAGES_PATH = '/kaggle/input/trashcan/images'
```

**Scenario C:** If images are directly in the folder (no images/ subfolder):
```python
ANNOTATIONS_PATH = '/kaggle/input/trashcan-annotations-coco-format'
IMAGES_PATH = '/kaggle/input/trashcan'
```

Update the paths in your notebook based on what you see!

---

### Step 7: Final Verification

After setting the correct paths, run this cell:

```python
import json
import os

# Set your paths here
ANNOTATIONS_PATH = '/kaggle/input/trashcan-annotations-coco-format'  # Adjust if needed
IMAGES_PATH = '/kaggle/input/trashcan/images'  # Adjust if needed

print("🔍 Final Verification\n")

# Check train.json
train_json = os.path.join(ANNOTATIONS_PATH, 'train.json')
if os.path.exists(train_json):
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    print(f"✅ train.json found: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
else:
    print(f"❌ train.json NOT found at: {train_json}")
    print(f"   Try: {ANNOTATIONS_PATH}/annotations/train.json")

# Check val.json
val_json = os.path.join(ANNOTATIONS_PATH, 'val.json')
if os.path.exists(val_json):
    with open(val_json, 'r') as f:
        val_data = json.load(f)
    print(f"✅ val.json found: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
else:
    print(f"❌ val.json NOT found at: {val_json}")

# Check images
train_imgs_path = os.path.join(IMAGES_PATH, 'train')
val_imgs_path = os.path.join(IMAGES_PATH, 'val')

if os.path.exists(train_imgs_path):
    train_count = len([f for f in os.listdir(train_imgs_path) if f.endswith('.jpg')])
    print(f"✅ Train images: {train_count} files")
else:
    print(f"❌ Train images NOT found at: {train_imgs_path}")

if os.path.exists(val_imgs_path):
    val_count = len([f for f in os.listdir(val_imgs_path) if f.endswith('.jpg')])
    print(f"✅ Val images: {val_count} files")
else:
    print(f"❌ Val images NOT found at: {val_imgs_path}")

print("\n" + "="*70)
if os.path.exists(train_json) and os.path.exists(val_json) and os.path.exists(train_imgs_path) and os.path.exists(val_imgs_path):
    print("🎉 PERFECT! All files found. Ready to train!")
else:
    print("⚠️  Some files not found. Check the paths above.")
print("="*70)
```

**Expected Output:**
```
✅ train.json found: 6065 images, 9540 annotations
✅ val.json found: 1147 images, 2588 annotations
✅ Train images: 6065 files
✅ Val images: 1147 files

🎉 PERFECT! All files found. Ready to train!
```

---

### Step 8: Start Training! 🚀

If verification passed, you're ready!

Click **"Run All"** in the Kaggle notebook!

The training will:
- ⏱️ Take ~6-8 hours for 100 epochs on T4 GPU
- 💾 Save checkpoints every 10 epochs to `/kaggle/working/checkpoints/`
- 🔄 Auto-resume if session disconnects (just re-run training cell)
- 📊 Log metrics to TensorBoard
- 🏆 Save best model based on validation mAP

---

## 🎯 Quick Command Summary

**In Kaggle Notebook, use these paths:**

```python
# Most likely configuration:
ANNOTATIONS_PATH = '/kaggle/input/trashcan-annotations-coco-format'
IMAGES_PATH = '/kaggle/input/trashcan/images'
```

**Dataset IDs:**
```python
# For adding datasets programmatically:
annotations_dataset = 'k2001hede/trashcan-annotations-coco-format'
images_dataset = 'k2001hede/trashcan'
```

---

## 📊 Monitor Training

### View TensorBoard
In notebook, run:
```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/runs
```

### Check Checkpoints
```python
import glob
checkpoints = glob.glob('/kaggle/working/checkpoints/*.pth')
for ckpt in checkpoints:
    print(ckpt)
```

---

## 🔧 Troubleshooting

### Problem: "FileNotFoundError: train.json"

**Solution:** Run the verification cell (Step 5) to see actual structure, then update paths.

### Problem: "No such file or directory: images/train"

**Solution:** Images might be at `/kaggle/input/trashcan/train` (no images/ folder). Update:
```python
IMAGES_PATH = '/kaggle/input/trashcan'
```

### Problem: Training is slow

**Solution:** 
- Make sure GPU is enabled (Settings → Accelerator → GPU T4)
- Check batch size in config (reduce if OOM errors)
- Current config uses batch_size=8 (optimized for T4)

### Problem: Session disconnects

**Solution:** 
- Training auto-saves checkpoints every 10 epochs
- Just re-run the training cell with `--resume` flag
- Or use the notebook's auto-resume feature

---

## 📥 Download Trained Model

After training completes:

1. **From Notebook UI:**
   - Navigate to `/kaggle/working/checkpoints/`
   - Right-click on `best.pth` → Download

2. **Programmatically:**
```python
from kaggle_secrets import UserSecretsClient
import shutil

# Copy to Kaggle output
shutil.copy('/kaggle/working/checkpoints/best.pth', 
            '/kaggle/working/yolo_udd_best.pth')
print("✅ Model saved to /kaggle/working/yolo_udd_best.pth")
```

---

## ✅ Final Checklist

- [x] Datasets uploaded to Kaggle ✓
- [ ] Notebook created with GPU enabled
- [ ] Both datasets added via "+ Add Data"
- [ ] Paths updated in Step 2 cell
- [ ] Verification cell passed (all ✅)
- [ ] Training started
- [ ] TensorBoard monitoring active
- [ ] Checkpoints saving successfully

---

## 🎉 You're Ready!

Your personalized paths:
```
Annotations: /kaggle/input/trashcan-annotations-coco-format
Images:      /kaggle/input/trashcan/images
```

Follow Steps 1-8 above and you'll be training in minutes! 🚀

Good luck with your training! 🌊🗑️🤖
