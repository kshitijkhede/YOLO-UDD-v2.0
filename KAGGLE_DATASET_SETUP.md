# 📦 How to Add Datasets in Kaggle - Step by Step Guide

**Last Updated:** November 2, 2025

---

## 🎯 Overview

You need to upload **2 datasets** to Kaggle:
1. **Annotations** (train.json + val.json) - ~28 MB
2. **Images** (train + val folders) - ~28.6 MB

---

## 📤 PART 1: Upload Annotations to Kaggle

### Step 1.1: Prepare Annotations Locally

On your computer, open a terminal and run:

```bash
cd /home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/annotations

# Create a zip file with annotations
zip annotations.zip train.json val.json

# Verify the zip was created
ls -lh annotations.zip
```

You should see: `annotations.zip` (~3.7 MB)

### Step 1.2: Upload to Kaggle Website

1. **Open Browser** → Go to **https://www.kaggle.com/datasets**
2. **Log in** to your Kaggle account
3. Click **"New Dataset"** button (blue button, top right)

### Step 1.3: Upload the Annotations File

On the upload page:

1. **Upload File**:
   - Click **"Select Files to Upload"** or drag & drop
   - Select `annotations.zip` from your computer
   - Wait for upload to complete ✅

2. **Fill Dataset Info**:
   - **Title**: `TrashCAN Annotations COCO Format`
   - **Subtitle** (optional): `COCO format annotations for TrashCAN dataset`
   - **Description**: 
     ```
     COCO format annotations for TrashCAN underwater trash detection dataset.
     Contains 6,065 training images and 1,147 validation images.
     22 object categories including various trash types and marine life.
     ```

3. **Settings**:
   - **Visibility**: Choose "Public" or "Private" (your choice)
   - **License**: CC0: Public Domain (or your preference)

4. Click **"Create"** button

### Step 1.4: Note Your Dataset Path

After creation, Kaggle will show you a URL like:
```
https://www.kaggle.com/datasets/YOUR_USERNAME/trashcan-annotations-coco-format
```

**Important:** Save this path! You'll need it in the notebook.

The dataset path will be:
```
YOUR_USERNAME/trashcan-annotations-coco-format
```

---

## 🖼️ PART 2: Upload Images to Kaggle

### Step 2.1: Prepare Images Locally

On your computer, create a zip file with images:

```bash
cd /home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan

# Create zip with proper folder structure
zip -r trashcan_images.zip images/

# This creates a zip containing:
# images/train/ (6065 images)
# images/val/ (1147 images)

# Verify
ls -lh trashcan_images.zip
```

**Note:** If the zip file is too large (>100 MB), you can split it:

```bash
# Option A: Zip train and val separately
cd /home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/images
zip -r train_images.zip train/
zip -r val_images.zip val/

# Option B: Create compressed zip
zip -r -9 trashcan_images.zip images/
```

### Step 2.2: Upload to Kaggle Website

1. **Go to** → **https://www.kaggle.com/datasets**
2. Click **"New Dataset"** button

### Step 2.3: Upload the Images

On the upload page:

1. **Upload File**:
   - Click **"Select Files to Upload"**
   - Select `trashcan_images.zip` (or both train_images.zip and val_images.zip)
   - Wait for upload (may take several minutes for larger files)

2. **Fill Dataset Info**:
   - **Title**: `TrashCAN Images`
   - **Subtitle**: `Underwater trash detection images`
   - **Description**: 
     ```
     Image dataset for TrashCAN underwater object detection.
     - 6,065 training images (640x480 resolution)
     - 1,147 validation images (640x480 resolution)
     - Various underwater trash and marine life objects
     ```

3. **Settings**:
   - **Visibility**: Public or Private
   - **License**: Your choice

4. Click **"Create"**

### Step 2.4: Note Your Dataset Path

After creation, note the dataset path:
```
YOUR_USERNAME/trashcan-images
```

---

## 🔧 PART 3: Update Paths in Kaggle Notebook

Now that your datasets are uploaded, you need to tell the notebook where to find them.

### Step 3.1: Open Your Notebook in Kaggle

1. Go to **https://www.kaggle.com/code**
2. Click **"New Notebook"**
3. Upload `YOLO_UDD_Kaggle_Training_Fixed.ipynb`
   - OR create new notebook and copy cells

### Step 3.2: Add Your Datasets to the Notebook

**Before running any code:**

1. Look at the **RIGHT SIDEBAR** in Kaggle notebook
2. Click **"+ Add Data"** button
3. Click **"Your Datasets"** tab
4. Search for your datasets:
   - `trashcan-annotations-coco-format`
   - `trashcan-images`
5. Click **"Add"** for both datasets

This adds them to `/kaggle/input/` in your notebook environment.

### Step 3.3: Find the Exact Paths

After adding datasets, you need to find their exact paths. Run this cell in your notebook:

```python
import os

print("=" * 70)
print("🔍 FINDING DATASET PATHS")
print("=" * 70)

# List all input datasets
print("\n📂 Available datasets in /kaggle/input/:\n")
for dataset in os.listdir('/kaggle/input'):
    dataset_path = f'/kaggle/input/{dataset}'
    print(f"\n📁 {dataset}/")
    
    # List contents
    try:
        contents = os.listdir(dataset_path)
        for item in contents[:10]:  # Show first 10 items
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                count = len(os.listdir(item_path))
                print(f"   📁 {item}/ ({count} items)")
            else:
                size = os.path.getsize(item_path) / (1024*1024)
                print(f"   📄 {item} ({size:.1f} MB)")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
```

### Step 3.4: Update the Paths in Step 2

Find **Step 2** in the notebook (the cell that starts with "# === MODIFY THESE PATHS ===")

**BEFORE (default):**
```python
# === MODIFY THESE PATHS TO MATCH YOUR KAGGLE DATASETS ===
ANNOTATIONS_PATH = '/kaggle/input/trashcan-annotations-coco-format/annotations'
IMAGES_PATH = '/kaggle/input/trashcan/images'
```

**AFTER (update based on your findings):**

Example if your username is `johndoe`:
```python
# === MODIFY THESE PATHS TO MATCH YOUR KAGGLE DATASETS ===
ANNOTATIONS_PATH = '/kaggle/input/trashcan-annotations-coco-format'
IMAGES_PATH = '/kaggle/input/trashcan-images/images'
```

**Common Path Patterns:**

| Your Dataset Name | Likely Path |
|------------------|-------------|
| `trashcan-annotations-coco-format` | `/kaggle/input/trashcan-annotations-coco-format/` |
| `trashcan-images` | `/kaggle/input/trashcan-images/images/` |
| `trashcan` (if you named it that) | `/kaggle/input/trashcan/images/` |

**Important:** The exact path depends on how Kaggle extracts your zip files.

---

## ✅ PART 4: Verify Everything Works

After updating the paths, run these verification cells:

### Cell 1: Check Annotations
```python
import json
import os

print("📋 Checking annotations...")

train_json = os.path.join(ANNOTATIONS_PATH, 'train.json')
val_json = os.path.join(ANNOTATIONS_PATH, 'val.json')

if os.path.exists(train_json):
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    print(f"✅ train.json found: {len(train_data['images'])} images")
else:
    print(f"❌ train.json NOT found at: {train_json}")
    print(f"\n🔍 Files in {ANNOTATIONS_PATH}:")
    print(os.listdir(ANNOTATIONS_PATH))

if os.path.exists(val_json):
    with open(val_json, 'r') as f:
        val_data = json.load(f)
    print(f"✅ val.json found: {len(val_data['images'])} images")
else:
    print(f"❌ val.json NOT found at: {val_json}")
```

### Cell 2: Check Images
```python
import os

print("🖼️  Checking images...")

train_imgs = os.path.join(IMAGES_PATH, 'train')
val_imgs = os.path.join(IMAGES_PATH, 'val')

if os.path.exists(train_imgs):
    count = len([f for f in os.listdir(train_imgs) if f.endswith('.jpg')])
    print(f"✅ Train images found: {count} files")
else:
    print(f"❌ Train images NOT found at: {train_imgs}")
    print(f"\n🔍 Contents of {IMAGES_PATH}:")
    print(os.listdir(IMAGES_PATH))

if os.path.exists(val_imgs):
    count = len([f for f in os.listdir(val_imgs) if f.endswith('.jpg')])
    print(f"✅ Val images found: {count} files")
else:
    print(f"❌ Val images NOT found at: {val_imgs}")
```

If you see all ✅ checkmarks, you're ready to train! 🎉

---

## 🔧 Troubleshooting

### Problem 1: "Annotations not found"

**Cause:** Path is wrong or files are in a subfolder

**Solution:**
```python
# Run this to explore the structure
import os
base = '/kaggle/input/trashcan-annotations-coco-format'
for root, dirs, files in os.walk(base):
    level = root.replace(base, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f'{subindent}{file}')
```

Then update `ANNOTATIONS_PATH` to match the actual location.

### Problem 2: "Images not found"

**Cause:** Zip extracted to different structure

**Solution:**
```python
# Explore image dataset structure
import os
base = '/kaggle/input/trashcan-images'
for root, dirs, files in os.walk(base):
    level = root.replace(base, '').count(os.sep)
    if level < 3:  # Don't go too deep
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
```

Then update `IMAGES_PATH` accordingly.

### Problem 3: "Dataset not showing up"

**Cause:** Forgot to add dataset in Kaggle UI

**Solution:**
1. Look at **right sidebar** in notebook
2. Click **"+ Add Data"**
3. Find and add your datasets

---

## 📝 Quick Reference

### Your Dataset URLs (fill these in):
```
Annotations: https://www.kaggle.com/datasets/YOUR_USERNAME/_____________
Images: https://www.kaggle.com/datasets/YOUR_USERNAME/_____________
```

### Your Dataset Paths (fill these in):
```python
ANNOTATIONS_PATH = '/kaggle/input/_______________'
IMAGES_PATH = '/kaggle/input/_______________/images'
```

### Verification Checklist:
- [ ] Annotations uploaded to Kaggle
- [ ] Images uploaded to Kaggle
- [ ] Both datasets added to notebook (right sidebar)
- [ ] Paths updated in Step 2 cell
- [ ] Verification cells show ✅ for all checks
- [ ] Ready to start training!

---

## 🎉 Next Steps

Once everything is verified:
1. ✅ Set notebook to **GPU T4 or P100** (Settings → Accelerator)
2. ✅ Enable **Internet** (Settings → Internet → ON)
3. ✅ Click **"Run All"** or run cells one by one
4. 🚀 **Start Training!**

---

**Need Help?** 
- Check the troubleshooting section above
- Run the exploration cells to see actual file structure
- Make sure dataset names match in Kaggle
