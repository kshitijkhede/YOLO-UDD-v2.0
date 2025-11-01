# 📤 How to Upload Dataset to Kaggle - Visual Guide

## 📍 **Your Files Location:**
```
/home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/annotations/annotations.zip
```

**File ready to upload:**
- ✅ `annotations.zip` (3.7 MB) - Contains train.json and val.json

---

## 🌐 **OPTION 1: Upload via Kaggle Website (Easiest)**

### **Step 1: Open Kaggle Datasets Page**

1. Open your web browser
2. Go to: **https://www.kaggle.com/datasets**
3. Log in to your Kaggle account (if not already logged in)

### **Step 2: Create New Dataset**

1. Click the **"New Dataset"** button (blue button, top right corner)
2. You'll see an upload page

### **Step 3: Upload Your File**

**Two ways to upload:**

#### Option A: Drag and Drop
1. Open your file manager
2. Navigate to: `/home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/annotations/`
3. Find `annotations.zip`
4. **Drag it** to the Kaggle upload area

#### Option B: Click to Upload
1. Click **"Upload"** or **"Select Files to Upload"**
2. Navigate to: `/home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/annotations/`
3. Select `annotations.zip`
4. Click **"Open"**

### **Step 4: Fill in Dataset Information**

Wait for the upload to complete (should take ~1 minute), then fill in:

**Required Fields:**
- **Title:** `TrashCAN Annotations COCO Format`
- **Subtitle:** `COCO format annotations for underwater debris detection`

**Description (copy and paste this):**
```
Annotations for TrashCAN 1.0 dataset in COCO format for underwater debris detection.

Dataset Contents:
- train.json: 6,065 training images with 9,540 annotations
- val.json: 1,147 validation images with 2,588 annotations
- Total Classes: 22 (underwater objects and debris)

Classes Include:
- Marine life: rov, plant, fish, starfish, shells, crab, eel, etc.
- Trash items: bottles, bags, pipes, clothing, containers, nets, ropes, wrappers, cans, cups, branches, wreckage, tarps, etc.

Format: COCO JSON format
Compatible with: YOLO-UDD v2.0 training pipeline
Project: https://github.com/kshitijkhede/YOLO-UDD-v2.0

Usage:
These annotations are designed for training object detection models to identify underwater debris and marine life. The annotations follow the standard COCO format with bounding boxes, categories, and image metadata.
```

**Settings:**
- **Visibility:** Choose "Public" (so you can access it easily) or "Private" (if you want to keep it restricted)
- **License:** Choose "CC0: Public Domain" or "Attribution 4.0 International"

### **Step 5: Create Dataset**

1. Review your information
2. Click **"Create"** button (bottom of page)
3. Wait a few seconds for processing

### **Step 6: Note Your Dataset Path**

After creation, Kaggle will show you the dataset page. The URL will look like:
```
https://www.kaggle.com/datasets/YOUR-USERNAME/trashcan-annotations-coco-format
```

**IMPORTANT:** Write down your dataset path:
```
YOUR-USERNAME/trashcan-annotations-coco-format
```

You'll need this exact path when setting up your training notebook!

---

## 🌐 **OPTION 2: Upload via Kaggle CLI (For Advanced Users)**

If you have Kaggle CLI installed:

### **Install Kaggle CLI:**
```bash
pip install kaggle
```

### **Setup API Credentials:**
1. Go to: https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New API Token"
4. Download `kaggle.json`
5. Move it to: `~/.kaggle/kaggle.json`
6. Set permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### **Create Dataset Metadata:**
```bash
cd /home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/annotations

cat > dataset-metadata.json << 'EOF'
{
  "title": "TrashCAN Annotations COCO Format",
  "id": "YOUR-USERNAME/trashcan-annotations-coco-format",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
```

### **Upload Dataset:**
```bash
kaggle datasets create -p /home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/annotations
```

---

## ✅ **Verify Upload Success**

After uploading, you should see:
- ✅ Dataset page with your title and description
- ✅ Files tab showing: `train.json` and `val.json`
- ✅ File sizes: ~22 MB and ~5.6 MB
- ✅ Dataset is accessible (public or private as you chose)

---

## 🚀 **Next Steps: Use in Kaggle Notebook**

Now that your dataset is uploaded:

### **1. Create New Notebook**
- Go to: https://www.kaggle.com/code
- Click "New Notebook"
- Enable GPU: Settings → Accelerator → GPU P100/T4
- Enable Internet: Settings → Internet → ON

### **2. Add Your Dataset**
- Click "+ Add Data" (right sidebar)
- Search: `YOUR-USERNAME/trashcan-annotations-coco-format`
- Click "Add"

### **3. Access Files in Code**
Your annotations will be at:
```python
annotations_path = '/kaggle/input/trashcan-annotations-coco-format'
train_json = f'{annotations_path}/train.json'
val_json = f'{annotations_path}/val.json'
```

### **4. Verify in Notebook**
Run this in your first cell:
```python
import os
import json

# Check if files exist
annotations_path = '/kaggle/input/trashcan-annotations-coco-format'
print(f"Checking path: {annotations_path}")
print(f"Path exists: {os.path.exists(annotations_path)}")

# List files
print("\nFiles in dataset:")
for file in os.listdir(annotations_path):
    file_path = os.path.join(annotations_path, file)
    size = os.path.getsize(file_path) / (1024*1024)  # MB
    print(f"  - {file} ({size:.1f} MB)")

# Load and verify train.json
with open(f'{annotations_path}/train.json', 'r') as f:
    train_data = json.load(f)

print(f"\n✅ Train annotations loaded!")
print(f"   Images: {len(train_data['images'])}")
print(f"   Annotations: {len(train_data['annotations'])}")
print(f"   Categories: {len(train_data['categories'])}")

# Load and verify val.json
with open(f'{annotations_path}/val.json', 'r') as f:
    val_data = json.load(f)

print(f"\n✅ Val annotations loaded!")
print(f"   Images: {len(val_data['images'])}")
print(f"   Annotations: {len(val_data['annotations'])}")
print(f"   Categories: {len(val_data['categories'])}")
```

Expected output:
```
Checking path: /kaggle/input/trashcan-annotations-coco-format
Path exists: True

Files in dataset:
  - train.json (22.6 MB)
  - val.json (5.9 MB)

✅ Train annotations loaded!
   Images: 6065
   Annotations: 9540
   Categories: 22

✅ Val annotations loaded!
   Images: 1147
   Annotations: 2588
   Categories: 22
```

---

## 🔧 **Troubleshooting**

### ❌ **"Upload failed"**
- Check your internet connection
- Try uploading just the zip file (not individual JSON files)
- File size limit: 20 GB (your 3.7 MB zip is fine)

### ❌ **"Dataset not found in notebook"**
- Make sure you clicked "+ Add Data" in the notebook
- Search for the exact dataset name
- Check if dataset is set to "Public" or you're logged in

### ❌ **"Files not extracting from zip"**
- Kaggle automatically extracts zip files
- If not, you can extract manually in notebook:
```python
import zipfile
with zipfile.ZipFile('/kaggle/input/trashcan-annotations-coco-format/annotations.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/annotations')
```

---

## 📊 **What About Images?**

**Good News:** You don't need to upload the images for initial training!

The TrashCAN dataset images are likely already available on Kaggle. Search for:
- "TrashCAN dataset"
- "Underwater debris detection"
- "Marine debris detection"

Or you can:
1. Use a public TrashCAN dataset on Kaggle (if available)
2. Upload your own images later (they're larger, will take longer)
3. Train with annotations only initially to verify the code works

To find existing datasets:
1. Go to: https://www.kaggle.com/datasets
2. Search: "TrashCAN" or "underwater debris"
3. Look for datasets with ~7,000 images
4. Add it to your notebook along with your annotations

---

## 💡 **Pro Tips**

1. **Keep the zip file:** Don't delete `annotations.zip` - you might need it again
2. **Test upload:** Create a "test" version first to make sure everything works
3. **Version control:** Kaggle supports dataset versions - you can update later
4. **Public vs Private:** 
   - Public = easier to access, can share with others
   - Private = only you can access, counts toward storage quota
5. **Storage limits:** Free tier has limits, but your 3.7 MB is tiny

---

## ✅ **Checklist**

Before uploading:
- [ ] File ready: `annotations.zip` (3.7 MB)
- [ ] Kaggle account created and logged in
- [ ] Internet connection stable

During upload:
- [ ] Navigate to kaggle.com/datasets
- [ ] Click "New Dataset"
- [ ] Upload `annotations.zip`
- [ ] Fill in title and description
- [ ] Choose visibility (public/private)
- [ ] Click "Create"

After upload:
- [ ] Dataset page loads successfully
- [ ] Can see train.json and val.json in Files tab
- [ ] Note down dataset path: `USERNAME/trashcan-annotations-coco-format`
- [ ] Ready to use in notebook!

---

## 🎉 **You're Ready!**

Once you see your dataset on Kaggle with the two JSON files visible, you're all set to create your training notebook!

Refer back to **KAGGLE_SETUP_GUIDE.md** for the complete training code.

---

*Last updated: November 1, 2025*
