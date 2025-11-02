# 📋 Kaggle Dataset Setup - Quick Cheat Sheet

## 🎯 TL;DR - 3 Simple Steps

### 1️⃣ Prepare Files Locally
```bash
# Annotations
cd /home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/annotations
zip annotations.zip train.json val.json

# Images
cd /home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan
zip -r trashcan_images.zip images/
```

### 2️⃣ Upload to Kaggle
1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset" → Upload `annotations.zip` → Title: "TrashCAN Annotations COCO Format"
3. Click "New Dataset" → Upload `trashcan_images.zip` → Title: "TrashCAN Images"

### 3️⃣ Update Notebook Paths
In Kaggle notebook, find Step 2 and update:

```python
# Replace these with your actual dataset names
ANNOTATIONS_PATH = '/kaggle/input/trashcan-annotations-coco-format'
IMAGES_PATH = '/kaggle/input/trashcan-images/images'
```

---

## 🔍 Finding Your Dataset Paths

After uploading to Kaggle, your datasets will be at:
```
/kaggle/input/YOUR-DATASET-NAME/
```

To find the exact path, add the dataset to your notebook:
1. Right sidebar → "+ Add Data"
2. Select your datasets
3. They appear as: `/kaggle/input/dataset-name-with-dashes/`

---

## 🧪 Test Your Paths

Run this in a Kaggle notebook cell:

```python
import os

# List your datasets
print("Your datasets:")
for d in os.listdir('/kaggle/input'):
    print(f"  /kaggle/input/{d}")

# Test annotations path
ANNOTATIONS_PATH = '/kaggle/input/YOUR-DATASET-NAME'  # Update this
print(f"\nAnnotations: {os.listdir(ANNOTATIONS_PATH)}")

# Test images path  
IMAGES_PATH = '/kaggle/input/YOUR-DATASET-NAME/images'  # Update this
print(f"Images: {os.listdir(IMAGES_PATH)}")
```

**Expected output:**
```
Your datasets:
  /kaggle/input/trashcan-annotations-coco-format
  /kaggle/input/trashcan-images

Annotations: ['train.json', 'val.json']
Images: ['train', 'val']
```

If you see this ✅ → Ready to train!

---

## 🚨 Common Issues

| Error | Fix |
|-------|-----|
| `FileNotFoundError: train.json` | Wrong path! Add `/annotations` or check structure |
| `Dataset not found` | Forgot to add dataset via "+ Add Data" button |
| `No such directory: images` | Try without `/images` at end |

---

## ✅ Final Checklist

Before starting training:
- [ ] Both datasets uploaded to Kaggle ✓
- [ ] Datasets added to notebook via "+ Add Data" ✓
- [ ] Paths updated in notebook Step 2 ✓
- [ ] Verification cell shows correct files ✓
- [ ] GPU enabled (T4 or P100) ✓
- [ ] Internet ON ✓

🚀 **Ready to train!**

---

**Full guide:** See `KAGGLE_DATASET_SETUP.md` for detailed instructions.
