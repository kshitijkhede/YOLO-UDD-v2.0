# 🚨 Dataset Issue - Resolution Guide

## Current Problem

Your TrashCan dataset diagnostic shows:
- ❌ **Missing COCO JSON files**: `instances_train_trashcan.json` and `instances_val_trashcan.json` not found
- ❌ **No images**: 0 image files found in the directories
- ❌ **Empty directories**: `train` and `val` are empty or just placeholder files

## Root Cause

The dataset is not properly downloaded and set up. The project expects:
1. COCO-format annotation JSON files
2. Actual underwater trash images
3. Proper directory structure

---

## 🎯 Solution Options

### **Option A: Get the Real TrashCan Dataset (RECOMMENDED)**

The TrashCan dataset is required for real training. Here's how to get it:

#### 1. Download from Official Source
- **Source**: University of Minnesota Digital Conservancy
- **URL**: https://conservancy.umn.edu/handle/11299/214865
- **Paper**: "Trash in the Ocean: A Deep Learning Approach" (IEEE OCEANS 2020)

#### 2. Alternative Sources
- **Kaggle**: Search for "TrashCan dataset underwater"
- **Roboflow**: May have pre-processed versions
- **Contact**: Reach out to the paper authors if needed

#### 3. Required Files
After downloading, you should have:
```
instances_train_trashcan.json  (COCO format annotations)
instances_val_trashcan.json    (COCO format annotations)
train/                         (thousands of underwater images)
val/                          (validation images)
```

#### 4. Setup Instructions
1. Download the dataset ZIP file
2. Extract it
3. Copy/move files to: `F:\MIR\project\YOLO-UDD-v2.0-main\data\trashcan\`
4. Ensure structure matches below

---

### **Option B: Create Synthetic Test Dataset**

For code testing without real data:

#### Prerequisites
Install Python and required packages first:
```powershell
# Install Python from python.org or Microsoft Store
# Then install required packages:
pip install pillow numpy
```

#### Run the Generator
```powershell
# Navigate to project directory
cd F:\MIR\project\YOLO-UDD-v2.0-main

# Generate dummy dataset
python scripts/create_dummy_dataset.py --num_train 50 --num_val 20
```

This creates synthetic colored rectangles simulating objects for code testing only.

---

## 📁 Required Directory Structure

After setup, your `data/trashcan/` should look like this:

```
F:\MIR\project\YOLO-UDD-v2.0-main\data\trashcan\
│
├── instances_train_trashcan.json    ← COCO annotations for training
├── instances_val_trashcan.json      ← COCO annotations for validation
│
└── images/
    ├── train/
    │   ├── frame_0001.jpg
    │   ├── frame_0002.jpg
    │   ├── frame_0003.jpg
    │   └── ... (many more images)
    │
    └── val/
        ├── frame_5001.jpg
        ├── frame_5002.jpg
        └── ... (validation images)
```

---

## 📋 COCO JSON Format Example

Your annotation files should follow this structure:

```json
{
  "info": {
    "description": "TrashCan Dataset 1.0",
    "version": "1.0",
    "year": 2020
  },
  
  "images": [
    {
      "id": 1,
      "file_name": "frame_0001.jpg",
      "height": 1080,
      "width": 1920,
      "date_captured": ""
    }
  ],
  
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [450, 320, 180, 220],
      "area": 39600,
      "iscrowd": 0,
      "segmentation": []
    }
  ],
  
  "categories": [
    {"id": 1, "name": "trash", "supercategory": "object"},
    {"id": 2, "name": "animal", "supercategory": "object"},
    {"id": 3, "name": "rov", "supercategory": "object"}
  ]
}
```

**Note**: `bbox` format is `[x, y, width, height]` in absolute pixel coordinates.

---

## ✅ Verification Steps

### Step 1: Check File Existence
```powershell
# Check if files exist
Test-Path "F:\MIR\project\YOLO-UDD-v2.0-main\data\trashcan\instances_train_trashcan.json"
Test-Path "F:\MIR\project\YOLO-UDD-v2.0-main\data\trashcan\instances_val_trashcan.json"

# Count images
(Get-ChildItem "F:\MIR\project\YOLO-UDD-v2.0-main\data\trashcan\images\train\*.jpg").Count
(Get-ChildItem "F:\MIR\project\YOLO-UDD-v2.0-main\data\trashcan\images\val\*.jpg").Count
```

### Step 2: Validate JSON Files (with Python)
```python
import json
import os

# Load training annotations
with open('data/trashcan/instances_train_trashcan.json', 'r') as f:
    train_data = json.load(f)

print(f"Training images: {len(train_data['images'])}")
print(f"Training annotations: {len(train_data['annotations'])}")
print(f"Categories: {train_data['categories']}")

# Load validation annotations
with open('data/trashcan/instances_val_trashcan.json', 'r') as f:
    val_data = json.load(f)

print(f"\nValidation images: {len(val_data['images'])}")
print(f"Validation annotations: {len(val_data['annotations'])}")
```

### Step 3: Test Dataset Loader
```python
from data.dataset import TrashCanDataset

# Create dataset
train_dataset = TrashCanDataset(
    data_dir='data/trashcan',
    split='train',
    img_size=640,
    augment=True
)

print(f"Dataset loaded successfully!")
print(f"Total training samples: {len(train_dataset)}")
print(f"Number of classes: {train_dataset.num_classes}")

# Test loading a sample
if len(train_dataset) > 0:
    sample = train_dataset[0]
    print(f"\nSample loaded:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Bboxes: {sample['bboxes'].shape}")
    print(f"  Labels: {sample['labels'].shape}")
```

---

## 🔍 Common Issues & Fixes

### Issue 1: "JSON file not found"
**Solution**: Ensure annotation files are in the correct location with exact filenames:
- `instances_train_trashcan.json`
- `instances_val_trashcan.json`

### Issue 2: "Could not load image"
**Solution**: 
- Verify image paths in JSON match actual filenames
- Ensure images are in `data/trashcan/images/train/` and `data/trashcan/images/val/`
- Check image file extensions (.jpg, .png)

### Issue 3: "Empty annotations"
**Solution**: 
- Open JSON files and verify they contain actual data
- Ensure `images` and `annotations` arrays are not empty
- Validate JSON syntax (no trailing commas, proper brackets)

### Issue 4: "Module not found" errors
**Solution**:
```powershell
# Install required packages
pip install -r requirements.txt
```

---

## 📞 Next Steps

1. **Install Python** if not already installed
   - Download from: https://www.python.org/downloads/
   - Or use Microsoft Store version

2. **Choose your path**:
   - **For real training**: Get the actual TrashCan dataset
   - **For code testing**: Run the dummy dataset generator

3. **Verify setup** using the commands above

4. **Start training**:
   ```powershell
   python scripts/train.py --config configs/train_config.yaml
   ```

---

## 📚 Additional Resources

- **TrashCan Paper**: https://arxiv.org/abs/2007.08097
- **COCO Format**: https://cocodataset.org/#format-data
- **PyTorch Datasets**: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- **Project Documentation**: See `DOCUMENTATION.md`

---

## 📝 Quick Reference Commands

```powershell
# Generate dummy dataset
python scripts/create_dummy_dataset.py

# Verify dataset
python -c "from data.dataset import TrashCanDataset; ds = TrashCanDataset('data/trashcan'); print(f'Size: {len(ds)}')"

# Train model
python scripts/train.py --config configs/train_config.yaml

# Run detection
python scripts/detect.py --weights checkpoints/best.pth --source test_image.jpg
```

---

**Remember**: The dummy dataset is only for testing code structure. For actual model performance, you need the real TrashCan dataset with genuine underwater imagery!
