# 📥 TrashCan Dataset Setup Instructions

## Problem
Your dataset directory is missing:
- ❌ COCO JSON annotation files
- ❌ Actual image files in train/val directories

## Solution

### Step 1: Download TrashCan Dataset

The TrashCan 1.0 dataset is available from J-Park Research:
- **Website**: https://conservancy.umn.edu/handle/11299/214865
- **Paper**: "Trash in the Ocean: A Deep Learning Approach" (IEEE OCEANS 2020)

### Step 2: Expected Directory Structure

After downloading, your structure should look like:

```
data/trashcan/
├── instances_train_trashcan.json    ← COCO format annotations
├── instances_val_trashcan.json      ← COCO format annotations
└── images/
    ├── train/
    │   ├── image_0001.jpg
    │   ├── image_0002.jpg
    │   └── ... (thousands of images)
    └── val/
        ├── image_0001.jpg
        ├── image_0002.jpg
        └── ... (validation images)
```

### Step 3: COCO JSON Format

The annotation files should be in COCO format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image_0001.jpg",
      "height": 1080,
      "width": 1920
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 250],
      "area": 50000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "trash"},
    {"id": 2, "name": "animal"},
    {"id": 3, "name": "rov"}
  ]
}
```

### Step 4: Verify Setup

Run this diagnostic script to verify your dataset:

```python
python -c "
import os
import json

data_dir = 'data/trashcan'

# Check annotations
train_ann = os.path.join(data_dir, 'instances_train_trashcan.json')
val_ann = os.path.join(data_dir, 'instances_val_trashcan.json')

print('Checking annotations...')
for ann_file in [train_ann, val_ann]:
    if os.path.exists(ann_file):
        with open(ann_file, 'r') as f:
            data = json.load(f)
            print(f'✓ {ann_file}')
            print(f'  - Images: {len(data.get(\"images\", []))}')
            print(f'  - Annotations: {len(data.get(\"annotations\", []))}')
    else:
        print(f'✗ {ann_file} NOT FOUND')

# Check images
train_dir = os.path.join(data_dir, 'images', 'train')
val_dir = os.path.join(data_dir, 'images', 'val')

print('\nChecking images...')
for img_dir in [train_dir, val_dir]:
    if os.path.isdir(img_dir):
        num_images = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        print(f'✓ {img_dir}: {num_images} images')
    else:
        print(f'✗ {img_dir} NOT A DIRECTORY')
"
```

## Alternative: Create Sample Dataset for Testing

If you can't access the real dataset, create a minimal test dataset:

```python
python scripts/create_dummy_dataset.py
```

This will create a small synthetic dataset for testing the code.
