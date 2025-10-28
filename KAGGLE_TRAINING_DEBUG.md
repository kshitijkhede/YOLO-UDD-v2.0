# Kaggle Training Debug Guide

## Current Status
We've identified and are debugging an issue where the training script shows it's looking for annotations in `/kaggle/input/trashcan-underwater-debris-detection/` instead of `/kaggle/working/trashcan/` despite passing the correct path.

## Recent Changes (Commits)

### 1. **Cell 4.5: Dataset Verification** (Commit: 668e5e2)
Added a new verification cell that runs BEFORE training starts:
- Checks if `/kaggle/working/trashcan/` exists
- Lists directory contents
- Verifies `annotations/train.json` and `val.json` are present
- Counts images in the dataset
- Provides clear success/error messages

**Purpose:** Confirms the restructured dataset is in place before training.

### 2. **Cell 6: Enhanced Training Cell** (Same commit)
Improved the training cell with pre-flight checks:
- Verifies dataset directory exists
- Checks annotation files are present
- Shows the **absolute path** being used
- Better error messages if files are missing
- Shows full command being executed

### 3. **Debug Prints in Dataset Loader** (Commit: 3d0d02d)
Added detailed logging to `data/dataset.py`:
- Shows the actual `data_dir` received by `TrashCanDataset`
- Shows which paths are being checked for annotations
- Helps trace where the path might be getting changed

## How to Use on Kaggle

### Step 1: Import Updated Notebook
1. Go to your Kaggle notebook
2. Click **File → Import Notebook**
3. Select **GitHub** tab
4. Enter: `kshitijkhede/YOLO-UDD-v2.0`
5. Select `YOLO_UDD_Kaggle.ipynb`

### Step 2: Run Cells Sequentially

```
Cell 1: Environment Setup
├─> Installs NumPy 1.26.4
└─> Clones repository

Cell 2: Install Dependencies
├─> Reinstalls packages against NumPy 1.x
└─> Prevents binary incompatibility

Cell 3: Setup Dataset ⭐ CRITICAL
├─> Detects Kaggle dataset structure
├─> Restructures to /kaggle/working/trashcan/
├─> Creates annotations/train.json and val.json
└─> Merges images into single folder

Cell 4: Build & Test Model
└─> Quick model verification

Cell 4.5: Verify Dataset Path 🆕 NEW
├─> Confirms restructured dataset exists
├─> Lists directory contents
├─> Verifies annotation files
└─> Counts images
⚠️ IMPORTANT: Check output carefully!

Cell 5: (Optional) Manual Model Test
└─> Can skip if Cell 4 worked

Cell 6: Start Training 🚀
├─> Pre-flight checks
├─> Shows absolute path being used
├─> Displays full command
└─> Starts training
⚠️ Watch the DEBUG output!

Cell 7: Check Results
└─> Run after training completes
```

### Step 3: What to Look For

When you run **Cell 4.5**, you should see:
```
Verifying dataset location at: /kaggle/working/trashcan/
✓ Dataset directory exists
Directory contents:
  - annotations/ (2 items)
  - images/ (7212 items)

Checking critical files:
  ✓ annotations/train.json exists (size: X bytes)
  ✓ annotations/val.json exists (size: Y bytes)

Image count: 7212 images found
✓ Dataset is ready for training!
```

When you run **Cell 6** (training), watch for these DEBUG messages:
```
[DEBUG] TrashCanDataset(train): data_dir = /kaggle/working/trashcan
[DEBUG] Checking TrashCAN format: /kaggle/working/trashcan/instances_train_trashcan.json
[DEBUG] Checking standard format: /kaggle/working/trashcan/annotations/train.json
✓ Loading annotations from: /kaggle/working/trashcan/annotations/train.json
```

## Expected Behavior

### ✅ SUCCESS - Should see:
```
CELL 6: Starting Training
======================================================================

Training Configuration:
  Epochs:       100
  Batch Size:   8
  Learning Rate: 0.01
  Dataset:      /kaggle/working/trashcan
  Save Dir:     /kaggle/working/runs/train

Checking annotations:
  train.json: Found
  val.json:   Found

Starting training...
Using absolute dataset path: /kaggle/working/trashcan
Command: python scripts/train.py --config configs/train_config.yaml --data-dir /kaggle/working/trashcan ...

[DEBUG] TrashCanDataset(train): data_dir = /kaggle/working/trashcan
[DEBUG] Checking standard format: /kaggle/working/trashcan/annotations/train.json
✓ Loading annotations from: /kaggle/working/trashcan/annotations/train.json
```

### ❌ FAILURE - What you saw before:
```
Warning: Annotation file not found: /kaggle/input/trashcan-underwater-debris-detection/annotations/train.json
```

## What the Debug Changes Will Tell Us

1. **Cell 4.5 Output**: 
   - Confirms dataset exists at correct location
   - Verifies files are in place
   - If this FAILS → Cell 3 didn't run or had issues

2. **Cell 6 Pre-flight Checks**:
   - Shows the exact path being passed to training script
   - If this shows `/kaggle/input/...` → Problem in Cell 6 code
   - If this shows `/kaggle/working/...` → Keep checking...

3. **DEBUG Prints from dataset.py**:
   - Shows what `data_dir` the TrashCanDataset actually receives
   - Shows which paths it checks for annotations
   - If this shows `/kaggle/input/...` → Path got changed somewhere between train.py and dataset.py
   - If this shows `/kaggle/working/...` but still fails → Annotations might be missing

## Possible Issues & Solutions

### Issue 1: Cell 4.5 shows dataset NOT found
**Solution:** Run Cell 3 again. It should restructure the dataset.

### Issue 2: Cell 6 pre-flight shows annotations NOT found
**Solution:** Run Cell 3 again. The restructuring might have failed.

### Issue 3: DEBUG shows `/kaggle/input/...` path
**Cause:** Something is overriding the `--data-dir` argument
**Next Step:** Check if `train_config.yaml` has hardcoded path (we checked, it doesn't)

### Issue 4: DEBUG shows correct path but file doesn't exist
**Cause:** Restructuring created wrong directory structure
**Next Step:** Manually check `/kaggle/working/trashcan/annotations/` exists

## Manual Verification Commands

If needed, you can add a code cell to manually check:

```python
import os
import json

# Check dataset structure
base = '/kaggle/working/trashcan'
print("Checking dataset structure:")
print(f"  Base exists: {os.path.exists(base)}")
print(f"  Annotations exists: {os.path.exists(os.path.join(base, 'annotations'))}")
print(f"  Images exists: {os.path.exists(os.path.join(base, 'images'))}")

# Check annotation files
train_json = os.path.join(base, 'annotations', 'train.json')
val_json = os.path.join(base, 'annotations', 'val.json')
print(f"\nAnnotation files:")
print(f"  train.json: {os.path.exists(train_json)}")
print(f"  val.json: {os.path.exists(val_json)}")

if os.path.exists(train_json):
    with open(train_json) as f:
        data = json.load(f)
        print(f"  train.json has {len(data.get('images', []))} images")
        print(f"  train.json has {len(data.get('annotations', []))} annotations")
```

## What to Report Back

Please run Cells 1-6 and share:

1. ✅ **Cell 3 output** - Did restructuring complete?
2. ✅ **Cell 4.5 output** - What does verification show?
3. ✅ **Cell 6 pre-flight** - What path is shown?
4. ✅ **Cell 6 DEBUG prints** - What does TrashCanDataset receive?
5. ✅ **Any error messages** - Full error text

This will help us pinpoint exactly where the path is getting changed!
