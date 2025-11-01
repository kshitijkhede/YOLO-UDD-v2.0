# 🚀 Complete Step-by-Step Guide: Run YOLO-UDD v2.0 on Kaggle

**Last Updated:** November 1, 2025

---

## 📋 **Prerequisites**

Before you start, make sure you have:
- ✅ Kaggle account (free): https://www.kaggle.com/
- ✅ Annotation files on your local machine:
  - `train.json` (22 MB) - `/home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/annotations/train.json`
  - `val.json` (5.6 MB) - `/home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/annotations/val.json`
- ✅ Image dataset (Optional - can be uploaded separately or used from existing Kaggle dataset)

---

## 🎯 **OPTION 1: Quick Start (Using Kaggle Dataset API)**

### **Step 1: Create Kaggle Dataset for Annotations**

#### 1.1 Prepare Annotations Locally
```bash
# On your local machine
cd /home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/annotations
zip annotations.zip train.json val.json
```

#### 1.2 Upload to Kaggle
1. Go to: https://www.kaggle.com/datasets
2. Click **"New Dataset"** (blue button, top right)
3. Click **"Upload"** or drag `annotations.zip`
4. Fill in details:
   - **Title**: `TrashCAN Annotations COCO Format`
   - **Subtitle**: `COCO format annotations for underwater debris detection`
   - **Description**: 
     ```
     Annotations for TrashCAN 1.0 dataset in COCO format.
     - train.json: 6,065 images, 9,540 annotations, 22 classes
     - val.json: 1,147 images, 2,588 annotations, 22 classes
     
     Classes: rov, plant, various animals (fish, starfish, shells, crab, eel, etc), 
     various trash items (clothing, pipe, bottle, bag, wrapper, can, cup, container, etc)
     ```
   - **Visibility**: Public or Private (your choice)
5. Click **"Create"**
6. **Note your dataset path**: `YOUR_USERNAME/trashcan-annotations-coco-format`

### **Step 2: Get TrashCAN Images Dataset**

You have two options:

#### Option A: Use Existing Kaggle Dataset (Recommended)
Search for "TrashCAN" or "underwater debris" on Kaggle and use existing dataset.

#### Option B: Upload Your Own Images
1. On your local machine:
   ```bash
   cd /home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan
   zip -r images.zip images/
   ```
2. Upload to Kaggle as a dataset (same process as annotations)
3. **Note the dataset path**: `YOUR_USERNAME/trashcan-images`

### **Step 3: Create Kaggle Notebook**

1. Go to: https://www.kaggle.com/code
2. Click **"New Notebook"** 
3. **Important Settings**:
   - Click **Settings** (gear icon, top right)
   - **Accelerator**: Select **GPU P100** or **GPU T4** ⚡
   - **Internet**: Turn **ON** 🌐
   - **Persistence**: Files Only (default)

### **Step 4: Add Datasets to Notebook**

1. In notebook, click **"+ Add Data"** (right sidebar)
2. Search and add your datasets:
   - `YOUR_USERNAME/trashcan-annotations-coco-format`
   - `YOUR_USERNAME/trashcan-images` (or existing TrashCAN dataset)
3. Datasets will be available at:
   - `/kaggle/input/trashcan-annotations-coco-format/`
   - `/kaggle/input/trashcan-images/` (or similar)

### **Step 5: Run Training Code**

Copy and paste the following cells into your Kaggle notebook:

#### Cell 1: Clone Repository
```python
# Clone YOLO-UDD v2.0 repository
!git clone https://github.com/kshitijkhede/YOLO-UDD-v2.0.git
%cd YOLO-UDD-v2.0

# Verify we're in the right place
!pwd
!ls -la
```

#### Cell 2: Install Dependencies
```python
# Install required packages
!pip install -q torch torchvision torchaudio
!pip install -q albumentations
!pip install -q pycocotools
!pip install -q tensorboard
!pip install -q tqdm
!pip install -q pyyaml

print("✅ All dependencies installed!")
```

#### Cell 3: Verify GPU
```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

#### Cell 4: Setup Dataset Structure
```python
import os
import shutil

# Create directory structure
os.makedirs('data/trashcan/annotations', exist_ok=True)
os.makedirs('data/trashcan/images', exist_ok=True)

# Copy annotations from Kaggle input
# ADJUST THESE PATHS based on your dataset names
annotations_path = '/kaggle/input/trashcan-annotations-coco-format'
images_path = '/kaggle/input/trashcan-images'  # or your image dataset path

# Copy annotations
if os.path.exists(annotations_path):
    shutil.copy(f'{annotations_path}/train.json', 'data/trashcan/annotations/train.json')
    shutil.copy(f'{annotations_path}/val.json', 'data/trashcan/annotations/val.json')
    print("✅ Annotations copied!")
    
    # Verify
    import json
    with open('data/trashcan/annotations/train.json', 'r') as f:
        train_data = json.load(f)
    with open('data/trashcan/annotations/val.json', 'r') as f:
        val_data = json.load(f)
    
    print(f"Train images: {len(train_data['images'])}")
    print(f"Train annotations: {len(train_data['annotations'])}")
    print(f"Val images: {len(val_data['images'])}")
    print(f"Val annotations: {len(val_data['annotations'])}")
else:
    print("❌ Annotations path not found! Check your dataset path.")

# Create symlink or copy images
if os.path.exists(images_path):
    # Symlink is faster than copying
    !ln -s {images_path}/images/* data/trashcan/images/
    print("✅ Images linked!")
else:
    print("⚠️  Images path not found. Will need to adjust.")
```

#### Cell 5: Create Kaggle Training Config
```python
# Create optimized config for Kaggle
import yaml

config = {
    'model': {
        'name': 'YOLO-UDD-v2.0',
        'num_classes': 22,
        'pretrained_path': None
    },
    'data': {
        'dataset_name': 'TrashCan-1.0',
        'data_dir': 'data/trashcan',
        'img_size': 640,
        'class_names': [
            "rov", "plant", "animal_fish", "animal_starfish", "animal_shells",
            "animal_crab", "animal_eel", "animal_etc", "trash_clothing", "trash_pipe",
            "trash_bottle", "trash_bag", "trash_snack_wrapper", "trash_can", "trash_cup",
            "trash_container", "trash_unknown_instance", "trash_branch", "trash_wreckage",
            "trash_tarp", "trash_rope", "trash_net"
        ],
        'train_split': 0.70,
        'val_split': 0.15,
        'test_split': 0.15
    },
    'training': {
        'epochs': 100,  # 100 epochs per session
        'batch_size': 16,
        'num_workers': 2,  # Reduced for Kaggle
        'optimizer': 'AdamW',
        'learning_rate': 0.01,
        'weight_decay': 0.0005,
        'scheduler': 'CosineAnnealing',
        'lr_min': 0.0001,
        'early_stopping_patience': 20,
        'grad_clip_norm': 10.0,
        'use_amp': True
    },
    'loss': {
        'lambda_box': 5.0,
        'lambda_obj': 1.0,
        'lambda_cls': 1.0,
        'focal_loss_gamma': 2.0,
        'iou_type': 'CIoU'
    },
    'augmentation': {
        'use_augmentation': True,
        'horizontal_flip_prob': 0.5,
        'color_jitter': True,
        'gaussian_blur': True,
        'underwater_augmentation': True
    },
    'checkpoints': {
        'save_dir': '/kaggle/working/checkpoints',
        'save_interval': 10,
        'save_best_only': False
    },
    'logging': {
        'use_tensorboard': True,
        'log_dir': '/kaggle/working/runs',
        'log_interval': 10
    },
    'eval': {
        'conf_threshold': 0.001,
        'nms_threshold': 0.6,
        'eval_interval': 5
    }
}

# Save config
os.makedirs('configs', exist_ok=True)
with open('configs/kaggle_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("✅ Kaggle config created!")
print("\nConfig summary:")
print(f"- Epochs: {config['training']['epochs']}")
print(f"- Batch size: {config['training']['batch_size']}")
print(f"- Image size: {config['data']['img_size']}")
print(f"- Checkpoints: {config['checkpoints']['save_dir']}")
```

#### Cell 6: Test Dataset Loading
```python
# Test if dataset loads correctly
from data.dataset import TrashCANDataset
from torch.utils.data import DataLoader

print("Testing dataset loading...")

try:
    # Create dataset
    train_dataset = TrashCANDataset(
        data_dir='data/trashcan',
        split='train',
        img_size=640,
        augment=False  # Test without augmentation first
    )
    
    print(f"✅ Train dataset loaded: {len(train_dataset)} images")
    
    # Test loading a sample
    sample = train_dataset[0]
    print(f"✅ Sample loaded successfully!")
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Bboxes: {len(sample['bboxes'])}")
    print(f"   Labels: {len(sample['labels'])}")
    
    # Test dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=train_dataset.collate_fn
    )
    
    batch = next(iter(train_loader))
    print(f"✅ Batch loaded successfully!")
    print(f"   Batch images shape: {batch['images'].shape}")
    print(f"   Batch size: {len(batch['bboxes'])}")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
```

#### Cell 7: Start Training
```python
# Start training with auto-resume
import os
import sys

# Check for existing checkpoint
checkpoint_dir = '/kaggle/working/checkpoints'
latest_checkpoint = None

if os.path.exists(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if checkpoints:
        # Find latest checkpoint
        checkpoints.sort()
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
        print(f"🔄 Found checkpoint: {latest_checkpoint}")
        print(f"   Will resume training from this checkpoint")
    else:
        print("🆕 No checkpoint found. Starting fresh training...")
else:
    print("🆕 Starting fresh training...")

# Run training
cmd = "python scripts/train.py --config configs/kaggle_config.yaml"
if latest_checkpoint:
    cmd += f" --resume {latest_checkpoint}"

print(f"\n🚀 Running: {cmd}\n")
!{cmd}
```

#### Cell 8: Monitor with TensorBoard (Optional)
```python
# Load TensorBoard in notebook
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/runs
```

#### Cell 9: Evaluate Best Model
```python
# Evaluate the best model
import glob

checkpoint_dir = '/kaggle/working/checkpoints'
checkpoints = glob.glob(os.path.join(checkpoint_dir, 'best*.pth'))

if checkpoints:
    best_checkpoint = checkpoints[0]
    print(f"📊 Evaluating: {best_checkpoint}")
    
    !python scripts/evaluate.py \
        --config configs/kaggle_config.yaml \
        --checkpoint {best_checkpoint} \
        --split val
else:
    print("❌ No checkpoint found for evaluation")
```

#### Cell 10: Test Detection on Sample Images
```python
# Run detection on validation images
import glob
import os

checkpoint_dir = '/kaggle/working/checkpoints'
checkpoints = glob.glob(os.path.join(checkpoint_dir, 'best*.pth'))

if checkpoints:
    best_checkpoint = checkpoints[0]
    
    # Create output directory
    os.makedirs('/kaggle/working/detections', exist_ok=True)
    
    print(f"🔍 Running detection with: {best_checkpoint}")
    
    !python scripts/detect.py \
        --config configs/kaggle_config.yaml \
        --checkpoint {best_checkpoint} \
        --source data/trashcan/images/val \
        --output /kaggle/working/detections \
        --conf-threshold 0.5 \
        --save-txt
    
    print("\n✅ Detection complete! Check /kaggle/working/detections/")
else:
    print("❌ No checkpoint found")
```

#### Cell 11: Visualize Results
```python
# Display sample detection results
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

detection_dir = '/kaggle/working/detections'
if os.path.exists(detection_dir):
    images = glob.glob(os.path.join(detection_dir, '*.jpg'))[:6]
    
    if images:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, img_path in enumerate(images):
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(os.path.basename(img_path))
        
        plt.tight_layout()
        plt.show()
        print(f"✅ Displayed {len(images)} detection results")
    else:
        print("⚠️  No detection images found")
else:
    print("⚠️  Detection directory not found")
```

#### Cell 12: Save Checkpoints
```python
# Checkpoints are automatically saved to /kaggle/working/checkpoints/
# These persist after the notebook ends!

import os
import glob

checkpoint_dir = '/kaggle/working/checkpoints'
if os.path.exists(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    print(f"💾 Saved checkpoints ({len(checkpoints)}):")
    for cp in sorted(checkpoints):
        size = os.path.getsize(cp) / (1024*1024)  # MB
        print(f"   - {os.path.basename(cp)} ({size:.1f} MB)")
    
    print("\n📥 These files are saved in /kaggle/working/checkpoints/")
    print("   They persist after this session ends!")
    print("   You can resume training in a new session by uploading them.")
else:
    print("❌ No checkpoints found")
```

### **Step 6: Monitor Training**

- **Check Progress**: Scroll down to see training logs in real-time
- **Expected Output**:
  ```
  Epoch [1/100] - Loss: 2.5432, mAP@50: 0.0234
  Epoch [2/100] - Loss: 2.1234, mAP@50: 0.0456
  ...
  ```
- **TensorBoard**: Run Cell 8 to see interactive graphs
- **Training Time**: ~6 hours for 100 epochs on T4 GPU

### **Step 7: Resume Training (Next Session)**

If your Kaggle session ends before training completes:

1. **Start new notebook session**
2. **Enable GPU and Internet** (same as Step 3)
3. **Add your datasets** (same as Step 4)
4. **Add previous checkpoint**:
   - Click "+ Add Data"
   - Select "Output" → Your previous notebook run
   - Checkpoints will be in `/kaggle/input/YOUR-NOTEBOOK-NAME/checkpoints/`
5. **Run all cells** - training auto-resumes from checkpoint!

---

## 🎯 **OPTION 2: Manual Upload (No Kaggle Dataset)**

If you prefer not to create a Kaggle dataset:

### **Step 1-3**: Same as Option 1 (Create notebook, enable GPU)

### **Step 4: Upload Files Directly in Notebook**

#### Cell 1: Setup Upload
```python
# Upload annotation files
from google.colab import files  # Works in Kaggle too!
import os

os.makedirs('data/trashcan/annotations', exist_ok=True)

print("📤 Upload train.json...")
uploaded = files.upload()
!mv train.json data/trashcan/annotations/

print("📤 Upload val.json...")
uploaded = files.upload()
!mv val.json data/trashcan/annotations/

print("✅ Annotations uploaded!")
```

### **Step 5**: Continue from Cell 5 in Option 1

**Note**: This option requires re-uploading annotations for each new session.

---

## 📊 **Expected Training Timeline**

### **Single Session (100 epochs, ~6 hours on T4):**
- **0-25 epochs** (1.5 hrs): Loss drops from 2.5 → 1.5, mAP ~25-30%
- **25-50 epochs** (1.5 hrs): Loss drops to 1.2, mAP ~40-45%
- **50-75 epochs** (1.5 hrs): Loss drops to 0.9, mAP ~50-55%
- **75-100 epochs** (1.5 hrs): Loss drops to 0.7, mAP ~55-60%

### **Multi-Session (300 epochs total):**
- **Session 1** (100 epochs): mAP ~55-60%
- **Session 2** (100 epochs): mAP ~65-70%
- **Session 3** (100 epochs): mAP ~70-75% ✅ **Production Ready!**

---

## 🔧 **Troubleshooting**

### **❌ "No module named 'data.dataset'"**
```python
# Add repository to Python path
import sys
sys.path.insert(0, '/kaggle/working/YOLO-UDD-v2.0')
```

### **❌ "CUDA out of memory"**
Reduce batch size in Cell 5:
```python
config['training']['batch_size'] = 8  # or even 4
```

### **❌ "Annotation files not found"**
Check your dataset path in Cell 4:
```python
!ls -la /kaggle/input/  # See what's available
# Adjust annotations_path variable
```

### **❌ "Images not found"**
Ensure image dataset is added and paths are correct:
```python
!ls -la /kaggle/input/YOUR-IMAGE-DATASET/
# Adjust images_path in Cell 4
```

### **⚠️ "Training is slow"**
- Verify GPU is enabled: Run Cell 3
- Reduce num_workers: Set to 0 or 1 in config
- Reduce image size: Try 512 instead of 640

---

## 💾 **Saving & Downloading Results**

### **Automatic Saves**
- Checkpoints: `/kaggle/working/checkpoints/`
- TensorBoard logs: `/kaggle/working/runs/`
- Detection results: `/kaggle/working/detections/`

### **Download Files**
```python
# Download best checkpoint
from IPython.display import FileLink
FileLink('/kaggle/working/checkpoints/best.pth')
```

Or manually:
1. Click **Output** tab (top right)
2. Browse to `checkpoints/`
3. Click download icon next to files

---

## 📈 **Monitoring Progress**

### **In Notebook**
- Watch console output for epoch-by-epoch progress
- Loss should decrease over time
- mAP should increase over time

### **TensorBoard**
```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/runs
```

Shows:
- Training/Validation Loss curves
- mAP progression
- Learning rate schedule
- Sample detections

---

## ✅ **Success Checklist**

Before you start:
- [ ] Kaggle account created
- [ ] Annotation files ready (train.json, val.json)
- [ ] Image dataset available (Kaggle dataset or upload)
- [ ] GPU enabled in notebook settings
- [ ] Internet enabled in notebook settings

After first epoch:
- [ ] No errors in console
- [ ] Checkpoint saved to `/kaggle/working/checkpoints/`
- [ ] TensorBoard shows training curves
- [ ] Loss is decreasing

After training completes:
- [ ] Best checkpoint downloaded
- [ ] Evaluation metrics > 50% mAP@50
- [ ] Detection visualizations look good
- [ ] Ready for deployment or further training

---

## 🎓 **Additional Tips**

1. **Save often**: Checkpoints save every 10 epochs automatically
2. **Monitor GPU**: Run `!nvidia-smi` to check GPU utilization
3. **Kaggle limits**: 30 hours GPU per week, 12-hour max session
4. **Internet required**: To clone GitHub repo and install packages
5. **Persistence**: Files in `/kaggle/working/` persist after session ends

---

## 📚 **Resources**

- **GitHub Repo**: https://github.com/kshitijkhede/YOLO-UDD-v2.0
- **Kaggle Docs**: https://www.kaggle.com/docs/notebooks
- **PyTorch Docs**: https://pytorch.org/docs/
- **TensorBoard Guide**: https://www.tensorflow.org/tensorboard

---

## 🎉 **You're Ready!**

Follow Option 1 or Option 2 above, and you'll have your model training in minutes!

**Questions?** Check the troubleshooting section or review the notebook cells carefully.

**Good luck with your underwater debris detection project!** 🌊🤖

---

*Last updated: November 1, 2025*
*YOLO-UDD v2.0 - Turbidity-Adaptive Underwater Debris Detection*
