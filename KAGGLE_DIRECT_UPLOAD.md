# 📤 Alternative Upload Method - Direct Upload in Kaggle Notebook

## ⚠️ If You Get "Permission Denied" for Dataset Creation

Use this method to upload annotations **directly** in your training notebook instead.

---

## 🚀 **METHOD: Direct Upload in Notebook (No Dataset Needed)**

### **Step 1: Create Kaggle Notebook First**

1. Go to: **https://www.kaggle.com/code**
2. Click **"New Notebook"**
3. **Settings** (gear icon):
   - ⚡ **Accelerator**: GPU P100 or T4
   - 🌐 **Internet**: ON
4. Click **"Save Version"** to save the notebook

---

### **Step 2: Upload Annotations Directly**

Copy and paste these cells into your Kaggle notebook:

#### **Cell 1: Clone Repository**
```python
# Clone YOLO-UDD v2.0 repository
!git clone https://github.com/kshitijkhede/YOLO-UDD-v2.0.git
%cd YOLO-UDD-v2.0

print("✅ Repository cloned!")
!pwd
```

#### **Cell 2: Install Dependencies**
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

#### **Cell 3: Setup Directory Structure**
```python
import os

# Create directories
os.makedirs('data/trashcan/annotations', exist_ok=True)
os.makedirs('data/trashcan/images/train', exist_ok=True)
os.makedirs('data/trashcan/images/val', exist_ok=True)

print("✅ Directory structure created!")
print("Please upload annotation files in the next cell...")
```

#### **Cell 4: Upload Annotation Files** ⭐ **IMPORTANT**
```python
# Upload your annotation files
from kaggle_secrets import UserSecretsClient
import json
import os

print("=" * 70)
print("📤 UPLOAD ANNOTATION FILES")
print("=" * 70)
print("\nYou need to upload these files from your local machine:")
print("  1. train.json (22 MB)")
print("  2. val.json (5.6 MB)")
print("\nLocation on your computer:")
print("  /home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/annotations/")
print("\n" + "=" * 70)
print("\nCLICK THE 'Add Data' BUTTON IN THE RIGHT SIDEBAR →")
print("Then select 'Upload' and choose your files.")
print("=" * 70)

# Instructions for user
print("\n⚠️  AFTER UPLOADING:")
print("1. Your files will appear in: /kaggle/input/")
print("2. Update the paths below with your upload location")
print("3. Run the next cell to copy them to the correct location")
```

#### **Cell 5: Manual File Upload Instructions**
```python
# ALTERNATIVE: Use Kaggle's file upload widget
# This creates a file upload button in the notebook

from IPython.display import display, HTML

upload_instructions = """
<div style="background-color: #f0f0f0; padding: 20px; border-radius: 5px; border: 2px solid #4CAF50;">
<h2>📤 Upload Your Annotation Files</h2>

<h3>Option A: Add Data Button (Recommended)</h3>
<ol>
    <li>Look at the <b>RIGHT SIDEBAR</b> of this notebook</li>
    <li>Click <b>"+ Add Data"</b> button</li>
    <li>Select <b>"Upload"</b></li>
    <li>Upload <b>train.json</b> (22 MB)</li>
    <li>Upload <b>val.json</b> (5.6 MB)</li>
    <li>Files will be in <code>/kaggle/input/</code></li>
</ol>

<h3>Option B: Copy from Public Dataset</h3>
<p>If someone has already uploaded TrashCAN annotations as a public dataset:</p>
<ol>
    <li>Click <b>"+ Add Data"</b></li>
    <li>Search for <b>"TrashCAN"</b> or <b>"underwater debris annotations"</b></li>
    <li>Add the dataset</li>
</ol>

<h3>Option C: Use Kaggle Dataset API</h3>
<p>If you uploaded to Kaggle datasets (needs phone verification):</p>
<ol>
    <li>Click <b>"+ Add Data"</b></li>
    <li>Search for your dataset</li>
    <li>Add it to the notebook</li>
</ol>

<p style="color: red; font-weight: bold;">⚠️ After upload, update the path in the next cell!</p>
</div>
"""

display(HTML(upload_instructions))
```

#### **Cell 6: Copy Uploaded Files**
```python
import shutil
import os
import json

print("Searching for uploaded annotation files...")

# Common upload locations in Kaggle
possible_paths = [
    '/kaggle/input/',
    '/kaggle/working/',
]

# Search for the files
train_json_path = None
val_json_path = None

for base_path in possible_paths:
    if os.path.exists(base_path):
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file == 'train.json':
                    train_json_path = os.path.join(root, file)
                    print(f"✅ Found train.json: {train_json_path}")
                elif file == 'val.json':
                    val_json_path = os.path.join(root, file)
                    print(f"✅ Found val.json: {val_json_path}")

# If found, copy to correct location
if train_json_path and val_json_path:
    print("\n📋 Copying annotations to project directory...")
    shutil.copy(train_json_path, 'data/trashcan/annotations/train.json')
    shutil.copy(val_json_path, 'data/trashcan/annotations/val.json')
    
    # Verify
    with open('data/trashcan/annotations/train.json', 'r') as f:
        train_data = json.load(f)
    with open('data/trashcan/annotations/val.json', 'r') as f:
        val_data = json.load(f)
    
    print(f"\n✅ Annotations copied and verified!")
    print(f"   Train images: {len(train_data['images'])}")
    print(f"   Train annotations: {len(train_data['annotations'])}")
    print(f"   Val images: {len(val_data['images'])}")
    print(f"   Val annotations: {len(val_data['annotations'])}")
    print(f"   Categories: {len(train_data['categories'])}")
else:
    print("\n❌ Annotation files not found!")
    print("\n⚠️  Please upload them using one of these methods:")
    print("   1. Click '+ Add Data' button → Upload")
    print("   2. Or manually specify paths below:")
    print("\n# MANUAL PATH SETUP:")
    print("# If you know where your files are, uncomment and edit:")
    print("# train_json_path = '/kaggle/input/YOUR-PATH/train.json'")
    print("# val_json_path = '/kaggle/input/YOUR-PATH/val.json'")
    print("# Then run the copy commands above manually")
```

#### **Cell 7: Verify GPU**
```python
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  WARNING: GPU not available! Enable GPU in Settings.")
```

#### **Cell 8: Create Training Config**
```python
import yaml
import os

config = {
    'model': {
        'name': 'YOLO-UDD-v2.0',
        'num_classes': 22,
        'pretrained_path': None
    },
    'data': {
        'dataset_name': 'TrashCAN-1.0',
        'data_dir': 'data/trashcan',
        'img_size': 640,
        'class_names': [
            "rov", "plant", "animal_fish", "animal_starfish", "animal_shells",
            "animal_crab", "animal_eel", "animal_etc", "trash_clothing", "trash_pipe",
            "trash_bottle", "trash_bag", "trash_snack_wrapper", "trash_can", "trash_cup",
            "trash_container", "trash_unknown_instance", "trash_branch", "trash_wreckage",
            "trash_tarp", "trash_rope", "trash_net"
        ]
    },
    'training': {
        'epochs': 100,
        'batch_size': 16,
        'num_workers': 2,
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

os.makedirs('configs', exist_ok=True)
with open('configs/kaggle_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("✅ Training config created!")
```

#### **Cell 9: Test Dataset Loading**
```python
from data.dataset import TrashCANDataset
from torch.utils.data import DataLoader

print("Testing dataset loading...")

try:
    train_dataset = TrashCANDataset(
        data_dir='data/trashcan',
        split='train',
        img_size=640,
        augment=False
    )
    
    print(f"✅ Train dataset loaded: {len(train_dataset)} images")
    
    # Test sample
    sample = train_dataset[0]
    print(f"✅ Sample loaded: image shape {sample['image'].shape}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n⚠️  Make sure annotations are uploaded and copied correctly!")
```

#### **Cell 10: Start Training**
```python
# Start training
checkpoint_dir = '/kaggle/working/checkpoints'
latest_checkpoint = None

if os.path.exists(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if checkpoints:
        checkpoints.sort()
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
        print(f"🔄 Resuming from: {latest_checkpoint}")

cmd = "python scripts/train.py --config configs/kaggle_config.yaml"
if latest_checkpoint:
    cmd += f" --resume {latest_checkpoint}"

print(f"🚀 Starting training: {cmd}\n")
!{cmd}
```

---

## 📝 **STEP-BY-STEP WORKFLOW:**

### **1. Create Notebook**
   - Go to: https://www.kaggle.com/code
   - New Notebook
   - Enable GPU + Internet

### **2. Copy All Cells**
   - Copy Cell 1-10 above into your notebook

### **3. Run Cells 1-5**
   - This sets everything up

### **4. Upload Your Files**
   - In Cell 5, you'll see upload instructions
   - Click **"+ Add Data"** (right sidebar)
   - Select **"Upload"**
   - Upload both files from:
     ```
     /home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/annotations/
     - train.json
     - val.json
     ```

### **5. Run Cell 6**
   - This automatically finds and copies your files

### **6. Run Cells 7-10**
   - Verifies setup and starts training!

---

## ✅ **Advantages of This Method:**

- ✅ **No dataset creation needed** - bypasses permission issue
- ✅ **Direct upload** - files go straight into notebook
- ✅ **Auto-detection** - Cell 6 finds your files automatically
- ✅ **Works immediately** - no phone verification needed
- ⚠️ **Limitation**: Need to re-upload for each new session

---

## 🔄 **For Next Session:**

When you start a new notebook session:

1. **Method A**: Upload files again (quick, ~1 minute)
2. **Method B**: Add previous notebook's output as data source
   - Your previous run's `/kaggle/working/` persists
   - Add it as a data source: `+ Add Data` → `Your Work` → `Previous notebook`

---

## 💡 **Alternative: Ask Someone to Share**

Search Kaggle for existing TrashCAN datasets:
- https://www.kaggle.com/datasets
- Search: "TrashCAN" or "underwater debris"
- If found, add it to your notebook (no upload needed!)

---

*This method works around the dataset creation permission issue!*
