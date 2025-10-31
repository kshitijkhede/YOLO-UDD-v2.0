# ============================================
# YOLO-UDD v2.0 Kaggle Training Notebook
# Dataset: Google Drive (trashcan.zip)
# FILE_ID: 17oRYriPgBnW9zowwmhImxdUpmHwOjgIp
# ============================================

# Cell 1: Download Dataset from Google Drive

import gdown
import zipfile
import os

# Your Google Drive FILE_ID
FILE_ID = "17oRYriPgBnW9zowwmhImxdUpmHwOjgIp"

# Download dataset
print("📥 Downloading dataset from Google Drive...")
url = f"https://drive.google.com/uc?id={FILE_ID}"
output = "trashcan.zip"
gdown.download(url, output, quiet=False)

# Extract dataset
print("\n📦 Extracting dataset...")
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall("data")

print("\n✅ Dataset ready!")
print(f"📂 Dataset location: /kaggle/working/data/")



# ============================================

# Cell 2: Clone YOLO-UDD Repository
print("📥 Cloning YOLO-UDD v2.0 repository...")


# Or upload your project files manually and skip this cell

# ============================================

# Cell 3: Install Dependencies
print("📦 Installing dependencies...")


# Check NumPy version (must be < 2.0)
import numpy as np
print(f"NumPy version: {np.__version__}")

if np.__version__.startswith('2.'):
    print("⚠️ Downgrading NumPy to 1.26.4...")
   
    print("✅ NumPy downgraded. Please restart kernel!")
else:
    print("✅ NumPy version is compatible")

# ============================================

# Cell 4: Verify GPU
import torch
print(f"🖥️ PyTorch version: {torch.__version__}")
print(f"🎮 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🎮 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️ No GPU detected, training will use CPU (very slow!)")

# ============================================

# Cell 5: Create Training Config
config_content = """
# YOLO-UDD Training Configuration (Kaggle GPU)

# Model
model:
  num_classes: 22  # Your dataset has 22 classes
  backbone: 'cspdarknet53'

# Dataset
dataset:
  name: 'trashcan'
  root_dir: '/kaggle/working/data'
  train_ann: 'instances_train_trashcan.json'
  val_ann: 'instances_val_trashcan.json'
  img_dir: 'images'

# Training
training:
  epochs: 50
  batch_size: 16
  img_size: 640
  num_workers: 4
  device: 'cuda'
  use_amp: true

# Optimizer
optimizer:
  type: 'adamw'
  lr: 0.001
  weight_decay: 0.0001

# Learning rate scheduler
scheduler:
  type: 'cosine'
  warmup_epochs: 3

# Loss weights
loss:
  box_weight: 5.0
  obj_weight: 1.0
  cls_weight: 1.0

# Data augmentation
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 10.0
  translate: 0.1
  scale: 0.5
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.0

# Checkpoints
checkpoint:
  save_dir: 'checkpoints'
  save_interval: 5

# Logging
logging:
  log_dir: 'logs'
  log_interval: 10
"""

with open('configs/train_config_kaggle.yaml', 'w') as f:
    f.write(config_content)

print("✅ Config created: configs/train_config_kaggle.yaml")

# ============================================

# Cell 6: Start Training
print("🚀 Starting training...")

# ============================================

# Cell 7: Download Checkpoints (After Training)
from IPython.display import FileLink
import glob

# Find latest checkpoint
checkpoints = sorted(glob.glob('checkpoints/*.pth'))
if checkpoints:
    latest = checkpoints[-1]
    print(f"📥 Download your trained model:")
    print(f"   {latest}")
    display(FileLink(latest))
else:
    print("⚠️ No checkpoints found. Training may have failed.")

# ============================================

# Cell 8: Visualize Training Progress
import matplotlib.pyplot as plt
import json

# Load training log
log_file = 'logs/train_log.json'
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        logs = [json.loads(line) for line in f]
    
    epochs = [log['epoch'] for log in logs]
    train_loss = [log['train_loss'] for log in logs]
    val_loss = [log['val_loss'] for log in logs]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    
    plt.subplot(1, 2, 2)
    mAP = [log.get('mAP', 0) for log in logs]
    plt.plot(epochs, mAP, label='mAP@0.5')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.title('Validation mAP')
    
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Training log not found")

print("\n✅ Training complete!")
print(f"📊 Total images trained: 6,065")
print(f"📊 Total epochs: 50")
print(f"📂 Checkpoints saved in: checkpoints/")
