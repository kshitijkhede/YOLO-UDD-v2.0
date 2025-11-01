# 🖼️ Uploading Images to Kaggle - Complete Guide

## 📊 **Your Situation:**

✅ **Annotations uploaded**: train.json (6,065 images), val.json (1,147 images)  
❌ **Images missing**: Need to upload actual image files

**Total size**: ~28.6 MB (24 MB train + 4.6 MB val)

---

## 🎯 **SOLUTION: Create and Upload Image Dataset**

Since the images are small (28.6 MB), you can easily upload them to Kaggle!

---

## 📦 **STEP 1: Prepare Images on Your Computer**

### **Option A: Create Single Zip File** ⭐ **RECOMMENDED**

```bash
# On your local machine
cd /home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan

# Create a zip file with proper structure
zip -r trashcan_images.zip images/train images/val

# Check the result
ls -lh trashcan_images.zip
```

This creates: `trashcan_images.zip` (~25-28 MB)

### **Option B: Create Separate Zips** (If Option A is too large)

```bash
cd /home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/images

# Zip train images
zip -r train_images.zip train/

# Zip val images  
zip -r val_images.zip val/

# Check sizes
ls -lh *.zip
```

---

## 🌐 **STEP 2: Upload Images to Kaggle**

### **Method 1: Try Creating Dataset Again** (May work now)

1. **Verify Phone Number First:**
   - Go to: https://www.kaggle.com/settings
   - Complete phone verification if needed
   - Wait 2-3 minutes

2. **Create Image Dataset:**
   - Go to: https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload: `trashcan_images.zip`
   - **Title**: `TrashCAN Images`
   - **Description**: 
     ```
     TrashCAN 1.0 underwater debris detection images.
     - 6,065 training images
     - 1,147 validation images
     - 22 object categories (marine life and debris)
     ```
   - Click "Create"
   - **Note your dataset path**: `YOUR-USERNAME/trashcan-images`

### **Method 2: Upload Directly in Notebook** (If dataset creation still fails)

In your Kaggle notebook:

1. **Click "+ Add Data"** (right sidebar)
2. **Select "Upload"**
3. **Upload** `trashcan_images.zip`
4. **Files appear at**: `/kaggle/input/YOUR-UPLOAD-NAME/`

---

## 📝 **STEP 3: Use Images in Your Notebook**

Add this cell to your Kaggle notebook after uploading:

### **Cell: Extract and Setup Images**

```python
import zipfile
import os
import shutil

print("=" * 70)
print("🖼️  SETTING UP IMAGES")
print("=" * 70)

# Find the uploaded zip file
uploaded_zip = None
search_paths = ['/kaggle/input/']

print("\n🔍 Searching for image zip file...")
for base_path in search_paths:
    if os.path.exists(base_path):
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if 'trashcan' in file.lower() and file.endswith('.zip'):
                    uploaded_zip = os.path.join(root, file)
                    print(f"✅ Found: {uploaded_zip}")
                    break
            if uploaded_zip:
                break

# Extract images
if uploaded_zip:
    print(f"\n📦 Extracting images from: {uploaded_zip}")
    
    # Extract to working directory first
    extract_path = '/kaggle/working/temp_images'
    os.makedirs(extract_path, exist_ok=True)
    
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    print("✅ Extraction complete!")
    
    # Find and move images to correct location
    print("\n📂 Organizing image directories...")
    
    # Create target directories
    os.makedirs('data/trashcan/images/train', exist_ok=True)
    os.makedirs('data/trashcan/images/val', exist_ok=True)
    
    # Find the extracted images
    for root, dirs, files in os.walk(extract_path):
        if 'train' in root:
            print(f"   Moving training images from {root}")
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(root, file)
                    dst = os.path.join('data/trashcan/images/train', file)
                    shutil.copy2(src, dst)
        elif 'val' in root:
            print(f"   Moving validation images from {root}")
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(root, file)
                    dst = os.path.join('data/trashcan/images/val', file)
                    shutil.copy2(src, dst)
    
    # Verify
    train_count = len([f for f in os.listdir('data/trashcan/images/train') if f.endswith(('.jpg', '.jpeg', '.png'))])
    val_count = len([f for f in os.listdir('data/trashcan/images/val') if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"\n✅ Images organized successfully!")
    print(f"   📊 Training images: {train_count}")
    print(f"   📊 Validation images: {val_count}")
    
    # Clean up temp directory
    shutil.rmtree(extract_path)
    print("   🗑️  Temporary files cleaned up")
    
else:
    print("\n❌ Image zip file not found!")
    print("\n⚠️  Please upload images using '+ Add Data' button:")
    print("   1. Click '+ Add Data' (right sidebar)")
    print("   2. Select 'Upload'")
    print("   3. Upload trashcan_images.zip")
    print("   4. Re-run this cell")
    
    print("\n📂 Current files in /kaggle/input/:")
    for root, dirs, files in os.walk('/kaggle/input/'):
        for file in files:
            print(f"   - {os.path.join(root, file)}")

print("\n" + "=" * 70)
```

---

## 🔄 **ALTERNATIVE: Use Public TrashCAN Dataset**

Instead of uploading, search for existing TrashCAN dataset:

### **In Your Kaggle Notebook:**

1. **Click "+ Add Data"** (right sidebar)
2. **Search for**: `TrashCAN` or `underwater debris`
3. **Look for datasets** with ~7,000 images
4. **Add it** to your notebook

Common dataset names to search:
- "TrashCAN"
- "TrashCAN 1.0"
- "underwater debris detection"
- "marine debris dataset"
- "JAMSTEC underwater"

If found, the images will be at: `/kaggle/input/DATASET-NAME/`

Then use a simpler cell to copy them:

```python
import os
import shutil

# Update this path to match the dataset you added
source_path = '/kaggle/input/DATASET-NAME/images'

# Copy to project directory
shutil.copytree(f'{source_path}/train', 'data/trashcan/images/train', dirs_exist_ok=True)
shutil.copytree(f'{source_path}/val', 'data/trashcan/images/val', dirs_exist_ok=True)

# Verify
train_count = len(os.listdir('data/trashcan/images/train'))
val_count = len(os.listdir('data/trashcan/images/val'))

print(f"✅ Images copied!")
print(f"   Training: {train_count} images")
print(f"   Validation: {val_count} images")
```

---

## ✅ **Complete Workflow Summary**

### **On Your Local Computer:**
```bash
# 1. Create image zip
cd /home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan
zip -r trashcan_images.zip images/train images/val

# File ready at:
# /home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/trashcan_images.zip
```

### **In Your Kaggle Notebook:**

```
1. Click "+ Add Data" → Upload → trashcan_images.zip
2. Add extraction cell (code above)
3. Run cell to extract and organize images
4. Continue with training!
```

---

## 📊 **Verify Everything is Ready**

Add this verification cell before training:

```python
import os
import json

print("=" * 70)
print("✅ FINAL DATASET VERIFICATION")
print("=" * 70)

# Check annotations
train_json = 'data/trashcan/annotations/train.json'
val_json = 'data/trashcan/annotations/val.json'

with open(train_json, 'r') as f:
    train_data = json.load(f)
with open(val_json, 'r') as f:
    val_data = json.load(f)

print(f"\n📋 Annotations:")
print(f"   Train: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
print(f"   Val: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")

# Check images
train_images = len([f for f in os.listdir('data/trashcan/images/train') if f.endswith(('.jpg', '.jpeg', '.png'))])
val_images = len([f for f in os.listdir('data/trashcan/images/val') if f.endswith(('.jpg', '.jpeg', '.png'))])

print(f"\n🖼️  Image Files:")
print(f"   Train: {train_images} files")
print(f"   Val: {val_images} files")

# Verify match
print(f"\n🔍 Verification:")
if len(train_data['images']) == train_images:
    print(f"   ✅ Train annotations match image count")
else:
    print(f"   ⚠️  Train mismatch: {len(train_data['images'])} annotations vs {train_images} images")

if len(val_data['images']) == val_images:
    print(f"   ✅ Val annotations match image count")
else:
    print(f"   ⚠️  Val mismatch: {len(val_data['images'])} annotations vs {val_images} images")

if train_images > 0 and val_images > 0:
    print(f"\n🎉 DATASET READY FOR TRAINING!")
else:
    print(f"\n❌ Images still missing - please upload them!")

print("=" * 70)
```

---

## 🎯 **Quick Reference**

| Task | Command/Action |
|------|----------------|
| Create zip locally | `zip -r trashcan_images.zip images/` |
| Upload to Kaggle | "+ Add Data" → Upload |
| File location | `/kaggle/input/YOUR-UPLOAD/` |
| Extract in notebook | Use extraction cell above |
| Verify dataset | Use verification cell above |

---

## 💡 **Pro Tips**

1. **Zip compression**: Images compress well, expect ~20-25 MB zip
2. **Upload time**: ~2-3 minutes for 28 MB
3. **Alternative**: Search for existing TrashCAN dataset first
4. **Storage**: Kaggle allows up to 20 GB uploads (you're using <1%)

---

## 🆘 **Troubleshooting**

### **"Upload too large"**
- Your 28 MB is fine! Kaggle allows up to 20 GB
- If issues, split into train_images.zip and val_images.zip

### **"Can't find uploaded zip"**
- Check: `!ls -la /kaggle/input/`
- Adjust search path in extraction cell

### **"Images not extracting"**
- Verify zip structure: `unzip -l trashcan_images.zip` locally
- Should show: `images/train/*.jpg` and `images/val/*.jpg`

---

*After uploading images, you'll be ready to train!* 🚀
