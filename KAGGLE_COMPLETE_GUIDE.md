# 🚀 Complete Kaggle Training Guide - YOLO-UDD v2.0

## 📋 **Prerequisites**

Before starting, ensure you have:
- ✅ Kaggle account (sign up at https://www.kaggle.com)
- ✅ Your dataset files ready on your local machine
- ✅ Internet connection for uploading files

---

## 📦 **STEP 1: Prepare Your Dataset Locally**

### **1.1 Verify Your Dataset Files**

Your dataset location:
```bash
/home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/
```

Contents:
```
trashcan/
├── annotations/
│   ├── train.json (22 MB - 6,065 images)
│   └── val.json (5.6 MB - 1,147 images)
└── images/
    ├── train/ (6,065 images)
    └── val/ (1,147 images)
```

### **1.2 Create Dataset Archive**

Open terminal and run:

```bash
cd /home/student/MIR/Project/YOLO-UDD-v2.0/data

# Create a single zip with proper structure
zip -r trashcan_dataset.zip trashcan/
```

This creates `trashcan_dataset.zip` (~30 MB) containing everything.

**Expected output:**
```
adding: trashcan/annotations/train.json
adding: trashcan/annotations/val.json
adding: trashcan/images/train/...
adding: trashcan/images/val/...
```

---

## 🌐 **STEP 2: Upload Dataset to Kaggle**

### **2.1 Go to Kaggle Datasets**

1. Open browser: https://www.kaggle.com/datasets
2. Click **"New Dataset"** button (top right)

### **2.2 Upload Your Dataset**

1. **Drag and drop** or click to upload: `trashcan_dataset.zip`
2. Wait for upload to complete (~2-5 minutes for 30MB)

### **2.3 Configure Dataset Settings**

Fill in the form:

**Title:** `TrashCAN Underwater Debris Dataset`

**Subtitle (optional):** `COCO format annotations for underwater trash detection`

**Description:**
```
TrashCAN 1.0 dataset for underwater debris detection.

Contents:
- 6,065 training images
- 1,147 validation images
- COCO format annotations (instance segmentation)
- 22 object categories

Dataset structure:
trashcan/
├── annotations/
│   ├── train.json
│   └── val.json
└── images/
    ├── train/
    └── val/
```

**Visibility:** Choose "Private" (recommended) or "Public"

Click **"Create"** button

### **2.4 Note Your Dataset Path**

After creation, you'll see your dataset URL:
```
https://www.kaggle.com/datasets/YOUR_USERNAME/trashcan-underwater-debris-dataset
```

Your dataset path will be:
```
YOUR_USERNAME/trashcan-underwater-debris-dataset
```

**⚠️ IMPORTANT:** Write this down! You'll need it in Step 4.

---

## 📓 **STEP 3: Create Kaggle Notebook**

### **3.1 Go to Kaggle Code**

1. Open browser: https://www.kaggle.com/code
2. Click **"New Notebook"** button

### **3.2 Upload Your Training Notebook**

**Option A: Upload from GitHub (Recommended)**

1. In the notebook, click **"File"** → **"Import Notebook"**
2. Select **"GitHub"** tab
3. Enter repository URL:
   ```
   https://github.com/kshitijkhede/YOLO-UDD-v2.0/blob/main/YOLO_UDD_Kaggle_Training.ipynb
   ```
4. Click **"Copy and Edit"**

**Option B: Upload Local File**

1. Download notebook from GitHub first:
   ```bash
   cd /home/student/MIR/Project/YOLO-UDD-v2.0
   # Notebook is already here: YOLO_UDD_Kaggle_Training.ipynb
   ```

2. In Kaggle, click **"File"** → **"Import Notebook"**
3. Select **"Upload"** tab
4. Choose `YOLO_UDD_Kaggle_Training.ipynb`

### **3.3 Rename Your Notebook (Optional)**

Click on notebook title and rename to:
```
YOLO-UDD Training Session 1
```

---

## ⚙️ **STEP 4: Configure Kaggle Environment**

### **4.1 Enable GPU**

**CRITICAL STEP!**

1. Click **"Settings"** (right panel) or **"⋮"** menu
2. Under **"Accelerator"**, select:
   - **GPU T4 x2** (recommended) or
   - **GPU P100** (if available)
3. **DO NOT select "None" or "TPU"** - Training requires GPU!

### **4.2 Enable Internet**

1. In **"Settings"** panel
2. Toggle **"Internet"** to **ON**
3. This allows cloning from GitHub

### **4.3 Add Your Dataset**

1. In **"Settings"** panel, find **"Add Data"** section
2. Click **"+ Add Data"** button
3. Select **"Your Datasets"** tab
4. Find and click your uploaded dataset: `trashcan-underwater-debris-dataset`
5. Click **"Add"** button

You should now see it in the **Input** section:
```
📁 /kaggle/input/trashcan-underwater-debris-dataset/
```

### **4.4 Verify Session Settings**

Your settings should look like:
```
✅ Accelerator: GPU T4 x2
✅ Internet: ON
✅ Dataset: trashcan-underwater-debris-dataset
✅ Language: Python
✅ Environment: Latest (Python 3.10+)
```

---

## 🎯 **STEP 5: Configure Dataset Path in Notebook**

### **5.1 Find Cell 4️⃣ - Dataset Setup**

Scroll to the cell that says:
```python
# 4️⃣ Setup Dataset Paths
```

### **5.2 Update the Dataset Path**

Find this line:
```python
KAGGLE_DATASET_PATH = "/kaggle/input/YOUR-DATASET-NAME"
```

**Replace** `YOUR-DATASET-NAME` with your actual dataset name:
```python
KAGGLE_DATASET_PATH = "/kaggle/input/trashcan-underwater-debris-dataset"
```

### **5.3 Verify the Dataset Structure**

The notebook expects this structure:
```
/kaggle/input/trashcan-underwater-debris-dataset/
└── trashcan/
    ├── annotations/
    │   ├── train.json
    │   └── val.json
    └── images/
        ├── train/
        └── val/
```

If your zip had a different structure, adjust paths accordingly.

---

## ▶️ **STEP 6: Start Training!**

### **6.1 Run All Cells**

**Method 1: Run All at Once (Recommended)**
1. Click **"Run All"** button at the top
2. Notebook will execute all cells sequentially

**Method 2: Run Cell by Cell**
1. Click on first cell
2. Press **Shift + Enter** to run and move to next cell
3. Repeat for each cell

### **6.2 What Each Cell Does**

| Cell | Description | Time | Expected Output |
|------|-------------|------|-----------------|
| 1️⃣ | Check GPU availability | 5s | GPU T4 x2 detected |
| 2️⃣ | Clone GitHub repository | 10s | Repository cloned |
| 3️⃣ | Install dependencies | 2m | Packages installed |
| 4️⃣ | Setup dataset paths | 5s | Paths verified |
| 5️⃣ | Create training config | 5s | Config saved |
| 6️⃣ | Verify dataset loading | 30s | 6065 train, 1147 val |
| 7️⃣ | Start training | **6h** | Training progress |
| 8️⃣ | Load TensorBoard | 10s | TensorBoard UI |
| 9️⃣ | Evaluate model | 5m | mAP scores |
| 🔟 | Test detection | 2m | Visualizations |
| 1️⃣1️⃣ | Save checkpoints | 1m | Files saved |
| 1️⃣2️⃣ | Show results summary | 5s | Training stats |

### **6.3 Monitor Training Progress**

#### **Training Output:**
You'll see output like:
```
Epoch [1/100]:  10%|████▏                | 100/1000 [02:30<22:30, 1.50s/it]
Train Loss: 2.345 | Val Loss: 2.123 | mAP@50: 0.342
```

#### **TensorBoard:**
After cell 8️⃣, you'll see interactive graphs:
- Training Loss (should decrease)
- Validation Loss (should decrease)
- mAP@50 (should increase)
- Learning Rate (cosine curve)

### **6.4 Expected Training Time**

**Session 1 (100 epochs):**
- GPU T4 x2: ~6 hours
- GPU P100: ~5 hours

**Progress checkpoints:**
- Epoch 25: mAP ~35-40% (1.5h)
- Epoch 50: mAP ~45-50% (3h)
- Epoch 75: mAP ~52-57% (4.5h)
- Epoch 100: mAP ~55-60% (6h) ✅

---

## 💾 **STEP 7: Save Your Checkpoints**

### **7.1 Automatic Saving**

The notebook automatically saves checkpoints to:
```
/kaggle/working/checkpoints/
├── best.pth        (best validation mAP)
├── latest.pth      (most recent epoch)
└── epoch_100.pth   (final epoch)
```

### **7.2 Download Checkpoints**

**Method 1: Kaggle Output Tab**
1. Wait for training to complete
2. Click **"Output"** tab (top right)
3. Download `checkpoints/best.pth` (~200 MB)

**Method 2: Via Code Cell**
Add and run this cell at the end:
```python
from IPython.display import FileLink
FileLink('/kaggle/working/checkpoints/best.pth')
```

### **7.3 Verify Checkpoint Files**

Run this cell to check:
```python
!ls -lh /kaggle/working/checkpoints/
```

Expected output:
```
-rw-r--r-- 1 root root 198M Nov 1 12:34 best.pth
-rw-r--r-- 1 root root 198M Nov 1 12:45 latest.pth
-rw-r--r-- 1 root root 198M Nov 1 12:45 epoch_100.pth
```

---

## 🔄 **STEP 8: Resume Training (Session 2)**

After first session completes (100 epochs), continue training:

### **8.1 Save Your Checkpoint**

Before session expires:
1. Go to **"Output"** tab
2. Download `checkpoints/best.pth` or `latest.pth`

### **8.2 Start New Session**

1. Go back to https://www.kaggle.com/code
2. Open your notebook again or click **"Edit"**
3. Re-enable GPU + Internet (settings reset each session)
4. Add dataset again

### **8.3 Upload Previous Checkpoint**

**Option A: Upload as New Dataset**
1. Create new dataset with checkpoint file
2. Add dataset to notebook
3. Copy checkpoint to working directory

**Option B: Via Kaggle API (Advanced)**
Upload checkpoint to Kaggle Datasets via API

### **8.4 Modify Training Config**

In cell 5️⃣, update:
```python
config = {
    'epochs': 200,              # Changed from 100 to 200
    'start_epoch': 100,         # Resume from epoch 100
    'resume': '/kaggle/input/your-checkpoint-dataset/best.pth',
    # ... rest of config
}
```

### **8.5 Run All Cells Again**

Training will resume from epoch 100 → 200.

---

## 📊 **STEP 9: Evaluate Results**

### **9.1 Check Final Metrics**

After training completes, cell 9️⃣ shows:

```
Evaluation Results:
===================
mAP@50: 0.XX
mAP@75: 0.XX
mAP@[50:95]: 0.XX

Per-Class Results:
------------------
Class 1 (bottle): AP = 0.XX
Class 2 (can): AP = 0.XX
...
```

### **9.2 View Detection Examples**

Cell 🔟 generates visualizations:
- Original images with ground truth boxes
- Predicted bounding boxes and labels
- Confidence scores

### **9.3 Download TensorBoard Logs**

```python
!zip -r tensorboard_logs.zip /kaggle/working/runs/
```

Then download from Output tab.

---

## 🎯 **Expected Results Timeline**

### **Session 1: 0-100 epochs (6 hours)**
```
✅ Training Loss: 2.5 → 1.2
✅ Validation Loss: 2.3 → 1.4
✅ mAP@50: 0% → 55-60%
✅ Model learns basic object detection
```

### **Session 2: 100-200 epochs (6 hours)**
```
✅ Training Loss: 1.2 → 0.8
✅ Validation Loss: 1.4 → 1.1
✅ mAP@50: 55% → 68-72%
✅ Model refines detections
```

### **Session 3: 200-300 epochs (6 hours)**
```
✅ Training Loss: 0.8 → 0.6
✅ Validation Loss: 1.1 → 0.9
✅ mAP@50: 68% → 72-76%
✅ Production-ready model! 🎉
```

---

## 🐛 **Troubleshooting**

### **❌ Error: "Out of Memory"**

**Solution 1:** Reduce batch size
```python
config['batch_size'] = 8  # or 4
```

**Solution 2:** Use GPU T4 x2 instead of single GPU

### **❌ Error: "Dataset not found"**

**Solution:** Check dataset path
```python
# Verify files exist
!ls /kaggle/input/
!ls /kaggle/input/trashcan-underwater-debris-dataset/
```

Update path in cell 4️⃣.

### **❌ Error: "No GPU available"**

**Solution:** 
1. Stop notebook
2. Settings → Accelerator → GPU T4 x2
3. Save and run again

### **❌ Error: "Internet is required"**

**Solution:**
1. Settings → Internet → ON
2. Accept Kaggle's terms
3. Run again

### **❌ Warning: "Session about to expire"**

**Solution:**
1. Quickly download checkpoints from Output tab
2. Session expires after 12 hours of inactivity
3. Save checkpoints every few hours

### **❌ Error: "git clone failed"**

**Solution:**
```python
# Manual clone in cell 2️⃣
!git clone https://github.com/kshitijkhede/YOLO-UDD-v2.0.git
%cd YOLO-UDD-v2.0
```

### **❌ Training is very slow**

**Solution:**
- Verify GPU is enabled (not CPU)
- Check GPU utilization: `!nvidia-smi`
- Reduce `num_workers` if CPU bottleneck:
  ```python
  config['num_workers'] = 2  # reduce from 4
  ```

---

## 📝 **Quick Checklist**

Before starting training, verify:

```
✅ Kaggle account created
✅ Dataset uploaded to Kaggle Datasets
✅ Notebook imported from GitHub
✅ GPU T4 x2 enabled
✅ Internet enabled
✅ Dataset added to notebook inputs
✅ Dataset path updated in cell 4️⃣
✅ All cells executed in order
✅ Training started (cell 7️⃣)
✅ TensorBoard monitoring (cell 8️⃣)
✅ Checkpoints saving automatically
```

---

## 🎓 **Pro Tips**

### **Tip 1: Monitor Actively**
Check notebook every 1-2 hours to ensure training is progressing.

### **Tip 2: Save Frequently**
Download checkpoints after 50 epochs, 75 epochs, etc. Don't wait until the end.

### **Tip 3: Use Multiple Browsers**
Keep TensorBoard open in one tab, training logs in another.

### **Tip 4: Notebook Versions**
Kaggle auto-saves versions. Go to **"File"** → **"Versions"** to see history.

### **Tip 5: GPU Quota**
Kaggle gives 30 hours/week GPU time. Plan your sessions:
- Session 1: Monday (6h)
- Session 2: Wednesday (6h)
- Session 3: Friday (6h)

### **Tip 6: Early Stopping**
If validation mAP plateaus for 20+ epochs, consider stopping early.

---

## 📚 **Additional Resources**

- **Kaggle Documentation:** https://www.kaggle.com/docs
- **Kaggle GPU Info:** https://www.kaggle.com/docs/notebooks#gpus
- **GitHub Repository:** https://github.com/kshitijkhede/YOLO-UDD-v2.0
- **TensorBoard Guide:** https://www.tensorflow.org/tensorboard

---

## 🎉 **Summary**

**Total time to complete training:**
- Dataset upload: 10 minutes
- Setup: 5 minutes
- Session 1 (0-100 epochs): 6 hours
- Session 2 (100-200 epochs): 6 hours
- Session 3 (200-300 epochs): 6 hours
- **Total: ~18-20 hours across 3 sessions**

**Final deliverable:**
- ✅ Trained model (`best.pth`)
- ✅ TensorBoard logs
- ✅ Evaluation metrics (mAP ~72-76%)
- ✅ Detection visualizations

---

## 🚀 **Ready to Start?**

1. Open terminal and create dataset zip
2. Upload to Kaggle Datasets
3. Import notebook from GitHub
4. Enable GPU + Internet + Add Dataset
5. Click **"Run All"**
6. Wait 6 hours
7. See results! 🎊

**Good luck with your training!** 🌊🗑️🤖

---

*Last Updated: November 1, 2025*
*YOLO-UDD v2.0 - Underwater Debris Detection*
