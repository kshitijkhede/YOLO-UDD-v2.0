# 🌐 Cloud Training Guide - Kaggle & Colab

## 🎯 **Which Platform Should You Use?**

### ✅ **Kaggle (RECOMMENDED)**
- **GPU Time**: 30 hours/week
- **Session Length**: Up to 12 hours
- **Storage**: Persistent between sessions
- **Best For**: Complete training (can finish in 2-3 sessions)

### ⚡ **Google Colab**
- **GPU Time**: Varies (free tier)
- **Session Length**: 2-3 hours then disconnects
- **Storage**: Need Google Drive for checkpoints
- **Best For**: Quick experiments or if no Kaggle account

---

## 📋 **Pre-Upload Checklist**

Before uploading to Kaggle/Colab, you need:

### 1. **Annotation Files** (from your local machine)
```bash
/home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/annotations/
├── train.json (22 MB)
└── val.json (5.6 MB)
```

### 2. **Image Files** (from your local machine)
```bash
/home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan/images/
├── train/ (6,065 images)
└── val/ (1,147 images)
```

---

## 🚀 **KAGGLE TRAINING (Recommended)**

### **Step 1: Prepare Dataset**

#### Option A: Create Kaggle Dataset (Recommended)
1. Zip your images:
   ```bash
   cd /home/student/MIR/Project/YOLO-UDD-v2.0/data/trashcan
   zip -r trashcan_images.zip images/
   ```

2. Upload to Kaggle:
   - Go to: https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload `trashcan_images.zip`
   - Title: "TrashCAN Underwater Images"
   - Make it public or private
   - Note the dataset path: `YOUR_USERNAME/trashcan-images`

#### Option B: Upload Annotations to Google Drive
1. Upload annotation files to Google Drive
2. Get shareable link
3. Use link in notebook to download

### **Step 2: Upload Notebook**
1. Go to: https://www.kaggle.com/code
2. Click "New Notebook"
3. Upload: `YOLO_UDD_Kaggle_Training.ipynb`
4. Or paste notebook content

### **Step 3: Configure Kaggle**
1. **Enable GPU**: 
   - Settings → Accelerator → GPU P100 or T4
2. **Enable Internet**: 
   - Settings → Internet → ON
3. **Add Dataset**:
   - Settings → Add Data → Your uploaded dataset

### **Step 4: Start Training**
1. Run all cells in order
2. Training will run for up to 12 hours
3. **Checkpoints save automatically** to `/kaggle/working/`
4. Download checkpoints at the end

### **Step 5: Resume Training (Next Session)**
1. Start new Kaggle notebook
2. Upload the same notebook
3. Upload previous checkpoint from `/kaggle/input/`
4. Notebook automatically detects and resumes

### **Expected Timeline:**
- **Session 1** (6 hours): 0-100 epochs → mAP ~50-55%
- **Session 2** (6 hours): 100-200 epochs → mAP ~65-70%
- **Session 3** (6 hours): 200-300 epochs → mAP ~70-75% ✅ Done!

---

## ⚡ **GOOGLE COLAB TRAINING**

### **Step 1: Upload to Google Drive**
1. Upload your annotations to Google Drive:
   ```
   /MyDrive/YOLO_UDD_Training/annotations/
   ├── train.json
   └── val.json
   ```

2. Upload your images:
   ```
   /MyDrive/trashcan_images/
   ├── train/ (6,065 images)
   └── val/ (1,147 images)
   ```

### **Step 2: Open Notebook in Colab**
1. Go to: https://colab.research.google.com
2. File → Upload notebook
3. Upload: `YOLO_UDD_Colab_Training.ipynb`

### **Step 3: Configure Colab**
1. **Enable GPU**: Runtime → Change runtime type → GPU (T4)
2. **Connect Drive**: Run cell 2️⃣ and authorize

### **Step 4: Start Training**
1. Run all cells in order
2. **IMPORTANT**: Training runs for 2-3 hours max
3. **Before timeout**: Run cell 9️⃣ to save to Drive!

### **Step 5: Resume Training (Multiple Sessions)**
1. Open same notebook in new session
2. Run all cells - automatically resumes from Drive
3. Repeat until training complete

### **Expected Timeline:**
- **Session 1** (2 hours): 0-50 epochs → Save to Drive
- **Session 2** (2 hours): 50-100 epochs → Save to Drive
- **Session 3** (2 hours): 100-150 epochs → Save to Drive
- **Session 4** (2 hours): 150-200 epochs → Save to Drive
- **Session 5** (2 hours): 200-250 epochs → Save to Drive
- **Session 6** (2 hours): 250-300 epochs → Final! ✅

---

## 📊 **Monitoring Training**

Both notebooks include TensorBoard for real-time monitoring:

### **In Kaggle:**
```python
%load_ext tensorboard
%tensorboard --logdir runs
```

### **In Colab:**
```python
%load_ext tensorboard
%tensorboard --logdir runs
```

**What to monitor:**
- Training Loss (should decrease)
- Validation Loss (should decrease)
- mAP (should increase)
- Learning Rate (follows cosine schedule)

---

## 💾 **Saving & Downloading Results**

### **From Kaggle:**
```python
# Checkpoints are in /kaggle/working/checkpoints/
# Download them at the end of session
from IPython.display import FileLink
FileLink('checkpoints/best.pth')
```

### **From Colab:**
```python
# Saved automatically to Google Drive
# Location: /MyDrive/YOLO_UDD_Training/checkpoints/
# Or download directly:
from google.colab import files
files.download('runs/*/checkpoints/best.pth')
```

---

## 🎯 **Training Strategy Comparison**

| Feature | Kaggle | Google Colab |
|---------|--------|--------------|
| GPU Time | 30 hrs/week | Variable |
| Session Length | 12 hours | 2-3 hours |
| Sessions Needed | 2-3 | 6+ |
| Auto-Resume | ✅ Easy | ⚠️ Manual |
| Storage | Persistent | Need Drive |
| **Recommendation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 🔧 **Troubleshooting**

### **"Out of Memory" Error**
Reduce batch size in notebook:
```python
config['batch_size'] = 8  # or 4
```

### **"Session Timeout" (Colab)**
- Always save checkpoints to Drive before timeout
- Use cell 9️⃣ regularly (every hour)

### **"Dataset Not Found"**
- Verify paths in notebook match your upload location
- Check file permissions (make dataset public or add to notebook)

### **"Slow Training"**
- Verify GPU is enabled (not CPU)
- Check GPU utilization: `!nvidia-smi`
- Reduce `num_workers` if CPU is bottleneck

---

## 📈 **Expected Results**

### **After 50 epochs** (~1 hour on T4):
- Training Loss: ~1.5-2.0
- Validation mAP@50: ~35-40%
- Status: Basic detection working

### **After 100 epochs** (~2 hours on T4):
- Training Loss: ~1.0-1.5
- Validation mAP@50: ~50-55%
- Status: Good detection quality

### **After 200 epochs** (~4 hours on T4):
- Training Loss: ~0.7-1.0
- Validation mAP@50: ~65-70%
- Status: Very good quality

### **After 300 epochs** (~6 hours on T4):
- Training Loss: ~0.5-0.8
- Validation mAP@50: ~70-75%
- Status: **Production ready!** ✅

---

## 🎓 **Quick Start Commands**

### **For Kaggle:**
1. Upload `YOLO_UDD_Kaggle_Training.ipynb`
2. Enable GPU + Internet
3. Add your dataset
4. Run all cells
5. ✅ Done!

### **For Colab:**
1. Upload files to Google Drive
2. Open `YOLO_UDD_Colab_Training.ipynb` in Colab
3. Enable GPU
4. Run all cells
5. Save checkpoints before timeout!
6. Repeat for multiple sessions

---

## 📚 **Additional Resources**

- **Kaggle Documentation**: https://www.kaggle.com/docs
- **Colab Documentation**: https://colab.research.google.com/
- **TensorBoard Guide**: https://www.tensorflow.org/tensorboard

---

## ✅ **Files to Upload**

From your local machine:
```
📦 Required Files:
├── 📓 YOLO_UDD_Kaggle_Training.ipynb (for Kaggle)
├── 📓 YOLO_UDD_Colab_Training.ipynb (for Colab)
├── 📄 train.json (22 MB)
├── 📄 val.json (5.6 MB)
└── 📁 images/ (6,065 train + 1,147 val images)
```

Everything else (code, models, configs) is automatically cloned from your GitHub repo!

---

## 🎉 **Summary**

- ✅ **Kaggle**: Best choice, fewer sessions, easier management
- ✅ **Colab**: Good alternative, needs more sessions, free GPU
- ✅ Both notebooks include auto-resume functionality
- ✅ Both save checkpoints automatically
- ✅ Both show results with TensorBoard

**Choose Kaggle for easiest experience!**

---

*Last Updated: November 1, 2025*
