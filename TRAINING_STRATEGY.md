# 🚀 Training Strategy for Limited GPU Time (Google Colab / Kaggle)

## 📋 Overview

Since Google Colab and Kaggle provide **2-3 hours of GPU time per session**, you'll need to train your YOLO-UDD v2.0 model across **multiple sessions**. This guide shows you how to do this efficiently.

---

## 🎯 Quick Strategy

### **Total Training Needed**: 300-500 epochs
### **GPU Time Available**: 2-3 hours per session
### **Epochs Per Session**: ~50 epochs (depends on GPU)
### **Total Sessions Needed**: 6-10 sessions

**Key Insight**: We'll save checkpoints to Google Drive, so you can resume training across sessions!

---

## 🔄 Multi-Session Training Workflow

### **Session 1** (First Time)
1. Open Colab notebook: `YOLO_UDD_Training_Colab.ipynb`
2. Enable GPU (Runtime → Change runtime type → T4 GPU)
3. Run all cells to train 50 epochs
4. Checkpoints auto-save to Google Drive
5. **Stop before timeout** (after ~2.5 hours)

### **Session 2-6** (Resume Training)
1. Open the same notebook
2. Run all cells again
3. Training **automatically resumes** from last checkpoint
4. Trains another 50 epochs
5. Repeat until you reach 300+ epochs

---

## 📊 Expected Progress Per Session

| Session | Epochs | Expected mAP | Training Time |
|---------|--------|--------------|---------------|
| 1       | 0-50   | 20-30%       | 2-3 hours     |
| 2       | 51-100 | 40-50%       | 2-3 hours     |
| 3       | 101-150| 55-60%       | 2-3 hours     |
| 4       | 151-200| 65-70%       | 2-3 hours     |
| 5       | 201-250| 70-72%       | 2-3 hours     |
| 6       | 251-300| 70-75%       | 2-3 hours     |

**Total Time**: 12-18 hours spread across 6 sessions

---

## 🎯 Two Platforms: Google Colab vs Kaggle

### **Google Colab** (Recommended for this project)

**Pros:**
- ✅ Free T4 GPU (2-3 hours)
- ✅ Easy Google Drive integration
- ✅ Simple checkpoint saving/loading
- ✅ Better for iterative training

**Cons:**
- ⚠️ Need to reconnect and rerun cells each session
- ⚠️ May disconnect during training

**Best For**: Multi-session training with checkpoints

### **Kaggle Notebooks**

**Pros:**
- ✅ Free GPU (30 hours/week)
- ✅ Can run longer sessions (9 hours)
- ✅ Better uptime

**Cons:**
- ⚠️ Harder to save checkpoints externally
- ⚠️ Weekly limit

**Best For**: If you can train in 2-3 long sessions

---

## 📦 Setup Instructions

### **For Google Colab**:

1. **Upload Dataset to Google Drive**:
   ```
   Google Drive/
   └── YOLO_UDD/
       ├── dataset/
       │   └── trashcan/
       │       ├── annotations/
       │       │   ├── train.json
       │       │   └── val.json
       │       └── images/
       │           ├── train/ (6,065 images)
       │           └── val/ (1,147 images)
       └── checkpoints/ (auto-created)
   ```

2. **Upload Notebook to Colab**:
   - Go to https://colab.research.google.com
   - Upload `YOLO_UDD_Training_Colab.ipynb`
   - Or: File → Upload notebook

3. **Enable GPU**:
   - Runtime → Change runtime type
   - Hardware accelerator → T4 GPU
   - Save

4. **Run All Cells** and let it train!

---

### **For Kaggle**:

1. **Upload Dataset to Kaggle**:
   - Zip your `data/trashcan` folder
   - Go to https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload the zip file

2. **Create New Notebook**:
   - Go to https://www.kaggle.com/code
   - Click "New Notebook"
   - Settings → Accelerator → GPU T4 x2
   - Add your dataset

3. **Upload Code**:
   - Copy your project code to Kaggle
   - Or clone from GitHub in the notebook

---

## 💾 Checkpoint Strategy

### **What Gets Saved**:
```
Google Drive/YOLO_UDD/checkpoints/
├── checkpoint_epoch_5.pth
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_15.pth
├── ...
├── checkpoint_epoch_50.pth
└── best_model.pth  (best validation mAP)
```

### **Auto-Resume Logic**:
The notebook automatically:
1. Checks for existing checkpoints
2. Loads the latest one
3. Resumes training from that epoch
4. Continues for 50 more epochs

---

## ⚡ Optimization Tips for Short Sessions

### **1. Reduce Batch Size**
```yaml
batch_size: 8  # Instead of 16
```
- Faster epochs
- More epochs per session
- Same final results

### **2. Use Mixed Precision Training**
```python
torch.cuda.amp.autocast()
```
- 2x faster training
- Uses less GPU memory

### **3. Reduce Image Size** (if needed)
```yaml
img_size: 512  # Instead of 640
```
- Faster training
- Slightly lower accuracy

### **4. Save More Frequently**
```yaml
save_freq: 5  # Save every 5 epochs
```
- Less data loss if disconnected

---

## 📱 Mobile Workflow

You can even train from your phone!

1. **Google Colab App** (Android/iOS)
   - Download Colab app
   - Open your notebook
   - Start training
   - Check progress later

2. **Kaggle App**
   - Similar workflow
   - Monitor training on mobile

---

## 🎯 Step-by-Step: First Training Session

### **Step 1**: Prepare Dataset (One-Time)

```bash
# On your local machine
cd /home/student/MIR/Project/YOLO-UDD-v2.0

# Zip the dataset
zip -r trashcan_dataset.zip data/trashcan/

# Upload to Google Drive:
# - Go to drive.google.com
# - Create folder: YOLO_UDD/dataset/
# - Upload and unzip trashcan_dataset.zip there
```

### **Step 2**: Upload Notebook

1. Go to https://colab.research.google.com
2. File → Upload notebook
3. Select `YOLO_UDD_Training_Colab.ipynb`

### **Step 3**: Start First Session

1. Enable GPU (Runtime → Change runtime type → T4 GPU)
2. Run first cell to check GPU
3. Run all cells (Runtime → Run all)
4. Wait 2-3 hours
5. Check results!

### **Step 4**: Resume in Next Session

1. Open same notebook
2. Run all cells again
3. Training automatically resumes!
4. Repeat until done

---

## 📊 Monitoring Progress

### **In Colab/Kaggle**:
```python
# View TensorBoard
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/YOLO_UDD/logs
```

### **After Each Session**:
```python
# Check progress
checkpoint = torch.load('latest_checkpoint.pth')
print(f"Completed epochs: {checkpoint['epoch']}")
print(f"Best mAP so far: {checkpoint['best_map']:.4f}")
```

---

## 🎓 Example Timeline

### **Day 1** (Weekend)
- **Morning**: Session 1 (Epochs 0-50)
- **Afternoon**: Session 2 (Epochs 51-100)
- **Evening**: Session 3 (Epochs 101-150)

### **Day 2** (Weekend)
- **Morning**: Session 4 (Epochs 151-200)
- **Afternoon**: Session 5 (Epochs 201-250)
- **Evening**: Session 6 (Epochs 251-300)

**Total**: 2 days, 300 epochs, trained model ready!

---

## 🚀 Quick Start Commands

### **Google Colab**:
```python
# Just run the notebook!
# Everything is automated
```

### **Manual Training** (if notebook doesn't work):
```python
# In Colab cell
!git clone https://github.com/kshitijkhede/YOLO-UDD-v2.0.git
%cd YOLO-UDD-v2.0
!pip install -r requirements.txt

# Link dataset from Drive
!ln -s /content/drive/MyDrive/YOLO_UDD/dataset/trashcan data/trashcan

# Train
!python scripts/train.py --config configs/colab_config.yaml
```

---

## 🎯 Final Results

After completing all sessions, you'll have:

✅ **Trained model** saved in Google Drive  
✅ **Training logs** with TensorBoard  
✅ **Evaluation metrics** (mAP ~70-75%)  
✅ **Detection results** on sample images  
✅ **Model weights** ready for deployment  

---

## 📚 Files Provided

1. **`YOLO_UDD_Training_Colab.ipynb`** - Ready-to-use Colab notebook
2. **`TRAINING_STRATEGY.md`** - This guide
3. **`configs/colab_config.yaml`** - Optimized config for Colab

---

## 🆘 Troubleshooting

### **Problem**: "No GPU available"
**Solution**: Runtime → Change runtime type → T4 GPU

### **Problem**: "Disconnected during training"
**Solution**: Checkpoints are saved! Just rerun all cells to resume.

### **Problem**: "Dataset not found"
**Solution**: Check Google Drive path, ensure dataset is uploaded

### **Problem**: "Out of memory"
**Solution**: Reduce batch_size to 4 in config

---

## ✨ Summary

**You CAN train your model with limited GPU time!**

- Use Google Colab (free GPU)
- Save checkpoints to Drive
- Train in 6-10 sessions of 2-3 hours each
- Total time: ~12-18 hours spread over 2-3 days
- Get 70-75% mAP results!

**Ready to start? Open `YOLO_UDD_Training_Colab.ipynb` in Google Colab!** 🚀
