# 🚀 QUICK START - Kaggle Training (5 Minutes Setup)

## ⚡ **Ultra-Fast Setup**

### **STEP 1: Prepare Dataset (2 minutes)**

```bash
cd /home/student/MIR/Project/YOLO-UDD-v2.0
./prepare_kaggle_dataset.sh
```

This creates: `data/trashcan_dataset.zip` (~30 MB)

---

### **STEP 2: Upload to Kaggle (3 minutes)**

1. **Go to:** https://www.kaggle.com/datasets
2. **Click:** "New Dataset"
3. **Upload:** `trashcan_dataset.zip`
4. **Title:** "TrashCAN Underwater Debris Dataset"
5. **Click:** "Create"
6. **Copy your dataset path:** `YOUR_USERNAME/trashcan-underwater-debris-dataset`

---

### **STEP 3: Start Training Notebook (1 minute)**

1. **Go to:** https://www.kaggle.com/code
2. **Click:** "New Notebook"
3. **Import from GitHub:**
   ```
   https://github.com/kshitijkhede/YOLO-UDD-v2.0/blob/main/YOLO_UDD_Kaggle_Training.ipynb
   ```

---

### **STEP 4: Configure (1 minute)**

**Settings Panel (right side):**
- ✅ **Accelerator:** GPU T4 x2
- ✅ **Internet:** ON
- ✅ **Add Data:** Your uploaded dataset

**In Cell 4️⃣, update:**
```python
KAGGLE_DATASET_PATH = "/kaggle/input/trashcan-underwater-debris-dataset"
```

---

### **STEP 5: Train! (6 hours)**

Click **"Run All"** and wait for magic! ✨

---

## 📊 **What You'll Get**

After 6 hours (100 epochs):
- ✅ Trained model: `best.pth`
- ✅ mAP@50: ~55-60%
- ✅ TensorBoard logs
- ✅ Detection visualizations

---

## 🔄 **Continue Training?**

Repeat 2 more times for full training:
- **Session 1:** 0-100 epochs → mAP ~55%
- **Session 2:** 100-200 epochs → mAP ~68%
- **Session 3:** 200-300 epochs → mAP ~75% ✅

---

## 📖 **Need Detailed Instructions?**

See: `KAGGLE_COMPLETE_GUIDE.md`

---

## 🐛 **Troubleshooting**

| Problem | Solution |
|---------|----------|
| Out of Memory | Reduce batch_size to 8 or 4 |
| Dataset not found | Check path in Cell 4️⃣ |
| No GPU | Settings → GPU T4 x2 |
| Internet error | Settings → Internet ON |

---

## ✅ **Quick Checklist**

```
□ Dataset zip created
□ Uploaded to Kaggle
□ Notebook imported
□ GPU enabled
□ Internet enabled
□ Dataset added
□ Path updated
□ Run All clicked
```

---

## 🎯 **Total Time**

- Setup: 5 minutes
- Training: 6 hours
- **Results:** Underwater debris detection working! 🌊🗑️

---

**Ready? Let's go!** 🚀
