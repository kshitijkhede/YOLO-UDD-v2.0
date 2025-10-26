# 🎯 Complete Training Guide for YOLO-UDD v2.0

## 📊 Current Status
- ✅ Model architecture: Working (695 MB checkpoint)
- ⚠️ Training progress: Only 1 epoch completed (need 9 more)
- 🎯 Goal: Complete full training with 10+ epochs

---

## 🚀 Option 1: Run Full Training Locally (Recommended for Complete Control)

### Quick Start:
```bash
cd /home/student/MIR/Project/YOLO-UDD-v2.0

# Run with default settings (10 epochs)
python3 run_full_training.py

# Or customize:
python3 run_full_training.py --epochs 30 --batch-size 8
```

### Advanced Options:
```bash
# Resume from existing checkpoint
python3 run_full_training.py --resume runs/train/checkpoints/latest.pt --epochs 10

# Train for longer (better results)
python3 run_full_training.py --epochs 50 --batch-size 8

# Use smaller batch size if GPU memory issues
python3 run_full_training.py --epochs 30 --batch-size 4
```

### Expected Output:
```
🚀 YOLO-UDD v2.0 - FULL TRAINING SETUP
======================================================================
📋 Training Configuration:
   Dataset:       data/trashcan
   Batch Size:    8
   Epochs:        10
   Learning Rate: 0.01
   Save Dir:      runs/full_training_10epochs_20251026_120000

⏱️  Estimated Time:
   Iterations/epoch: 721
   Total time:       ~60 minutes (1.0 hours)
======================================================================

Training progress will show in real-time...
Epoch 1/10: 100%|████████████| 721/721 [06:23<00:00, 1.89it/s]
Epoch 2/10: 100%|████████████| 721/721 [06:21<00:00, 1.89it/s]
...
```

---

## 🌐 Option 2: Run Full Training on Google Colab

### Step-by-Step:

1. **Open Updated Notebook:**
   ```
   https://colab.research.google.com/github/kshitijkhede/YOLO-UDD-v2.0/blob/main/YOLO_UDD_Colab.ipynb
   ```

2. **Enable GPU:**
   - Runtime → Change runtime type → GPU (T4 or better)

3. **Run Cells Sequentially:**
   - Cell 1-9: Setup environment and dataset
   - Cell 10-12: Check GPU and clear cache
   - Cell 13: Test model architecture
   - Cell 14-15: Configure training (already set to 10 epochs)
   - **Cell 19: START TRAINING** ⭐ **This is the main cell!**

4. **Monitor Progress:**
   - Watch epoch progress in real-time
   - Training will take ~1-1.5 hours for 10 epochs
   - Results auto-saved to Google Drive

5. **Important:** 
   - ⚠️ **Do NOT interrupt the training cell!**
   - ⚠️ Keep the Colab tab active (prevents disconnection)
   - ⚠️ Ensure stable internet connection

---

## 📈 Recommended Training Configurations

### Quick Test (10 epochs - ~1 hour):
```python
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.01
```
**Use case:** Initial testing, quick results

### Standard Training (30 epochs - ~3 hours):
```python
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.01
```
**Use case:** Good baseline performance

### Full Training (50 epochs - ~5 hours):
```python
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.01
```
**Use case:** Best performance, production-ready

### Safe Mode (GPU memory constrained):
```python
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 0.01
```
**Use case:** Limited GPU memory, prevents OOM errors

---

## 🔍 Monitoring Training Progress

### 1. Real-time Terminal Output:
```
Epoch 1/10:  10%|█         | 72/721 [00:38<05:45, 1.88it/s]
Loss: 2.345, Val Loss: 1.987
```

### 2. TensorBoard (Visual):
```bash
tensorboard --logdir=runs/full_training_10epochs_*/logs/
```
Then open: http://localhost:6006

### 3. Check Saved Checkpoints:
```bash
ls -lh runs/full_training_*/checkpoints/
```
You should see:
- `epoch_1.pt`, `epoch_2.pt`, ... , `epoch_10.pt`
- `best.pt` (best performing model)
- `latest.pt` (most recent checkpoint)

---

## ✅ Verification After Training

### 1. Check Epoch Count:
```bash
python3 << 'EOF'
import torch
checkpoint = torch.load('runs/full_training_*/checkpoints/latest.pt', map_location='cpu')
print(f"Completed epochs: {checkpoint.get('epoch', 'unknown')}")
print(f"Best fitness: {checkpoint.get('best_fitness', 'unknown')}")
EOF
```

### 2. Expected Output:
```
Completed epochs: 10
Best fitness: 0.892
```

### 3. Verify All Checkpoints:
```bash
find runs/full_training_* -name "epoch_*.pt" | wc -l
```
Should output: `10` (or your configured epoch count)

---

## 🎯 Next Steps After Full Training

### 1. Evaluate Model Performance:
```bash
python3 scripts/evaluate.py \
    --checkpoint runs/full_training_*/checkpoints/best.pt \
    --data-dir data/trashcan \
    --save-dir runs/evaluation/
```

### 2. Test on New Images:
```bash
python3 scripts/detect.py \
    --checkpoint runs/full_training_*/checkpoints/best.pt \
    --source path/to/test/images/ \
    --save-dir runs/detections/ \
    --conf-threshold 0.5
```

### 3. Analyze Results:
```bash
# View training curves
tensorboard --logdir=runs/full_training_*/logs/

# Check metrics
cat runs/evaluation/metrics.txt
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: Training Stops After 1 Epoch
**Cause:** Script interrupted or error occurred  
**Solution:** 
```bash
# Resume from checkpoint
python3 run_full_training.py --resume runs/train/checkpoints/latest.pt --epochs 10
```

### Issue 2: GPU Out of Memory
**Cause:** Batch size too large  
**Solution:**
```bash
# Reduce batch size
python3 run_full_training.py --batch-size 4 --epochs 10
```

### Issue 3: Training Too Slow (3+ hours expected)
**Cause:** CPU training or low GPU  
**Solution:**
- Check GPU: `nvidia-smi`
- Ensure GPU is used in code
- Use Google Colab with GPU runtime

### Issue 4: Colab Disconnects During Training
**Cause:** Idle timeout or poor connection  
**Solution:**
- Keep Colab tab active
- Use Colab Pro for longer runtime
- Enable auto-clicker extension
- Results are saved to Drive (can resume)

---

## 📊 Expected Results After Full Training

### With 10 Epochs:
- ✅ Basic convergence achieved
- ✅ Model can detect underwater debris
- ⚠️ May need more epochs for best performance

### With 30 Epochs:
- ✅ Good performance on validation set
- ✅ Stable loss curves
- ✅ Ready for testing and deployment

### With 50+ Epochs:
- ✅ Best possible performance
- ✅ Production-ready model
- ✅ Optimal detection accuracy

---

## 🚀 Quick Start Commands

### Local Training:
```bash
# Standard 10-epoch training
python3 run_full_training.py

# Extended 30-epoch training (recommended)
python3 run_full_training.py --epochs 30

# Full 50-epoch training (best results)
python3 run_full_training.py --epochs 50
```

### After Training:
```bash
# Evaluate
python3 scripts/evaluate.py --checkpoint runs/full_training_*/checkpoints/best.pt

# Test detection
python3 scripts/detect.py --checkpoint runs/full_training_*/checkpoints/best.pt --source test_images/

# View logs
tensorboard --logdir=runs/full_training_*/logs/
```

---

## 📞 Support

If you encounter issues:
1. Check this guide first
2. Review error messages carefully
3. Try the troubleshooting solutions
4. Check GPU status: `nvidia-smi`
5. Verify dataset: `python3 scripts/train.py --test`

**Repository:** https://github.com/kshitijkhede/YOLO-UDD-v2.0

---

**Last Updated:** October 26, 2025  
**Status:** Ready for full training ✅
