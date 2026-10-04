# ⚡ ULTRA FAST Training - Get Results in 2-3 Hours!

**Goal**: Train on a small subset of data to quickly verify everything works and see initial results

---

## 🎯 What This Does

Instead of training on:
- **Full dataset**: 6,065 training images × 100 epochs = 75 hours
- **Fast config**: 6,065 images × 50 epochs = 25 hours

You train on:
- **Quick subset**: ~1,200 images (20%) × 20 epochs = **2-3 hours!** ⚡

---

## 📊 Comparison

| Config | Dataset Size | Epochs | Image Size | Time | Expected mAP |
|--------|-------------|--------|------------|------|--------------|
| **Original** | 6,065 images | 100 | 640px | ~75 hrs | 82-85% |
| **Fast** | 6,065 images | 50 | 640px | ~25 hrs | 82-84% |
| **Quick** ⚡ | 1,200 images | 20 | 512px | **2-3 hrs** | 65-75% |

---

## 🚀 How to Use

### Option 1: Automatic Subset Creation (Recommended)

The dataset loader will automatically use a subset when you use `train_config_quick.yaml`:

```bash
# On Kaggle or Local:
python scripts/train.py --config configs/train_config_quick.yaml
```

The config has `use_subset: true` and `subset_ratio: 0.2`, which tells the dataloader to randomly sample 20% of images each epoch.

### Option 2: Create Physical Subset (For Multiple Tests)

If you want to create a permanent subset directory:

```bash
# Create 20% subset
python scripts/create_subset.py --ratio 0.2 --seed 42

# This creates: data/trashcan_subset_20/
# Then train on it:
python scripts/train.py --config configs/train_config_quick.yaml
```

---

## 📁 Files Created

1. **`configs/train_config_quick.yaml`**
   - 20% data subset
   - 20 epochs
   - 512px images (faster)
   - Minimal augmentation
   - Batch size 32

2. **`scripts/create_subset.py`**
   - Creates physical subset if needed
   - Random sampling with seed
   - Preserves annotation format

---

## ⏱️ Expected Timeline

### On Kaggle T4 GPU:

**Quick Config** (~2-3 hours total):
```
Setup: 5 min
Epoch 1/20:  ████░░░░░░░░░░░ | 8 min  | Loss: 245.32
Epoch 5/20:  ████░░░░░░░░░░░ | 8 min  | mAP: 0.35
Epoch 10/20: ████░░░░░░░░░░░ | 8 min  | mAP: 0.55
Epoch 15/20: ████░░░░░░░░░░░ | 8 min  | mAP: 0.68
Epoch 20/20: ████░░░░░░░░░░░ | 8 min  | mAP: 0.70-0.75 ✅

Total: ~2.5 hours
```

**Per Epoch**: ~8 minutes (vs 30-45 min on full data)

---

## 🎯 What You'll Get

### After 2-3 Hours:

✅ **Working model** - Proves your pipeline works  
✅ **Initial results** - mAP ~65-75% (lower than target but functional)  
✅ **Trained weights** - Can visualize detections  
✅ **Quick validation** - See if TAFM module helps  
✅ **Fast iteration** - Test changes quickly

### Not Suitable For:

❌ Final paper/thesis results (need full training)  
❌ Accurate performance comparison  
❌ Publication-quality metrics  
❌ Production deployment

---

## 💡 Use Cases

### 1. **Initial Testing** ✅
```bash
# First time testing if your code works
python scripts/train.py --config configs/train_config_quick.yaml
```
**Result**: Know in 2-3 hours if everything works!

### 2. **Debugging** ✅
```bash
# Testing a code change
python scripts/train.py --config configs/train_config_quick.yaml --epochs 5
```
**Result**: 30-40 minutes to verify fix works

### 3. **Hyperparameter Search** ✅
```bash
# Test different learning rates quickly
for lr in 0.001 0.01 0.1; do
  python scripts/train.py --config configs/train_config_quick.yaml --lr $lr
done
```
**Result**: 6-9 hours to test 3 learning rates (vs 75+ hours)

### 4. **Demo/Presentation** ✅
- Show live training in progress
- Have results ready quickly
- Demonstrate the system works

---

## 🔄 Progressive Training Strategy

**Recommended approach:**

### Phase 1: Quick Test (2-3 hours)
```bash
python scripts/train.py --config configs/train_config_quick.yaml
```
**Goal**: Verify everything works, get initial results

### Phase 2: Fast Training (25 hours)
```bash
python scripts/train.py --config configs/train_config_fast.yaml
```
**Goal**: Get good results (82-84% mAP) reasonably fast

### Phase 3: Full Training (Optional - 75 hours)
```bash
python scripts/train.py --config configs/train_config.yaml
```
**Goal**: Squeeze out last 1-2% mAP for publication

---

## 🎨 Visualizing Quick Results

After quick training completes:

```python
# Load best model
checkpoint = torch.load('checkpoints_quick/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Run detection on validation images
python scripts/detect.py \
    --checkpoint checkpoints_quick/best.pt \
    --source data/trashcan/images/val/ \
    --output results_quick/
```

Even with 65-75% mAP, you'll see:
- ✅ Model detects most objects
- ✅ Bounding boxes mostly accurate  
- ✅ TAFM adapts to turbidity
- ✅ System is functional

---

## ⚙️ Quick Config Settings

```yaml
# What makes it fast:
data:
  use_subset: true        # Use only 20% of data
  subset_ratio: 0.2       # 1,200 images instead of 6,065
  img_size: 512           # Smaller images (vs 640)

training:
  epochs: 20              # Few epochs (vs 50-100)
  batch_size: 32          # Larger batches (faster)
  
augmentation:
  # Minimal augmentation for speed
  horizontal_flip: 0.5
  # Most augmentations disabled

evaluation:
  eval_interval: 2        # Validate less often
```

---

## 📊 Expected Results Comparison

After training completes, you can expect:

| Metric | Quick (20% data) | Fast (100% data) | Full (100% data) |
|--------|------------------|------------------|------------------|
| **mAP@50** | ~0.75-0.80 | ~0.88-0.90 | ~0.90-0.92 |
| **mAP@50:95** | ~0.65-0.75 | ~0.82-0.84 | ~0.84-0.86 |
| **Precision** | ~0.70-0.75 | ~0.85-0.87 | ~0.87-0.89 |
| **Recall** | ~0.68-0.72 | ~0.82-0.84 | ~0.84-0.86 |
| **Time** | 2-3 hours ⚡ | 25 hours | 75 hours |

---

## 🚦 Decision Guide

### Use Quick Config If:
- ✅ First time running the project
- ✅ Testing if code works
- ✅ Debugging issues
- ✅ Need results today
- ✅ Limited GPU quota
- ✅ Just want to see it work

### Use Fast Config If:
- ✅ Need good results (>82% mAP)
- ✅ For project submission
- ✅ Have 1-2 days
- ✅ Want optimal speed/accuracy balance

### Use Full Config If:
- ✅ Final paper/thesis
- ✅ Publication quality needed
- ✅ Have 3-4 days
- ✅ Want absolute best results

---

## 🎯 Example Workflow

**Day 1 Morning** (2-3 hours):
```bash
# Quick test to verify everything works
python scripts/train.py --config configs/train_config_quick.yaml
```
✅ Results: mAP ~70%, confirms system works!

**Day 1 Afternoon** → **Day 2**:
```bash
# Full fast training for good results
python scripts/train.py --config configs/train_config_fast.yaml
```
✅ Results: mAP ~83%, ready for submission!

---

## 📝 Summary

**Quick Config gives you:**
- ⚡ **2-3 hour training** (vs 25-75 hours)
- ✅ **Functional results** (65-75% mAP)
- 🎯 **Fast validation** of your code
- 💡 **Quick iteration** for debugging
- 🎨 **Demo-ready** visualizations

**Perfect for:**
- Initial testing
- Debugging
- Quick demos
- Learning the system
- Fast experimentation

**Then use `train_config_fast.yaml` for your actual project results!**

---

## 🚀 Start Now!

```bash
# Local:
cd /home/student/MIR/Project/YOLO-UDD-v2.0
source venv/bin/activate
python scripts/train.py --config configs/train_config_quick.yaml

# Kaggle: Just use train_config_quick.yaml in your notebook!
```

**Get results in 2-3 hours!** ⚡🎉
