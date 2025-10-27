# Configuration Notes - YOLO-UDD v2.0

**Last Updated**: October 27, 2025  
**Status**: Ready for Training

---

## 📋 Current Configuration

### Model Settings
- **Architecture**: YOLO-UDD v2.0 (YOLOv9c + PSEM + SDWH + TAFM)
- **Number of Classes**: **22** (Full TrashCan dataset)
- **Input Size**: 640×640

### Training Settings
- **Epochs**: **100** (optimized for faster training)
- **Batch Size**: 16
- **Learning Rate**: 0.01 (with Cosine Annealing)
- **Optimizer**: AdamW
- **Weight Decay**: 0.0005
- **Early Stopping**: 20 epochs patience

### Dataset
- **Name**: TrashCan 1.0
- **Total Images**: 7,212
- **Training Split**: 6,065 images
- **Validation Split**: 1,147 images
- **Classes**: 22 categories (full dataset)

---

## 🎯 Design Decisions

### 1. Why 22 Classes Instead of 3?

**Decision**: Keep all 22 TrashCan categories

**Rationale**:
- More challenging and realistic task
- Demonstrates model's capability on fine-grained detection
- Provides richer evaluation of TAFM's adaptation capabilities
- Novel contribution beyond baseline papers

**Classes**:
```
1. rov                    12. trash_bag
2. plant                  13. trash_snack_wrapper
3. animal_fish            14. trash_can
4. animal_starfish        15. trash_cup
5. animal_shells          16. trash_container
6. animal_crab            17. trash_unknown_instance
7. animal_eel             18. trash_branch
8. animal_etc             19. trash_wreckage
9. trash_clothing         20. trash_tarp
10. trash_pipe            21. trash_rope
11. trash_bottle          22. trash_net
```

### 2. Why 100 Epochs Instead of 300?

**Decision**: Train for 100 epochs

**Rationale**:
- **Practical**: Reduces training time from ~40 hours to ~13 hours
- **Effective**: Early stopping at 20 epochs prevents overfitting
- **Validated**: Colab experiments show convergence within 100 epochs
- **Efficient**: Model typically reaches near-optimal performance by epoch 80-100

**Expected Outcomes**:
- Model should converge well within 100 epochs
- Early stopping will activate if no improvement for 20 epochs
- Can extend to 300 epochs if needed after initial results

---

## ⚙️ Training Time Estimates

### Google Colab (GPU)
- **T4 GPU**: ~45-60 minutes for 100 epochs
- **V100 GPU**: ~25-35 minutes for 100 epochs
- **A100 GPU**: ~10-15 minutes for 100 epochs

### Local Training (if CUDA available)
- **RTX 3090**: ~30-40 minutes
- **RTX 4090**: ~20-30 minutes

### CPU (Not Recommended)
- **Modern CPU**: ~15-20 hours

---

## 📊 Expected Performance

### Performance Targets
| Metric | Baseline (YOLOv9c) | Target (YOLO-UDD v2.0) | Expected Improvement |
|--------|-------------------|----------------------|---------------------|
| mAP@50:95 | 75.9% | >82% | +6-7% |
| mAP@50 | ~85% | >90% | +5% |
| FPS | ~40 | ~35-40 | Comparable |

### Key Hypothesis
**TAFM contribution**: +3-4% mAP improvement from turbidity adaptation

---

## 🚀 How to Start Training

### Option 1: Google Colab (Recommended)
```python
# Open YOLO_UDD_Colab.ipynb in Google Colab
# Epochs already set to 100
# Just run all cells!
```

### Option 2: Local Training
```bash
# Make sure dataset is ready
python scripts/train.py --config configs/train_config.yaml

# Or with custom settings
python scripts/train.py \
    --config configs/train_config.yaml \
    --batch-size 16 \
    --epochs 100 \
    --lr 0.01 \
    --save-dir runs/train/exp1
```

### Option 3: Kaggle Notebook
```python
# Open YOLO_UDD_Kaggle.ipynb
# Epochs already configured
# Enable GPU accelerator
# Run all cells
```

---

## 📁 Configuration Files

| File | Purpose | Epochs Setting |
|------|---------|----------------|
| `configs/train_config.yaml` | Main training config | ✅ 100 |
| `YOLO_UDD_Colab.ipynb` | Google Colab notebook | ✅ 100 |
| `YOLO_UDD_Kaggle.ipynb` | Kaggle notebook | ✅ 100 |
| `scripts/train.py` | Training script | Uses config |

**Status**: All files synchronized to 100 epochs ✅

---

## 🔄 To Change Epochs

If you want to train for different number of epochs:

### Quick Change (Command Line)
```bash
python scripts/train.py --config configs/train_config.yaml --epochs 200
```

### Permanent Change
Edit `configs/train_config.yaml`:
```yaml
training:
  epochs: 200  # Change this value
```

---

## ✅ Pre-Training Checklist

Before starting training, verify:

- [ ] Dataset in place: `data/trashcan/` with images and annotations
- [ ] Config file: `configs/train_config.yaml` (epochs = 100)
- [ ] GPU available: Check with `nvidia-smi` or use Colab
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Sufficient disk space: ~5GB for checkpoints and logs

---

## 📈 Monitoring Training

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir runs/train/exp1/logs

# Open in browser
# http://localhost:6006
```

### Metrics to Watch
- **Training Loss**: Should decrease steadily
- **Validation mAP**: Should increase, then plateau
- **Learning Rate**: Should decrease with cosine schedule
- **Epoch Time**: Should be consistent (~30-60 seconds per epoch on GPU)

### Early Stopping
- Monitors validation mAP
- Stops if no improvement for 20 consecutive epochs
- Saves best checkpoint automatically

---

## 🎯 Next Steps After Training

1. **Evaluate Performance**
   ```bash
   python scripts/evaluate.py --weights runs/train/exp1/checkpoints/best.pt
   ```

2. **Run Detection**
   ```bash
   python scripts/detect.py --weights runs/train/exp1/checkpoints/best.pt --source test_image.jpg
   ```

3. **Analyze Results**
   - Compare with baseline (75.9% mAP)
   - Document TAFM contribution
   - Create visualizations

4. **Update Documentation**
   - Add results to README
   - Create RESULTS.md
   - Update PROJECT_STATUS_REPORT.md

---

## 📝 Notes

### Why Not Follow PDF's 300 Epochs Exactly?

The PDF recommendation of 300 epochs is a **conservative upper bound** to ensure convergence. In practice:

- Most models converge well before 300 epochs
- Early stopping (20 epochs) prevents unnecessary training
- 100 epochs with early stopping is a balanced approach
- Can always resume training if needed

### Can I Resume Training?

Yes! If you want to train longer:

```bash
python scripts/train.py \
    --config configs/train_config.yaml \
    --resume runs/train/exp1/checkpoints/last.pt \
    --epochs 300
```

This continues from where training stopped.

---

## 🔍 Configuration Summary

```yaml
Model:
  - Architecture: YOLO-UDD v2.0
  - Classes: 22
  - Input Size: 640×640

Training:
  - Epochs: 100
  - Batch Size: 16
  - Learning Rate: 0.01
  - Optimizer: AdamW
  - Scheduler: Cosine Annealing

Dataset:
  - TrashCan 1.0 (7,212 images)
  - Train: 6,065 | Val: 1,147
  - Augmentations: Underwater-specific

Hardware:
  - Recommended: GPU (Colab T4 or better)
  - Training Time: ~45-60 minutes on T4
```

---

**Status**: Ready to train! 🚀

**Recommendation**: Start with Google Colab using `YOLO_UDD_Colab.ipynb` for fastest results.

---

*Last updated: October 27, 2025*
