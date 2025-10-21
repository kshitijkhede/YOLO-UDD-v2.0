# YOLO-UDD v2.0 Project Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Details](#architecture-details)
3. [Implementation Guide](#implementation-guide)
4. [Training Guide](#training-guide)
5. [Evaluation Guide](#evaluation-guide)
6. [Troubleshooting](#troubleshooting)

## Introduction

YOLO-UDD v2.0 is a state-of-the-art underwater debris detection model that addresses the unique challenges of the marine environment through three novel contributions:

1. **TAFM (Turbidity-Adaptive Fusion Module)**: Dynamically adjusts feature fusion based on water clarity
2. **PSEM (Partial Semantic Encoding Module)**: Enhances multi-scale feature representation
3. **SDWH (Split Dimension Weighting Head)**: Applies multi-stage attention for robust detection

## Architecture Details

### Complete Dataflow

```
Input Image (640×640×3)
    ↓
YOLOv9c Backbone
    ↓
Multi-scale Features [P3, P4, P5, P6]
    ↓
PSEM-enhanced Neck (PANet)
    ↓
TAFM (Turbidity Adaptation)
    ↓
SDWH Detection Head
    ↓
Predictions [BBox, Objectness, Class]
```

### Module Specifications

#### 1. TAFM (Turbidity-Adaptive Fusion Module)

**Purpose**: Adapt feature fusion to water conditions

**Input**: 
- Original image [B, 3, 640, 640]
- Neck features [B, C, H, W]

**Output**:
- Adapted features [B, C, H, W]
- Turbidity score [B, 1, 1, 1]

**Key Formula**: 
$$w_{adapt} = \sigma(Turb \cdot \alpha + (1-Turb) \cdot \beta)$$

where:
- $Turb$ is the estimated turbidity score [0, 1]
- $\alpha$ are learned parameters for murky conditions
- $\beta$ are learned parameters for clear conditions
- $\sigma$ is the sigmoid activation

#### 2. PSEM (Partial Semantic Encoding Module)

**Purpose**: Enhanced multi-scale feature fusion

**Principle**: $f(x) = Conv(Residual(x)) + x$

**Components**:
- Dual-branch processing (standard + dilated convolutions)
- Channel attention mechanism
- Spatial attention mechanism
- Residual connections

#### 3. SDWH (Split Dimension Weighting Head)

**Purpose**: Attention-based detection with noise suppression

**Three-stage Attention**:
1. **Level-wise**: Scale-specific weighting
2. **Spatial-wise**: Self-attention for location focus
   $$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$
3. **Channel-wise**: Squeeze-and-excitation for semantic tasks

## Implementation Guide

### Setting Up the Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Dataset Structure

Organize the TrashCan 1.0 dataset as follows:

```
data/trashcan/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

### Model Initialization

```python
from models import build_yolo_udd

# Build model
model = build_yolo_udd(
    num_classes=3,
    pretrained='path/to/coco_weights.pt'  # Optional
)

# Check model info
info = model.get_model_info()
for key, value in info.items():
    print(f"{key}: {value}")
```

## Training Guide

### Basic Training

```bash
python scripts/train.py \
    --config configs/train_config.yaml \
    --data-dir data/trashcan \
    --save-dir runs/train/exp1
```

### Advanced Training Options

```bash
python scripts/train.py \
    --config configs/train_config.yaml \
    --data-dir data/trashcan \
    --batch-size 16 \
    --epochs 300 \
    --lr 0.01 \
    --pretrained weights/yolov9c.pt \
    --save-dir runs/train/exp2
```

### Training Configuration

Key hyperparameters (from Section 5.2):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 16 | Batch size for training |
| `epochs` | 300 | Maximum training epochs |
| `learning_rate` | 0.01 | Initial learning rate |
| `weight_decay` | 0.0005 | AdamW weight decay |
| `early_stopping_patience` | 20 | Early stopping patience |

### Monitoring Training

```bash
# Start TensorBoard
tensorboard --logdir runs/train/exp1/logs

# View at http://localhost:6006
```

## Evaluation Guide

### Standard Evaluation

```bash
python scripts/evaluate.py \
    --weights runs/train/exp1/checkpoints/best.pt \
    --data-dir data/trashcan \
    --save-dir runs/eval/exp1
```

### Baseline Comparison

```bash
python scripts/evaluate.py \
    --weights runs/train/exp1/checkpoints/best.pt \
    --data-dir data/trashcan \
    --save-dir runs/eval/exp1 \
    --compare-baseline
```

This will output:
- Precision, Recall, mAP@50, mAP@50:95
- Comparison with YOLOv9c baseline (75.9%)
- Achievement toward target (>82%)

### Metrics Explanation

**Precision**: Proportion of correct positive predictions
$$P = \frac{TP}{TP + FP}$$

**Recall**: Proportion of actual positives detected
$$R = \frac{TP}{TP + FN}$$

**mAP@50**: Mean Average Precision at IoU threshold 0.50

**mAP@50:95**: Mean Average Precision averaged over IoU thresholds 0.50 to 0.95

**FPS**: Frames per second for real-time feasibility

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution**: Reduce batch size
```bash
python scripts/train.py --batch-size 8  # Instead of 16
```

#### 2. Slow Data Loading

**Solution**: Increase num_workers
```bash
python scripts/train.py --config configs/train_config.yaml
# Edit config: num_workers: 8
```

#### 3. Poor Initial Performance

**Solution**: Use pretrained weights
```bash
python scripts/train.py --pretrained weights/yolov9c_coco.pt
```

#### 4. Model Not Converging

**Checklist**:
- [ ] Verify data augmentations are not too aggressive
- [ ] Check learning rate (try 0.001 instead of 0.01)
- [ ] Ensure proper normalization
- [ ] Verify loss weights are balanced

### Performance Optimization

#### Mixed Precision Training

Enable in config:
```yaml
training:
  use_amp: true
```

Benefits:
- 2-3x faster training
- 30-50% less memory usage
- Minimal accuracy impact

#### Gradient Accumulation

For larger effective batch sizes:
```python
# In training loop
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Expected Results

Based on the project plan (Section 7.1):

### Timeline

| Month | Milestone | Expected mAP |
|-------|-----------|--------------|
| 1-2 | Baseline YOLOv9c + PSEM/SDWH | 78.7% |
| 3-4 | TAFM Integration | 80-81% |
| 5 | Full YOLO-UDD v2.0 | >82% |

### Performance Targets

| Model | mAP@50:95 | Improvement |
|-------|-----------|-------------|
| YOLOv9c (Baseline) | 75.9% | - |
| +PSEM/SDWH | ~78.7% | +2.8% |
| +TAFM (YOLO-UDD v2.0) | >82% | +6-7% |

## Citation

If you use this code, please cite:

```bibtex
@article{yoloud2025,
  title={YOLO-UDD v2.0: A Turbidity-Adaptive Architecture for 
         High-Fidelity Underwater Debris Detection},
  author={Your Name},
  journal={TBD},
  year={2025}
}
```

## Support

For issues or questions:
1. Check this documentation
2. Review closed issues on GitHub
3. Open a new issue with:
   - Error message
   - Environment details
   - Minimal reproducible example
