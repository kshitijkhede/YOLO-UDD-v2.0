#!/bin/bash
# TRAIN YOLO-UDD v2.0
# Simple script to start training with sensible defaults

echo "🚀 Starting YOLO-UDD v2.0 Training..."
echo ""

# Activate environment
source venv/bin/activate

# Create experiment name with timestamp
EXPERIMENT_NAME="yolo_udd_$(date +%Y%m%d_%H%M%S)"

echo "📝 Experiment: $EXPERIMENT_NAME"
echo "💾 Checkpoints will be saved to: runs/$EXPERIMENT_NAME/"
echo ""

# Start training
python3 scripts/train.py \
    --config configs/train_config.yaml \
    --experiment-name "$EXPERIMENT_NAME" \
    --resume ""

echo ""
echo "✅ Training completed!"
echo "📊 View results: tensorboard --logdir runs/$EXPERIMENT_NAME/"
