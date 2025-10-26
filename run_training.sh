#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run training
python scripts/train.py \
    --config configs/train_config.yaml \
    --data-dir data/trashcan \
    --batch-size 4 \
    --epochs 50 \
    --save-dir runs/train
