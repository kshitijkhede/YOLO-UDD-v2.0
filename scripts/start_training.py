#!/usr/bin/env python3
"""
Complete Training Pipeline for YOLO-UDD v2.0
This script implements the full training protocol from the project plan
"""

import os
import sys
import torch
import yaml
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*80)
print("YOLO-UDD v2.0 - Complete Training Pipeline")
print("="*80)

# Check CUDA availability
if torch.cuda.is_available():
    print(f"✓ CUDA Available: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠ WARNING: CUDA not available. Training will be slow on CPU.")

# Import project modules
try:
    from models import build_yolo_udd
    from data.dataset import create_dataloaders
    from utils.loss import YOLOUDDLoss
    from utils.metrics import compute_metrics_coco, compute_metrics
    from utils.nms import batched_nms
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Error importing modules: {e}")
    sys.exit(1)

# Verify dataset exists
data_dir = project_root / "data" / "trashcan"
if not data_dir.exists():
    print(f"✗ Error: Dataset not found at {data_dir}")
    sys.exit(1)

annotations_dir = data_dir / "annotations"
if not (annotations_dir / "train.json").exists():
    print(f"✗ Error: train.json not found in {annotations_dir}")
    sys.exit(1)

print(f"✓ Dataset found at {data_dir}")

# Test model creation
print("\nTesting model creation...")
try:
    test_model = build_yolo_udd(num_classes=22)  # Full TrashCAN dataset
    print(f"✓ Model created successfully")
    
    # Test forward pass
    test_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        test_output = test_model(test_input)
    print(f"✓ Forward pass successful")
    print(f"  Output scales: {len(test_output)}")
    
    del test_model, test_input, test_output
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
except Exception as e:
    print(f"✗ Error in model test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test loss function
print("\nTesting loss function...")
try:
    test_loss_fn = YOLOUDDLoss(num_classes=3)
    print(f"✓ Loss function created successfully")
    
    # Create dummy data for loss test
    predictions = [
        (torch.randn(1, 4, 80, 80), torch.randn(1, 1, 80, 80), torch.randn(1, 3, 80, 80)),
        (torch.randn(1, 4, 40, 40), torch.randn(1, 1, 40, 40), torch.randn(1, 3, 40, 40)),
        (torch.randn(1, 4, 20, 20), torch.randn(1, 1, 20, 20), torch.randn(1, 3, 20, 20)),
    ]
    target_boxes = [torch.tensor([[320.0, 320.0, 100.0, 100.0]])]
    target_labels = [torch.tensor([0])]
    
    loss_dict = test_loss_fn(predictions, target_boxes, target_labels)
    print(f"✓ Loss computation successful")
    print(f"  Total loss: {loss_dict['total_loss']:.4f}")
    
    del test_loss_fn, predictions, target_boxes, target_labels, loss_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
except Exception as e:
    print(f"✗ Error in loss test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("ALL PRE-FLIGHT CHECKS PASSED ✓")
print("="*80)

print("\n🚀 Ready to start training!")
print("\nNext steps:")
print("1. Review training configuration in configs/train_config.yaml")
print("2. Run training with:")
print("   python scripts/train.py --config configs/train_config.yaml")
print("\nOr start quick test training (10 epochs, batch=4):")
print("   python scripts/train.py --config configs/train_config.yaml --epochs 10 --batch-size 4")

print("\n" + "="*80)

# Offer to start training now
response = input("\nStart training now? (yes/no): ").strip().lower()
if response in ['yes', 'y']:
    print("\n🚀 Starting training...")
    
    # Load config
    config_path = project_root / "configs" / "train_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse arguments for quick override
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    args, _ = parser.parse_known_args()
    
    # Override config if provided
    if args.epochs:
        config['training']['epochs'] = args.epochs
        print(f"  Overriding epochs: {args.epochs}")
    
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        print(f"  Overriding batch size: {args.batch_size}")
    
    # Build training config
    train_config = {
        'num_classes': config['model']['num_classes'],
        'pretrained_path': config['model'].get('pretrained_path'),
        'data_dir': config['data']['data_dir'],
        'img_size': config['data']['img_size'],
        'batch_size': config['training']['batch_size'],
        'num_workers': config['training'].get('num_workers', 4),
        'epochs': config['training']['epochs'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training'].get('weight_decay', 0.0005),
        'save_dir': str(project_root / 'runs' / 'train'),
        'use_amp': config['training'].get('use_amp', True),
        'early_stopping_patience': config['training'].get('early_stopping_patience', 20)
    }
    
    print(f"\nTraining Configuration:")
    print(f"  Dataset: {train_config['data_dir']}")
    print(f"  Classes: {train_config['num_classes']}")
    print(f"  Epochs: {train_config['epochs']}")
    print(f"  Batch Size: {train_config['batch_size']}")
    print(f"  Learning Rate: {train_config['learning_rate']}")
    print(f"  Image Size: {train_config['img_size']}")
    print(f"  Mixed Precision: {train_config['use_amp']}")
    print(f"  Save Directory: {train_config['save_dir']}")
    
    # Import and run trainer
    from scripts.train import Trainer
    
    try:
        trainer = Trainer(train_config)
        trainer.train()
        
        print("\n" + "="*80)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nCheckpoints saved to: {train_config['save_dir']}/checkpoints/")
        print(f"TensorBoard logs: {train_config['save_dir']}/logs/")
        print(f"\nTo view training curves:")
        print(f"  tensorboard --logdir {train_config['save_dir']}/logs/")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("  Latest checkpoint saved automatically")
    except Exception as e:
        print(f"\n\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("\nTraining cancelled. Run the command above when ready.")

