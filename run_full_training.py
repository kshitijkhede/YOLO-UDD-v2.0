#!/usr/bin/env python3
"""
Full Training Script for YOLO-UDD v2.0
This script runs complete training with proper checkpointing and monitoring.
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_full_training(
    data_dir='data/trashcan',
    batch_size=8,
    epochs=10,
    learning_rate=0.01,
    save_dir=None,
    resume=None
):
    """
    Run full training with all epochs completed.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Training batch size (8 recommended for GPU, 4 for safety)
        epochs: Number of epochs to train (10 minimum, 30+ recommended)
        learning_rate: Learning rate for optimizer
        save_dir: Directory to save results (auto-generated if None)
        resume: Path to checkpoint to resume from (optional)
    """
    
    # Create save directory with timestamp
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f'runs/full_training_{epochs}epochs_{timestamp}'
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Build training command
    cmd = [
        sys.executable, 'scripts/train.py',
        '--config', 'configs/train_config.yaml',
        '--data-dir', data_dir,
        '--batch-size', str(batch_size),
        '--epochs', str(epochs),
        '--lr', str(learning_rate),
        '--save-dir', save_dir
    ]
    
    # Add resume flag if checkpoint provided
    if resume:
        cmd.extend(['--resume', resume])
    
    print("="*70)
    print("🚀 YOLO-UDD v2.0 - FULL TRAINING SETUP")
    print("="*70)
    print(f"\n📋 Training Configuration:")
    print(f"   Dataset:       {data_dir}")
    print(f"   Batch Size:    {batch_size}")
    print(f"   Epochs:        {epochs}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Save Dir:      {save_dir}")
    if resume:
        print(f"   Resume From:   {resume}")
    print(f"\n⏱️  Estimated Time:")
    iterations_per_epoch = 5769 // batch_size
    estimated_time = (iterations_per_epoch * epochs * 0.5) / 60  # ~0.5s per iteration
    print(f"   Iterations/epoch: {iterations_per_epoch}")
    print(f"   Total time:       ~{estimated_time:.1f} minutes ({estimated_time/60:.1f} hours)")
    print(f"\n💡 Tips:")
    print(f"   • Training will save checkpoint after each epoch")
    print(f"   • Monitor progress in real-time")
    print(f"   • Press Ctrl+C to stop (progress will be saved)")
    print("="*70)
    
    # Confirmation
    response = input("\n🚦 Start training? (yes/y to continue, anything else to cancel): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Training cancelled.")
        return False
    
    print("\n" + "="*70)
    print("🎯 STARTING TRAINING...")
    print("="*70)
    print(f"Command: {' '.join(cmd)}\n")
    
    # Track start time
    start_time = time.time()
    
    try:
        # Run training (output shown in real-time)
        result = subprocess.run(cmd, check=True)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        elapsed_minutes = elapsed_time / 60
        elapsed_hours = elapsed_minutes / 60
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📊 Results:")
        print(f"   Total time:     {elapsed_hours:.2f} hours ({elapsed_minutes:.1f} minutes)")
        print(f"   Epochs:         {epochs} (all completed)")
        print(f"   Checkpoints:    {save_dir}/checkpoints/")
        print(f"   Logs:           {save_dir}/logs/")
        print(f"\n🎉 Your model is now fully trained and ready to use!")
        print("="*70)
        
        return True
        
    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        elapsed_minutes = elapsed_time / 60
        
        print("\n" + "="*70)
        print("⚠️  TRAINING INTERRUPTED BY USER")
        print("="*70)
        print(f"   Time elapsed: {elapsed_minutes:.1f} minutes")
        print(f"   Checkpoints saved in: {save_dir}/checkpoints/")
        print(f"\n💡 You can resume training later with:")
        print(f"   python run_full_training.py --resume {save_dir}/checkpoints/latest.pt")
        print("="*70)
        
        return False
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        elapsed_minutes = elapsed_time / 60
        
        print("\n" + "="*70)
        print("❌ TRAINING FAILED")
        print("="*70)
        print(f"   Exit code:    {e.returncode}")
        print(f"   Time elapsed: {elapsed_minutes:.1f} minutes")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Check GPU memory: nvidia-smi")
        print(f"   2. Try smaller batch size (--batch-size 4 or 2)")
        print(f"   3. Check dataset: python scripts/train.py --data-dir {data_dir} --test")
        print(f"   4. Review error messages above")
        print("="*70)
        
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run full YOLO-UDD training')
    parser.add_argument('--data-dir', default='data/trashcan', help='Dataset directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size (8 recommended)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (10 minimum, 30+ recommended)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--save-dir', default=None, help='Save directory (auto if not specified)')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    success = run_full_training(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        resume=args.resume
    )
    
    sys.exit(0 if success else 1)
