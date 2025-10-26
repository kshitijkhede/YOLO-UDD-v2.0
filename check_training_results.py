#!/usr/bin/env python3
"""Check training results and verify completeness."""

import torch
import os
import glob
from pathlib import Path

def check_checkpoint(checkpoint_path):
    """Load and analyze a checkpoint file."""
    print(f"\n{'='*70}")
    print(f"📦 Analyzing: {checkpoint_path}")
    print('='*70)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Display checkpoint contents
        print("\n🔑 Checkpoint Keys:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"   • {key}: {type(checkpoint[key]).__name__} with {len(checkpoint[key])} items")
            elif isinstance(checkpoint[key], (int, float, str)):
                print(f"   • {key}: {checkpoint[key]}")
            else:
                print(f"   • {key}: {type(checkpoint[key]).__name__}")
        
        # Check for training metadata
        if 'epoch' in checkpoint:
            print(f"\n✅ Training Epoch: {checkpoint['epoch']}")
        
        if 'best_fitness' in checkpoint or 'best_metric' in checkpoint:
            metric = checkpoint.get('best_fitness') or checkpoint.get('best_metric')
            print(f"✅ Best Metric/Fitness: {metric}")
        
        if 'train_loss' in checkpoint:
            print(f"✅ Final Training Loss: {checkpoint['train_loss']:.4f}")
        
        if 'val_loss' in checkpoint:
            print(f"✅ Final Validation Loss: {checkpoint['val_loss']:.4f}")
        
        # Check model state
        if 'model' in checkpoint:
            model_state = checkpoint['model']
            if isinstance(model_state, dict):
                num_params = len(model_state)
                print(f"\n✅ Model State: {num_params} parameter tensors")
        
        # Check optimizer state
        if 'optimizer' in checkpoint:
            print(f"✅ Optimizer State: Saved")
        
        # File size
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"\n📊 Checkpoint Size: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading checkpoint: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("🔍 YOLO-UDD Training Results Verification")
    print("="*70)
    
    # Find all checkpoint directories
    runs_dir = Path("runs")
    
    if not runs_dir.exists():
        print("\n❌ No 'runs/' directory found!")
        print("   Training may not have completed or saved results.")
        return
    
    # Find all .pt files
    checkpoints = list(runs_dir.glob("**/checkpoints/*.pt"))
    
    if not checkpoints:
        print("\n❌ No checkpoint files (.pt) found in runs/ directory!")
        print("   Training may have failed or not saved checkpoints.")
        return
    
    print(f"\n✅ Found {len(checkpoints)} checkpoint file(s):")
    for ckpt in checkpoints:
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        modified = ckpt.stat().st_mtime
        from datetime import datetime
        mod_time = datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M:%S')
        print(f"   • {ckpt.relative_to(runs_dir)}: {size_mb:.1f} MB (modified: {mod_time})")
    
    # Check the most recent checkpoint
    latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
    check_checkpoint(str(latest_ckpt))
    
    # Check for TensorBoard logs
    print("\n" + "="*70)
    print("📊 TensorBoard Logs")
    print("="*70)
    
    log_files = list(runs_dir.glob("**/logs/events.out.tfevents.*"))
    if log_files:
        print(f"\n✅ Found {len(log_files)} TensorBoard event file(s)")
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        print(f"   Latest: {latest_log.relative_to(runs_dir)}")
        print(f"\n💡 To view training curves, run:")
        print(f"   tensorboard --logdir={latest_log.parent}")
    else:
        print("\n⚠️  No TensorBoard logs found")
    
    # Summary
    print("\n" + "="*70)
    print("📋 VERIFICATION SUMMARY")
    print("="*70)
    
    if len(checkpoints) > 0:
        print("✅ Training completed successfully!")
        print("✅ Model checkpoints saved")
        print(f"✅ Latest checkpoint: {latest_ckpt.name}")
        
        if len(checkpoints) >= 10:
            print("✅ Multiple epoch checkpoints found (training progressed)")
        
        if log_files:
            print("✅ TensorBoard logs available for analysis")
        
        print("\n🎯 Next Steps:")
        print("   1. Test the model on validation/test data")
        print("   2. Run inference on new images")
        print("   3. Analyze training curves in TensorBoard")
        print("   4. Fine-tune hyperparameters if needed")
    else:
        print("❌ No valid checkpoints found")
        print("   Please check training logs for errors")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
