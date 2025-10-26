#!/usr/bin/env python3
"""
Analyze and visualize training outputs from checkpoints
"""
import torch
import os
import sys
from pathlib import Path

def analyze_checkpoint(checkpoint_path):
    """Detailed checkpoint analysis"""
    print("="*70)
    print(f"📊 ANALYZING: {checkpoint_path}")
    print("="*70)
    
    try:
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        print("\n📦 CHECKPOINT CONTENTS:")
        print(f"   Keys available: {list(ckpt.keys())}")
        
        print("\n🔢 TRAINING METRICS:")
        if 'epoch' in ckpt:
            print(f"   ✅ Epoch: {ckpt['epoch']}")
        else:
            print("   ⚠️  Epoch: Not found")
            
        if 'train_loss' in ckpt:
            print(f"   📉 Training Loss: {ckpt['train_loss']:.4f}")
        else:
            print("   ⚠️  Training Loss: Not saved")
            
        if 'val_loss' in ckpt:
            print(f"   📉 Validation Loss: {ckpt['val_loss']:.4f}")
        else:
            print("   ⚠️  Validation Loss: Not saved")
            
        if 'best_fitness' in ckpt:
            fitness = ckpt['best_fitness']
            if isinstance(fitness, (int, float)):
                print(f"   🎯 Best Fitness/mAP: {fitness:.4f}")
            else:
                print(f"   🎯 Best Fitness: {fitness}")
        else:
            print("   ⚠️  Best Fitness: Not saved")
        
        print("\n🏗️  MODEL INFORMATION:")
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
            num_params = sum(p.numel() for p in state_dict.values())
            print(f"   ✅ Model layers: {len(state_dict)} layers")
            print(f"   ✅ Total parameters: {num_params:,}")
            print(f"   ✅ Model size: {num_params * 4 / (1024**2):.2f} MB (float32)")
        
        if 'optimizer' in ckpt:
            print(f"   ✅ Optimizer state: Saved")
            if 'param_groups' in ckpt['optimizer']:
                lr = ckpt['optimizer']['param_groups'][0].get('lr', 'N/A')
                print(f"   📊 Learning rate: {lr}")
        
        print("\n⚙️  HYPERPARAMETERS:")
        if 'hyperparameters' in ckpt:
            for key, value in ckpt['hyperparameters'].items():
                print(f"   • {key}: {value}")
        else:
            print("   ⚠️  Hyperparameters not saved in checkpoint")
        
        # File size
        size_mb = os.path.getsize(checkpoint_path) / (1024**2)
        print(f"\n💾 FILE SIZE: {size_mb:.2f} MB")
        
        return ckpt
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return None

def find_training_runs():
    """Find all training run directories"""
    runs_dir = Path('runs')
    if not runs_dir.exists():
        print("❌ No 'runs' directory found")
        return []
    
    training_dirs = []
    for subdir in runs_dir.iterdir():
        if subdir.is_dir():
            checkpoint_dir = subdir / 'checkpoints'
            if checkpoint_dir.exists():
                training_dirs.append(subdir)
    
    return sorted(training_dirs, key=lambda x: x.stat().st_mtime, reverse=True)

def main():
    print("🔍 YOLO-UDD TRAINING OUTPUT ANALYZER")
    print("="*70)
    
    # Find training runs
    training_dirs = find_training_runs()
    
    if not training_dirs:
        print("❌ No training runs found in 'runs' directory")
        return
    
    print(f"\n📁 FOUND {len(training_dirs)} TRAINING RUN(S):\n")
    for i, run_dir in enumerate(training_dirs, 1):
        mod_time = run_dir.stat().st_mtime
        from datetime import datetime
        mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"   {i}. {run_dir.name}")
        print(f"      Modified: {mod_date}")
        
        # List checkpoints
        checkpoint_dir = run_dir / 'checkpoints'
        checkpoints = list(checkpoint_dir.glob('*.pt'))
        print(f"      Checkpoints: {len(checkpoints)} files")
        
        for ckpt in sorted(checkpoints):
            size_mb = ckpt.stat().st_size / (1024**2)
            print(f"         • {ckpt.name} ({size_mb:.1f} MB)")
        print()
    
    # Analyze the most recent run's best checkpoint
    latest_run = training_dirs[0]
    print("="*70)
    print(f"📊 ANALYZING LATEST RUN: {latest_run.name}")
    print("="*70)
    
    # Try to find best.pt, then latest.pt
    checkpoint_dir = latest_run / 'checkpoints'
    best_ckpt = checkpoint_dir / 'best.pt'
    latest_ckpt = checkpoint_dir / 'latest.pt'
    
    if best_ckpt.exists():
        print("\n🏆 ANALYZING BEST CHECKPOINT:\n")
        analyze_checkpoint(best_ckpt)
    elif latest_ckpt.exists():
        print("\n📌 ANALYZING LATEST CHECKPOINT:\n")
        analyze_checkpoint(latest_ckpt)
    else:
        # Find any epoch checkpoint
        epoch_ckpts = sorted(checkpoint_dir.glob('epoch_*.pt'))
        if epoch_ckpts:
            print(f"\n📌 ANALYZING {epoch_ckpts[-1].name}:\n")
            analyze_checkpoint(epoch_ckpts[-1])
        else:
            print("\n❌ No checkpoint files found!")
    
    # Check for TensorBoard logs
    print("\n" + "="*70)
    print("📊 TENSORBOARD LOGS:")
    print("="*70)
    
    log_dir = latest_run / 'logs'
    if log_dir.exists():
        event_files = list(log_dir.glob('events.out.tfevents.*'))
        print(f"   ✅ Found {len(event_files)} TensorBoard event file(s)")
        print(f"\n   📍 To view results, run:")
        print(f"      tensorboard --logdir={latest_run}/logs/")
        print(f"      Then open: http://localhost:6006")
    else:
        print("   ⚠️  No TensorBoard logs found")
    
    print("\n" + "="*70)
    print("📚 For detailed explanation, see: UNDERSTANDING_OUTPUTS.md")
    print("="*70)

if __name__ == '__main__':
    main()
