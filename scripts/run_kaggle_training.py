"""
Robust Kaggle training helper for YOLO-UDD v2.0

Usage (from a notebook cell):
  !python scripts/run_kaggle_training.py --data-dir /kaggle/working/trashcan --epochs 100 --batch-size 8

What it does:
- Checks/fixes NumPy compatibility and asks for kernel restart if needed.
- Detects GPU availability and switches to safe CPU defaults when no GPU is present.
- Installs only missing non-PyTorch dependencies (avoids reinstalling PyTorch on Kaggle).
- Attempts to locate COCO-style annotation files and fallbacks to common filenames.
- Runs `scripts/train.py` with safe CLI overrides and prints captured stdout/stderr on failure.

This script is intended to be run from Kaggle notebooks (or directly on Linux). It
is conservative about changing the environment and aims to make the training step
more robust without editing the notebook JSON.
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import shutil
import glob

def check_numpy():
    try:
        import numpy as np
        v = np.__version__
        if v.startswith('2.'):
            print(f"⚠️  NumPy {v} detected. This repo prefers NumPy 1.x (1.26.4).")
            print("Attempting to install NumPy 1.26.4. After install, PLEASE RESTART the kernel and re-run the notebook.")
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'numpy'])
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy==1.26.4'])
            print("Installed NumPy 1.26.4. Exiting so you can restart the kernel.")
            sys.exit(0)
        else:
            print(f"✅ NumPy {v} OK")
    except Exception as e:
        print(f"NumPy check failed: {e}")
        print("Installing NumPy 1.26.4 and exiting for restart")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy==1.26.4'])
        sys.exit(0)

def install_requirements(skip_torch=True):
    # Install the smaller set of dependencies that often cause runtime failures on Kaggle
    pkgs = [
        'opencv-python-headless>=4.7.0',
        'albumentations>=1.3.0',
        'pycocotools>=2.0.6',
        'tensorboard>=2.12.0',
        'tqdm',
        'pyyaml',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'pandas',
        'gdown'
    ]

    if skip_torch:
        print("ℹ️  Skipping PyTorch install (Kaggle already provides a compatible build).")

    to_install = []
    for pkg in pkgs:
        name = pkg.split('>=')[0]
        # quick heuristic: try import for some packages
        try:
            if name in ('pycocotools',):
                import pycocotools as _  # type: ignore
            elif name == 'albumentations':
                import albumentations as _  # type: ignore
            elif name == 'opencv-python-headless':
                import cv2 as _  # type: ignore
            elif name == 'gdown':
                import gdown as _  # type: ignore
            else:
                # skip deep import checks for others
                pass
        except Exception:
            to_install.append(pkg)

    if to_install:
        print(f"🔧 Installing missing packages: {to_install}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + to_install)
    else:
        print("✅ Required non-PyTorch packages already installed")

def find_annotations(dataset_dir: str) -> tuple[str, str]:
    # Common COCO-style names
    candidates = [
        ('annotations/train.json', 'annotations/val.json'),
        ('annotations/instances_train_trashcan.json', 'annotations/instances_val_trashcan.json'),
        ('instances_train_trashcan.json', 'instances_val_trashcan.json'),
        ('train/annotations.json', 'val/annotations.json')
    ]

    for train_rel, val_rel in candidates:
        t = os.path.join(dataset_dir, train_rel)
        v = os.path.join(dataset_dir, val_rel)
        if os.path.exists(t) and os.path.exists(v):
            return t, v

    # Fallback: look for any large json file in dataset dir
    json_files = glob.glob(os.path.join(dataset_dir, '**', '*.json'), recursive=True)
    if json_files:
        # choose the two largest jsons (likely train/val)
        json_files_sorted = sorted(json_files, key=lambda p: os.path.getsize(p), reverse=True)
        if len(json_files_sorted) >= 2:
            return json_files_sorted[0], json_files_sorted[1]

    raise FileNotFoundError(f"Could not find train/val annotation JSONs under {dataset_dir}. Searched common paths and found: {json_files}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/kaggle/working/trashcan')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='/kaggle/working/runs/train')
    args = parser.parse_args()

    # Step 1: NumPy check
    check_numpy()

    # Step 2: install non-PyTorch deps (conservative)
    install_requirements(skip_torch=True)

    # Step 3: determine device
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    device = 'cuda' if has_cuda else 'cpu'
    print(f"Device chosen: {device}")

    # If CPU-only, apply safer defaults to avoid OOM and long runs
    epochs = args.epochs
    batch_size = args.batch_size
    if device == 'cpu':
        print("⚠️  No GPU detected. Switching to CPU-safe defaults: batch_size=1, img_size=320, epochs=min(epochs,10)")
        batch_size = 1
        epochs = min(epochs, 10)

    # Step 4: verify dataset
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset path not found: {args.data_dir}")

    print(f"Dataset directory: {args.data_dir}")
    try:
        train_ann, val_ann = find_annotations(args.data_dir)
        print(f"Found annotations:\n  Train: {train_ann}\n  Val:   {val_ann}")
    except Exception as e:
        print(f"⚠️  Warning: {e}")
        print("Proceeding — scripts/train.py will attempt to create dataloaders; ensure dataset structure matches repository expectations.")

    # Step 5: Build command
    cmd = [
        sys.executable, 'scripts/train.py',
        '--config', 'configs/train_config.yaml',
        '--data-dir', os.path.abspath(args.data_dir),
        '--batch-size', str(batch_size),
        '--epochs', str(epochs),
        '--lr', str(args.lr),
        '--save-dir', args.save_dir
    ]
    if args.pretrained:
        cmd += ['--pretrained', args.pretrained]

    print('\nRunning training with command:')
    print(' '.join(cmd))

    # Step 6: run and capture output
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print('\n=== STDOUT ===')
        print(proc.stdout[:10000])
        if proc.returncode != 0:
            print('\n=== STDERR (truncated) ===')
            print(proc.stderr[-10000:])
            print(f"\nTraining failed with return code {proc.returncode}")
            sys.exit(proc.returncode)
        else:
            print('\nTraining finished successfully')
    except Exception as ex:
        print(f"Failed to run training: {ex}")
        raise

if __name__ == '__main__':
    main()
