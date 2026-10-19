"""
Dataset Verification Script for YOLO-UDD v2.0

Run this before uploading to Google Drive to ensure your dataset is correctly structured.

Usage:
    python scripts/verify_dataset.py --dataset-dir path/to/trashcan

This will check:
- Required files exist
- Image counts
- Annotation format
- Recommended ZIP structure
"""

import os
import json
import argparse
from pathlib import Path


def check_file_exists(path, description):
    """Check if file exists and print result"""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"  {status} {description}: {path}")
    return exists


def count_images(directory):
    """Count image files in directory"""
    if not os.path.exists(directory):
        return 0
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]
    return len(images)


def verify_annotation_file(json_path):
    """Verify COCO-style annotation file"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        has_images = 'images' in data
        has_annotations = 'annotations' in data
        has_categories = 'categories' in data
        
        num_images = len(data.get('images', []))
        num_annotations = len(data.get('annotations', []))
        num_categories = len(data.get('categories', []))
        
        print(f"    📊 Images: {num_images:,}")
        print(f"    📊 Annotations: {num_annotations:,}")
        print(f"    📊 Categories: {num_categories}")
        
        return has_images and has_annotations and has_categories
    except Exception as e:
        print(f"    ❌ Error reading JSON: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Verify YOLO-UDD dataset structure')
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to dataset directory (e.g., data/trashcan)')
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    print("=" * 70)
    print("🔍 YOLO-UDD Dataset Verification")
    print("=" * 70)
    print(f"\n📂 Dataset directory: {dataset_dir.absolute()}")
    
    if not dataset_dir.exists():
        print(f"\n❌ ERROR: Directory not found: {dataset_dir}")
        return False
    
    # Check required structure
    print("\n" + "=" * 70)
    print("📋 Checking Required Files")
    print("=" * 70)
    
    all_ok = True
    
    # Check annotation files (try multiple naming conventions)
    train_json_candidates = [
        dataset_dir / 'instances_train_trashcan.json',
        dataset_dir / 'train.json',
        dataset_dir / 'annotations' / 'train.json',
        dataset_dir / 'annotations' / 'instances_train.json'
    ]
    
    val_json_candidates = [
        dataset_dir / 'instances_val_trashcan.json',
        dataset_dir / 'val.json',
        dataset_dir / 'annotations' / 'val.json',
        dataset_dir / 'annotations' / 'instances_val.json'
    ]
    
    train_json = None
    val_json = None
    
    print("\n🔍 Looking for training annotations...")
    for candidate in train_json_candidates:
        if candidate.exists():
            train_json = candidate
            print(f"  ✅ Found: {candidate.relative_to(dataset_dir)}")
            break
    
    if not train_json:
        print("  ❌ Training annotations not found!")
        print("     Searched for:")
        for c in train_json_candidates:
            print(f"       - {c.relative_to(dataset_dir)}")
        all_ok = False
    else:
        print("\n  📄 Verifying training annotations...")
        if not verify_annotation_file(train_json):
            all_ok = False
    
    print("\n🔍 Looking for validation annotations...")
    for candidate in val_json_candidates:
        if candidate.exists():
            val_json = candidate
            print(f"  ✅ Found: {candidate.relative_to(dataset_dir)}")
            break
    
    if not val_json:
        print("  ❌ Validation annotations not found!")
        print("     Searched for:")
        for c in val_json_candidates:
            print(f"       - {c.relative_to(dataset_dir)}")
        all_ok = False
    else:
        print("\n  📄 Verifying validation annotations...")
        if not verify_annotation_file(val_json):
            all_ok = False
    
    # Check images directories
    print("\n" + "=" * 70)
    print("📸 Checking Image Directories")
    print("=" * 70)
    
    images_dir = dataset_dir / 'images'
    # Images can live under <dir>/images/<split> OR directly <dir>/<split>
    train_images_dir = images_dir / 'train' if (images_dir / 'train').exists() else dataset_dir / 'train'
    val_images_dir = images_dir / 'val' if (images_dir / 'val').exists() else dataset_dir / 'val'

    if not train_images_dir.exists() and not val_images_dir.exists():
        print(f"\n❌ Images directory not found: {images_dir}")
        all_ok = False
    else:
        print(f"\n✅ Images directory found: {images_dir.relative_to(dataset_dir)}")
        
        # Count training images
        train_count = count_images(train_images_dir)
        if train_count > 0:
            print(f"  ✅ Training images: {train_count:,} images")
        else:
            print(f"  ❌ No training images found in: {train_images_dir}")
            all_ok = False
        
        # Count validation images
        val_count = count_images(val_images_dir)
        if val_count > 0:
            print(f"  ✅ Validation images: {val_count:,} images")
        else:
            print(f"  ❌ No validation images found in: {val_images_dir}")
            all_ok = False
    
    # Calculate dataset size
    print("\n" + "=" * 70)
    print("📊 Dataset Statistics")
    print("=" * 70)
    
    total_size = 0
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            filepath = os.path.join(root, file)
            total_size += os.path.getsize(filepath)
    
    size_mb = total_size / (1024 * 1024)
    size_gb = total_size / (1024 * 1024 * 1024)
    
    if size_gb >= 1:
        print(f"\n📦 Total dataset size: {size_gb:.2f} GB")
    else:
        print(f"\n📦 Total dataset size: {size_mb:.1f} MB")
    
    # Final recommendation
    print("\n" + "=" * 70)
    print("✨ Recommendations")
    print("=" * 70)
    
    if all_ok:
        print("\n✅ Dataset structure looks good!")
        print("\n📤 Ready to upload to Google Drive:")
        print("   1. ZIP this directory")
        print("   2. Upload to Google Drive")
        print("   3. Share with 'Anyone with the link'")
        print("   4. Copy the FILE_ID from the share link")
        print("   5. Use FILE_ID in Kaggle notebook")
        
        # Show expected structure
        print("\n📂 Expected ZIP structure:")
        print("   trashcan.zip")
        print("   └── trashcan/")
        print("       ├── instances_train_trashcan.json")
        print("       ├── instances_val_trashcan.json")
        print("       └── images/")
        print("           ├── train/")
        print("           │   ├── image1.jpg")
        print("           │   └── ...")
        print("           └── val/")
        print("               ├── image1.jpg")
        print("               └── ...")
        
        if size_gb > 5:
            print(f"\n⚠️  Large dataset ({size_gb:.1f} GB):")
            print("   - Upload may take a while")
            print("   - Consider splitting into parts if > 15GB")
            print("   - Or upload directly as Kaggle dataset")
        
        return True
    else:
        print("\n❌ Dataset has issues that need to be fixed!")
        print("\n📋 Required structure:")
        print("   dataset/")
        print("   ├── instances_train_trashcan.json (or train.json)")
        print("   ├── instances_val_trashcan.json (or val.json)")
        print("   └── images/")
        print("       ├── train/ (with .jpg/.png images)")
        print("       └── val/ (with .jpg/.png images)")
        
        return False


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
