#!/usr/bin/env python3
"""
Fix dataset by filtering out missing images from annotations
"""
import json
import os
from pathlib import Path

def fix_annotations(ann_file, img_dir, output_file):
    """Filter annotations to only include existing images"""
    print(f"\n🔍 Processing: {ann_file}")
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    total_images = len(data['images'])
    total_annotations = len(data['annotations'])
    
    # Check which images exist
    valid_images = []
    valid_image_ids = set()
    missing_count = 0
    
    for img_info in data['images']:
        img_path = os.path.join(img_dir, img_info['file_name'])
        if os.path.exists(img_path):
            valid_images.append(img_info)
            valid_image_ids.add(img_info['id'])
        else:
            missing_count += 1
            if missing_count <= 10:  # Show first 10
                print(f"  Missing: {img_info['file_name']}")
    
    if missing_count > 10:
        print(f"  ... and {missing_count - 10} more missing files")
    
    # Filter annotations to only valid images
    valid_annotations = [
        ann for ann in data['annotations']
        if ann['image_id'] in valid_image_ids
    ]
    
    # Create new filtered dataset
    filtered_data = {
        'images': valid_images,
        'annotations': valid_annotations,
        'categories': data['categories']
    }
    
    # Save filtered annotations
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f)
    
    print(f"\n  ✅ Results:")
    print(f"     Images: {total_images} → {len(valid_images)} ({missing_count} removed)")
    print(f"     Annotations: {total_annotations} → {len(valid_annotations)}")
    print(f"     Saved to: {output_file}")
    
    return len(valid_images), missing_count

def main():
    print("="*70)
    print("🔧 YOLO-UDD Dataset Fixer")
    print("="*70)
    print("\nThis script will:")
    print("  1. Scan for missing image files")
    print("  2. Create filtered annotations")
    print("  3. Allow training to proceed\n")
    
    base_dir = Path('data/trashcan')
    
    # Process train set
    train_ann = base_dir / 'annotations' / 'train.json'
    train_img_dir = base_dir / 'images' / 'train'
    train_out = base_dir / 'annotations' / 'train_fixed.json'
    
    if train_ann.exists():
        train_valid, train_missing = fix_annotations(
            str(train_ann), str(train_img_dir), str(train_out)
        )
    else:
        print(f"❌ Train annotations not found: {train_ann}")
        return
    
    # Process val set
    val_ann = base_dir / 'annotations' / 'val.json'
    val_img_dir = base_dir / 'images' / 'val'
    val_out = base_dir / 'annotations' / 'val_fixed.json'
    
    if val_ann.exists():
        val_valid, val_missing = fix_annotations(
            str(val_ann), str(val_img_dir), str(val_out)
        )
    else:
        print(f"\n⚠️  Val annotations not found: {val_ann}")
        val_valid, val_missing = 0, 0
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"\nTraining Set:")
    print(f"  Valid images: {train_valid}")
    print(f"  Missing images: {train_missing}")
    print(f"  Success rate: {train_valid/(train_valid+train_missing)*100:.1f}%")
    
    if val_valid > 0:
        print(f"\nValidation Set:")
        print(f"  Valid images: {val_valid}")
        print(f"  Missing images: {val_missing}")
        print(f"  Success rate: {val_valid/(val_valid+val_missing)*100:.1f}%")
    
    print("\n" + "="*70)
    print("✅ DATASET FIXED!")
    print("="*70)
    print("\nTo use fixed dataset, update dataset.py to load:")
    print(f"  • {train_out}")
    print(f"  • {val_out}")
    print("\nOr run training with --train-ann and --val-ann flags")
    print("="*70)

if __name__ == '__main__':
    main()
