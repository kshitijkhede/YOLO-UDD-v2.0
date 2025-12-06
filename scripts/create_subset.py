#!/usr/bin/env python3
"""
Create a subset of TrashCAN dataset for quick training tests
Usage: python scripts/create_subset.py --ratio 0.2 --seed 42
"""

import json
import argparse
import random
from pathlib import Path
import shutil

def create_subset(data_dir, ratio=0.2, seed=42):
    """
    Create a subset of the dataset by copying annotations and creating symlinks
    
    Args:
        data_dir: Path to data directory (e.g., 'data/trashcan')
        ratio: Fraction of data to keep (0.2 = 20%)
        seed: Random seed for reproducibility
    """
    data_dir = Path(data_dir)
    random.seed(seed)
    
    print(f"🎯 Creating {ratio*100:.0f}% subset of TrashCAN dataset")
    print(f"📂 Data directory: {data_dir}")
    print(f"🌱 Seed: {seed}\n")
    
    # Create subset directory
    subset_dir = data_dir.parent / f"trashcan_subset_{int(ratio*100)}"
    subset_dir.mkdir(exist_ok=True)
    (subset_dir / "annotations").mkdir(exist_ok=True)
    (subset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (subset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val']:
        print(f"📊 Processing {split} split...")
        
        # Load original annotations
        ann_file = data_dir / "annotations" / f"{split}.json"
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Get all image IDs
        all_image_ids = [img['id'] for img in data['images']]
        
        # Randomly sample subset
        num_keep = int(len(all_image_ids) * ratio)
        selected_ids = set(random.sample(all_image_ids, num_keep))
        
        # Filter images
        subset_images = [img for img in data['images'] if img['id'] in selected_ids]
        
        # Filter annotations
        subset_annotations = [ann for ann in data['annotations'] 
                            if ann['image_id'] in selected_ids]
        
        # Create subset annotation file
        subset_data = {
            'images': subset_images,
            'annotations': subset_annotations,
            'categories': data['categories']
        }
        
        subset_ann_file = subset_dir / "annotations" / f"{split}.json"
        with open(subset_ann_file, 'w') as f:
            json.dump(subset_data, f)
        
        print(f"  ✅ {split}: {len(data['images'])} → {len(subset_images)} images")
        print(f"  ✅ {split}: {len(data['annotations'])} → {len(subset_annotations)} annotations")
        
        # Create symlinks for selected images
        img_src_dir = data_dir / "images" / split
        img_dst_dir = subset_dir / "images" / split
        
        if img_src_dir.is_symlink():
            # If source is a symlink, resolve it
            img_src_dir = img_src_dir.resolve()
        
        copied = 0
        for img in subset_images:
            src = img_src_dir / img['file_name']
            dst = img_dst_dir / img['file_name']
            
            if src.exists() and not dst.exists():
                # Create symlink instead of copying to save space
                try:
                    dst.symlink_to(src.resolve())
                    copied += 1
                except Exception as e:
                    # If symlink fails, copy the file
                    shutil.copy2(src, dst)
                    copied += 1
        
        print(f"  ✅ Linked {copied} images\n")
    
    print(f"✅ Subset created at: {subset_dir}")
    print(f"\n📝 To use subset, update your config:")
    print(f"   data_dir: \"{subset_dir}\"")
    
    return subset_dir

def main():
    parser = argparse.ArgumentParser(description='Create dataset subset for quick training')
    parser.add_argument('--data-dir', type=str, default='data/trashcan',
                      help='Path to original dataset')
    parser.add_argument('--ratio', type=float, default=0.2,
                      help='Fraction of data to keep (0.2 = 20%%)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    create_subset(args.data_dir, args.ratio, args.seed)

if __name__ == '__main__':
    main()
