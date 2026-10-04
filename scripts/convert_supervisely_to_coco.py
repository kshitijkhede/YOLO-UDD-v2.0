"""
Convert Supervisely format dataset to COCO format for YOLO-UDD

This script converts the dataset from:
  dataset/
  ├── instance train/
  │   ├── ann/ (JSON files)
  │   └── img/ (images)
  └── instance val/
      ├── ann/ (JSON files)
      └── img/ (images)

To COCO format:
  trashcan/
  ├── instances_train_trashcan.json
  ├── instances_val_trashcan.json
  └── images/
      ├── train/
      └── val/

Usage:
    python scripts/convert_supervisely_to_coco.py --input-dir f:/MIR/project/dataset --output-dir data/trashcan_converted
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse


def polygon_to_bbox(points):
    """Convert polygon points to bounding box [x, y, width, height]"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_min, y_min, width, height]


def rectangle_to_bbox(points):
    """Convert rectangle points to bounding box"""
    # Rectangle has 2 points: top-left and bottom-right
    x1, y1 = points[0]
    x2, y2 = points[1]
    
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    
    return [x_min, y_min, width, height]


def convert_supervisely_to_coco(input_dir, output_dir, split='train'):
    """Convert Supervisely annotations to COCO format"""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Paths
    if split == 'train':
        ann_dir = input_dir / 'instance train' / 'ann'
        img_dir = input_dir / 'instance train' / 'img'
    else:
        ann_dir = input_dir / 'instance val' / 'ann'
        img_dir = input_dir / 'instance val' / 'img'
    
    output_img_dir = output_dir / 'images' / split
    output_img_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO format structure
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # Get all class names from annotations
    class_names = set()
    print(f"\n[{split.upper()}] Scanning for class names...")
    for json_file in ann_dir.glob('*.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
            for obj in data.get('objects', []):
                class_names.add(obj['classTitle'])
    
    # Create categories
    class_names = sorted(list(class_names))
    for idx, class_name in enumerate(class_names, start=1):
        coco_data['categories'].append({
            'id': idx,
            'name': class_name,
            'supercategory': 'object'
        })
    
    print(f"  Found {len(class_names)} classes: {class_names}")
    
    # Create class name to ID mapping
    class_to_id = {name: idx for idx, name in enumerate(class_names, start=1)}
    
    # Convert annotations
    image_id = 1
    annotation_id = 1
    
    json_files = list(ann_dir.glob('*.json'))
    print(f"\n[{split.upper()}] Converting {len(json_files)} annotations...")
    
    for json_file in tqdm(json_files, desc=f'Converting {split}'):
        # Read Supervisely annotation
        with open(json_file, 'r') as f:
            ann_data = json.load(f)
        
        # Get image filename (remove .json extension)
        img_filename = json_file.stem  # e.g., "vid_000002_frame0000023.jpg"
        img_path = img_dir / img_filename
        
        if not img_path.exists():
            print(f"  Warning: Image not found: {img_path}")
            continue
        
        # Copy image to output directory
        output_img_path = output_img_dir / img_filename
        if not output_img_path.exists():
            shutil.copy2(img_path, output_img_path)
        
        # Add image to COCO
        img_width = ann_data['size']['width']
        img_height = ann_data['size']['height']
        
        coco_data['images'].append({
            'id': image_id,
            'file_name': img_filename,
            'width': img_width,
            'height': img_height
        })
        
        # Convert objects to COCO annotations
        for obj in ann_data.get('objects', []):
            class_name = obj['classTitle']
            category_id = class_to_id[class_name]
            
            # Convert geometry to bounding box
            geometry_type = obj['geometryType']
            points = obj['points']['exterior']
            
            if geometry_type == 'polygon':
                bbox = polygon_to_bbox(points)
            elif geometry_type == 'rectangle':
                bbox = rectangle_to_bbox(points)
            else:
                print(f"  Warning: Unknown geometry type: {geometry_type}")
                continue
            
            # Calculate area
            area = bbox[2] * bbox[3]
            
            # Add annotation
            coco_data['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': bbox,
                'area': area,
                'iscrowd': 0
            })
            
            annotation_id += 1
        
        image_id += 1
    
    return coco_data


def main():
    parser = argparse.ArgumentParser(description='Convert Supervisely to COCO format')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory with Supervisely format (e.g., f:/MIR/project/dataset)')
    parser.add_argument('--output-dir', type=str, default='data/trashcan_converted',
                        help='Output directory for COCO format')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 70)
    print("🔄 Supervisely to COCO Converter")
    print("=" * 70)
    print(f"\n📂 Input:  {input_dir}")
    print(f"📂 Output: {output_dir}")
    
    # Check input directory
    if not input_dir.exists():
        print(f"\n❌ Input directory not found: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert training set
    print("\n" + "=" * 70)
    print("📊 Converting Training Set")
    print("=" * 70)
    train_coco = convert_supervisely_to_coco(input_dir, output_dir, 'train')
    
    train_output_file = output_dir / 'instances_train_trashcan.json'
    with open(train_output_file, 'w') as f:
        json.dump(train_coco, f, indent=2)
    
    print(f"\n✅ Saved: {train_output_file}")
    print(f"   Images: {len(train_coco['images']):,}")
    print(f"   Annotations: {len(train_coco['annotations']):,}")
    print(f"   Categories: {len(train_coco['categories'])}")
    
    # Convert validation set
    print("\n" + "=" * 70)
    print("📊 Converting Validation Set")
    print("=" * 70)
    val_coco = convert_supervisely_to_coco(input_dir, output_dir, 'val')
    
    val_output_file = output_dir / 'instances_val_trashcan.json'
    with open(val_output_file, 'w') as f:
        json.dump(val_coco, f, indent=2)
    
    print(f"\n✅ Saved: {val_output_file}")
    print(f"   Images: {len(val_coco['images']):,}")
    print(f"   Annotations: {len(val_coco['annotations']):,}")
    print(f"   Categories: {len(val_coco['categories'])}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ Conversion Complete!")
    print("=" * 70)
    
    total_images = len(train_coco['images']) + len(val_coco['images'])
    total_annotations = len(train_coco['annotations']) + len(val_coco['annotations'])
    
    print(f"\n📊 Total Statistics:")
    print(f"   Total images: {total_images:,}")
    print(f"   Total annotations: {total_annotations:,}")
    print(f"   Categories: {len(train_coco['categories'])}")
    
    print(f"\n📁 Output structure:")
    print(f"   {output_dir}/")
    print(f"   ├── instances_train_trashcan.json")
    print(f"   ├── instances_val_trashcan.json")
    print(f"   └── images/")
    print(f"       ├── train/ ({len(train_coco['images'])} images)")
    print(f"       └── val/ ({len(val_coco['images'])} images)")
    
    print(f"\n📋 Categories found:")
    for cat in train_coco['categories']:
        print(f"   {cat['id']}: {cat['name']}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Verify converted dataset:")
    print(f"      python scripts/verify_dataset.py --dataset-dir {output_dir}")
    print(f"   2. ZIP the dataset:")
    print(f"      Compress-Archive -Path '{output_dir}' -DestinationPath 'trashcan.zip'")
    print(f"   3. Upload to Google Drive")
    print(f"   4. Train in Kaggle!")


if __name__ == '__main__':
    main()
