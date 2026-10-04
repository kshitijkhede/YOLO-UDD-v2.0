"""
Create a minimal dummy TrashCan dataset for testing
"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw
import random


def create_dummy_image(image_id, size=(640, 480), num_objects=3):
    """
    Create a dummy underwater image with colored rectangles as objects
    
    Args:
        image_id: Image identifier
        size: Image size (width, height)
        num_objects: Number of objects to draw
        
    Returns:
        image: PIL Image
        annotations: List of annotation dictionaries
    """
    # Create base image with underwater-like color
    img = Image.new('RGB', size, color=(20, 60, 100))
    draw = ImageDraw.Draw(img)
    
    annotations = []
    
    # Colors for different classes
    class_colors = {
        1: (200, 100, 50),   # trash - brownish
        2: (100, 150, 200),  # animal - bluish
        3: (150, 150, 150),  # rov - grayish
    }
    
    for i in range(num_objects):
        # Random position and size
        x = random.randint(50, size[0] - 150)
        y = random.randint(50, size[1] - 150)
        w = random.randint(50, 100)
        h = random.randint(50, 100)
        
        # Random class
        category_id = random.randint(1, 3)
        color = class_colors[category_id]
        
        # Draw rectangle
        draw.rectangle([x, y, x + w, y + h], fill=color, outline=(255, 255, 255))
        
        # Create annotation
        annotations.append({
            "id": image_id * 100 + i,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0
        })
    
    return img, annotations


def create_dummy_dataset(output_dir='data/trashcan', num_train=50, num_val=20):
    """
    Create a complete dummy TrashCan dataset
    
    Args:
        output_dir: Output directory for dataset
        num_train: Number of training images
        num_val: Number of validation images
    """
    print(f"Creating dummy TrashCan dataset in {output_dir}...")
    
    # Create directory structure
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    
    # Categories
    categories = [
        {"id": 1, "name": "trash", "supercategory": "object"},
        {"id": 2, "name": "animal", "supercategory": "object"},
        {"id": 3, "name": "rov", "supercategory": "object"}
    ]
    
    # Create training set
    print(f"Generating {num_train} training images...")
    train_images = []
    train_annotations = []
    
    for i in range(num_train):
        image_id = i + 1
        filename = f"train_{image_id:04d}.jpg"
        
        # Generate image and annotations
        img, anns = create_dummy_image(image_id, num_objects=random.randint(2, 5))
        
        # Save image
        img_path = os.path.join(output_dir, 'images', 'train', filename)
        img.save(img_path, 'JPEG')
        
        # Add to dataset
        train_images.append({
            "id": image_id,
            "file_name": filename,
            "height": img.height,
            "width": img.width
        })
        train_annotations.extend(anns)
    
    # Save training annotations
    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories
    }
    
    train_json_path = os.path.join(output_dir, 'instances_train_trashcan.json')
    with open(train_json_path, 'w') as f:
        json.dump(train_coco, f, indent=2)
    print(f"✓ Saved {train_json_path}")
    print(f"  - {len(train_images)} images")
    print(f"  - {len(train_annotations)} annotations")
    
    # Create validation set
    print(f"\nGenerating {num_val} validation images...")
    val_images = []
    val_annotations = []
    
    for i in range(num_val):
        image_id = i + 1
        filename = f"val_{image_id:04d}.jpg"
        
        # Generate image and annotations
        img, anns = create_dummy_image(image_id, num_objects=random.randint(2, 5))
        
        # Save image
        img_path = os.path.join(output_dir, 'images', 'val', filename)
        img.save(img_path, 'JPEG')
        
        # Add to dataset
        val_images.append({
            "id": image_id,
            "file_name": filename,
            "height": img.height,
            "width": img.width
        })
        val_annotations.extend(anns)
    
    # Save validation annotations
    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories
    }
    
    val_json_path = os.path.join(output_dir, 'instances_val_trashcan.json')
    with open(val_json_path, 'w') as f:
        json.dump(val_coco, f, indent=2)
    print(f"✓ Saved {val_json_path}")
    print(f"  - {len(val_images)} images")
    print(f"  - {len(val_annotations)} annotations")
    
    print(f"\n✅ Dummy dataset created successfully!")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training:   {len(train_images)} images, {len(train_annotations)} objects")
    print(f"   Validation: {len(val_images)} images, {len(val_annotations)} objects")
    print(f"   Classes:    {len(categories)} (trash, animal, rov)")
    
    print(f"\n📂 Directory structure:")
    print(f"   {output_dir}/")
    print(f"   ├── instances_train_trashcan.json")
    print(f"   ├── instances_val_trashcan.json")
    print(f"   └── images/")
    print(f"       ├── train/ ({num_train} images)")
    print(f"       └── val/ ({num_val} images)")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Verify dataset: python -c 'from data.dataset import TrashCanDataset; ds = TrashCanDataset(\"data/trashcan\"); print(f\"Dataset size: {{len(ds)}}\")'")
    print(f"   2. Start training: python scripts/train.py --config configs/train_config.yaml")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create dummy TrashCan dataset for testing')
    parser.add_argument('--output_dir', type=str, default='data/trashcan',
                        help='Output directory for dataset')
    parser.add_argument('--num_train', type=int, default=50,
                        help='Number of training images')
    parser.add_argument('--num_val', type=int, default=20,
                        help='Number of validation images')
    
    args = parser.parse_args()
    
    create_dummy_dataset(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val
    )
