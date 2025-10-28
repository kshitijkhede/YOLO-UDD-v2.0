"""
TrashCan 1.0 Dataset Loader
Implements data loading and underwater-specific augmentations as described in Section 4
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json


class TrashCanDataset(Dataset):
    """
    TrashCan 1.0 Dataset with 3-class configuration
    Classes: Trash, Animal, ROV
    
    Args:
        data_dir (str): Root directory of TrashCan dataset
        split (str): 'train', 'val', or 'test'
        img_size (int): Input image size (default: 640)
        augment (bool): Whether to apply augmentations
    """
    
    def __init__(self, data_dir, split='train', img_size=640, augment=True):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        
        # DEBUG: Print the data_dir being used
        print(f"[DEBUG] TrashCanDataset({split}): data_dir = {os.path.abspath(data_dir)}")
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Define class mapping for 3-class configuration
        self.class_map = {
            'trash': 0,
            'animal': 1,
            'rov': 2
        }
        self.num_classes = len(self.class_map)
        
        # Setup augmentation pipeline
        self.transform = self._get_transforms()
        
    def _load_annotations(self):
        """Load dataset annotations from TrashCAN COCO format"""
        # Try TrashCAN format first: instances_train_trashcan.json
        ann_file = os.path.join(self.data_dir, f'instances_{self.split}_trashcan.json')
        print(f"[DEBUG] Checking TrashCAN format: {ann_file}")
        
        # Fallback to standard format
        if not os.path.exists(ann_file):
            ann_file = os.path.join(self.data_dir, 'annotations', f'{self.split}.json')
            print(f"[DEBUG] Checking standard format: {ann_file}")
        
        annotations = []
        
        if os.path.exists(ann_file):
            print(f"✓ Loading annotations from: {ann_file}")
            with open(ann_file, 'r') as f:
                data = json.load(f)
                # COCO format has 'images' and 'annotations' keys
                if isinstance(data, dict) and 'images' in data:
                    annotations = data
                else:
                    annotations = data
        else:
            print(f"✗ Warning: Annotation file not found: {ann_file}")
        
        return annotations
    
    def _get_transforms(self):
        """
        Get augmentation pipeline with underwater-specific transformations
        As described in Section 4.3
        """
        if self.augment:
            return A.Compose([
                # Geometric augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.RandomResizedCrop(height=self.img_size, width=self.img_size, size=(self.img_size, self.img_size), scale=(0.8, 1.0), p=0.5),
                
                # Underwater-specific photometric augmentations
                # Simulate color casting at depth
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1,
                    p=0.7
                ),
                
                # Simulate water turbidity with blur and haze
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.MedianBlur(blur_limit=7, p=1.0),
                ], p=0.5),
                
                # Simulate varying lighting conditions
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
                
                # Simulate sensor noise
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.ISONoise(p=1.0),
                ], p=0.3),
                
                # Additional underwater effects
                A.RGBShift(r_shift_limit=20, g_shift_limit=10, b_shift_limit=20, p=0.4),
                A.ChannelShuffle(p=0.1),
                
                # Normalization and tensor conversion
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            # Validation/test transforms (no augmentation)
            return A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __len__(self):
        if isinstance(self.annotations, dict) and 'images' in self.annotations:
            return len(self.annotations['images'])
        return len(self.annotations) if self.annotations else 0
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            dict: {
                'image': torch.Tensor [3, H, W],
                'bboxes': torch.Tensor [N, 4],
                'labels': torch.Tensor [N],
                'image_id': int
            }
        """
        # Get image info from COCO format
        if isinstance(self.annotations, dict) and 'images' in self.annotations:
            img_info = self.annotations['images'][idx]
            image_id = img_info['id']
            img_filename = img_info['file_name']
            
            # Load image
            # Try: data/trashcan/images/train/filename.jpg
            img_path = os.path.join(self.data_dir, 'images', self.split, img_filename)
            image = cv2.imread(img_path)
            if image is None:
                # Fallback: try data/trashcan/train/filename.jpg
                img_path = os.path.join(self.data_dir, self.split, img_filename)
                image = cv2.imread(img_path)
            if image is None:
                # Fallback: try data/trashcan/filename.jpg
                img_path = os.path.join(self.data_dir, img_filename)
                image = cv2.imread(img_path)
            
            if image is None:
                print(f"Warning: Could not load image {img_path}, using blank image")
                image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get annotations for this image
            img_anns = [ann for ann in self.annotations['annotations'] if ann['image_id'] == image_id]
            
            # Extract bboxes and labels
            bboxes = []
            class_labels = []
            
            for ann in img_anns:
                # COCO format: [x, y, width, height] in absolute coordinates
                x, y, w, h = ann['bbox']
                img_h, img_w = image.shape[:2]
                
                # Convert to YOLO format: [x_center, y_center, width, height] normalized
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                # Clip to valid range [0, 1] to handle rounding errors
                x_center = np.clip(x_center, 0.0, 1.0)
                y_center = np.clip(y_center, 0.0, 1.0)
                norm_w = np.clip(norm_w, 0.0, 1.0)
                norm_h = np.clip(norm_h, 0.0, 1.0)
                
                bboxes.append([x_center, y_center, norm_w, norm_h])
                
                # Map category_id to class label
                category_id = ann['category_id']
                # Assuming categories are 1-indexed in COCO, convert to 0-indexed
                class_labels.append(category_id - 1 if category_id > 0 else 0)
            
            # Convert to numpy arrays
            bboxes = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32)
            class_labels = np.array(class_labels, dtype=np.int64) if class_labels else np.zeros((0,), dtype=np.int64)
        else:
            # Fallback: use dummy data if annotations not in expected format
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            bboxes = np.zeros((0, 4), dtype=np.float32)
            class_labels = np.zeros((0,), dtype=np.int64)
            image_id = idx
        
        # Apply transforms
        if self.transform and len(bboxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            image = transformed['image']
            # Clip bboxes to [0, 1] range to handle rounding errors from transforms
            transformed_bboxes = np.array(transformed['bboxes'], dtype=np.float32) if len(transformed['bboxes']) > 0 else np.zeros((0, 4), dtype=np.float32)
            if len(transformed_bboxes) > 0:
                transformed_bboxes = np.clip(transformed_bboxes, 0.0, 1.0)
            bboxes = torch.tensor(transformed_bboxes, dtype=torch.float32)
            class_labels = torch.tensor(transformed['class_labels'], dtype=torch.long) if len(transformed['class_labels']) > 0 else torch.zeros((0,), dtype=torch.long)
        elif self.transform:
            # No bboxes, just transform image
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros((0,), dtype=torch.long)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            class_labels = torch.tensor(class_labels, dtype=torch.long)
        
        return {
            'image': image,
            'bboxes': bboxes,
            'labels': class_labels,
            'image_id': image_id
        }


def collate_fn(batch):
    """
    Custom collate function for batching variable-length bounding boxes
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        dict: Batched data
    """
    images = []
    bboxes = []
    labels = []
    image_ids = []
    
    for sample in batch:
        images.append(sample['image'])
        bboxes.append(sample['bboxes'])
        labels.append(sample['labels'])
        image_ids.append(sample['image_id'])
    
    images = torch.stack(images, 0)
    
    return {
        'images': images,
        'bboxes': bboxes,  # List of tensors (variable length)
        'labels': labels,  # List of tensors (variable length)
        'image_ids': image_ids
    }


def create_dataloaders(data_dir, batch_size=16, num_workers=4, img_size=640):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir (str): Root directory of dataset
        batch_size (int): Batch size
        num_workers (int): Number of dataloader workers
        img_size (int): Input image size
        
    Returns:
        dict: Dictionary containing train, val, and test dataloaders
    """
    # Create datasets
    train_dataset = TrashCanDataset(data_dir, split='train', img_size=img_size, augment=True)
    val_dataset = TrashCanDataset(data_dir, split='val', img_size=img_size, augment=False)
    test_dataset = TrashCanDataset(data_dir, split='test', img_size=img_size, augment=False)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == '__main__':
    # Test dataset and augmentations
    dataset = TrashCanDataset(
        data_dir='./data/trashcan',
        split='train',
        img_size=640,
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Class mapping: {dataset.class_map}")
    
    # Test loading a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"BBoxes shape: {sample['bboxes'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
