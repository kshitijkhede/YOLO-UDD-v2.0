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
        """Load dataset annotations"""
        ann_file = os.path.join(self.data_dir, 'annotations', f'{self.split}.json')
        
        # Placeholder - actual implementation would load from JSON
        # For now, return empty list
        annotations = []
        
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                annotations = json.load(f)
        
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
        # Placeholder implementation
        # Actual implementation would load image and annotations
        
        # For demo purposes, create dummy data
        image = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        bboxes = np.array([[0.5, 0.5, 0.2, 0.2]])  # Dummy bbox in YOLO format
        class_labels = np.array([0])
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            image = transformed['image']
            bboxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            class_labels = torch.tensor(transformed['class_labels'], dtype=torch.long)
        
        return {
            'image': image,
            'bboxes': bboxes,
            'labels': class_labels,
            'image_id': idx
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
