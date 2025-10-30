"""
Training Script for YOLO-UDD v2.0
Implements training protocol as described in Section 5
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import yaml
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_yolo_udd
from data.dataset import create_dataloaders
from utils.metrics import compute_metrics
from utils.loss import YOLOUDDLoss


class Trainer:
    """
    Trainer class for YOLO-UDD v2.0
    
    Implements training protocol from Section 5.2:
    - AdamW optimizer
    - Cosine annealing learning rate schedule
    - Early stopping
    - Transfer learning from COCO pretrained weights
    """
    
    def __init__(self, config):
        self.config = config
        # Allow forcing device via config
        if config.get('device'):
            self.device = torch.device(config['device'])
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.setup_directories()
        
        # Build model
        self.model = build_yolo_udd(
            num_classes=config['num_classes'],
            pretrained=config.get('pretrained_path', None)
        ).to(self.device)
        
        # Setup data loaders
        self.dataloaders = create_dataloaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            img_size=config['img_size']
        )
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['learning_rate'] * 0.01
        )
        
        # Setup loss function
        self.criterion = YOLOUDDLoss(num_classes=config['num_classes'])
        
        # Training state
        self.current_epoch = 0
        self.best_map = 0.0
        self.early_stop_counter = 0
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
    def setup_directories(self):
        """Create necessary directories"""
        self.save_dir = Path(self.config['save_dir'])
        self.log_dir = self.save_dir / 'logs'
        self.checkpoint_dir = self.save_dir / 'checkpoints'
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.dataloaders['train'], desc=f'Epoch {self.current_epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            bboxes = batch['bboxes']
            labels = batch['labels']
            
            # Forward pass
            predictions, turb_score = self.model(images)
            
            # Compute loss
            loss_dict = self.criterion(predictions, bboxes, labels)
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'bbox_loss': loss_dict['bbox_loss'],
                'obj_loss': loss_dict['obj_loss'],
                'cls_loss': loss_dict['cls_loss'],
                'turb': turb_score.mean().item()
            })
            
            # Log to tensorboard
            global_step = self.current_epoch * len(self.dataloaders['train']) + batch_idx
            self.writer.add_scalar('Train/loss', loss.item(), global_step)
            self.writer.add_scalar('Train/bbox_loss', loss_dict['bbox_loss'], global_step)
            self.writer.add_scalar('Train/obj_loss', loss_dict['obj_loss'], global_step)
            self.writer.add_scalar('Train/cls_loss', loss_dict['cls_loss'], global_step)
            self.writer.add_scalar('Train/turbidity', turb_score.mean().item(), global_step)
        
        avg_loss = total_loss / len(self.dataloaders['train'])
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.dataloaders['val'], desc='Validation')
        for batch in pbar:
            images = batch['images'].to(self.device)
            bboxes = batch['bboxes']
            labels = batch['labels']
            
            # Forward pass
            predictions, turb_score = self.model(images)
            
            # Compute loss
            loss_dict = self.criterion(predictions, bboxes, labels)
            loss = loss_dict['total_loss']
            
            total_loss += loss.item()
            
            # Store predictions for metrics
            all_predictions.extend(predictions)
            all_targets.extend(list(zip(bboxes, labels)))
        
        avg_loss = total_loss / len(self.dataloaders['val'])
        
        # Compute metrics
        metrics = compute_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint with mAP: {self.best_map:.4f}")
    
    def train(self):
        """Main training loop"""
        print("=" * 80)
        print("Starting YOLO-UDD v2.0 Training")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Model: {self.model.get_model_info()['Architecture']}")
        print(f"Training samples: {len(self.dataloaders['train'].dataset)}")
        print(f"Validation samples: {len(self.dataloaders['val'].dataset)}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print("=" * 80)
        
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            print(f"\nEpoch {epoch}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
            print(f"mAP@50: {metrics['map50']:.4f} | mAP@50:95: {metrics['map']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Tensorboard logging
            self.writer.add_scalar('Val/loss', val_loss, epoch)
            self.writer.add_scalar('Val/precision', metrics['precision'], epoch)
            self.writer.add_scalar('Val/recall', metrics['recall'], epoch)
            self.writer.add_scalar('Val/mAP50', metrics['map50'], epoch)
            self.writer.add_scalar('Val/mAP', metrics['map'], epoch)
            self.writer.add_scalar('Train/learning_rate', current_lr, epoch)
            
            # Check for improvement
            is_best = metrics['map'] > self.best_map
            if is_best:
                self.best_map = metrics['map']
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.early_stop_counter >= self.config['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best mAP: {self.best_map:.4f}")
                break
        
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best mAP@50:95: {self.best_map:.4f}")
        print("=" * 80)
        
        self.writer.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train YOLO-UDD v2.0')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--pretrained', type=str, help='Path to pretrained weights')
    parser.add_argument('--save-dir', type=str, default='runs/train',
                        help='Directory to save results')
    parser.add_argument('--img-size', type=int, help='Image size for training')
    parser.add_argument('--num-workers', type=int, help='Number of dataloader workers')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Force device (cpu or cuda)')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    
    # Load config
    if os.path.exists(args.config):
        yaml_config = load_config(args.config)
        # Flatten nested YAML structure
        config = {
            'num_classes': yaml_config.get('model', {}).get('num_classes', 3),
            'pretrained_path': yaml_config.get('model', {}).get('pretrained_path', None),
            'data_dir': yaml_config.get('data', {}).get('data_dir', 'data/trashcan'),
            'img_size': yaml_config.get('data', {}).get('img_size', 640),
            'batch_size': yaml_config.get('training', {}).get('batch_size', 16),
            'epochs': yaml_config.get('training', {}).get('epochs', 300),
            'learning_rate': yaml_config.get('training', {}).get('learning_rate', 0.01),
            'weight_decay': yaml_config.get('training', {}).get('weight_decay', 0.0005),
            'num_workers': yaml_config.get('training', {}).get('num_workers', 4),
            'early_stopping_patience': yaml_config.get('training', {}).get('early_stopping_patience', 20),
            'save_dir': 'runs/train'
        }
    else:
        # Default configuration as per Section 5.2
        config = {
            'num_classes': 3,
            'data_dir': 'data/trashcan',
            'img_size': 640,
            'batch_size': 16,
            'epochs': 300,
            'learning_rate': 0.01,
            'weight_decay': 0.0005,
            'num_workers': 4,
            'early_stopping_patience': 20,
            'save_dir': 'runs/train'
        }
    
    # Override with command line arguments
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    if args.pretrained:
        config['pretrained_path'] = args.pretrained
    if args.save_dir:
        config['save_dir'] = args.save_dir
    if args.img_size:
        config['img_size'] = args.img_size
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
    if args.device:
        config['device'] = args.device
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
