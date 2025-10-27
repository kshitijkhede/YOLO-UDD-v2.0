"""
Loss Functions for YOLO-UDD v2.0 with Proper Target Assignment
Implements composite loss as described in Section 3.4:
- EIoU Loss for bounding box regression
- Varifocal Loss for classification
- BCE Loss for objectness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .target_assignment import build_targets


class EIoULoss(nn.Module):
    """Efficient IoU Loss for bounding box regression"""
    
    def __init__(self):
        super(EIoULoss, self).__init__()
    
    def forward(self, pred_boxes, target_boxes, reduction='mean'):
        """
        Args:
            pred_boxes: [N, 4] in format (x, y, w, h)
            target_boxes: [N, 4] in format (x, y, w, h)
        Returns:
            loss: EIoU loss value
        """
        if pred_boxes.numel() == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
            
        # Convert to (x1, y1, x2, y2)
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
        
        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
        
        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-7)
        
        # Enclosing box
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        enclose_w = enclose_x2 - enclose_x1 + 1e-7
        enclose_h = enclose_y2 - enclose_y1 + 1e-7
        
        # Center distance
        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        target_cx = (target_x1 + target_x2) / 2
        target_cy = (target_y1 + target_y2) / 2
        
        center_dist = ((pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2)
        diagonal_dist = enclose_w ** 2 + enclose_h ** 2 + 1e-7
        
        # Width and height difference
        w_diff = (pred_boxes[:, 2] - target_boxes[:, 2]) ** 2
        h_diff = (pred_boxes[:, 3] - target_boxes[:, 3]) ** 2
        
        # EIoU Loss
        loss = 1 - iou + center_dist / diagonal_dist + w_diff / (enclose_w ** 2) + h_diff / (enclose_h ** 2)
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss


class YOLOUDDLoss(nn.Module):
    """
    Composite loss function for YOLO-UDD v2.0 with proper target assignment
    """
    
    def __init__(self, num_classes=3, lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0):
        super(YOLOUDDLoss, self).__init__()
        
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        
        # Loss components
        self.bbox_loss_fn = EIoULoss()
        self.obj_loss_fn = nn.BCELoss(reduction='none')
        self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, predictions, target_boxes, target_labels):
        """
        Compute composite loss with proper target assignment
        
        Args:
            predictions: List of (bbox, obj, cls) predictions for each scale
            target_boxes: List of target bounding boxes for each image
            target_labels: List of target labels for each image
        
        Returns:
            dict: Loss components
        """
        device = predictions[0][0].device
        
        # Build targets using proper assignment
        targets = build_targets(predictions, target_boxes, target_labels, 
                              img_size=640, num_classes=self.num_classes)
        
        total_bbox_loss = 0.0
        total_obj_loss = 0.0
        total_cls_loss = 0.0
        total_pos_samples = 0
        
        # Process each scale
        for scale_idx, (bbox_pred, obj_pred, cls_pred) in enumerate(predictions):
            bbox_target = targets['bbox_targets'][scale_idx]
            obj_target = targets['obj_targets'][scale_idx]
            cls_target = targets['cls_targets'][scale_idx]
            pos_mask = targets['pos_masks'][scale_idx]
            
            # Flatten spatial dimensions
            B, _, H, W = bbox_pred.shape
            bbox_pred_flat = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            bbox_target_flat = bbox_target.permute(0, 2, 3, 1).reshape(-1, 4)
            pos_mask_flat = pos_mask.view(-1)
            
            # 1. Bounding Box Loss (only on positive samples)
            if pos_mask_flat.sum() > 0:
                bbox_loss = self.bbox_loss_fn(
                    bbox_pred_flat[pos_mask_flat],
                    bbox_target_flat[pos_mask_flat]
                )
                total_bbox_loss += bbox_loss
                total_pos_samples += pos_mask_flat.sum().item()
            
            # 2. Objectness Loss (all samples)
            # Clamp raw predictions before sigmoid to prevent NaN/Inf
            obj_pred_clamped = torch.clamp(obj_pred, min=-50, max=50)
            obj_pred_sigmoid = torch.sigmoid(obj_pred_clamped)
            
            # Ensure strict [0, 1] range for both inputs
            obj_pred_sigmoid = torch.clamp(obj_pred_sigmoid, min=1e-7, max=1.0-1e-7)
            obj_target_safe = torch.clamp(obj_target, min=0.0, max=1.0)
            
            # Handle NaN/Inf values
            if torch.isnan(obj_pred_sigmoid).any() or torch.isinf(obj_pred_sigmoid).any():
                obj_pred_sigmoid = torch.nan_to_num(obj_pred_sigmoid, nan=0.5, posinf=1.0-1e-7, neginf=1e-7)
            if torch.isnan(obj_target_safe).any() or torch.isinf(obj_target_safe).any():
                obj_target_safe = torch.nan_to_num(obj_target_safe, nan=0.0)
            
            obj_loss = self.obj_loss_fn(obj_pred_sigmoid, obj_target_safe)
            
            # Weight positive and negative samples
            pos_weight = 2.0
            neg_weight = 0.5
            obj_loss = torch.where(obj_target > 0.5, 
                                  obj_loss * pos_weight,
                                  obj_loss * neg_weight)
            total_obj_loss += obj_loss.mean()
            
            # 3. Classification Loss (only on positive samples)
            if pos_mask.sum() > 0:
                cls_pred_pos = cls_pred * pos_mask.float()
                cls_target_pos = cls_target * pos_mask.float()
                
                cls_loss = self.cls_loss_fn(cls_pred_pos, cls_target_pos)
                cls_loss = cls_loss * pos_mask.float()
                
                if pos_mask.sum() > 0:
                    cls_loss = cls_loss.sum() / pos_mask.sum()
                    total_cls_loss += cls_loss
        
        # Average losses across scales
        num_scales = len(predictions)
        bbox_loss = total_bbox_loss / max(num_scales, 1)
        obj_loss = total_obj_loss / max(num_scales, 1)
        cls_loss = total_cls_loss / max(num_scales, 1) if total_cls_loss > 0 else torch.tensor(0.0, device=device)
        
        # Weighted composite loss
        total_loss = (
            self.lambda_box * bbox_loss +
            self.lambda_obj * obj_loss +
            self.lambda_cls * cls_loss
        )
        
        return {
            'total_loss': total_loss,
            'bbox_loss': bbox_loss.item() if torch.is_tensor(bbox_loss) else bbox_loss,
            'obj_loss': obj_loss.item() if torch.is_tensor(obj_loss) else obj_loss,
            'cls_loss': cls_loss.item() if torch.is_tensor(cls_loss) else cls_loss,
            'pos_samples': total_pos_samples
        }


if __name__ == '__main__':
    print("Testing YOLO-UDD Loss with proper target assignment...")
    
    # Create dummy predictions
    predictions = [
        (torch.randn(2, 4, 80, 80), torch.randn(2, 1, 80, 80), torch.randn(2, 3, 80, 80)),
        (torch.randn(2, 4, 40, 40), torch.randn(2, 1, 40, 40), torch.randn(2, 3, 40, 40)),
        (torch.randn(2, 4, 20, 20), torch.randn(2, 1, 20, 20), torch.randn(2, 3, 20, 20)),
    ]
    
    # Create dummy targets (normalized coordinates)
    target_boxes = [
        torch.tensor([[320.0, 320.0, 128.0, 192.0], [192.0, 448.0, 96.0, 96.0]]),
        torch.tensor([[384.0, 256.0, 160.0, 160.0]])
    ]
    target_labels = [
        torch.tensor([0, 2]),
        torch.tensor([1])
    ]
    
    # Test loss
    criterion = YOLOUDDLoss(num_classes=3)
    loss_dict = criterion(predictions, target_boxes, target_labels)
    
    print(f"✓ Loss calculation successful!")
    print(f"  Total Loss: {loss_dict['total_loss']:.4f}")
    print(f"  BBox Loss: {loss_dict['bbox_loss']:.4f}")
    print(f"  Obj Loss: {loss_dict['obj_loss']:.4f}")
    print(f"  Cls Loss: {loss_dict['cls_loss']:.4f}")
    print(f"  Positive Samples: {loss_dict['pos_samples']}")
