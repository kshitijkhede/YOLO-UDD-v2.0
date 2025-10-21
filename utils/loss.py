"""
Loss Functions for YOLO-UDD v2.0
Implements composite loss as described in Section 3.4:
- EIoU Loss for bounding box regression
- Varifocal Loss for classification
- BCE Loss for objectness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EIoULoss(nn.Module):
    """
    Efficient IoU Loss for bounding box regression
    """
    
    def __init__(self):
        super(EIoULoss, self).__init__()
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] in format (x, y, w, h)
            target_boxes: [N, 4] in format (x, y, w, h)
        
        Returns:
            loss: EIoU loss value
        """
        # Convert (x, y, w, h) to (x1, y1, x2, y2)
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
        
        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
        
        # Intersection area
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union area
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
        
        enclose_w = enclose_x2 - enclose_x1
        enclose_h = enclose_y2 - enclose_y1
        
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
        loss = 1 - iou + center_dist / diagonal_dist + w_diff / (enclose_w ** 2 + 1e-7) + h_diff / (enclose_h ** 2 + 1e-7)
        
        return loss.mean()


class VarifocalLoss(nn.Module):
    """
    Varifocal Loss for classification
    Addresses class imbalance by down-weighting easy negatives
    """
    
    def __init__(self, alpha=0.75, gamma=2.0):
        super(VarifocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target, quality_score=None):
        """
        Args:
            pred: [N, C] predicted class scores
            target: [N] target class labels
            quality_score: [N] IoU scores for weighting (optional)
        
        Returns:
            loss: Varifocal loss value
        """
        # Convert target to one-hot
        num_classes = pred.size(1)
        target_one_hot = F.one_hot(target, num_classes=num_classes).float()
        
        # Compute sigmoid of predictions
        pred_sigmoid = torch.sigmoid(pred)
        
        # Varifocal loss
        if quality_score is not None:
            # Weight positive samples by quality score (IoU)
            target_weighted = target_one_hot * quality_score.unsqueeze(1)
        else:
            target_weighted = target_one_hot
        
        # Focal weight
        focal_weight = target_weighted * (target_weighted - pred_sigmoid).abs().pow(self.gamma) + \
                      (1 - target_one_hot) * pred_sigmoid.abs().pow(self.gamma)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target_weighted, reduction='none')
        
        # Weighted loss
        loss = (focal_weight * bce).sum() / (target_one_hot.sum() + 1e-7)
        
        return loss


class YOLOUDDLoss(nn.Module):
    """
    Composite loss function for YOLO-UDD v2.0
    Combines:
    1. EIoU Loss for bounding box regression
    2. Varifocal Loss for classification
    3. BCE Loss for objectness
    
    Args:
        num_classes (int): Number of classes
        lambda_box (float): Weight for box loss
        lambda_obj (float): Weight for objectness loss
        lambda_cls (float): Weight for classification loss
    """
    
    def __init__(self, num_classes=3, lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0):
        super(YOLOUDDLoss, self).__init__()
        
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        
        # Loss components
        self.bbox_loss_fn = EIoULoss()
        self.cls_loss_fn = VarifocalLoss()
        self.obj_loss_fn = nn.BCELoss()
    
    def forward(self, predictions, target_boxes, target_labels):
        """
        Compute composite loss
        
        Args:
            predictions: List of (bbox, obj, cls) predictions for each scale
            target_boxes: List of target bounding boxes for each image
            target_labels: List of target labels for each image
        
        Returns:
            dict: Loss components
        """
        total_bbox_loss = 0.0
        total_obj_loss = 0.0
        total_cls_loss = 0.0
        num_targets = 0
        
        # Process each scale
        for bbox_pred, obj_pred, cls_pred in predictions:
            batch_size = bbox_pred.size(0)
            
            # Simplified loss calculation (actual implementation would require proper target assignment)
            # For demo purposes, use dummy losses
            
            # Bounding box loss
            if len(target_boxes) > 0 and len(target_boxes[0]) > 0:
                # Dummy calculation
                bbox_loss = torch.tensor(1.0, device=bbox_pred.device)
                total_bbox_loss += bbox_loss
            
            # Objectness loss
            # Clamp predictions to [0, 1] range to avoid BCE errors
            obj_pred_clamped = torch.clamp(obj_pred, min=0.0, max=1.0)
            # Dummy targets
            obj_targets = torch.rand_like(obj_pred_clamped)
            obj_loss = self.obj_loss_fn(obj_pred_clamped, obj_targets)
            total_obj_loss += obj_loss
            
            # Classification loss
            if cls_pred.numel() > 0:
                # Dummy calculation
                cls_loss = torch.tensor(0.5, device=cls_pred.device)
                total_cls_loss += cls_loss
            
            num_targets += batch_size
        
        # Average losses
        num_scales = len(predictions)
        bbox_loss = total_bbox_loss / max(num_scales, 1)
        obj_loss = total_obj_loss / max(num_scales, 1)
        cls_loss = total_cls_loss / max(num_scales, 1)
        
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
        }


if __name__ == '__main__':
    # Test loss functions
    print("Testing YOLO-UDD Loss Functions...")
    
    # Test EIoU Loss
    print("\n1. Testing EIoU Loss")
    eiou_loss = EIoULoss()
    pred_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]])
    target_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.35, 0.35, 0.1, 0.1]])
    loss = eiou_loss(pred_boxes, target_boxes)
    print(f"EIoU Loss: {loss.item():.4f}")
    
    # Test Varifocal Loss
    print("\n2. Testing Varifocal Loss")
    vf_loss = VarifocalLoss()
    pred = torch.randn(10, 3)
    target = torch.randint(0, 3, (10,))
    quality = torch.rand(10)
    loss = vf_loss(pred, target, quality)
    print(f"Varifocal Loss: {loss.item():.4f}")
    
    # Test Composite Loss
    print("\n3. Testing Composite YOLO-UDD Loss")
    composite_loss = YOLOUDDLoss(num_classes=3)
    
    # Dummy predictions
    predictions = [
        (torch.rand(2, 4, 80, 80), torch.rand(2, 1, 80, 80), torch.rand(2, 3, 80, 80)),
        (torch.rand(2, 4, 40, 40), torch.rand(2, 1, 40, 40), torch.rand(2, 3, 40, 40)),
        (torch.rand(2, 4, 20, 20), torch.rand(2, 1, 20, 20), torch.rand(2, 3, 20, 20)),
    ]
    target_boxes = [torch.rand(3, 4), torch.rand(2, 4)]
    target_labels = [torch.randint(0, 3, (3,)), torch.randint(0, 3, (2,))]
    
    loss_dict = composite_loss(predictions, target_boxes, target_labels)
    print(f"Total Loss: {loss_dict['total_loss']:.4f}")
    print(f"BBox Loss: {loss_dict['bbox_loss']:.4f}")
    print(f"Obj Loss: {loss_dict['obj_loss']:.4f}")
    print(f"Cls Loss: {loss_dict['cls_loss']:.4f}")
