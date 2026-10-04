"""
Target Assignment for YOLO-UDD v2.0
Implements simplified anchor-free assignment strategy
"""

import torch
import torch.nn.functional as F


def box_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes
    
    Args:
        boxes1: [N, 4] in format (x, y, w, h)
        boxes2: [M, 4] in format (x, y, w, h)
    
    Returns:
        iou: [N, M] IoU matrix
    """
    # Convert to (x1, y1, x2, y2)
    boxes1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    boxes1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    boxes1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    boxes1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
    
    boxes2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    boxes2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    boxes2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    boxes2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
    
    # Calculate intersection
    inter_x1 = torch.max(boxes1_x1.unsqueeze(1), boxes2_x1.unsqueeze(0))
    inter_y1 = torch.max(boxes1_y1.unsqueeze(1), boxes2_y1.unsqueeze(0))
    inter_x2 = torch.min(boxes1_x2.unsqueeze(1), boxes2_x2.unsqueeze(0))
    inter_y2 = torch.min(boxes1_y2.unsqueeze(1), boxes2_y2.unsqueeze(0))
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union
    boxes1_area = (boxes1_x2 - boxes1_x1) * (boxes1_y2 - boxes1_y1)
    boxes2_area = (boxes2_x2 - boxes2_x1) * (boxes2_y2 - boxes2_y1)
    union_area = boxes1_area.unsqueeze(1) + boxes2_area.unsqueeze(0) - inter_area
    
    iou = inter_area / (union_area + 1e-7)
    
    return iou


def assign_targets_simple(predictions, target_boxes, target_labels, num_classes=3, iou_threshold=0.5):
    """
    Simplified target assignment for YOLO-UDD
    Assigns ground truth boxes to predictions based on IoU and spatial location
    
    Args:
        predictions: List of (bbox_pred, obj_pred, cls_pred) for each scale
        target_boxes: List of [N, 4] target boxes per image in batch
        target_labels: List of [N] target labels per image in batch
        num_classes: Number of classes
        iou_threshold: IoU threshold for positive assignment
    
    Returns:
        assigned_targets: Dictionary containing assigned targets for each scale
    """
    device = predictions[0][0].device
    batch_size = predictions[0][0].size(0)
    
    assigned_targets = {
        'bbox_targets': [],
        'obj_targets': [],
        'cls_targets': [],
        'pos_masks': []
    }
    
    for scale_idx, (bbox_pred, obj_pred, cls_pred) in enumerate(predictions):
        # bbox_pred: [B, 4, H, W]
        # obj_pred: [B, 1, H, W]
        # cls_pred: [B, num_classes, H, W]
        
        B, _, H, W = bbox_pred.shape
        
        # Initialize targets
        bbox_target = torch.zeros_like(bbox_pred)
        obj_target = torch.zeros_like(obj_pred)
        cls_target = torch.zeros(B, num_classes, H, W, device=device)
        pos_mask = torch.zeros(B, 1, H, W, dtype=torch.bool, device=device)
        
        # Process each image in batch
        for b in range(batch_size):
            if len(target_boxes[b]) == 0:
                continue
                
            gt_boxes = target_boxes[b].to(device)  # [N_gt, 4]
            gt_labels = target_labels[b].to(device)  # [N_gt]
            
            # Create grid coordinates normalized to [0, 1]
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, 1, H, device=device),
                torch.linspace(0, 1, W, device=device),
                indexing='ij'
            )
            
            # For each ground truth box, find best matching grid cell
            for gt_idx in range(len(gt_boxes)):
                gt_box = gt_boxes[gt_idx]  # [4] - (x, y, w, h) normalized
                gt_label = gt_labels[gt_idx].long()
                
                # Find grid cell that contains box center
                gt_x, gt_y = gt_box[0], gt_box[1]
                grid_i = int(gt_y * H)
                grid_j = int(gt_x * W)
                
                # Clamp to valid range
                grid_i = min(max(0, grid_i), H - 1)
                grid_j = min(max(0, grid_j), W - 1)
                
                # Assign target to this grid cell
                bbox_target[b, :, grid_i, grid_j] = gt_box
                obj_target[b, 0, grid_i, grid_j] = 1.0
                cls_target[b, gt_label, grid_i, grid_j] = 1.0
                pos_mask[b, 0, grid_i, grid_j] = True
                
                # Also assign to neighboring cells for better coverage
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = grid_i + di, grid_j + dj
                        if 0 <= ni < H and 0 <= nj < W and not pos_mask[b, 0, ni, nj]:
                            # Calculate distance to see if this neighbor should be positive
                            dist = ((ni / H - gt_y) ** 2 + (nj / W - gt_x) ** 2) ** 0.5
                            if dist < 0.15:  # Within 15% distance
                                bbox_target[b, :, ni, nj] = gt_box
                                obj_target[b, 0, ni, nj] = max(0.5, 1.0 - dist * 3)
                                cls_target[b, gt_label, ni, nj] = 1.0
                                pos_mask[b, 0, ni, nj] = True
        
        assigned_targets['bbox_targets'].append(bbox_target)
        assigned_targets['obj_targets'].append(obj_target)
        assigned_targets['cls_targets'].append(cls_target)
        assigned_targets['pos_masks'].append(pos_mask)
    
    return assigned_targets


def build_targets(predictions, target_boxes, target_labels, img_size=640, num_classes=3):
    """
    Build targets for all scales with proper normalization
    
    Args:
        predictions: Model predictions at different scales
        target_boxes: List of target boxes per image (absolute coordinates)
        target_labels: List of target labels per image
        img_size: Input image size
        num_classes: Number of classes
    
    Returns:
        Dictionary of assigned targets
    """
    # Normalize target boxes to [0, 1]
    normalized_boxes = []
    for boxes in target_boxes:
        if len(boxes) > 0:
            norm_boxes = boxes.clone().float()
            norm_boxes[:, [0, 2]] /= img_size  # Normalize x, w
            norm_boxes[:, [1, 3]] /= img_size  # Normalize y, h
            normalized_boxes.append(norm_boxes)
        else:
            normalized_boxes.append(torch.empty(0, 4))
    
    return assign_targets_simple(predictions, normalized_boxes, target_labels, num_classes)


if __name__ == '__main__':
    print("Testing target assignment...")
    
    # Create dummy predictions
    predictions = [
        (torch.randn(2, 4, 80, 80), torch.randn(2, 1, 80, 80), torch.randn(2, 3, 80, 80)),
        (torch.randn(2, 4, 40, 40), torch.randn(2, 1, 40, 40), torch.randn(2, 3, 40, 40)),
        (torch.randn(2, 4, 20, 20), torch.randn(2, 1, 20, 20), torch.randn(2, 3, 20, 20)),
    ]
    
    # Create dummy targets
    target_boxes = [
        torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.7, 0.15, 0.15]]),
        torch.tensor([[0.6, 0.4, 0.25, 0.25]])
    ]
    target_labels = [
        torch.tensor([0, 2]),
        torch.tensor([1])
    ]
    
    targets = assign_targets_simple(predictions, target_boxes, target_labels)
    
    print(f"✓ Target assignment successful!")
    print(f"  Number of scales: {len(targets['bbox_targets'])}")
    print(f"  Positive samples per scale:")
    for i, mask in enumerate(targets['pos_masks']):
        print(f"    Scale {i+1}: {mask.sum().item()} positive locations")
