"""
Non-Maximum Suppression (NMS) for YOLO-UDD v2.0
Implements IoU-based NMS for post-processing predictions
"""

import torch


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: [N, 4] in format (x, y, w, h) - normalized coordinates
        boxes2: [M, 4] in format (x, y, w, h) - normalized coordinates
    
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
    
    # Compute intersection
    inter_x1 = torch.max(boxes1_x1.unsqueeze(1), boxes2_x1.unsqueeze(0))
    inter_y1 = torch.max(boxes1_y1.unsqueeze(1), boxes2_y1.unsqueeze(0))
    inter_x2 = torch.min(boxes1_x2.unsqueeze(1), boxes2_x2.unsqueeze(0))
    inter_y2 = torch.min(boxes1_y2.unsqueeze(1), boxes2_y2.unsqueeze(0))
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Compute union
    area1 = (boxes1_x2 - boxes1_x1) * (boxes1_y2 - boxes1_y1)
    area2 = (boxes2_x2 - boxes2_x1) * (boxes2_y2 - boxes2_y1)
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-7)
    
    return iou


def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    
    Args:
        boxes: [N, 4] bounding boxes in (x, y, w, h) format
        scores: [N] confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        keep_indices: Indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    
    # Sort by scores (descending)
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep = []
    while len(sorted_indices) > 0:
        # Keep highest scoring box
        current = sorted_indices[0]
        keep.append(current.item())
        
        if len(sorted_indices) == 1:
            break
        
        # Compute IoU with remaining boxes
        current_box = boxes[current].unsqueeze(0)
        remaining_boxes = boxes[sorted_indices[1:]]
        
        ious = box_iou(current_box, remaining_boxes).squeeze(0)
        
        # Keep boxes with IoU < threshold
        mask = ious < iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long)


def batched_nms(predictions, conf_threshold=0.25, iou_threshold=0.5, max_det=300):
    """
    Apply NMS to batched predictions
    
    Args:
        predictions: List of (bbox, obj, cls) tensors for each scale
        conf_threshold: Minimum confidence threshold
        iou_threshold: IoU threshold for NMS
        max_det: Maximum number of detections per image
    
    Returns:
        detections: List of detections per image, each containing:
            - boxes: [N, 4] filtered bounding boxes
            - scores: [N] confidence scores
            - classes: [N] class predictions
    """
    # Aggregate predictions from all scales
    all_boxes = []
    all_obj_scores = []
    all_cls_scores = []
    all_cls_indices = []
    
    for bbox_pred, obj_pred, cls_pred in predictions:
        B, _, H, W = bbox_pred.shape
        
        # Reshape predictions
        bbox = bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)
        obj = torch.sigmoid(torch.clamp(obj_pred, min=-50, max=50)).permute(0, 2, 3, 1).reshape(B, -1)
        cls = torch.sigmoid(cls_pred).permute(0, 2, 3, 1).reshape(B, -1, cls_pred.shape[1])
        
        all_boxes.append(bbox)
        all_obj_scores.append(obj)
        all_cls_scores.append(cls)
    
    # Concatenate all scales
    boxes = torch.cat(all_boxes, dim=1)  # [B, N_total, 4]
    obj_scores = torch.cat(all_obj_scores, dim=1)  # [B, N_total]
    cls_scores = torch.cat(all_cls_scores, dim=1)  # [B, N_total, C]
    
    # Get class predictions and scores
    cls_max_scores, cls_indices = torch.max(cls_scores, dim=2)  # [B, N_total]
    
    # Combined confidence (objectness * class score)
    conf_scores = obj_scores * cls_max_scores
    
    # Process each image in batch
    batch_detections = []
    for b in range(boxes.shape[0]):
        # Filter by confidence
        mask = conf_scores[b] >= conf_threshold
        if mask.sum() == 0:
            batch_detections.append({
                'boxes': torch.empty(0, 4, device=boxes.device),
                'scores': torch.empty(0, device=boxes.device),
                'classes': torch.empty(0, dtype=torch.long, device=boxes.device)
            })
            continue
        
        img_boxes = boxes[b][mask]
        img_scores = conf_scores[b][mask]
        img_classes = cls_indices[b][mask]
        
        # Apply NMS per class
        keep_indices = []
        for c in range(cls_scores.shape[2]):
            class_mask = img_classes == c
            if class_mask.sum() == 0:
                continue
            
            class_boxes = img_boxes[class_mask]
            class_scores = img_scores[class_mask]
            class_indices = torch.where(class_mask)[0]
            
            # Apply NMS
            nms_indices = nms(class_boxes, class_scores, iou_threshold)
            keep_indices.append(class_indices[nms_indices])
        
        if len(keep_indices) > 0:
            keep_indices = torch.cat(keep_indices)
            
            # Sort by score and limit to max_det
            sorted_indices = torch.argsort(img_scores[keep_indices], descending=True)[:max_det]
            final_indices = keep_indices[sorted_indices]
            
            batch_detections.append({
                'boxes': img_boxes[final_indices],
                'scores': img_scores[final_indices],
                'classes': img_classes[final_indices]
            })
        else:
            batch_detections.append({
                'boxes': torch.empty(0, 4, device=boxes.device),
                'scores': torch.empty(0, device=boxes.device),
                'classes': torch.empty(0, dtype=torch.long, device=boxes.device)
            })
    
    return batch_detections


if __name__ == '__main__':
    print("Testing NMS implementation...")
    
    # Create dummy predictions
    predictions = [
        (torch.randn(2, 4, 80, 80), torch.randn(2, 1, 80, 80), torch.randn(2, 3, 80, 80)),
        (torch.randn(2, 4, 40, 40), torch.randn(2, 1, 40, 40), torch.randn(2, 3, 40, 40)),
        (torch.randn(2, 4, 20, 20), torch.randn(2, 1, 20, 20), torch.randn(2, 3, 20, 20)),
    ]
    
    # Apply NMS
    detections = batched_nms(predictions, conf_threshold=0.5, iou_threshold=0.5)
    
    print(f"✓ NMS successful!")
    print(f"  Batch size: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"  Image {i}: {len(det['boxes'])} detections")
