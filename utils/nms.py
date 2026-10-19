"""
Non-Maximum Suppression (NMS) for YOLO-UDD v2.0
Implements IoU-based NMS for post-processing predictions
"""

import torch

# torchvision provides a fused (C++/CUDA) NMS kernel. We use it when available
# because the pure-Python fallback below is O(N^2) with per-iteration GPU->CPU
# syncs, which is catastrophically slow when an (early / untrained) model emits
# thousands of candidate boxes per image.
try:
    from torchvision.ops import batched_nms as _tv_batched_nms
    _HAS_TV = True
except Exception:  # torchvision missing / broken
    _HAS_TV = False


def _xywh_to_xyxy(b):
    """[N,4] (x_center, y_center, w, h) -> (x1, y1, x2, y2)."""
    x, y, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=1)


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


def batched_nms(predictions, conf_threshold=0.25, iou_threshold=0.5, max_det=300,
                max_nms=3000):
    """
    Apply NMS to batched predictions.

    Args:
        predictions: List of (bbox, obj, cls) tensors for each scale
        conf_threshold: Minimum confidence threshold
        iou_threshold: IoU threshold for NMS
        max_det: Maximum number of detections kept per image
        max_nms: Max candidate boxes fed into NMS per image (capped by score).
            Critical early in training, when an untrained model passes almost
            every anchor through the confidence filter.

    Returns:
        detections: List of detections per image, each a dict with:
            - boxes: [N, 4] in (x_center, y_center, w, h) format
            - scores: [N] confidence scores
            - classes: [N] class predictions
    """
    # Aggregate predictions from all scales
    all_boxes = []
    all_obj_scores = []
    all_cls_scores = []

    for bbox_pred, obj_pred, cls_pred in predictions:
        B, _, H, W = bbox_pred.shape
        C = cls_pred.shape[1]
        bbox = bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)
        # obj_pred is ALREADY sigmoid-activated by the objectness head (do not
        # sigmoid again). cls_pred are raw logits, so they do get a sigmoid here.
        obj = obj_pred.clamp(0, 1).permute(0, 2, 3, 1).reshape(B, -1)
        cls = torch.sigmoid(cls_pred).permute(0, 2, 3, 1).reshape(B, -1, C)

        all_boxes.append(bbox)
        all_obj_scores.append(obj)
        all_cls_scores.append(cls)

    # Concatenate all scales
    boxes = torch.cat(all_boxes, dim=1)        # [B, N_total, 4]
    obj_scores = torch.cat(all_obj_scores, dim=1)   # [B, N_total]
    cls_scores = torch.cat(all_cls_scores, dim=1)   # [B, N_total, C]

    # Class prediction + combined confidence (objectness * class score)
    cls_max_scores, cls_indices = torch.max(cls_scores, dim=2)  # [B, N_total]
    conf_scores = obj_scores * cls_max_scores

    def _empty():
        return {
            'boxes': torch.empty(0, 4, device=boxes.device),
            'scores': torch.empty(0, device=boxes.device),
            'classes': torch.empty(0, dtype=torch.long, device=boxes.device),
        }

    batch_detections = []
    for b in range(boxes.shape[0]):
        # Filter by confidence
        mask = conf_scores[b] >= conf_threshold
        if not bool(mask.any()):
            batch_detections.append(_empty())
            continue

        img_boxes = boxes[b][mask]
        img_scores = conf_scores[b][mask]
        img_classes = cls_indices[b][mask]

        # Cap the number of candidates BEFORE NMS (highest-scoring first). This
        # bounds cost regardless of how many anchors slip past conf_threshold.
        if img_scores.numel() > max_nms:
            top = torch.topk(img_scores, max_nms).indices
            img_boxes, img_scores, img_classes = img_boxes[top], img_scores[top], img_classes[top]

        xyxy = _xywh_to_xyxy(img_boxes)

        if _HAS_TV:
            # Class-aware NMS in one fused kernel; returns indices sorted by score.
            keep = _tv_batched_nms(xyxy, img_scores, img_classes, iou_threshold)
        else:
            # Pure-Python fallback (slow): per-class NMS on xywh boxes.
            keep_parts = []
            for c in torch.unique(img_classes):
                cm = img_classes == c
                ci = torch.where(cm)[0]
                k = nms(img_boxes[cm], img_scores[cm], iou_threshold)
                keep_parts.append(ci[k])
            if keep_parts:
                keep = torch.cat(keep_parts)
                keep = keep[torch.argsort(img_scores[keep], descending=True)]
            else:
                batch_detections.append(_empty())
                continue

        keep = keep[:max_det]
        batch_detections.append({
            'boxes': img_boxes[keep],
            'scores': img_scores[keep],
            'classes': img_classes[keep],
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
