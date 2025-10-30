"""
COCO-style Evaluation Metrics for YOLO-UDD v2.0
Implements proper mAP@50, mAP@50:95, Precision, Recall
"""

import torch
import numpy as np
from collections import defaultdict


def box_iou_xyxy(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes in (x1, y1, x2, y2) format
    
    Args:
        boxes1: [N, 4] torch tensor
        boxes2: [M, 4] torch tensor
    
    Returns:
        iou: [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    inter_x1 = torch.maximum(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.maximum(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.minimum(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.minimum(boxes1[:, None, 3], boxes2[:, 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    union_area = area1[:, None] + area2 - inter_area
    
    iou = inter_area / (union_area + 1e-7)
    return iou


def box_xywh_to_xyxy(boxes):
    """Convert boxes from (x, y, w, h) to (x1, y1, x2, y2) format"""
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def compute_ap(recall, precision):
    """
    Compute Average Precision using all-point interpolation (COCO-style)
    
    Args:
        recall: np.array of recall values
        precision: np.array of precision values
    
    Returns:
        ap: Average precision
    """
    # Add sentinel values at the beginning and end
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    
    # Compute precision envelope (maximum precision for all recalls >= r)
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    
    # Calculate area under PR curve
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    
    return ap


def compute_metrics_coco(detections, targets, num_classes=3, iou_thresholds=None):
    """
    Compute COCO-style metrics from detections and ground truth
    
    Args:
        detections: List of detection dicts per image, each with:
            - 'boxes': [N, 4] in (x, y, w, h) format
            - 'scores': [N] confidence scores
            - 'classes': [N] class indices
        targets: List of target tuples per image (boxes, labels):
            - boxes: [M, 4] in (x, y, w, h) format
            - labels: [M] class indices
        num_classes: Number of classes
        iou_thresholds: List of IoU thresholds (default: 0.5 to 0.95 step 0.05)
    
    Returns:
        dict: Metrics including precision, recall, mAP@50, mAP@50:95
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)
    
    # Collect all detections and ground truths by class
    all_detections = defaultdict(list)  # class_id -> list of (img_id, score, box)
    all_ground_truths = defaultdict(list)  # class_id -> list of (img_id, box)
    
    for img_id, (det, (gt_boxes, gt_labels)) in enumerate(zip(detections, targets)):
        # Process detections
        if len(det['boxes']) > 0:
            det_boxes_xyxy = box_xywh_to_xyxy(det['boxes'])
            for box, score, cls in zip(det_boxes_xyxy, det['scores'], det['classes']):
                all_detections[cls.item()].append({
                    'img_id': img_id,
                    'score': score.item(),
                    'box': box.cpu().numpy()
                })
        
        # Process ground truths
        if len(gt_boxes) > 0:
            gt_boxes_xyxy = box_xywh_to_xyxy(gt_boxes)
            for box, cls in zip(gt_boxes_xyxy, gt_labels):
                all_ground_truths[cls.item()].append({
                    'img_id': img_id,
                    'box': box.cpu().numpy() if torch.is_tensor(box) else box
                })
    
    # Compute AP for each class and IoU threshold
    aps = {}
    for iou_thresh in iou_thresholds:
        aps[iou_thresh] = []
        
        for cls in range(num_classes):
            if cls not in all_detections or cls not in all_ground_truths:
                continue
            
            # Sort detections by confidence
            dets = sorted(all_detections[cls], key=lambda x: x['score'], reverse=True)
            gts = all_ground_truths[cls]
            
            # Track which ground truths have been matched
            matched = defaultdict(set)  # img_id -> set of matched gt indices
            
            tp = np.zeros(len(dets))
            fp = np.zeros(len(dets))
            
            for det_idx, det in enumerate(dets):
                img_id = det['img_id']
                det_box = torch.tensor(det['box']).unsqueeze(0)
                
                # Get ground truths for this image and class
                img_gts = [gt for gt in gts if gt['img_id'] == img_id]
                
                if len(img_gts) == 0:
                    fp[det_idx] = 1
                    continue
                
                # Compute IoU with all ground truths
                gt_boxes = torch.tensor([gt['box'] for gt in img_gts])
                ious = box_iou_xyxy(det_box, gt_boxes).squeeze(0)
                
                # Find best matching ground truth
                max_iou, max_idx = ious.max(dim=0)
                
                if max_iou >= iou_thresh and max_idx.item() not in matched[img_id]:
                    tp[det_idx] = 1
                    matched[img_id].add(max_idx.item())
                else:
                    fp[det_idx] = 1
            
            # Compute precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / len(gts) if len(gts) > 0 else np.zeros_like(tp_cumsum)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
            
            # Compute AP
            ap = compute_ap(recalls, precisions)
            aps[iou_thresh].append(ap)
    
    # Calculate summary metrics
    map_50 = np.mean(aps[0.5]) if 0.5 in aps and len(aps[0.5]) > 0 else 0.0
    map_75 = np.mean(aps[0.75]) if 0.75 in aps and len(aps[0.75]) > 0 else 0.0
    map_50_95 = np.mean([np.mean(v) for v in aps.values() if len(v) > 0]) if aps else 0.0
    
    # Calculate overall precision and recall
    total_tp = sum(len([d for d in all_detections[cls] if d]) for cls in all_detections)
    total_gt = sum(len(all_ground_truths[cls]) for cls in all_ground_truths)
    total_det = sum(len(all_detections[cls]) for cls in all_detections)
    
    precision = total_tp / total_det if total_det > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'map50': map_50,
        'map75': map_75,
        'map': map_50_95,
    }


def compute_metrics(predictions, targets):
    """
    Wrapper for backward compatibility with training script
    
    Args:
        predictions: Raw model predictions (will be ignored for now)
        targets: List of (boxes, labels) tuples
    
    Returns:
        dict: Metrics (returns zeros since we need NMS-processed detections)
    """
    # This is called during validation with raw predictions
    # Return placeholder values until we integrate NMS
    return {
        'precision': 0.0,
        'recall': 0.0,
        'map50': 0.0,
        'map75': 0.0,
        'map': 0.0,
    }


if __name__ == '__main__':
    print("Testing COCO-style metrics...")
    
    # Create dummy detections (after NMS)
    detections = [
        {
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.3, 0.1, 0.1]]),
            'scores': torch.tensor([0.9, 0.7]),
            'classes': torch.tensor([0, 1])
        },
        {
            'boxes': torch.tensor([[0.4, 0.6, 0.15, 0.25]]),
            'scores': torch.tensor([0.85]),
            'classes': torch.tensor([2])
        }
    ]
    
    # Create dummy ground truths
    targets = [
        (torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.3, 0.1, 0.1]]), 
         torch.tensor([0, 1])),
        (torch.tensor([[0.4, 0.6, 0.15, 0.25]]), 
         torch.tensor([2]))
    ]
    
    # Compute metrics
    metrics = compute_metrics_coco(detections, targets, num_classes=3)
    
    print("✓ Metrics computation successful!")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  mAP@50: {metrics['map50']:.4f}")
    print(f"  mAP@50:95: {metrics['map']:.4f}")
    print(f"  mAP@75: {metrics['map75']:.4f}")
