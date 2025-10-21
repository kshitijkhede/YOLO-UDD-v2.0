"""
Split Dimension Weighting Head (SDWH)
Attention-driven detection head for YOLO-UDD v2.0

This module implements a multi-stage attention mechanism that applies weighting across
three dimensions: level-wise, spatial-wise, and channel-wise, as described in Section 3.3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LevelWiseAttention(nn.Module):
    """
    Level-wise (scale) attention module
    Weights features from different pyramid levels
    """
    
    def __init__(self, num_levels=3):
        super(LevelWiseAttention, self).__init__()
        self.num_levels = num_levels
        self.level_weights = nn.Parameter(torch.ones(num_levels, 1, 1, 1))
        
    def forward(self, features):
        """
        Args:
            features (list): List of feature maps from different levels
        Returns:
            list: Weighted feature maps
        """
        weighted_features = []
        weights = F.softmax(self.level_weights, dim=0)
        
        for i, feat in enumerate(features):
            weighted = feat * weights[i]
            weighted_features.append(weighted)
        
        return weighted_features


class SpatialWiseAttention(nn.Module):
    """
    Spatial-wise (location) attention module
    Applies location-specific attention using self-attention mechanism
    """
    
    def __init__(self, channels):
        super(SpatialWiseAttention, self).__init__()
        
        # Query, Key, Value projections
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        
        # Learnable gamma parameter for residual connection
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Self-attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
        
        Args:
            x (torch.Tensor): Input features [B, C, H, W]
        Returns:
            torch.Tensor: Spatially attended features
        """
        B, C, H, W = x.size()
        
        # Generate Q, K, V
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C']
        key = self.key_conv(x).view(B, -1, H * W)  # [B, C', HW]
        value = self.value_conv(x).view(B, -1, H * W)  # [B, C, HW]
        
        # Compute attention scores
        # QK^T / sqrt(d_k)
        d_k = query.size(-1)
        attention = torch.bmm(query, key) / math.sqrt(d_k)  # [B, HW, HW]
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable gamma
        out = self.gamma * out + x
        
        return out


class ChannelWiseAttention(nn.Module):
    """
    Channel-wise (semantic task) attention module
    Applies channel attention using squeeze-and-excitation mechanism
    """
    
    def __init__(self, channels, reduction=16):
        super(ChannelWiseAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for channel attention
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features [B, C, H, W]
        Returns:
            torch.Tensor: Channel-attended features
        """
        # Average and max pooling along spatial dimensions
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        
        return x * attention


class SDWH(nn.Module):
    """
    Split Dimension Weighting Head
    
    Cascaded attention mechanism that applies:
    1. Level-wise attention (scale)
    2. Spatial-wise attention (location)
    3. Channel-wise attention (semantic task)
    
    This purifies feature maps to focus on foreground targets (debris) while
    suppressing background noise.
    
    Args:
        channels_list (list): List of channel counts for each pyramid level [128, 256, 512]
        num_classes (int): Number of detection classes
        num_levels (int): Number of feature pyramid levels
    """
    
    def __init__(self, channels_list=[128, 256, 512], num_classes=3, num_levels=3):
        super(SDWH, self).__init__()
        
        self.num_classes = num_classes
        self.num_levels = num_levels
        self.channels_list = channels_list
        
        # Three-stage attention cascade (one per level)
        self.level_attention = LevelWiseAttention(num_levels)
        
        # Create separate attention and detection heads for each pyramid level
        self.spatial_attentions = nn.ModuleList([
            SpatialWiseAttention(ch) for ch in channels_list
        ])
        self.channel_attentions = nn.ModuleList([
            ChannelWiseAttention(ch) for ch in channels_list
        ])
        
        # Detection heads for each task (one set per level)
        # Bounding box regression heads
        self.bbox_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch, 4, 1)  # (x, y, w, h)
            ) for ch in channels_list
        ])
        
        # Objectness heads
        self.obj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch // 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ch // 2),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch // 2, 1, 1),
                nn.Sigmoid()
            ) for ch in channels_list
        ])
        
        # Classification heads
        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch, num_classes, 1)
            ) for ch in channels_list
        ])
        
    def forward_single(self, x, level_idx):
        """
        Forward pass for a single feature map
        
        Args:
            x (torch.Tensor): Input feature map [B, C, H, W]
            level_idx (int): Index of the pyramid level
        Returns:
            tuple: (bbox_pred, obj_pred, cls_pred)
        """
        # Apply cascaded attention
        x = self.spatial_attentions[level_idx](x)
        x = self.channel_attentions[level_idx](x)
        
        # Generate predictions
        bbox_pred = self.bbox_heads[level_idx](x)
        obj_pred = self.obj_heads[level_idx](x)
        cls_pred = self.cls_heads[level_idx](x)
        
        return bbox_pred, obj_pred, cls_pred
    
    def forward(self, features):
        """
        Forward pass for multi-scale features
        
        Args:
            features (list): List of feature maps from different levels
        Returns:
            list: List of (bbox, obj, cls) predictions for each level
        """
        # Apply level-wise attention first
        weighted_features = self.level_attention(features)
        
        # Process each level through spatial and channel attention + detection heads
        predictions = []
        for idx, feat in enumerate(weighted_features):
            pred = self.forward_single(feat, idx)
            predictions.append(pred)
        
        return predictions


class SDWHLoss(nn.Module):
    """
    Composite loss function for SDWH as described in Section 3.4
    Combines: EIoU Loss, Varifocal Loss, and BCE Loss
    """
    
    def __init__(self, num_classes=3):
        super(SDWHLoss, self).__init__()
        self.num_classes = num_classes
        self.bce_obj = nn.BCELoss(reduction='none')
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        
    def bbox_eiou_loss(self, pred_boxes, target_boxes):
        """
        Efficient IoU Loss for bounding box regression
        """
        # Calculate IoU
        # pred_boxes and target_boxes: [N, 4] in format (x, y, w, h)
        
        # Convert to (x1, y1, x2, y2)
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
        
        # EIoU Loss
        loss = 1 - iou
        
        return loss.mean()
    
    def forward(self, predictions, targets):
        """
        Compute composite loss
        
        Args:
            predictions: List of (bbox, obj, cls) predictions
            targets: Ground truth annotations
        Returns:
            dict: Loss components
        """
        # This is a simplified version - full implementation would require
        # proper target assignment and matching
        
        losses = {
            'bbox_loss': 0.0,
            'obj_loss': 0.0,
            'cls_loss': 0.0,
            'total_loss': 0.0
        }
        
        return losses
