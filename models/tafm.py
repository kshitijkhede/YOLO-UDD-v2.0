"""
Turbidity-Adaptive Fusion Module (TAFM)
Novel contribution for YOLO-UDD v2.0

This module dynamically adjusts feature fusion based on real-time water turbidity conditions.
As described in Section 6 of the project plan, TAFM uses a lightweight CNN to estimate
turbidity and adaptively weights features for optimal performance in varying water conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TAFM(nn.Module):
    """
    Turbidity-Adaptive Fusion Module
    
    Dynamically adjusts feature fusion strategy based on estimated water turbidity.
    
    Args:
        channels (int): Number of input feature channels
        reduction (int): Channel reduction ratio for efficiency (default: 16)
    """
    
    def __init__(self, channels, reduction=16):
        super(TAFM, self).__init__()
        
        # Lightweight CNN to estimate turbidity from image characteristics
        # Analyzes color histograms and high-frequency components
        self.turbidity_estimator = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Output turbidity score in range [0, 1]
        )
        
        # Learned parameters for clear water fusion strategy (beta)
        # These parameters emphasize fine-grained color and texture features
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1))
        
        # Learned parameters for murky water fusion strategy (alpha)
        # These parameters emphasize shape and strong edge features
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        
        # Channel attention for semantic refinement
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, image, neck_features):
        """
        Forward pass of TAFM
        
        Args:
            image (torch.Tensor): Original input image [B, 3, H, W]
            neck_features (torch.Tensor): Feature maps from neck [B, C, H', W']
            
        Returns:
            torch.Tensor: Adapted features with turbidity-aware weighting
        """
        # Estimate turbidity from downsampled image for efficiency
        # Turbidity score: 0 = clear water, 1 = murky water
        image_downsampled = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=False)
        turb_score = self.turbidity_estimator(image_downsampled)  # [B, 1, 1, 1]
        
        # Calculate adaptive weights using learned parameters
        # w_adapt = σ(Turb * α + (1-Turb) * β)
        # Where σ is sigmoid, α tunes murky conditions, β tunes clear conditions
        w_adapt = torch.sigmoid(
            turb_score * self.alpha + (1 - turb_score) * self.beta
        )
        
        # Apply channel attention for semantic task-specific weighting
        channel_weights = self.channel_attention(neck_features)
        
        # Combine turbidity-adaptive weighting with channel attention
        adapted_features = neck_features * w_adapt * channel_weights
        
        # Residual connection to preserve original information
        adapted_features = adapted_features + neck_features
        
        return adapted_features, turb_score
    
    def get_turbidity_score(self, image):
        """
        Get turbidity score for a given image
        
        Args:
            image (torch.Tensor): Input image [B, 3, H, W]
            
        Returns:
            torch.Tensor: Turbidity score [B, 1, 1, 1]
        """
        image_downsampled = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=False)
        return self.turbidity_estimator(image_downsampled)


class MultiScaleTAFM(nn.Module):
    """
    Multi-scale version of TAFM for handling features at different resolutions
    """
    
    def __init__(self, channels_list):
        """
        Args:
            channels_list (list): List of channel sizes for each scale
        """
        super(MultiScaleTAFM, self).__init__()
        self.tafm_modules = nn.ModuleList([
            TAFM(channels) for channels in channels_list
        ])
        
    def forward(self, image, feature_list):
        """
        Apply TAFM to multi-scale features
        
        Args:
            image (torch.Tensor): Original input image
            feature_list (list): List of feature maps at different scales
            
        Returns:
            list: Adapted features at each scale
            torch.Tensor: Average turbidity score
        """
        adapted_features = []
        turb_scores = []
        
        for tafm, features in zip(self.tafm_modules, feature_list):
            adapted, turb = tafm(image, features)
            adapted_features.append(adapted)
            turb_scores.append(turb)
        
        # Average turbidity score across scales
        avg_turb_score = torch.mean(torch.stack(turb_scores), dim=0)
        
        return adapted_features, avg_turb_score
