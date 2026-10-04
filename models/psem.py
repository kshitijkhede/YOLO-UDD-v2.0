"""
Partial Semantic Encoding Module (PSEM)
Enhanced feature integration for YOLO-UDD v2.0

This module enhances multi-scale feature fusion using dual-branch structure
with residual connections and partial convolutions, as described in Section 3.2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv(nn.Module):
    """
    Partial Convolution for efficient channel-wise semantic processing
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(PartialConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return self.bn(self.conv(x))


class PSEM(nn.Module):
    """
    Partial Semantic Encoding Module
    
    Enhances multi-scale feature fusion through dual-branch architecture with
    residual connections. Follows the principle: f(x) = Conv(Residual(x)) + x
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_branches (int): Number of parallel processing branches (default: 2)
    """
    
    def __init__(self, in_channels, out_channels, num_branches=2):
        super(PSEM, self).__init__()
        
        self.num_branches = num_branches
        
        # Split channels for partial convolutions
        self.split_channels = in_channels // num_branches
        
        # Branch 1: Standard convolution path for preserving spatial details
        self.branch1 = nn.Sequential(
            nn.Conv2d(self.split_channels, self.split_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.split_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.split_channels, self.split_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.split_channels),
        )
        
        # Branch 2: Dilated convolution path for expanded receptive field
        self.branch2 = nn.Sequential(
            nn.Conv2d(self.split_channels, self.split_channels, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(self.split_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.split_channels, self.split_channels, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(self.split_channels),
        )
        
        # Channel attention for semantic weighting
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention for location-specific weighting
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # Final fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # Residual projection if channels change
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        """
        Forward pass of PSEM
        
        Args:
            x (torch.Tensor): Input features [B, C, H, W]
            
        Returns:
            torch.Tensor: Enhanced fused features [B, C_out, H, W]
        """
        identity = x
        
        # Split input for partial convolutions
        x_splits = torch.split(x, self.split_channels, dim=1)
        
        # Process through parallel branches
        branch_outputs = []
        branch_outputs.append(self.branch1(x_splits[0]) + x_splits[0])  # Residual connection
        branch_outputs.append(self.branch2(x_splits[1]) + x_splits[1])  # Residual connection
        
        # Concatenate branch outputs
        x_concat = torch.cat(branch_outputs, dim=1)
        
        # Apply channel attention
        ca_weights = self.channel_attention(x_concat)
        x_ca = x_concat * ca_weights
        
        # Apply spatial attention
        # Compute max and avg pooling along channel dimension
        max_pool = torch.max(x_ca, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        spatial_features = torch.cat([max_pool, avg_pool], dim=1)
        sa_weights = self.spatial_attention(spatial_features)
        x_sa = x_ca * sa_weights
        
        # Final fusion
        x_fused = self.fusion_conv(x_sa)
        
        # Residual connection with original input
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        
        # Final output: f(x) = Conv(Residual(x)) + x
        output = x_fused + identity
        
        return output


class PSEMNeck(nn.Module):
    """
    PSEM-enhanced neck for multi-scale feature fusion in YOLO
    Replaces standard PANet convolutions with PSEM modules
    """
    
    def __init__(self, channels_list):
        """
        Args:
            channels_list (list): List of (in_channels, out_channels) tuples for each scale
        """
        super(PSEMNeck, self).__init__()
        
        self.psem_modules = nn.ModuleList([
            PSEM(in_ch, out_ch) for in_ch, out_ch in channels_list
        ])
        
    def forward(self, features):
        """
        Apply PSEM to multi-scale features
        
        Args:
            features (list): List of feature maps at different scales
            
        Returns:
            list: Enhanced feature maps at each scale
        """
        enhanced_features = []
        
        for psem, feat in zip(self.psem_modules, features):
            enhanced = psem(feat)
            enhanced_features.append(enhanced)
        
        return enhanced_features


class PSEMBlock(nn.Module):
    """
    Lightweight PSEM block for integration into existing architectures
    """
    
    def __init__(self, channels):
        super(PSEMBlock, self).__init__()
        self.psem = PSEM(channels, channels)
        
    def forward(self, x):
        return self.psem(x)
