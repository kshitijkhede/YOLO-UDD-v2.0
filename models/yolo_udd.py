"""
YOLO-UDD v2.0: Turbidity-Adaptive Architecture for Underwater Debris Detection

Main model architecture integrating:
- YOLOv9c backbone
- PSEM (Partial Semantic Encoding Module) in neck
- SDWH (Split Dimension Weighting Head)
- TAFM (Turbidity-Adaptive Fusion Module)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .psem import PSEM, PSEMNeck
from .sdwh import SDWH
from .tafm import TAFM, MultiScaleTAFM


class ConvModule(nn.Module):
    """Basic convolutional module used in YOLO"""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    """Cross Stage Partial Block for efficient feature extraction"""
    
    def __init__(self, in_channels, out_channels, num_blocks=3):
        super(CSPBlock, self).__init__()
        hidden_channels = out_channels // 2
        
        self.conv1 = ConvModule(in_channels, hidden_channels, 1)
        self.conv2 = ConvModule(in_channels, hidden_channels, 1)
        
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                ConvModule(hidden_channels, hidden_channels, 3, 1, 1),
                ConvModule(hidden_channels, hidden_channels, 3, 1, 1)
            ) for _ in range(num_blocks)
        ])
        
        self.conv3 = ConvModule(2 * hidden_channels, out_channels, 1)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.blocks(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.conv3(x)


class YOLOUDDBackbone(nn.Module):
    """
    YOLOv9c-based backbone for feature extraction
    Produces multi-scale features at different resolutions
    """
    
    def __init__(self, in_channels=3):
        super(YOLOUDDBackbone, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            ConvModule(in_channels, 32, 3, 2, 1),
            ConvModule(32, 64, 3, 2, 1)
        )
        
        # Stage 1: 160x160
        self.stage1 = CSPBlock(64, 128, num_blocks=3)
        self.downsample1 = ConvModule(128, 128, 3, 2, 1)
        
        # Stage 2: 80x80
        self.stage2 = CSPBlock(128, 256, num_blocks=6)
        self.downsample2 = ConvModule(256, 256, 3, 2, 1)
        
        # Stage 3: 40x40
        self.stage3 = CSPBlock(256, 512, num_blocks=6)
        self.downsample3 = ConvModule(512, 512, 3, 2, 1)
        
        # Stage 4: 20x20
        self.stage4 = CSPBlock(512, 1024, num_blocks=3)
        
    def forward(self, x):
        """
        Args:
            x: Input image [B, 3, 640, 640]
        Returns:
            list: Multi-scale features [P3, P4, P5]
        """
        x = self.stem(x)
        
        x = self.stage1(x)
        p3 = self.downsample1(x)  # 80x80, 128 channels
        
        x = self.stage2(p3)
        p4 = self.downsample2(x)  # 40x40, 256 channels
        
        x = self.stage3(p4)
        p5 = self.downsample3(x)  # 20x20, 512 channels
        
        x = self.stage4(p5)  # 20x20, 1024 channels
        
        return [p3, p4, p5, x]


class YOLOUDDNeck(nn.Module):
    """
    PSEM-enhanced neck with Path Aggregation Network (PANet)
    Integrates TAFM for turbidity-adaptive fusion
    """
    
    def __init__(self, channels_list=[128, 256, 512, 1024]):
        super(YOLOUDDNeck, self).__init__()
        
        # Lateral connections to reduce channels before concatenation
        self.lateral6 = ConvModule(1024, 512, 1, 1, 0)  # Reduce p6 channels to match p5
        self.lateral4 = ConvModule(512, 256, 1, 1, 0)   # 1x1 conv to match p4 channels
        self.lateral3 = ConvModule(256, 128, 1, 1, 0)   # 1x1 conv to match p3 channels
        
        # Top-down pathway with PSEM
        # No upsample for p6->p5 since they're the same spatial size (20x20)
        self.psem1 = PSEM(512 + 512, 512)  # Merge p6 and p5
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.psem2 = PSEM(256 + 256, 256)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.psem3 = PSEM(128 + 128, 128)
        
        # Bottom-up pathway with PSEM
        self.down1 = ConvModule(128, 128, 3, 2, 1)
        self.psem4 = PSEM(128 + 256, 256)
        
        self.down2 = ConvModule(256, 256, 3, 2, 1)
        self.psem5 = PSEM(256 + 512, 512)
        
        # TAFM for turbidity-adaptive fusion
        self.tafm = MultiScaleTAFM([128, 256, 512])
        
    def forward(self, image, features):
        """
        Args:
            image: Original input image [B, 3, 640, 640]
            features: [p3, p4, p5, p6] from backbone
        Returns:
            list: Enhanced features for detection [P3, P4, P5]
            torch.Tensor: Turbidity score
        """
        p3, p4, p5, p6 = features
        
        # Top-down pathway with lateral connections
        # p6 and p5 are both 20x20, so we concatenate without upsampling
        x = self.lateral6(p6)  # 20x20, 512ch
        x = torch.cat([x, p5], dim=1)  # 20x20, 1024ch
        p5_out = self.psem1(x)  # 20x20, 512ch
        
        # Reduce p5_out channels before upsampling to 40x40
        x = self.lateral4(p5_out)  # 20x20, 256ch
        x = self.up2(x)  # 40x40, 256ch
        x = torch.cat([x, p4], dim=1)  # 40x40, 512ch
        p4_out = self.psem2(x)  # 40x40, 256ch
        
        # Reduce p4_out channels before upsampling to 80x80
        x = self.lateral3(p4_out)  # 40x40, 128ch
        x = self.up3(x)  # 80x80, 128ch
        x = torch.cat([x, p3], dim=1)  # 80x80, 256ch
        p3_out = self.psem3(x)  # 80x80, 128ch
        
        # Bottom-up pathway
        x = self.down1(p3_out)
        x = torch.cat([x, p4_out], dim=1)
        p4_final = self.psem4(x)
        
        x = self.down2(p4_final)
        x = torch.cat([x, p5_out], dim=1)
        p5_final = self.psem5(x)
        
        # Apply TAFM for turbidity-adaptive fusion
        neck_features = [p3_out, p4_final, p5_final]
        adapted_features, turb_score = self.tafm(image, neck_features)
        
        return adapted_features, turb_score


class YOLOUD(nn.Module):
    """
    YOLO-UDD v2.0: Complete architecture for underwater debris detection
    
    Architecture:
    - Input: 640x640 RGB image
    - Backbone: YOLOv9c-based feature extractor
    - Neck: PSEM-enhanced PANet with TAFM
    - Head: SDWH attention-based detection head
    
    Args:
        num_classes (int): Number of detection classes (default: 3 for TrashCan dataset)
        pretrained (str): Path to pretrained weights (optional)
    """
    
    def __init__(self, num_classes=3, pretrained=None):
        super(YOLOUD, self).__init__()
        
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = YOLOUDDBackbone(in_channels=3)
        
        # Neck with PSEM and TAFM
        self.neck = YOLOUDDNeck()
        
        # Detection head with SDWH - accepts list of channels for each pyramid level
        self.head = SDWH(channels_list=[128, 256, 512], num_classes=num_classes, num_levels=3)
        
        # Initialize weights
        self._initialize_weights()
        
        # Load pretrained weights if provided
        if pretrained:
            self.load_pretrained(pretrained)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input images [B, 3, 640, 640]
            
        Returns:
            tuple: 
                - predictions: List of (bbox, obj, cls) for each scale
                - turb_score: Estimated turbidity score
        """
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Enhance features with PSEM and adapt with TAFM
        neck_features, turb_score = self.neck(x, features)
        
        # Generate detections with SDWH
        predictions = self.head(neck_features)
        
        return predictions, turb_score
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def load_pretrained(self, path):
        """Load pretrained weights from COCO"""
        try:
            state_dict = torch.load(path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'Architecture': 'YOLO-UDD v2.0',
            'Backbone': 'YOLOv9c',
            'Neck': 'PSEM-enhanced PANet + TAFM',
            'Head': 'SDWH',
            'Total Parameters': f'{total_params:,}',
            'Trainable Parameters': f'{trainable_params:,}',
            'Input Size': '640x640',
            'Output Classes': self.num_classes
        }
        
        return info


def build_yolo_udd(num_classes=3, pretrained=None):
    """
    Build YOLO-UDD v2.0 model
    
    Args:
        num_classes (int): Number of classes
        pretrained (str): Path to pretrained weights
        
    Returns:
        YOLOUD: Model instance
    """
    model = YOLOUD(num_classes=num_classes, pretrained=pretrained)
    return model


if __name__ == '__main__':
    # Test model
    model = build_yolo_udd(num_classes=3)
    
    # Print model info
    info = model.get_model_info()
    print("\n=== YOLO-UDD v2.0 Model Information ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 640, 640)
    predictions, turb_score = model(dummy_input)
    
    print(f"\nTurbidity Score: {turb_score.item():.4f}")
    print(f"Number of detection scales: {len(predictions)}")
    for i, (bbox, obj, cls) in enumerate(predictions):
        print(f"Scale {i+1} - BBox: {bbox.shape}, Obj: {obj.shape}, Cls: {cls.shape}")
