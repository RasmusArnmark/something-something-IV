"""2D Model definitions for Something-Something V2"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from typing import Optional


class ResNet2D(nn.Module):
    """
    2D ResNet model for video classification
    Supports single-frame or multi-frame with temporal pooling
    """
    
    def __init__(
        self,
        num_classes: int = 174,
        arch: str = 'resnet18',
        pretrained: bool = True,
        temporal_pooling: str = 'avg'  # 'avg', 'max', or 'none'
    ):
        super(ResNet2D, self).__init__()
        
        self.num_classes = num_classes
        self.arch = arch
        self.temporal_pooling = temporal_pooling
        
        # Load pretrained backbone
        if arch == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet18(weights=weights)
            feature_dim = 512
        elif arch == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet50(weights=weights)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        # Replace final fully connected layer
        self.backbone.fc = nn.Linear(feature_dim, num_classes)
        
        # Store feature dimension for Grad-CAM
        self.feature_dim = feature_dim
        
        # Store the last conv layer for Grad-CAM
        self.target_layer = self.backbone.layer4[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, C, H, W) for single frame
               or (B, C, T, H, W) for multi-frame
        
        Returns:
            logits: Output tensor of shape (B, num_classes)
        """
        # Check if input has temporal dimension
        if x.dim() == 5:  # (B, C, T, H, W)
            batch_size, channels, num_frames, height, width = x.shape
            
            # Reshape to process all frames at once
            x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
            x = x.reshape(batch_size * num_frames, channels, height, width)  # (B*T, C, H, W)
            
            # Forward through backbone
            logits = self.backbone(x)  # (B*T, num_classes)
            
            # Reshape back
            logits = logits.view(batch_size, num_frames, self.num_classes)  # (B, T, num_classes)
            
            # Temporal pooling
            if self.temporal_pooling == 'avg':
                logits = logits.mean(dim=1)  # (B, num_classes)
            elif self.temporal_pooling == 'max':
                logits = logits.max(dim=1)[0]  # (B, num_classes)
            else:
                raise ValueError(f"Unknown temporal pooling: {self.temporal_pooling}")
        
        else:  # (B, C, H, W) - single frame
            logits = self.backbone(x)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before final classification layer
        Useful for transfer learning to 3D models
        """
        # Remove the final FC layer temporarily
        fc = self.backbone.fc
        self.backbone.fc = nn.Identity()
        
        features = self.backbone(x)
        
        # Restore FC layer
        self.backbone.fc = fc
        
        return features


def create_2d_model(config: dict) -> nn.Module:
    """
    Factory function to create 2D model from config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        model: PyTorch model
    """
    model_cfg = config['model']
    
    model = ResNet2D(
        num_classes=model_cfg['num_classes'],
        arch=model_cfg['name'],
        pretrained=model_cfg['pretrained']
    )
    
    return model


if __name__ == '__main__':
    # Test the model
    model = ResNet2D(num_classes=174, arch='resnet18', pretrained=True)
    
    # Test single frame
    x_single = torch.randn(2, 3, 224, 224)
    out_single = model(x_single)
    print(f"Single frame input shape: {x_single.shape}")
    print(f"Single frame output shape: {out_single.shape}")
    
    # Test multi-frame
    x_multi = torch.randn(2, 3, 8, 224, 224)
    out_multi = model(x_multi)
    print(f"Multi-frame input shape: {x_multi.shape}")
    print(f"Multi-frame output shape: {out_multi.shape}")
