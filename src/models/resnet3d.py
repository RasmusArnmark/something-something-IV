"""3D Model definitions for Something-Something V2"""
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, r2plus1d_18, R3D_18_Weights, R2Plus1D_18_Weights
from typing import Optional
import math


class ResNet3D(nn.Module):
    """
    3D ResNet model for video classification
    """
    
    def __init__(
        self,
        num_classes: int = 174,
        arch: str = 'r3d_18',
        pretrained: bool = False,
        pretrained_2d_path: Optional[str] = None
    ):
        super(ResNet3D, self).__init__()
        
        self.num_classes = num_classes
        self.arch = arch
        
        # Load 3D backbone
        if arch == 'r3d_18':
            weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
            self.backbone = r3d_18(weights=weights)
            feature_dim = 512
        elif arch == 'r2plus1d_18':
            weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
            self.backbone = r2plus1d_18(weights=weights)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        # Replace final fully connected layer
        self.backbone.fc = nn.Linear(feature_dim, num_classes)
        
        # Load 2D pretrained weights if provided
        if pretrained_2d_path:
            self.load_2d_weights(pretrained_2d_path)
        
        # Store feature dimension
        self.feature_dim = feature_dim
        
        # Store the last conv layer for Grad-CAM
        # For R3D and R(2+1)D, the last layer is in layer4
        self.target_layer = self.backbone.layer4[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        
        Returns:
            logits: Output tensor of shape (B, num_classes)
        """
        return self.backbone(x)
    
    def load_2d_weights(self, checkpoint_path: str):
        """
        Load 2D pretrained weights and inflate to 3D
        
        This is a simplified version. For proper inflation:
        - 2D conv kernels (out, in, h, w) are inflated to (out, in, t, h, w)
        - by repeating along temporal dimension and dividing by t
        """
        print(f"Loading 2D weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get 2D state dict
        if 'model_state_dict' in checkpoint:
            state_dict_2d = checkpoint['model_state_dict']
        else:
            state_dict_2d = checkpoint
        
        # Get 3D state dict
        state_dict_3d = self.backbone.state_dict()
        
        # Inflate 2D weights to 3D
        inflated_state_dict = {}
        for name, param_3d in state_dict_3d.items():
            # Try to find corresponding 2D weight
            # Remove 'backbone.' prefix if present
            name_2d = name.replace('backbone.', '')
            
            if name_2d in state_dict_2d:
                param_2d = state_dict_2d[name_2d]
                
                # Check if it's a conv layer that needs inflation
                if len(param_2d.shape) == 4 and len(param_3d.shape) == 5:
                    # Inflate: (out, in, h, w) -> (out, in, t, h, w)
                    t = param_3d.shape[2]
                    param_inflated = param_2d.unsqueeze(2).repeat(1, 1, t, 1, 1) / t
                    inflated_state_dict[name] = param_inflated
                    print(f"Inflated {name}: {param_2d.shape} -> {param_inflated.shape}")
                elif param_2d.shape == param_3d.shape:
                    # Same shape, just copy
                    inflated_state_dict[name] = param_2d
                else:
                    print(f"Shape mismatch for {name}: 2D={param_2d.shape}, 3D={param_3d.shape}")
            else:
                print(f"No 2D weight found for {name}")
        
        # Load inflated weights
        self.backbone.load_state_dict(inflated_state_dict, strict=False)
        print("Successfully loaded and inflated 2D weights to 3D model")


class I3D(nn.Module):
    """
    Inflated 3D ConvNet (I3D) architecture
    Based on "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
    """
    
    def __init__(self, num_classes: int = 174, pretrained_2d_path: Optional[str] = None):
        super(I3D, self).__init__()
        
        self.num_classes = num_classes
        
        # Use R3D as base and modify if needed
        self.backbone = r3d_18()
        self.backbone.fc = nn.Linear(512, num_classes)
        
        if pretrained_2d_path:
            self.load_2d_weights(pretrained_2d_path)
        
        self.target_layer = self.backbone.layer4[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def load_2d_weights(self, checkpoint_path: str):
        """Load and inflate 2D weights"""
        print(f"Loading 2D weights for I3D from {checkpoint_path}")
        # Similar to ResNet3D inflation
        # This is a placeholder - implement based on your 2D model structure


def create_3d_model(config: dict) -> nn.Module:
    """
    Factory function to create 3D model from config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        model: PyTorch model
    """
    model_cfg = config['model']
    
    # Check if we should load 2D pretrained weights
    pretrained_2d = None
    if isinstance(model_cfg.get('pretrained'), str):
        pretrained_2d = model_cfg['pretrained']
        pretrained = False
    else:
        pretrained = model_cfg.get('pretrained', False)
    
    if model_cfg['name'] == 'i3d':
        model = I3D(
            num_classes=model_cfg['num_classes'],
            pretrained_2d_path=pretrained_2d
        )
    else:
        model = ResNet3D(
            num_classes=model_cfg['num_classes'],
            arch=model_cfg['name'],
            pretrained=pretrained,
            pretrained_2d_path=pretrained_2d
        )
    
    return model


if __name__ == '__main__':
    # Test the model
    model = ResNet3D(num_classes=174, arch='r3d_18', pretrained=False)
    
    # Test input
    x = torch.randn(2, 3, 16, 112, 112)  # (B, C, T, H, W)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
