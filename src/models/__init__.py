"""Model architectures for video classification"""
from .resnet2d import ResNet2D, create_2d_model
from .resnet3d import ResNet3D, I3D, create_3d_model

__all__ = ['ResNet2D', 'create_2d_model', 'ResNet3D', 'I3D', 'create_3d_model']
