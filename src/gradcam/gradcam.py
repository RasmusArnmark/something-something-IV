"""
Grad-CAM implementation for 2D and 3D models
Supports both spatial-only and spatiotemporal visualizations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from matplotlib import cm


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    
    Args:
        model: PyTorch model
        target_layer: Layer to compute gradients for (usually last conv layer)
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward activations"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM for the input
        
        Args:
            input_tensor: Input tensor of shape (1, C, H, W) for 2D
                         or (1, C, T, H, W) for 3D
            target_class: Target class index. If None, use predicted class
        
        Returns:
            cam: Grad-CAM heatmap(s)
                 For 2D: (H, W)
                 For 3D: (T, H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]
        
        # Check if 2D or 3D
        if gradients.dim() == 3:  # 2D case: (C, H, W)
            # Compute channel-wise weights
            weights = gradients.mean(dim=(1, 2))  # (C,)
            
            # Weighted sum of activations
            cam = torch.sum(weights.view(-1, 1, 1) * activations, dim=0)  # (H, W)
            
            # Apply ReLU
            cam = F.relu(cam)
            
            # Normalize
            cam = cam.cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
        elif gradients.dim() == 4:  # 3D case: (C, T, H, W)
            # Option 1: Global pooling over space and time
            # weights = gradients.mean(dim=(1, 2, 3))  # (C,)
            # cam = torch.sum(weights.view(-1, 1, 1, 1) * activations, dim=0)  # (T, H, W)
            
            # Option 2: Pooling over space only, keep temporal dimension
            weights = gradients.mean(dim=(2, 3))  # (C, T)
            
            # Compute CAM for each time step
            cam_frames = []
            for t in range(activations.shape[1]):
                cam_t = torch.sum(weights[:, t].view(-1, 1, 1) * activations[:, t], dim=0)  # (H, W)
                cam_t = F.relu(cam_t)
                cam_frames.append(cam_t)
            
            cam = torch.stack(cam_frames, dim=0)  # (T, H, W)
            
            # Normalize each frame
            cam = cam.cpu().numpy()
            for t in range(cam.shape[0]):
                frame = cam[t]
                cam[t] = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        
        else:
            raise ValueError(f"Unexpected gradient dimension: {gradients.dim()}")
        
        return cam
    
    def remove_hooks(self):
        """Remove forward and backward hooks"""
        self.forward_hook.remove()
        self.backward_hook.remove()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks
    """
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Compute alpha weights (Grad-CAM++ improvement)
        # alpha_k = grad^2 / (2*grad^2 + sum(act * grad^3))
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        
        if gradients.dim() == 3:  # 2D
            alpha = grad_2 / (2 * grad_2 + (activations * grad_3).sum(dim=(1, 2), keepdim=True) + 1e-8)
            weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))
            cam = torch.sum(weights.view(-1, 1, 1) * activations, dim=0)
            cam = F.relu(cam)
            cam = cam.cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        elif gradients.dim() == 4:  # 3D
            alpha = grad_2 / (2 * grad_2 + (activations * grad_3).sum(dim=(2, 3), keepdim=True) + 1e-8)
            weights = (alpha * F.relu(gradients)).sum(dim=(2, 3))
            
            cam_frames = []
            for t in range(activations.shape[1]):
                cam_t = torch.sum(weights[:, t].view(-1, 1, 1) * activations[:, t], dim=0)
                cam_t = F.relu(cam_t)
                cam_frames.append(cam_t)
            
            cam = torch.stack(cam_frames, dim=0)
            cam = cam.cpu().numpy()
            for t in range(cam.shape[0]):
                frame = cam[t]
                cam[t] = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        
        return cam


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image
    
    Args:
        image: Original image (H, W, 3) in range [0, 255]
        heatmap: Heatmap (H, W) in range [0, 1]
        alpha: Transparency of heatmap
        colormap: OpenCV colormap
    
    Returns:
        overlayed: Image with heatmap overlay (H, W, 3)
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = np.uint8(255 * image)
    
    # Overlay
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed


def visualize_gradcam_2d(
    model: nn.Module,
    image: torch.Tensor,
    target_layer: nn.Module,
    target_class: Optional[int] = None,
    original_image: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize Grad-CAM for 2D model
    
    Args:
        model: PyTorch model
        image: Input tensor (1, C, H, W)
        target_layer: Target layer for Grad-CAM
        target_class: Target class index
        original_image: Original image for overlay (H, W, 3)
        save_path: Path to save visualization
    
    Returns:
        overlay: Image with Grad-CAM overlay
    """
    # Generate Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_cam(image, target_class)
    grad_cam.remove_hooks()
    
    # Get original image
    if original_image is None:
        # Denormalize and convert tensor to numpy
        img = image[0].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        img = np.uint8(255 * img)
    else:
        img = original_image
    
    # Overlay heatmap
    overlay = overlay_heatmap_on_image(img, cam)
    
    # Save if requested
    if save_path:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap='jet')
        plt.title('Grad-CAM')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return overlay


def visualize_gradcam_3d(
    model: nn.Module,
    video: torch.Tensor,
    target_layer: nn.Module,
    target_class: Optional[int] = None,
    original_frames: Optional[List[np.ndarray]] = None,
    save_path: Optional[str] = None
) -> List[np.ndarray]:
    """
    Visualize Grad-CAM for 3D model
    
    Args:
        model: PyTorch model
        video: Input tensor (1, C, T, H, W)
        target_layer: Target layer for Grad-CAM
        target_class: Target class index
        original_frames: List of original frames [(H, W, 3), ...]
        save_path: Path to save visualization
    
    Returns:
        overlays: List of frames with Grad-CAM overlay
    """
    # Generate Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    cam_3d = grad_cam.generate_cam(video, target_class)  # (T, H, W)
    grad_cam.remove_hooks()
    
    # Process each frame
    overlays = []
    num_frames = cam_3d.shape[0]
    
    for t in range(num_frames):
        # Get original frame
        if original_frames is not None and t < len(original_frames):
            img = original_frames[t]
        else:
            # Extract from video tensor
            frame = video[0, :, t].cpu().numpy().transpose(1, 2, 0)
            frame = (frame - frame.min()) / (frame.max() - frame.min())
            img = np.uint8(255 * frame)
        
        # Overlay heatmap
        overlay = overlay_heatmap_on_image(img, cam_3d[t])
        overlays.append(overlay)
    
    # Save visualization
    if save_path:
        # Create grid visualization
        n_cols = min(4, num_frames)
        n_rows = (num_frames + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, overlay in enumerate(overlays):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f'Frame {idx}')
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for idx in range(len(overlays), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return overlays
