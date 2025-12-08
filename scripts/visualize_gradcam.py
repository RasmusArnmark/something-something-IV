"""Generate Grad-CAM visualizations for trained models"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.resnet2d import create_2d_model
from models.resnet3d import create_3d_model
from gradcam.gradcam import GradCAM, visualize_gradcam_2d, visualize_gradcam_3d
from data.dataset import SomethingSomethingV2Dataset, get_transforms


def visualize_2d_samples(
    model,
    dataset,
    num_samples=10,
    output_dir='outputs/gradcam_2d',
    device='cuda'
):
    """Generate Grad-CAM for 2D model samples"""
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    model.to(device)
    
    # Get target layer
    target_layer = model.target_layer
    
    for idx in range(min(num_samples, len(dataset))):
        print(f"Processing sample {idx + 1}/{num_samples}")
        
        # Get sample
        frames, label, metadata = dataset[idx]
        
        # Add batch dimension
        frames_batch = frames.unsqueeze(0).to(device)
        
        # Generate visualization
        save_path = os.path.join(output_dir, f'sample_{idx}_class_{label}.png')
        visualize_gradcam_2d(
            model,
            frames_batch,
            target_layer,
            target_class=label,
            save_path=save_path
        )
        
        print(f"Saved visualization to {save_path}")
        print(f"Video ID: {metadata['video_id']}, Label: {metadata['label_name']}")


def visualize_3d_samples(
    model,
    dataset,
    num_samples=10,
    output_dir='outputs/gradcam_3d',
    device='cuda'
):
    """Generate Grad-CAM for 3D model samples"""
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    model.to(device)
    
    # Get target layer
    target_layer = model.target_layer
    
    for idx in range(min(num_samples, len(dataset))):
        print(f"Processing sample {idx + 1}/{num_samples}")
        
        # Get sample
        video, label, metadata = dataset[idx]
        
        # Add batch dimension
        video_batch = video.unsqueeze(0).to(device)
        
        # Generate visualization
        save_path = os.path.join(output_dir, f'sample_{idx}_class_{label}.png')
        visualize_gradcam_3d(
            model,
            video_batch,
            target_layer,
            target_class=label,
            save_path=save_path
        )
        
        print(f"Saved visualization to {save_path}")
        print(f"Video ID: {metadata['video_id']}, Label: {metadata['label_name']}")


def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations')
    parser.add_argument('--model_type', type=str, required=True, choices=['2d', '3d'],
                        help='Model type (2d or 3d)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='outputs/gradcam',
                        help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create model
    if args.model_type == '2d':
        model = create_2d_model(config)
    else:
        model = create_3d_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Validation accuracy: {checkpoint.get('val_acc1', 'N/A')}")
    
    # Create dataset (validation set)
    data_cfg = config['data']
    aug_cfg = config.get('augmentation', {})
    
    transform = get_transforms(
        data_cfg['img_size'],
        is_train=False,
        config=aug_cfg
    )
    
    dataset = SomethingSomethingV2Dataset(
        data_root=data_cfg['dataset_path'],
        json_file=os.path.join(data_cfg['dataset_path'], data_cfg['json_val']),
        labels_file=os.path.join(data_cfg['dataset_path'], data_cfg['json_labels']),
        num_frames=data_cfg['num_frames'],
        frame_sampling=data_cfg['frame_sampling'],
        img_size=data_cfg['img_size'],
        temporal_stride=data_cfg.get('temporal_stride', 1),
        transform=transform,
        is_train=False
    )
    
    # Generate visualizations
    if args.model_type == '2d':
        visualize_2d_samples(model, dataset, args.num_samples, args.output_dir, device)
    else:
        visualize_3d_samples(model, dataset, args.num_samples, args.output_dir, device)
    
    print(f"\nDone! Visualizations saved to {args.output_dir}")


if __name__ == '__main__':
    main()
