"""
Inference script for Something-Something V2 with 3D CNNs.
"""

import argparse
import torch
import yaml
import cv2
import numpy as np
from src.models import resnet3d_18, resnet3d_34, resnet3d_50


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with 3D CNN on video')
    parser.add_argument('--video', type=str, required=True,
                        help='path to input video file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/resnet3d_18.yaml',
                        help='path to config file')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use')
    return parser.parse_args()


def load_video(video_path, num_frames=16, spatial_size=224, temporal_stride=2):
    """Load and preprocess video for inference"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    # Sample frames
    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError(f"Video {video_path} contains no frames")
    
    if total_frames < num_frames * temporal_stride:
        indices = np.linspace(0, max(0, total_frames - 1), num_frames).astype(int)
    else:
        start_idx = (total_frames - num_frames * temporal_stride) // 2
        indices = np.arange(start_idx, start_idx + num_frames * temporal_stride, temporal_stride)
    
    sampled_frames = [frames[i] for i in indices]
    
    # Preprocess frames
    processed_frames = []
    for frame in sampled_frames:
        frame = cv2.resize(frame, (spatial_size, spatial_size))
        frame = frame.astype(np.float32) / 255.0
        processed_frames.append(frame)
    
    # Stack frames: (T, H, W, C) -> (C, T, H, W)
    video_tensor = np.stack(processed_frames, axis=0)
    video_tensor = np.transpose(video_tensor, (3, 0, 1, 2))
    video_tensor = torch.from_numpy(video_tensor).float()
    
    # Add batch dimension
    video_tensor = video_tensor.unsqueeze(0)
    
    return video_tensor


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    
    if model_name == 'resnet3d_18':
        model = resnet3d_18(num_classes=num_classes)
    elif model_name == 'resnet3d_34':
        model = resnet3d_34(num_classes=num_classes)
    elif model_name == 'resnet3d_50':
        model = resnet3d_50(num_classes=num_classes)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f'Model: {model_name}')
    print(f'Checkpoint epoch: {checkpoint["epoch"]}')
    
    # Load video
    print(f'Loading video: {args.video}')
    video_tensor = load_video(
        args.video,
        num_frames=config['data']['num_frames'],
        spatial_size=config['data']['spatial_size'],
        temporal_stride=config['data']['temporal_stride']
    )
    video_tensor = video_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5, dim=1)
        
        print('\nTop-5 Predictions:')
        for i in range(5):
            prob = top5_prob[0][i].item()
            idx = top5_idx[0][i].item()
            print(f'{i+1}. Class {idx}: {prob*100:.2f}%')


if __name__ == '__main__':
    main()
