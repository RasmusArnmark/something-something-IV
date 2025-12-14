"""Evaluation script that extracts frames on-the-fly from local videos"""
import os
import sys
import argparse
import yaml
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.resnet3d import create_3d_model
from src.utils.metrics import AverageMeter, accuracy


def extract_frames_from_video(video_path, num_frames=16):
    """Extract frames uniformly from video"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # Uniform sampling
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to 112x112
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
    
    cap.release()
    
    if len(frames) != num_frames:
        return None
    
    return np.array(frames)


def preprocess_frames(frames):
    """Preprocess frames for model input"""
    # Convert to float and normalize
    frames = frames.astype(np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frames = (frames - mean) / std
    
    # Convert to tensor (T, H, W, C) -> (C, T, H, W)
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    
    return frames


def evaluate(model, video_dir, labels_file, label_map_file, device, num_samples=None):
    """Evaluate model on videos"""
    model.eval()
    
    # Load label mapping
    with open(label_map_file, 'r') as f:
        label_map = json.load(f)
    
    # Load labels
    with open(labels_file, 'r') as f:
        data = json.load(f)
    
    if num_samples:
        data = data[:num_samples]
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    successful = 0
    failed = 0
    
    with torch.no_grad():
        pbar = tqdm(data, desc='Evaluating')
        for item in pbar:
            video_id = item['id']
            label = item['template'].replace('[', '').replace(']', '')
            
            # Find video file
            video_path = Path(video_dir) / f"{video_id}.webm"
            if not video_path.exists():
                failed += 1
                continue
            
            # Extract frames
            frames = extract_frames_from_video(video_path, num_frames=16)
            if frames is None:
                failed += 1
                continue
            
            # Preprocess
            inputs = preprocess_frames(frames).unsqueeze(0).to(device)
            
            # Get label index from template (same as dataset loader)
            label_text = item['template'].replace('[', '').replace(']', '')
            if label_text not in label_map:
                failed += 1
                continue
            label_idx = int(label_map[label_text])
            label_idx = torch.tensor([label_idx]).to(device)
            
            # Forward pass
            outputs = model(inputs)
            acc1, acc5 = accuracy(outputs, label_idx, topk=(1, 5))
            
            top1.update(acc1.item(), 1)
            top5.update(acc5.item(), 1)
            successful += 1
            
            pbar.set_postfix({
                'top1': f'{top1.avg:.2f}',
                'top5': f'{top5.avg:.2f}',
                'success': successful,
                'failed': failed
            })
    
    return top1.avg, top5.avg, successful, failed


def main():
    parser = argparse.ArgumentParser(description='Evaluate model locally with on-the-fly frame extraction')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--video_dir', type=str, 
                        default='data/videos/20bn-something-something-v2',
                        help='Directory containing videos')
    parser.add_argument('--labels', type=str,
                        default='data/labels_filtered/validation_filtered.json',
                        help='Path to labels file')
    parser.add_argument('--label_map', type=str,
                        default='data/labels_filtered/labels.json',
                        help='Path to label mapping file')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/mps/cpu)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create model
    model = create_3d_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded checkpoint from {args.checkpoint}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch'] + 1}")
    if 'val_acc1' in checkpoint:
        print(f"Checkpoint validation accuracy: {checkpoint['val_acc1']:.2f}%")
    
    # Evaluate
    print(f"\nEvaluating on videos from {args.video_dir}")
    print(f"Labels from {args.labels}")
    if args.num_samples:
        print(f"Evaluating on {args.num_samples} samples")
    
    top1, top5, successful, failed = evaluate(
        model, args.video_dir, args.labels, args.label_map, device, args.num_samples
    )
    
    print(f"\nResults:")
    print(f"Successfully processed: {successful} videos")
    print(f"Failed: {failed} videos")
    print(f"Top-1 Accuracy: {top1:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")


if __name__ == '__main__':
    main()
