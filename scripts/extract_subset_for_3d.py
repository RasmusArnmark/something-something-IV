#!/usr/bin/env python3
"""
Extract a subset of videos with more frames for 3D training
"""
import os
import sys
import argparse
import json
from pathlib import Path

def create_subset_labels(labels_dir, output_dir, num_videos, seed=42):
    """
    Create a subset of training labels for 3D training
    
    Args:
        labels_dir: Directory containing original label files
        output_dir: Output directory for subset labels
        num_videos: Number of videos to include in subset
        seed: Random seed for reproducibility
    """
    import random
    random.seed(seed)
    
    # Load full training labels
    train_json = Path(labels_dir) / 'train.json'
    val_json = Path(labels_dir) / 'validation.json'
    labels_json = Path(labels_dir) / 'labels.json'
    
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    
    # Randomly sample videos
    train_subset = random.sample(train_data, min(num_videos, len(train_data)))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save subset
    output_train = Path(output_dir) / 'train_3d_subset.json'
    with open(output_train, 'w') as f:
        json.dump(train_subset, f)
    
    # Copy validation and labels as-is
    with open(val_json, 'r') as f:
        val_data = json.load(f)
    output_val = Path(output_dir) / 'validation_3d_subset.json'
    with open(output_val, 'w') as f:
        json.dump(val_data[:2500], f)  # Use same 2500 val videos
    
    import shutil
    shutil.copy(labels_json, Path(output_dir) / 'labels.json')
    
    print(f"Created 3D subset:")
    print(f"  Train: {len(train_subset):,} videos")
    print(f"  Val: 2,500 videos")
    print(f"  Saved to: {output_dir}")
    
    # Save video IDs for frame extraction
    video_ids = [item['id'] for item in train_subset]
    output_ids = Path(output_dir) / 'train_video_ids.txt'
    with open(output_ids, 'w') as f:
        f.write('\n'.join(video_ids))
    print(f"  Video IDs saved to: {output_ids}")
    
    return video_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create subset for 3D training')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Directory containing label JSON files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for subset labels')
    parser.add_argument('--num_videos', type=int, default=20000,
                        help='Number of training videos to include')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    create_subset_labels(args.labels_dir, args.output_dir, args.num_videos, args.seed)
    print("\n✓ Done! Now extract frames with:")
    print(f"  python scripts/extract_frames.py \\")
    print(f"    --videos_dir data/videos/20bn-something-something-v2 \\")
    print(f"    --output_dir data/images_3d \\")
    print(f"    --frames_per_video 16 \\")
    print(f"    --num_workers 8 \\")
    print(f"    --video_list {args.output_dir}/train_video_ids.txt")
