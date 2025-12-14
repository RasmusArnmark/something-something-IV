#!/usr/bin/env python3
"""Filter train/validation JSON files to only include videos with extracted frames"""
import json
import os
from pathlib import Path

def filter_labels(frames_dir, labels_dir, output_dir):
    """Filter label files to only include videos that have extracted frames"""
    
    # Get list of video IDs that have extracted frames
    frames_path = Path(frames_dir)
    extracted_video_ids = set()
    
    print(f"Scanning {frames_dir} for extracted videos...")
    for video_dir in frames_path.iterdir():
        if video_dir.is_dir():
            extracted_video_ids.add(video_dir.name)
    
    print(f"Found {len(extracted_video_ids):,} extracted videos")
    
    # Filter train.json
    train_json = Path(labels_dir) / 'train.json'
    if train_json.exists():
        print(f"\nFiltering {train_json}...")
        with open(train_json, 'r') as f:
            train_data = json.load(f)
        
        filtered_train = [item for item in train_data if item['id'] in extracted_video_ids]
        
        output_train = Path(output_dir) / 'train_filtered.json'
        with open(output_train, 'w') as f:
            json.dump(filtered_train, f)
        
        print(f"  Original: {len(train_data):,} samples")
        print(f"  Filtered: {len(filtered_train):,} samples")
        print(f"  Saved to: {output_train}")
    
    # Filter validation.json
    val_json = Path(labels_dir) / 'validation.json'
    if val_json.exists():
        print(f"\nFiltering {val_json}...")
        with open(val_json, 'r') as f:
            val_data = json.load(f)
        
        filtered_val = [item for item in val_data if item['id'] in extracted_video_ids]
        
        output_val = Path(output_dir) / 'validation_filtered.json'
        with open(output_val, 'w') as f:
            json.dump(filtered_val, f)
        
        print(f"  Original: {len(val_data):,} samples")
        print(f"  Filtered: {len(filtered_val):,} samples")
        print(f"  Saved to: {output_val}")
    
    # Copy labels.json as-is
    labels_json = Path(labels_dir) / 'labels.json'
    if labels_json.exists():
        import shutil
        output_labels = Path(output_dir) / 'labels.json'
        shutil.copy(labels_json, output_labels)
        print(f"\nCopied {labels_json} to {output_labels}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Filter label files based on extracted frames')
    parser.add_argument('--frames_dir', type=str, required=True,
                        help='Directory containing extracted frames')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Directory containing original label JSON files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save filtered label JSON files')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    filter_labels(args.frames_dir, args.labels_dir, args.output_dir)
    print("\n✓ Done!")
