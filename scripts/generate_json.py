"""Generate JSON metadata files from image directories"""
import json
from pathlib import Path
import os

def generate_json_from_dirs(images_dir, output_json):
    """
    Generate JSON file from image directory structure
    Assumes each subdirectory is a video ID
    """
    images_path = Path(images_dir)
    videos = []
    
    # Get all subdirectories (video IDs)
    video_dirs = [d for d in images_path.iterdir() if d.is_dir()]
    
    for i, video_dir in enumerate(sorted(video_dirs)):
        video_id = video_dir.name
        # Count frames
        frames = list(video_dir.glob('*.jpg'))
        
        if len(frames) > 0:
            # Create dummy label - you may need to adjust this based on actual labels
            videos.append({
                'id': video_id,
                'template': '[action_placeholder]',  # Placeholder
                'num_frames': len(frames)
            })
    
    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(videos, f, indent=2)
    
    print(f"Generated {output_json} with {len(videos)} videos")
    return videos

if __name__ == '__main__':
    # Generate train.json
    train_videos = generate_json_from_dirs(
        'data/images_new',
        'data/videos/labels/train.json'
    )
    
    # Generate val.json
    val_videos = generate_json_from_dirs(
        'data/images_val',
        'data/videos/labels/val.json'
    )
    
    print(f"Total: {len(train_videos)} train videos, {len(val_videos)} validation videos")
