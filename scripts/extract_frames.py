#!/usr/bin/env python3
"""
Script to extract frames from Something-Something V2 videos
Converts .webm videos to individual .jpg frames
"""
import os
import sys
import argparse
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def extract_frames_from_video(video_path, output_dir, max_frames=None, uniform_sample=True):
    """
    Extract frames from a single video file
    
    Args:
        video_path: Path to .webm video file
        output_dir: Directory to save extracted frames
        max_frames: Maximum number of frames to extract (None = all)
        uniform_sample: If True and max_frames is set, sample frames uniformly across video
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get total frame count if we need to sample uniformly
        if max_frames and uniform_sample:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > max_frames:
                # Calculate indices to sample uniformly
                frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
            else:
                # If video has fewer frames than requested, take all
                frame_indices = list(range(total_frames))
        else:
            frame_indices = None
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if we should save this frame
            should_save = True
            if frame_indices is not None:
                should_save = frame_count in frame_indices
            elif max_frames and saved_count >= max_frames:
                break
            
            if should_save:
                # Save frame as JPEG
                frame_filename = os.path.join(output_dir, f'{saved_count:05d}.jpg')
                cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        return saved_count > 0
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return False


def process_video_wrapper(args):
    """Wrapper for multiprocessing"""
    video_path, frames_dir, max_frames = args
    video_id = video_path.stem
    output_dir = frames_dir / video_id
    
    # Skip if already processed
    if output_dir.exists() and len(list(output_dir.glob('*.jpg'))) > 0:
        return True
    
    return extract_frames_from_video(video_path, output_dir, max_frames=max_frames)


def main():
    parser = argparse.ArgumentParser(description='Extract frames from Something-Something V2 videos')
    parser.add_argument('--videos_dir', type=str, required=True,
                        help='Directory containing .webm video files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save extracted frames')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Maximum number of videos to process (for testing)')
    parser.add_argument('--frames_per_video', type=int, default=None,
                        help='Maximum number of frames to extract per video (None = all frames). '
                             'Frames are sampled uniformly across the video.')
    parser.add_argument('--train_json', type=str, 
                        default='data/videos/labels/train.json',
                        help='Path to train.json file to filter training videos')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'validation', 'test', 'all'],
                        help='Which split to extract frames from (default: train)')
    args = parser.parse_args()
    
    videos_dir = Path(args.videos_dir)
    frames_dir = Path(args.output_dir)
    
    # Load train/validation/test split if specified
    train_video_ids = None
    if args.split != 'all':
        if args.split == 'train':
            split_file = Path(args.train_json)
        else:
            split_file = Path(args.train_json).parent / f"{args.split}.json"
        
        if split_file.exists():
            print(f"Loading {args.split} video IDs from {split_file}...")
            with open(split_file, 'r') as f:
                split_data = json.load(f)
                train_video_ids = set([item['id'] for item in split_data])
            print(f"Found {len(train_video_ids)} videos in {args.split} split")
        else:
            print(f"Warning: {split_file} not found, processing all videos")
    
    # Get list of video files
    video_files = list(videos_dir.glob('*.webm'))
    print(f"Found {len(video_files)} total video files")
    
    # Filter by split if specified
    if train_video_ids is not None:
        video_files = [vf for vf in video_files if vf.stem in train_video_ids]
        print(f"Filtered to {len(video_files)} videos in {args.split} split")
    
    if args.max_videos:
        video_files = video_files[:args.max_videos]
        print(f"Processing first {len(video_files)} videos")
    
    if args.frames_per_video:
        print(f"Extracting {args.frames_per_video} frames per video (uniformly sampled)")
    else:
        print("Extracting all frames from each video")
    
    # Create output directory
    os.makedirs(frames_dir, exist_ok=True)
    
    # Process videos in parallel
    print(f"Using {args.num_workers} workers...")
    
    # video_files already contains full Path objects from glob
    process_args = [(vf, frames_dir, args.frames_per_video) for vf in video_files]
    
    with mp.Pool(args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_video_wrapper, process_args),
            total=len(process_args),
            desc='Extracting frames'
        ))
    
    successful = sum(results)
    print(f"\nSuccessfully processed {successful}/{len(video_files)} videos")
    print(f"Frames saved to: {frames_dir}")


if __name__ == '__main__':
    main()
