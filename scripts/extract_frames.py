#!/usr/bin/env python3
"""
Script to extract frames from Something-Something V2 videos
Converts .webm videos to individual .jpg frames
"""
import os
import sys
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def extract_frames_from_video(video_path, output_dir, max_frames=None):
    """
    Extract frames from a single video file
    
    Args:
        video_path: Path to .webm video file
        output_dir: Directory to save extracted frames
        max_frames: Maximum number of frames to extract (None = all)
    
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
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            # Save frame as JPEG
            frame_filename = os.path.join(output_dir, f'{frame_count:05d}.jpg')
            cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_count += 1
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return False


def process_video_wrapper(args):
    """Wrapper for multiprocessing"""
    video_path, frames_dir = args
    video_id = video_path.stem
    output_dir = frames_dir / video_id
    
    # Skip if already processed
    if output_dir.exists() and len(list(output_dir.glob('*.jpg'))) > 0:
        return True
    
    return extract_frames_from_video(video_path, output_dir)


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
    args = parser.parse_args()
    
    videos_dir = Path(args.videos_dir)
    frames_dir = Path(args.output_dir)
    
    # Get list of video files
    video_files = list(videos_dir.glob('*.webm'))
    print(f"Found {len(video_files)} video files")
    
    if args.max_videos:
        video_files = video_files[:args.max_videos]
        print(f"Processing first {len(video_files)} videos")
    
    # Create output directory
    os.makedirs(frames_dir, exist_ok=True)
    
    # Process videos in parallel
    print(f"Extracting frames using {args.num_workers} workers...")
    
    # video_files already contains full Path objects from glob
    process_args = [(vf, frames_dir) for vf in video_files]
    
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
