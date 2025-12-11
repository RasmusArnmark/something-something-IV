"""Data loading utilities for Something-Something V2"""
import os
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class SomethingSomethingV2Dataset(Dataset):
    """Dataset class for Something-Something V2"""
    
    def __init__(self, images_dir, json_path, img_size=224, num_frames=1, split='train', transform=None):
        """
        Args:
            images_dir: Path to directory containing frame images
            json_path: Path to JSON file with video metadata
            img_size: Image size for resizing
            num_frames: Number of frames to sample from each video
            split: 'train' or 'val'
            transform: Image transformations
        """
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        self.num_frames = num_frames
        self.split = split
        self.transform = transform
        
        # Load video metadata
        with open(json_path, 'r') as f:
            self.video_list = json.load(f)
        
        # Create label mapping
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Build class mapping from video labels
        classes = set()
        for item in self.video_list:
            classes.add(item['template'].replace('[', '').replace(']', ''))
        
        for idx, cls in enumerate(sorted(classes)):
            self.class_to_idx[cls] = idx
            self.idx_to_class[idx] = cls
    
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx):
        item = self.video_list[idx]
        video_id = item['id']
        label = item['template'].replace('[', '').replace(']', '')
        label_idx = self.class_to_idx[label]
        
        # Load frames
        video_dir = self.images_dir / str(video_id)
        frame_files = sorted(list(video_dir.glob('*.jpg')))
        
        if len(frame_files) == 0:
            # Return dummy data if frames not found
            frames = torch.zeros(3, self.img_size, self.img_size)
        else:
            # Sample frames uniformly
            if len(frame_files) >= self.num_frames:
                indices = [int(i * len(frame_files) / self.num_frames) for i in range(self.num_frames)]
            else:
                indices = list(range(len(frame_files)))
            
            frames = []
            for i in indices:
                if i < len(frame_files):
                    img = Image.open(frame_files[i]).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    else:
                        img = transforms.ToTensor()(img)
                    frames.append(img)
            
            # Pad if necessary
            while len(frames) < self.num_frames:
                frames.append(torch.zeros(3, self.img_size, self.img_size))
            
            # Stack frames - if num_frames is 1, just return single frame; otherwise stack
            if self.num_frames == 1:
                frames = frames[0]
            else:
                frames = torch.stack(frames)
        
        metadata = {
            'video_id': video_id,
            'label': label
        }
        
        return frames, label_idx, metadata


def create_dataloaders(config):
    """Create train and validation dataloaders"""
    
    data_cfg = config['data']
    aug_cfg = config['augmentation']
    
    dataset_path = Path(data_cfg['dataset_path'])
    train_dir = dataset_path / data_cfg['train_dir']
    val_dir = dataset_path / data_cfg['val_dir']
    train_json = dataset_path / data_cfg['json_train']
    val_json = dataset_path / data_cfg['json_val']
    
    # Define transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if aug_cfg['normalize']:
        train_transform = transforms.Compose([
            transforms.Resize((data_cfg['img_size'], data_cfg['img_size'])),
            transforms.RandomCrop(data_cfg['img_size']) if aug_cfg['random_crop'] else transforms.CenterCrop(data_cfg['img_size']),
            transforms.RandomHorizontalFlip() if aug_cfg['horizontal_flip'] else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2) if aug_cfg['color_jitter'] else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            normalize
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((data_cfg['img_size'], data_cfg['img_size'])),
            transforms.CenterCrop(data_cfg['img_size']),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((data_cfg['img_size'], data_cfg['img_size'])),
            transforms.ToTensor()
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((data_cfg['img_size'], data_cfg['img_size'])),
            transforms.ToTensor()
        ])
    
    # Create datasets
    train_dataset = SomethingSomethingV2Dataset(
        images_dir=train_dir,
        json_path=train_json,
        img_size=data_cfg['img_size'],
        num_frames=data_cfg['num_frames'],
        split='train',
        transform=train_transform
    )
    
    val_dataset = SomethingSomethingV2Dataset(
        images_dir=val_dir,
        json_path=val_json,
        img_size=data_cfg['img_size'],
        num_frames=data_cfg['num_frames'],
        split='val',
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        num_workers=data_cfg['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader
