"""
Training script for Something-Something V2 with 3D CNNs.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models import resnet3d_18, resnet3d_34, resnet3d_50
from src.data import SomethingSomethingV2
from src.utils import accuracy, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D CNN on Something-Something V2')
    parser.add_argument('--config', type=str, default='configs/resnet3d_18.yaml',
                        help='path to config file')
    parser.add_argument('--data-root', type=str, required=True,
                        help='path to dataset root directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use')
    return parser.parse_args()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (videos, labels) in enumerate(pbar):
        videos = videos.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(videos)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        
        # Update metrics
        total_loss += loss.item()
        total_acc1 += acc1.item()
        total_acc5 += acc5.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc1': f'{acc1.item():.2f}%',
            'acc5': f'{acc5.item():.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_acc1 = total_acc1 / len(train_loader)
    avg_acc5 = total_acc5 / len(train_loader)
    
    logger.info(f'Train Epoch {epoch}: Loss={avg_loss:.4f}, Acc@1={avg_acc1:.2f}%, Acc@5={avg_acc5:.2f}%')
    
    return avg_loss, avg_acc1, avg_acc5


def validate(model, val_loader, criterion, device, epoch, logger):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Validation')
        for videos, labels in pbar:
            videos = videos.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            
            # Update metrics
            total_loss += loss.item()
            total_acc1 += acc1.item()
            total_acc5 += acc5.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc1': f'{acc1.item():.2f}%',
                'acc5': f'{acc5.item():.2f}%'
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_acc1 = total_acc1 / len(val_loader)
    avg_acc5 = total_acc5 / len(val_loader)
    
    logger.info(f'Validation Epoch {epoch}: Loss={avg_loss:.4f}, Acc@1={avg_acc1:.2f}%, Acc@5={avg_acc5:.2f}%')
    
    return avg_loss, avg_acc1, avg_acc5


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    os.makedirs('logs', exist_ok=True)
    logger = setup_logger('train', log_file='logs/train.log')
    logger.info(f'Config: {config}')
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
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
    
    model = model.to(device)
    logger.info(f'Model: {model_name}')
    
    # Create datasets
    train_dataset = SomethingSomethingV2(
        data_root=args.data_root,
        split='train',
        num_frames=config['data']['num_frames'],
        spatial_size=config['data']['spatial_size'],
        temporal_stride=config['data']['temporal_stride']
    )
    
    val_dataset = SomethingSomethingV2(
        data_root=args.data_root,
        split='validation',
        num_frames=config['data']['num_frames'],
        spatial_size=config['data']['spatial_size'],
        temporal_stride=config['data']['temporal_stride']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['lr_step_size'],
        gamma=config['training']['lr_gamma']
    )
    
    # Setup tensorboard
    log_dir = config.get('training', {}).get('log_dir', 'runs')
    writer = SummaryWriter(log_dir=log_dir)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_acc1 = 0.0
    
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc1 = checkpoint['best_acc1']
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss, train_acc1, train_acc5 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )
        
        # Validate
        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion, device, epoch, logger
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train_top1', train_acc1, epoch)
        writer.add_scalar('Accuracy/train_top5', train_acc5, epoch)
        writer.add_scalar('Accuracy/val_top1', val_acc1, epoch)
        writer.add_scalar('Accuracy/val_top5', val_acc5, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc1': best_acc1,
            'config': config
        }
        
        torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch}.pth')
        
        if is_best:
            torch.save(checkpoint, 'checkpoints/best_model.pth')
            logger.info(f'Best model saved with Acc@1={best_acc1:.2f}%')
    
    writer.close()
    logger.info('Training completed!')


if __name__ == '__main__':
    main()
