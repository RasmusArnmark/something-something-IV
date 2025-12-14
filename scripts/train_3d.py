"""Training script for 3D models on Something-Something V2"""
import os
import sys
import argparse
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.resnet3d import create_3d_model
from src.data import create_dataloaders
from src.utils.metrics import AverageMeter, accuracy


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
    config: dict
) -> tuple:
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (inputs, labels, metadata) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        
        # Update meters
        batch_size = inputs.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'top1': f'{top1.avg:.2f}',
            'top5': f'{top5.avg:.2f}'
        })
        
        # Log to wandb
        if batch_idx % config['logging']['print_freq'] == 0:
            wandb.log({
                'train/loss': losses.avg,
                'train/top1': top1.avg,
                'train/top5': top5.avg,
                'epoch': epoch,
                'batch': batch_idx
            })
    
    return losses.avg, top1.avg, top5.avg


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Validate the model"""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for inputs, labels, metadata in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Compute accuracy
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            
            # Update meters
            batch_size = inputs.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'top1': f'{top1.avg:.2f}',
                'top5': f'{top5.avg:.2f}'
            })
    
    return losses.avg, top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser(description='Train 3D model on Something-Something V2')
    parser.add_argument('--config', type=str, default='configs/config_3d.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained_2d', type=str, default=None,
                        help='Path to pretrained 2D model for weight inflation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--wandb_project', type=str, default='something-something-3d',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity (team/user) name')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override pretrained path if provided
    if args.pretrained_2d:
        config['model']['pretrained'] = args.pretrained_2d
        config['model']['inflate_from_2d'] = True
    
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=config,
        name=f"3d-{config['model']['name']}"
    )
    
    # Create model
    model = create_3d_model(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['momentum'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")
    
    # Learning rate scheduler
    if config['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    elif config['training']['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['epochs'] // 3,
            gamma=0.1
        )
    else:
        scheduler = None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc1 = 0.0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc1 = checkpoint.get('best_acc1', 0.0)
        print(f"Resumed from epoch {start_epoch}, best acc1: {best_acc1:.2f}")
    
    # Create checkpoint directory
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_acc1, train_acc5 = train_epoch(
            model, train_loader, criterion, optimizer, epoch, device, config
        )
        
        # Validate
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, device)
        
        print(f"Train - Loss: {train_loss:.4f}, Top-1: {train_acc1:.2f}%, Top-5: {train_acc5:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Top-1: {val_acc1:.2f}%, Top-5: {val_acc5:.2f}%")
        
        # Log to wandb
        wandb.log({
            'val/loss': val_loss,
            'val/top1': val_acc1,
            'val/top5': val_acc5,
            'train/loss': train_loss,
            'train/top1': train_acc1,
            'train/top5': train_acc5,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch
        })
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        
        if (epoch + 1) % config['logging']['save_freq'] == 0 or is_best:
            checkpoint_path = os.path.join(
                config['logging']['checkpoint_dir'],
                f'checkpoint_3d_epoch_{epoch + 1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc1': val_acc1,
                'val_acc5': val_acc5,
                'best_acc1': best_acc1,
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            if is_best:
                best_path = os.path.join(
                    config['logging']['checkpoint_dir'],
                    'best_3d_model.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc1': val_acc1,
                    'val_acc5': val_acc5,
                    'config': config
                }, best_path)
                print(f"New best model! Saved to {best_path}")
    
    print(f"\nTraining complete! Best Top-1 Accuracy: {best_acc1:.2f}%")
    
    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
