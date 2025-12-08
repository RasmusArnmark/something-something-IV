"""Quick evaluation script to test model performance"""
import os
import sys
import argparse
import yaml
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.resnet2d import create_2d_model
from models.resnet3d import create_3d_model
from data.dataset import create_dataloaders
from utils.metrics import AverageMeter, accuracy


def evaluate(model, dataloader, device):
    """Evaluate model on a dataset"""
    model.eval()
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for inputs, labels, metadata in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            
            batch_size = inputs.size(0)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
    
    return top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model_type', type=str, required=True, choices=['2d', '3d'],
                        help='Model type (2d or 3d)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    if args.model_type == '2d':
        model = create_2d_model(config)
    else:
        model = create_3d_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded checkpoint from {args.checkpoint}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'val_acc1' in checkpoint:
        print(f"Checkpoint validation accuracy: {checkpoint['val_acc1']:.2f}%")
    
    # Create dataloader (validation set)
    _, val_loader = create_dataloaders(config)
    
    # Evaluate
    print("\nEvaluating model...")
    top1, top5 = evaluate(model, val_loader, device)
    
    print(f"\nResults:")
    print(f"Top-1 Accuracy: {top1:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")


if __name__ == '__main__':
    main()
