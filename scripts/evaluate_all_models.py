"""Comprehensive evaluation script for all three 3D models with visualization"""
import os
import sys
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
    
    cap.release()
    
    if len(frames) != num_frames:
        return None
    
    return np.array(frames)


def preprocess_frames(frames):
    """Preprocess frames for model input"""
    frames = frames.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frames = (frames - mean) / std
    
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    
    return frames


def evaluate_model(model, video_dir, labels_file, label_map_file, device, num_samples=None):
    """Evaluate model and collect detailed statistics"""
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
    
    # Per-class accuracy
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    
    # Confusion data
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        pbar = tqdm(data, desc='Evaluating')
        for item in pbar:
            video_id = item['id']
            
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
            
            # Get label index from template
            label_text = item['template'].replace('[', '').replace(']', '')
            if label_text not in label_map:
                failed += 1
                continue
            label_idx = int(label_map[label_text])
            label_tensor = torch.tensor([label_idx]).to(device)
            
            # Forward pass
            outputs = model(inputs)
            acc1, acc5 = accuracy(outputs, label_tensor, topk=(1, 5))
            
            top1.update(acc1.item(), 1)
            top5.update(acc5.item(), 1)
            successful += 1
            
            # Get prediction
            _, pred = outputs.topk(1, 1, True, True)
            pred_idx = pred.item()
            
            predictions.append(pred_idx)
            ground_truths.append(label_idx)
            
            # Per-class stats
            per_class_total[label_text] += 1
            if pred_idx == label_idx:
                per_class_correct[label_text] += 1
            
            pbar.set_postfix({
                'top1': f'{top1.avg:.2f}',
                'top5': f'{top5.avg:.2f}',
                'success': successful,
                'failed': failed
            })
    
    return {
        'top1': top1.avg,
        'top5': top5.avg,
        'successful': successful,
        'failed': failed,
        'per_class_correct': dict(per_class_correct),
        'per_class_total': dict(per_class_total),
        'predictions': predictions,
        'ground_truths': ground_truths
    }


def plot_comparison_charts(results, output_dir='evaluation_results'):
    """Generate comparison charts"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    model_names = list(results.keys())
    
    # 1. Accuracy Comparison Bar Chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    top1_scores = [results[m]['top1'] for m in model_names]
    top5_scores = [results[m]['top5'] for m in model_names]
    
    bars1 = ax.bar(x - width/2, top1_scores, width, label='Top-1 Accuracy', color='#3498db')
    bars2 = ax.bar(x + width/2, top5_scores, width, label='Top-5 Accuracy', color='#2ecc71')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Success/Failure Pie Charts
    fig, axes = plt.subplots(1, len(model_names), figsize=(15, 5))
    if len(model_names) == 1:
        axes = [axes]
    
    colors = ['#2ecc71', '#e74c3c']
    
    for idx, (model_name, ax) in enumerate(zip(model_names, axes)):
        successful = results[model_name]['successful']
        failed = results[model_name]['failed']
        
        ax.pie([successful, failed], labels=['Successful', 'Failed'], 
               autopct='%1.1f%%', colors=colors, startangle=90,
               textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title(f'{model_name}\n({successful} videos)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/success_failure.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Per-class accuracy for top/bottom performing classes
    for model_name, result in results.items():
        per_class_acc = {}
        for class_name, total in result['per_class_total'].items():
            correct = result['per_class_correct'].get(class_name, 0)
            per_class_acc[class_name] = (correct / total * 100) if total > 0 else 0
        
        # Sort by accuracy
        sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
        
        # Top 10 and Bottom 10
        top_10 = sorted_classes[:10]
        bottom_10 = sorted_classes[-10:]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top 10
        classes, accs = zip(*top_10) if top_10 else ([], [])
        classes = [c[:50] + '...' if len(c) > 50 else c for c in classes]
        y_pos = np.arange(len(classes))
        ax1.barh(y_pos, accs, color='#2ecc71')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(classes, fontsize=9)
        ax1.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax1.set_title(f'{model_name} - Top 10 Performing Classes', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(accs):
            ax1.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
        
        # Bottom 10
        classes, accs = zip(*bottom_10) if bottom_10 else ([], [])
        classes = [c[:50] + '...' if len(c) > 50 else c for c in classes]
        y_pos = np.arange(len(classes))
        ax2.barh(y_pos, accs, color='#e74c3c')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(classes, fontsize=9)
        ax2.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax2.set_title(f'{model_name} - Bottom 10 Performing Classes', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(accs):
            ax2.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(f'{output_dir}/{safe_name}_per_class.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n📊 Charts saved to {output_dir}/")


def print_beautiful_table(results):
    """Print beautiful ASCII tables"""
    
    # Main comparison table
    print("\n" + "="*80)
    print("🎯 MODEL PERFORMANCE COMPARISON".center(80))
    print("="*80)
    
    # Header
    print(f"\n{'Model':<30} {'Top-1 Acc':<15} {'Top-5 Acc':<15} {'Success':<10} {'Failed':<10}")
    print("-"*80)
    
    # Data rows
    for model_name, result in results.items():
        print(f"{model_name:<30} {result['top1']:>6.2f}%{'':<8} {result['top5']:>6.2f}%{'':<8} "
              f"{result['successful']:>6}{'':<4} {result['failed']:>6}{'':<4}")
    
    print("="*80)
    
    # Best model
    best_model = max(results.items(), key=lambda x: x[1]['top1'])
    print(f"\n🏆 Best Model: {best_model[0]} with {best_model[1]['top1']:.2f}% Top-1 Accuracy")
    
    # Detailed statistics for each model
    for model_name, result in results.items():
        print(f"\n{'='*80}")
        print(f"📈 DETAILED STATISTICS: {model_name}".center(80))
        print(f"{'='*80}")
        
        print(f"\n  Overall Performance:")
        print(f"    • Top-1 Accuracy: {result['top1']:.2f}%")
        print(f"    • Top-5 Accuracy: {result['top5']:.2f}%")
        print(f"    • Successfully Evaluated: {result['successful']} videos")
        print(f"    • Failed: {result['failed']} videos")
        
        # Per-class statistics
        if result['per_class_total']:
            per_class_acc = {}
            for class_name, total in result['per_class_total'].items():
                correct = result['per_class_correct'].get(class_name, 0)
                per_class_acc[class_name] = (correct / total * 100) if total > 0 else 0
            
            sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n  Top 5 Best Performing Classes:")
            for i, (class_name, acc) in enumerate(sorted_classes[:5], 1):
                short_name = (class_name[:60] + '...') if len(class_name) > 60 else class_name
                print(f"    {i}. {short_name:<63} {acc:>6.2f}%")
            
            print(f"\n  Top 5 Worst Performing Classes:")
            for i, (class_name, acc) in enumerate(sorted_classes[-5:], 1):
                short_name = (class_name[:60] + '...') if len(class_name) > 60 else class_name
                print(f"    {i}. {short_name:<63} {acc:>6.2f}%")
    
    print(f"\n{'='*80}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate all three 3D models')
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
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to use (cuda/mps/cpu)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"🖥️  Using device: {device}")
    
    # Model configurations
    models = [
        {
            'name': 'From Scratch',
            'config': 'configs/config_3d.yaml',
            'checkpoint': 'ssh_checkpoints/checkpoints/best_3d_model.pth'
        },
        {
            'name': 'Kinetics Pretrained',
            'config': 'configs/config_3d_pretrained.yaml',
            'checkpoint': 'ssh_checkpoints/checkpoints_pretrained/best_3d_model.pth'
        },
        {
            'name': '2D-Inflated',
            'config': 'configs/config_3d_2d_inflated.yaml',
            'checkpoint': 'ssh_checkpoints/checkpoints_2d_inflated/best_3d_model.pth'
        }
    ]
    
    results = {}
    
    # Evaluate each model
    for model_info in models:
        print(f"\n{'='*80}")
        print(f"🔍 Evaluating: {model_info['name']}".center(80))
        print(f"{'='*80}")
        
        # Load config
        import yaml
        with open(model_info['config'], 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        from src.models.resnet3d import create_3d_model
        model = create_3d_model(config)
        
        # Load checkpoint
        checkpoint = torch.load(model_info['checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"✅ Loaded checkpoint from {model_info['checkpoint']}")
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch'] + 1}")
        if 'val_acc1' in checkpoint:
            print(f"   Checkpoint validation accuracy: {checkpoint['val_acc1']:.2f}%")
        
        # Evaluate
        result = evaluate_model(
            model, args.video_dir, args.labels, args.label_map, 
            device, args.num_samples
        )
        
        results[model_info['name']] = result
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print tables
    print_beautiful_table(results)
    
    # Generate charts
    plot_comparison_charts(results, args.output_dir)
    
    # Save results to JSON
    os.makedirs(args.output_dir, exist_ok=True)
    results_json = {
        name: {
            'top1': res['top1'],
            'top5': res['top5'],
            'successful': res['successful'],
            'failed': res['failed'],
            'per_class_accuracy': {
                cls: (res['per_class_correct'].get(cls, 0) / total * 100)
                for cls, total in res['per_class_total'].items()
            }
        }
        for name, res in results.items()
    }
    
    with open(f'{args.output_dir}/results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"💾 Results saved to {args.output_dir}/results.json")
    print(f"\n✨ Evaluation complete! Check {args.output_dir}/ for all outputs.\n")


if __name__ == '__main__':
    main()
