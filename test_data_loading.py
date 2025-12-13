#!/usr/bin/env python3
"""Quick test script to diagnose data loading"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 50)
print("Testing data loading...")
print("=" * 50)

import yaml
print("✓ YAML loaded")

import torch
print(f"✓ PyTorch loaded (version: {torch.__version__})")

from data.dataset import create_dataloaders
print("✓ Dataset module imported")

# Load config
config_path = 'configs/config_2d_test.yaml'
print(f"\nLoading config from {config_path}...")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
print("✓ Config loaded")

print(f"\nFrames path: {config['data']['frames_path']}")
print(f"Labels path: {config['data']['labels_path']}")
print(f"Batch size: {config['data']['batch_size']}")

print("\nCreating dataloaders (this may take a while)...")
try:
    train_loader, val_loader = create_dataloaders(config)
    print(f"✓ Train loader created: {len(train_loader.dataset)} samples")
    print(f"✓ Val loader created: {len(val_loader.dataset)} samples")
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    print("\nTesting first batch...")
    for batch_idx, (inputs, labels, metadata) in enumerate(train_loader):
        print(f"✓ First batch loaded:")
        print(f"  - Input shape: {inputs.shape}")
        print(f"  - Labels shape: {labels.shape}")
        print(f"  - Sample IDs: {metadata['id'][:3]}")
        break
    
    print("\n" + "=" * 50)
    print("✓ All tests passed! Ready to train.")
    print("=" * 50)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
