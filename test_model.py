"""
Simple test script to verify model architecture and forward pass.
"""

import torch
from src.models import resnet3d_18, resnet3d_34, resnet3d_50


def test_model(model, model_name):
    """Test a model with dummy input"""
    print(f"\nTesting {model_name}...")
    
    # Create dummy input: (batch_size, channels, depth, height, width)
    batch_size = 2
    channels = 3
    depth = 16  # number of frames
    height = 224
    width = 224
    
    dummy_input = torch.randn(batch_size, channels, depth, height, width)
    
    # Set model to eval mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Verify output shape
    assert output.shape == (batch_size, 174), f"Expected output shape (2, 174), got {output.shape}"
    print(f"✓ {model_name} test passed!")
    
    return True


def main():
    print("="*60)
    print("Testing 3D ResNet Models for Something-Something V2")
    print("="*60)
    
    # Test ResNet3D-18
    model_18 = resnet3d_18(num_classes=174)
    test_model(model_18, "ResNet3D-18")
    
    # Test ResNet3D-34
    model_34 = resnet3d_34(num_classes=174)
    test_model(model_34, "ResNet3D-34")
    
    # Test ResNet3D-50
    model_50 = resnet3d_50(num_classes=174)
    test_model(model_50, "ResNet3D-50")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


if __name__ == '__main__':
    main()
