# Quick Start Guide

This guide will help you get started with training 3D CNNs on the Something-Something V2 dataset.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU with at least 8GB VRAM (16GB recommended)
- 50GB+ free disk space for dataset
- 16GB+ RAM

## Step 1: Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/RasmusArnmark/something-something-IV.git
cd something-something-IV
pip install -r requirements.txt
```

## Step 2: Dataset Setup

1. Download the Something-Something V2 dataset from [the official website](https://developer.qualcomm.com/software/ai-datasets/something-something)

2. Extract and organize the dataset:

```bash
# Create data directory
mkdir -p data/videos data/annotations

# Extract videos to data/videos/
# Place annotation files in data/annotations/

# Your structure should look like:
# data/
# ├── videos/
# │   ├── 1.webm
# │   ├── 2.webm
# │   └── ...
# └── annotations/
#     ├── something-something-v2-train.json
#     ├── something-something-v2-validation.json
#     └── something-something-v2-labels.json
```

## Step 3: Verify Installation

Test that the models work correctly:

```bash
python test_model.py
```

You should see output like:
```
Testing ResNet3D-18...
Input shape: torch.Size([2, 3, 16, 224, 224])
Output shape: torch.Size([2, 174])
✓ ResNet3D-18 test passed!
```

## Step 4: Start Training

### Option A: Train with default settings (ResNet3D-18)

```bash
python train.py --data-root data/ --config configs/resnet3d_18.yaml
```

### Option B: Train with different model

For ResNet3D-34 (better accuracy, more memory):
```bash
python train.py --data-root data/ --config configs/resnet3d_34.yaml
```

For ResNet3D-50 (best accuracy, most memory):
```bash
python train.py --data-root data/ --config configs/resnet3d_50.yaml
```

### Option C: Train with custom GPU

```bash
python train.py --data-root data/ --config configs/resnet3d_18.yaml --gpu 1
```

## Step 5: Monitor Training

### View logs
```bash
tail -f logs/train.log
```

### View TensorBoard
```bash
tensorboard --logdir runs
```

Then open http://localhost:6006 in your browser to see:
- Training/validation loss
- Top-1 and Top-5 accuracy
- Learning rate schedule

## Step 6: Run Inference

After training, test your model on a video:

```bash
python inference.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/best_model.pth \
    --config configs/resnet3d_18.yaml
```

Output:
```
Top-5 Predictions:
1. Class 42: 87.32%
2. Class 15: 5.21%
3. Class 89: 3.12%
4. Class 7: 2.01%
5. Class 134: 1.45%
```

## Training Tips

### If you run out of memory:

1. **Reduce batch size** in config file:
   ```yaml
   training:
     batch_size: 4  # or even 2
   ```

2. **Reduce number of frames**:
   ```yaml
   data:
     num_frames: 8  # instead of 16
   ```

3. **Use smaller model**: Start with ResNet3D-18

4. **Enable gradient checkpointing** (modify train.py if needed)

### For faster training:

1. **Use multiple workers**:
   ```yaml
   training:
     num_workers: 8  # adjust based on CPU cores
   ```

2. **Use mixed precision training** (requires PyTorch AMP)

3. **Increase batch size** if you have enough memory

### Resume training:

```bash
python train.py \
    --data-root data/ \
    --config configs/resnet3d_18.yaml \
    --resume checkpoints/checkpoint_epoch_10.pth
```

## Expected Results

Training times (approximate, on NVIDIA RTX 3090):
- **ResNet3D-18**: ~24 hours for 50 epochs
- **ResNet3D-34**: ~36 hours for 50 epochs
- **ResNet3D-50**: ~48 hours for 50 epochs

Expected validation accuracy (after 50 epochs):
- **ResNet3D-18**: ~45-50% Top-1
- **ResNet3D-34**: ~48-53% Top-1
- **ResNet3D-50**: ~50-55% Top-1

Note: These are rough estimates. Actual results depend on:
- Hyperparameter tuning
- Data augmentation
- Training duration
- Hardware specifications

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch_size in config file

### Issue: "Video not found"
**Solution**: Check that video files are in `data/videos/` and have correct extensions (.webm or .mp4)

### Issue: "No module named 'src'"
**Solution**: Make sure you're running from the repository root directory

### Issue: Training is too slow
**Solution**: 
- Increase num_workers in config
- Use SSD instead of HDD for dataset
- Use smaller spatial_size (e.g., 112 instead of 224)

### Issue: Loss is NaN
**Solution**:
- Reduce learning_rate in config
- Check for corrupted videos in dataset
- Enable gradient clipping

## Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, batch sizes, etc.
2. **Data augmentation**: Add spatial/temporal augmentation in the dataset loader
3. **Model improvements**: Try other architectures like I3D or SlowFast
4. **Mixed precision**: Implement FP16 training for faster training
5. **Multi-GPU training**: Distribute training across multiple GPUs

## Additional Resources

- [2D vs 3D Convolutions](2D_vs_3D_convolutions.md) - Detailed comparison
- [Something-Something V2 Paper](https://arxiv.org/abs/1706.04261)
- [C3D Paper](https://arxiv.org/abs/1412.0767)
- [I3D Paper](https://arxiv.org/abs/1705.07750)

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review the detailed README.md
3. Open an issue on GitHub with:
   - Error message
   - Full command you ran
   - Your environment (OS, Python version, GPU, etc.)

Happy training! 🚀
