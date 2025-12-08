# Quick Start Guide

This guide will help you get up and running quickly with the Something-Something V2 project.

## 1. Environment Setup (5 minutes)

```bash
# Clone and navigate to the repository
cd something-something-IV

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Dataset Preparation

### Option A: Full Dataset (Recommended for final results)

1. Register and download from: https://developer.qualcomm.com/software/ai-datasets/something-something
2. Extract frames (or download pre-extracted frames)
3. Place in `data/something-something-v2/` following the structure in README.md

### Option B: Subset for Testing (Quick start)

For initial testing, you can:
1. Use a small subset of videos (e.g., first 1000 training videos)
2. Modify the JSON files to only include these videos
3. Update configs to use this subset

## 3. Configuration

Edit `configs/config_2d.yaml` and `configs/config_3d.yaml`:

```yaml
data:
  dataset_path: "./data/something-something-v2"  # Update this path
  batch_size: 32  # Adjust based on your GPU memory
  num_workers: 4   # Adjust based on your CPU cores
```

## 4. Training Workflow

### Step 1: Train 2D Baseline (1-2 days on single GPU)

```bash
# Train on single frames
python scripts/train_2d.py --config configs/config_2d.yaml --device cuda

# Monitor training
tensorboard --logdir runs/
```

**Expected time**: 1-2 days for 50 epochs on full dataset

### Step 2: Train 3D Model (2-3 days on single GPU)

**Option A: From scratch**
```bash
python scripts/train_3d.py --config configs/config_3d.yaml --device cuda
```

**Option B: With 2D pretraining (recommended)**
```bash
python scripts/train_3d.py \
    --config configs/config_3d.yaml \
    --pretrained_2d checkpoints/best_2d_model.pth \
    --device cuda
```

**Expected time**: 2-3 days for 50 epochs on full dataset

### Step 3: Generate Grad-CAM Visualizations (30 minutes)

```bash
# 2D model visualizations
python scripts/visualize_gradcam.py \
    --model_type 2d \
    --config configs/config_2d.yaml \
    --checkpoint checkpoints/best_2d_model.pth \
    --num_samples 50 \
    --output_dir outputs/gradcam_2d

# 3D model visualizations
python scripts/visualize_gradcam.py \
    --model_type 3d \
    --config configs/config_3d.yaml \
    --checkpoint checkpoints/best_3d_model.pth \
    --num_samples 50 \
    --output_dir outputs/gradcam_3d
```

### Step 4: Evaluate Models

```bash
# Evaluate 2D model
python scripts/evaluate.py \
    --model_type 2d \
    --config configs/config_2d.yaml \
    --checkpoint checkpoints/best_2d_model.pth

# Evaluate 3D model
python scripts/evaluate.py \
    --model_type 3d \
    --config configs/config_3d.yaml \
    --checkpoint checkpoints/best_3d_model.pth
```

## 5. Quick Debugging Tips

### If you're short on time or resources:

1. **Reduce dataset size**: Edit JSON files to use subset
2. **Reduce epochs**: Set `training.epochs: 10` in config
3. **Smaller batch size**: Set `data.batch_size: 8` or `4`
4. **Smaller image size**: Set `data.img_size: 112` for 2D model
5. **Fewer frames**: Set `data.num_frames: 8` for 3D model

### Test with toy data:

```bash
# Create a minimal test with just a few videos
# Edit configs/config_2d.yaml:
training:
  epochs: 2  # Just 2 epochs for testing

data:
  batch_size: 4
  num_frames: 1
  img_size: 112
```

## 6. Common Issues and Solutions

### Out of Memory Error
- Reduce `batch_size` in config
- Reduce `img_size` in config
- For 3D: reduce `num_frames`

### "Import torch could not be resolved"
- These are Pylance warnings, ignore if torch is installed
- Check with: `python -c "import torch; print(torch.__version__)"`

### Slow data loading
- Reduce `num_workers` if CPU bottleneck
- Ensure data is on fast storage (SSD)
- Consider caching frames

### Training not improving
- Check learning rate (try 1e-3, 1e-4)
- Verify data augmentation isn't too aggressive
- For Something-Something: don't use horizontal flip!

## 7. Expected Timeline

**Minimum viable project (1 week)**:
- Day 1: Setup + small subset training
- Day 2-3: 2D model on subset
- Day 4-5: 3D model on subset
- Day 6: Grad-CAM visualizations
- Day 7: Analysis and report

**Full project (2-3 weeks)**:
- Week 1: 2D model training + evaluation
- Week 2: 3D model training (scratch + pretrained)
- Week 3: Grad-CAM analysis + report writing

## 8. What to Include in Report

### Minimum Requirements:
- 2D baseline results (accuracy, sample Grad-CAMs)
- 3D model results (accuracy, temporal Grad-CAMs)
- Comparison of attention patterns
- Discussion of findings

### For Stronger Report:
- Multiple model architectures compared
- Transfer learning ablation study
- Quantitative analysis of Grad-CAM
- Failure case analysis
- Class-specific attention patterns

## 9. Files You'll Generate

After running everything, you should have:

```
checkpoints/
├── best_2d_model.pth
├── best_3d_model.pth
├── checkpoint_2d_epoch_*.pth
└── checkpoint_3d_epoch_*.pth

outputs/
├── gradcam_2d/
│   ├── sample_0_class_*.png
│   └── ...
└── gradcam_3d/
    ├── sample_0_class_*.png
    └── ...

runs/  # TensorBoard logs
```

## 10. Next Steps

1. **Analyze results**: Look at Grad-CAM visualizations
2. **Find patterns**: What do 2D vs 3D models attend to?
3. **Select examples**: Choose interesting cases for report
4. **Compare**: 2D vs 3D, correct vs incorrect, easy vs hard
5. **Write**: Document your findings

## Need Help?

Check the main README.md for:
- Detailed API documentation
- Architecture details
- Mathematical formulations
- References and citations

Good luck! 🚀
