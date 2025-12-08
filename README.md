# Something-Something V2: 2D→3D Model Analysis with Grad-CAM

This repository implements a comprehensive research project for video understanding on the Something-Something V2 dataset, progressing from 2D baseline models to 3D spatiotemporal models with Grad-CAM interpretability analysis.

## 📋 Research Question

**"How does moving from 2D to 3D architectures affect what the model attends to (via Grad-CAM) on Something-Something V2, and does 2D pretraining help 3D performance/interpretability?"**

## 🎯 Project Overview

This project follows a structured approach:

1. **2D Baseline**: Train ResNet-18/50 on single frames or multi-frame with temporal pooling
2. **3D Model**: Train R3D/R(2+1)D on video clips with spatiotemporal convolutions
3. **Transfer Learning**: Inflate 2D pretrained weights to initialize 3D models
4. **Interpretability**: Visualize attention using Grad-CAM for both 2D and 3D models
5. **Analysis**: Compare what models focus on (objects, hands, motion) across architectures

## 📁 Project Structure

```
something-something-IV/
├── configs/                 # Configuration files
│   ├── config_2d.yaml      # 2D model config
│   └── config_3d.yaml      # 3D model config
├── src/                     # Source code
│   ├── data/               # Data loading and preprocessing
│   │   └── dataset.py      # Something-Something V2 dataset loader
│   ├── models/             # Model architectures
│   │   ├── resnet2d.py     # 2D ResNet models
│   │   └── resnet3d.py     # 3D ResNet models (R3D, R(2+1)D, I3D)
│   ├── gradcam/            # Grad-CAM implementations
│   │   └── gradcam.py      # Grad-CAM and Grad-CAM++
│   └── utils/              # Utility functions
│       └── metrics.py      # Training metrics and helpers
├── scripts/                # Training and evaluation scripts
│   ├── train_2d.py         # Train 2D models
│   ├── train_3d.py         # Train 3D models
│   └── visualize_gradcam.py # Generate Grad-CAM visualizations
├── notebooks/              # Jupyter notebooks for analysis
├── checkpoints/            # Model checkpoints
├── outputs/                # Visualization outputs
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Something-Something V2 dataset

### Installation

1. Clone the repository:
```bash
git clone https://github.com/RasmusArnmark/something-something-IV.git
cd something-something-IV
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Setup

1. Download Something-Something V2 dataset from [official website](https://developer.qualcomm.com/software/ai-datasets/something-something)

2. Extract frames from videos (or download pre-extracted frames)

3. Organize the dataset as follows:
```
data/something-something-v2/
├── train/              # Training video frames
│   ├── 1/              # Video ID folders
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   └── ...
│   └── ...
├── validation/         # Validation video frames
│   └── ...
├── train.json         # Training annotations
├── validation.json    # Validation annotations
└── labels.json        # Label mapping
```

4. Update dataset paths in configuration files (`configs/config_2d.yaml` and `configs/config_3d.yaml`)

## 📊 Training

### Stage 1: Train 2D Model

Train a 2D ResNet model on single frames or multi-frame with temporal pooling:

```bash
python scripts/train_2d.py --config configs/config_2d.yaml --device cuda
```

**Key parameters to adjust in `configs/config_2d.yaml`:**
- `model.name`: `resnet18` or `resnet50`
- `data.frame_sampling`: `single` (single frame) or `multi` (multiple frames)
- `data.num_frames`: Number of frames to sample (1 for single frame)
- `training.learning_rate`: Initial learning rate (e.g., 1e-3)

The best model will be saved to `checkpoints/best_2d_model.pth`.

### Stage 2: Train 3D Model

Train a 3D model from scratch or with 2D initialization:

**Option A: Train from scratch**
```bash
python scripts/train_3d.py --config configs/config_3d.yaml --device cuda
```

**Option B: Train with 2D pretrained weights (transfer learning)**
```bash
python scripts/train_3d.py \
    --config configs/config_3d.yaml \
    --pretrained_2d checkpoints/best_2d_model.pth \
    --device cuda
```

**Key parameters in `configs/config_3d.yaml`:**
- `model.name`: `r3d_18`, `r2plus1d_18`, or `i3d`
- `data.num_frames`: Number of frames per clip (e.g., 16)
- `data.img_size`: Spatial resolution (e.g., 112 for memory efficiency)
- `training.batch_size`: Smaller batch size for 3D (e.g., 4-8)

### Resume Training

To resume training from a checkpoint:
```bash
python scripts/train_2d.py --config configs/config_2d.yaml --resume checkpoints/checkpoint_2d_epoch_10.pth
```

## 🔍 Grad-CAM Visualization

Generate Grad-CAM visualizations to understand model attention:

### For 2D Models
```bash
python scripts/visualize_gradcam.py \
    --model_type 2d \
    --config configs/config_2d.yaml \
    --checkpoint checkpoints/best_2d_model.pth \
    --num_samples 20 \
    --output_dir outputs/gradcam_2d
```

### For 3D Models
```bash
python scripts/visualize_gradcam.py \
    --model_type 3d \
    --config configs/config_3d.yaml \
    --checkpoint checkpoints/best_3d_model.pth \
    --num_samples 20 \
    --output_dir outputs/gradcam_3d
```

This will generate visualizations showing:
- Original frames
- Grad-CAM heatmaps
- Overlayed visualizations
- For 3D: temporal evolution of attention across frames

## 📈 Experimental Design

### Experiments to Run

1. **Baseline 2D Model**
   - Train ResNet-18 on single frames
   - Evaluate top-1/top-5 accuracy
   - Generate Grad-CAM for correct predictions, failures, and ambiguous classes
   - Analyze: Does it focus on objects, hands, background?

2. **Spatiotemporal 3D Model**
   - Train R3D-18 on 16-frame clips
   - Evaluate accuracy
   - Generate temporal Grad-CAM visualizations
   - Compare attention patterns with 2D model

3. **2D→3D Transfer Learning**
   - Train 3D model initialized from 2D weights
   - Compare with 3D trained from scratch
   - Analyze: Does pretraining improve performance and interpretability?

4. **Comparative Analysis**
   - Visualize side-by-side comparisons of 2D vs 3D attention
   - Focus on temporal reasoning tasks (e.g., "pulling left to right")
   - Document differences in what models attend to

## 🎓 For Your Report

### Suggested Report Structure

1. **Introduction**
   - Motivation: Why temporal reasoning matters for Something-Something
   - Research question

2. **Related Work**
   - 2D vs 3D CNNs for video
   - Grad-CAM and interpretability
   - Transfer learning in video understanding

3. **Methodology**
   - Dataset description
   - 2D model architecture and training
   - 3D model architecture and weight inflation
   - Grad-CAM implementation (2D and 3D)

4. **Experiments**
   - Experimental setup
   - Baseline results (2D)
   - 3D model results
   - Transfer learning results

5. **Grad-CAM Analysis**
   - Qualitative visualization comparison
   - Case studies (correct, incorrect, ambiguous)
   - Attention patterns: static vs temporal cues

6. **Discussion**
   - What do models learn?
   - How does 3D improve over 2D?
   - Does pretraining help?
   - Limitations

7. **Conclusion**
   - Key findings
   - Future work

## 🛠️ Key Features

### Data Loading
- Flexible frame sampling (single, uniform, random)
- Support for 2D and 3D inputs
- Efficient video frame loading with caching
- Configurable augmentations (crop, flip, color jitter)

### Models
- **2D**: ResNet-18/50 with optional temporal pooling
- **3D**: R3D-18, R(2+1)D-18, I3D
- Weight inflation: 2D→3D transfer
- Easy configuration via YAML files

### Grad-CAM
- Grad-CAM and Grad-CAM++ implementations
- Support for both 2D and 3D models
- Temporal Grad-CAM visualization (per-frame heatmaps)
- Automatic overlay generation

### Training
- Multi-GPU support
- TensorBoard logging
- Automatic checkpointing
- Resume from checkpoint
- Configurable optimizers and schedulers

## 📊 Expected Results

### Performance Expectations
- **2D single frame**: ~30-40% top-1 (limited by lack of temporal info)
- **2D multi-frame**: ~35-45% top-1 (simple temporal pooling)
- **3D from scratch**: ~45-55% top-1 (full spatiotemporal)
- **3D with 2D pretraining**: ~48-58% top-1 (transfer learning boost)

### Grad-CAM Insights
- **2D models**: Focus on objects and scene context, struggle with direction
- **3D models**: Attend to motion, hands, and temporal cues
- **Pretrained 3D**: May show more focused attention faster

## 🔧 Tips & Troubleshooting

### Memory Issues
- Reduce batch size in config
- Use smaller image size (112x112 instead of 224x224)
- Reduce number of frames for 3D
- Use gradient checkpointing (implement if needed)

### Slow Training
- Use fewer workers if I/O is bottleneck
- Ensure data is on fast storage (SSD)
- Consider mixed precision training (add to scripts if needed)
- Start with subset of classes for debugging

### Grad-CAM Issues
- Verify target layer is correct (last conv layer)
- Check that gradients flow properly
- Normalize heatmaps properly
- Try Grad-CAM++ if Grad-CAM is noisy

## 📚 References

- **Something-Something Dataset**: Goyal et al. "The 'something something' video database for learning and evaluating visual common sense" (ICCV 2017)
- **Grad-CAM**: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
- **I3D**: Carreira & Zisserman "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset" (CVPR 2017)
- **R(2+1)D**: Tran et al. "A Closer Look at Spatiotemporal Convolutions for Action Recognition" (CVPR 2018)

## 📝 Citation

If you use this code for your research, please cite:

```bibtex
@misc{something-something-iv,
  author = {Rasmus Arnmark},
  title = {Something-Something V2: 2D to 3D Model Analysis with Grad-CAM},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/RasmusArnmark/something-something-IV}
}
```

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the repository owner.

## 🙏 Acknowledgments

- Something-Something V2 dataset from TwentyBN/Qualcomm
- PyTorch and torchvision teams
- Grad-CAM implementation inspiration from pytorch-grad-cam

---

**Good luck with your project! 🚀**