# Something-Something V2 with 3D Convolutions

This project implements 3D Convolutional Neural Networks (3D CNNs) for video action recognition on the Something-Something V2 dataset. The repository provides a complete pipeline for training and inference using 3D ResNet architectures.

## Overview

The Something-Something V2 dataset is a large-scale video dataset for action recognition that contains 174 classes of temporal human-object interactions. Unlike traditional 2D CNNs that process each frame independently, 3D CNNs capture both spatial and temporal information by applying convolutions across the temporal dimension as well.

### Key Features

- **3D ResNet Models**: Implementation of ResNet3D-18, ResNet3D-34, and ResNet3D-50
- **Temporal Modeling**: 3D convolutions for capturing motion and temporal dynamics
- **Efficient Data Loading**: Optimized video loading and preprocessing pipeline
- **Flexible Configuration**: YAML-based configuration for easy experimentation
- **Training Pipeline**: Complete training loop with validation and checkpointing
- **Inference Support**: Easy-to-use inference script for video classification

## Project Structure

```
.
├── configs/                    # Configuration files
│   ├── resnet3d_18.yaml       # ResNet3D-18 configuration
│   ├── resnet3d_34.yaml       # ResNet3D-34 configuration
│   └── resnet3d_50.yaml       # ResNet3D-50 configuration
├── src/                        # Source code
│   ├── models/                # Model architectures
│   │   ├── __init__.py
│   │   └── resnet3d.py        # 3D ResNet implementation
│   ├── data/                  # Data loaders
│   │   ├── __init__.py
│   │   └── something_something_v2.py  # Dataset loader
│   └── utils/                 # Utilities
│       ├── __init__.py
│       ├── metrics.py         # Evaluation metrics
│       └── logger.py          # Logging utilities
├── train.py                   # Training script
├── inference.py               # Inference script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RasmusArnmark/something-something-IV.git
cd something-something-IV
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Download the Something-Something V2 dataset from the [official website](https://developer.qualcomm.com/software/ai-datasets/something-something)

2. Organize the dataset in the following structure:
```
data/
├── videos/
│   ├── 1.webm
│   ├── 2.webm
│   └── ...
└── annotations/
    ├── something-something-v2-train.json
    ├── something-something-v2-validation.json
    └── something-something-v2-labels.json
```

## Training

Train a model using the provided training script:

```bash
python train.py --config configs/resnet3d_18.yaml --data-root /path/to/dataset
```

### Training Options

- `--config`: Path to configuration file (default: `configs/resnet3d_18.yaml`)
- `--data-root`: Path to dataset root directory (required)
- `--resume`: Path to checkpoint to resume training from (optional)
- `--gpu`: GPU ID to use (default: 0)

### Configuration Files

You can modify the YAML configuration files to adjust:
- Model architecture (resnet3d_18, resnet3d_34, resnet3d_50)
- Number of frames to sample per video
- Spatial resolution
- Batch size
- Learning rate and training hyperparameters

Example configuration:
```yaml
model:
  name: resnet3d_18
  num_classes: 174

data:
  num_frames: 16
  spatial_size: 224
  temporal_stride: 2

training:
  batch_size: 8
  num_epochs: 50
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.0001
```

## Inference

Run inference on a video file:

```bash
python inference.py --video /path/to/video.mp4 --checkpoint checkpoints/best_model.pth --config configs/resnet3d_18.yaml
```

### Inference Options

- `--video`: Path to input video file (required)
- `--checkpoint`: Path to model checkpoint (required)
- `--config`: Path to configuration file (default: `configs/resnet3d_18.yaml`)
- `--gpu`: GPU ID to use (default: 0)

## Model Architecture

### 3D ResNet

The 3D ResNet architecture extends the traditional 2D ResNet by replacing 2D convolutions with 3D convolutions. This allows the network to:

1. **Capture Temporal Information**: Learn motion patterns and temporal dynamics
2. **Process Video Clips**: Take multiple frames as input and process them jointly
3. **Hierarchical Feature Learning**: Extract features at multiple spatiotemporal scales

**Input Shape**: `(batch_size, channels, depth, height, width)`
- `channels`: 3 (RGB)
- `depth`: Number of frames (e.g., 16)
- `height` and `width`: Spatial dimensions (e.g., 224x224)

**Output**: Class probabilities for 174 action categories

### Available Models

- **ResNet3D-18**: Lightweight model with 2 blocks per layer [2, 2, 2, 2]
- **ResNet3D-34**: Medium-sized model with [3, 4, 6, 3] blocks
- **ResNet3D-50**: Deeper model with bottleneck blocks [3, 4, 6, 3]

## Performance Monitoring

Training progress is logged to:
- Console output with tqdm progress bars
- Log files in `logs/` directory
- TensorBoard logs in `runs/` directory

To view TensorBoard logs:
```bash
tensorboard --logdir runs
```

## Checkpoints

Model checkpoints are saved in the `checkpoints/` directory:
- `checkpoint_epoch_X.pth`: Checkpoint after epoch X
- `best_model.pth`: Best model based on validation accuracy

Each checkpoint contains:
- Model weights
- Optimizer state
- Scheduler state
- Training configuration
- Best validation accuracy

## 2D vs 3D Convolutions

### 2D Convolutions (Traditional Approach)
- Process each frame independently
- Limited temporal modeling
- Rely on recurrent layers (LSTM/GRU) for temporal information
- Lower computational cost

### 3D Convolutions (This Implementation)
- Process multiple frames jointly
- Direct temporal modeling through 3D kernels
- Capture short-term motion patterns
- Higher computational cost but better temporal understanding

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 16GB+ RAM for data loading
- Storage for dataset and checkpoints

## Citation

If you use this code or the Something-Something V2 dataset, please cite:

```bibtex
@inproceedings{goyal2017something,
  title={The" something something" video database for learning and evaluating visual common sense},
  author={Goyal, Raghav and Ebrahimi Kahou, Samira and Michalski, Vincent and Materzynska, Joanna and Westphal, Susanne and Kim, Heuna and Haenel, Valentin and Fruend, Ingo and Yianilos, Peter and Mueller-Freitag, Moritz and others},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={5842--5850},
  year={2017}
}
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Something-Something V2 dataset by TwentyBN
- ResNet architecture by Kaiming He et al.
- 3D CNN concepts from C3D and I3D papers