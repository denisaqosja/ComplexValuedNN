# Complex-Valued Neural Networks (CVNN)

A PyTorch implementation of Complex-Valued Convolutional Neural Networks for radar signal processing and Range-Doppler map reconstruction.

## Overview

This project implements complex-valued neural networks with a focus on:
- **Complex-valued UNet architecture** for image reconstruction
- **Range-Doppler map processing** from measured radar data
- **Configurable training pipelines** with YAML-based configuration management

## Features

-  Complex-valued convolutional layers and operations
-  Real-valued and complex-valued UNet variants
-  TensorBoard logging and real-time visualization
-  Flexible YAML-based configuration management
-  Comprehensive evaluation and testing pipeline

## Project Structure

```
CVNN/
├── src/
│   ├── train_unet.py              # Real-valued UNet training
│   ├── train_complex_unet.py      # Complex-valued UNet training
│   ├── test_complex_unet.py       # Evaluation and testing
│   ├── rdmaps_loader.py           # Data loading utilities
│   ├── data/
│   │   ├── data_util.py
│   │   ├── helpers.py
│   │   ├── processing_util.py
│   │   └── transform_util.py
│   └── model/
│       ├── complex_unet.py        # Complex-valued UNet
│       ├── unet.py                # Real-valued UNet
│       ├── schedulers.py          # LR schedulers
│       ├── setup_model.py         # Model utilities
│       └── tensorboard_writer.py  # TensorBoard logging
├── configs/
│   └── cvnn.yaml                  # Configuration file
├── notebooks/                     # Analysis notebooks
├── experiments/                   # Auto-generated outputs
└── README.md                      # Project documentation
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-org/CVNN.git
cd CVNN
```

2. **Create conda environment from file**:
```bash
conda env create -f environment.yml
conda activate radarsigproc
```

## Configuration

All parameters are managed through `configs/cvnn.yaml`:


## Training

### Train Complex-Valued UNet

```bash
python src/train_complex_unet.py
```

**Training pipeline**:

When you run the training script, the following sequence occurs:
- Configuration is loaded from `configs/cvnn.yaml`
- A timestamped experiment directory is automatically created 
- The configuration is saved to the experiment directory, preserving exactly what settings were used
- Model architecture, optimizer, and data loaders are initialized based on the config
- Training begins with TensorBoard logging enabled for real-time monitoring
- Model checkpoints are saved every 5 epochs
- Validation is performed after each epoch to track progress


## Evaluation & Testing

Evaluate trained model on test data:

```bash
python src/test_complex_unet.py
```

Edit the script to specify:
- `exp_name`: Experiment directory name
- `ckpt`: Checkpoint filename (e.g., "epoch_100.pth")

**Pipeline**:
- Loads config from experiment directory
- Loads checkpoint
- Runs inference on test set
- Logs metrics to TensorBoard
- Generates magnitude map visualizations

## Model Architecture

### Complex-Valued UNet

```
Input (B, 1, H, W)
    ↓
Encoder (downsampling blocks with attention)
    ↓
Bottleneck (multi-scale features)
    ↓
Decoder (upsampling with skip connections)
    ↓
Output (B, 1, H, W)
```

**Key Components**:
- **Complex Convolution**: Preserves phase information
- **modReLU Activation**: Magnitude-phase decoupled ReLU for complex-valued networks
- **Attention Mechanism**: Spatial focus
- **Complex Weight Initialization**: Xavier and He uniform initialization adapted for complex-valued tensors

## Authors

**Denisa Qosja** - denisaqosja97@gmail.com

---

**Project Status**: Active Development  
**Last Updated**: March 29, 2026  
**Version**: 1.0.0
