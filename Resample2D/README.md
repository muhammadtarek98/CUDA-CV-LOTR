# Resample2D

2D resampling operations for spatial transformations and geometric warping.

## Description

This module provides efficient 2D resampling operations implemented in CUDA. It enables spatial transformations and geometric warping of images and feature maps with sub-pixel accuracy.

## Installation

```bash
cd Resample2D
python setup.py build develop
```

## Features

- CUDA-accelerated 2D resampling
- Sub-pixel accuracy with interpolation
- Supports arbitrary spatial transformations
- Differentiable for gradient-based learning

## Usage

```python
import torch
from Resample2D.Resample2D import Resample2d

# Create resampling module
resampler = Resample2d()

# Apply spatial transformation
input_image = torch.randn(1, 3, 256, 256).cuda()
sampling_grid = torch.randn(1, 2, 256, 256).cuda()  # x,y coordinates
resampled = resampler(input_image, sampling_grid)
```

## Use Cases

- Spatial transformer networks
- Image warping and rectification
- Geometric augmentation
- View synthesis

## Build Requirements

- CUDA >= 10.1
- PyTorch >= 1.8
- C++17 compatible compiler
- Supported CUDA architectures: 7.5, 8.9, 9.0
