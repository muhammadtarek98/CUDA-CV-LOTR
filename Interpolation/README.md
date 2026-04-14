# Interpolation

Custom interpolation operations for feature map upsampling and warping.

## Description

This module provides custom interpolation operations implemented in CUDA for efficient upsampling and warping of feature maps. It offers optimized alternatives to standard PyTorch interpolation with fine-grained control over the interpolation process.

## Installation

```bash
cd Interpolation
python setup.py build develop
```

## Features

- CUDA-accelerated interpolation
- Custom interpolation kernels
- Efficient upsampling for feature maps
- Differentiable for backpropagation

## Usage

```python
import torch
from Interpolation.Interpolation import InterpolationModule

# Create interpolation module
interp = InterpolationModule(requires_grad=True)

# Apply interpolation
input_features = torch.randn(1, 64, 32, 32).cuda()
upsampled = interp(input_features)
```

## Use Cases

- Feature map upsampling in CNNs
- Image super-resolution
- Feature pyramid networks
- Multi-scale processing

## Build Requirements

- CUDA >= 10.1
- PyTorch >= 1.8
- C++17 compatible compiler
- Supported CUDA architectures: 7.5, 8.9, 9.0
