# FilterInterpolation

Filter-based interpolation using learnable convolution kernels for adaptive upsampling.

## Description

This module implements filter-based interpolation that uses learnable convolution filters for adaptive upsampling. Unlike standard interpolation, the filters can be learned from data, allowing the network to adapt the upsampling operation to the specific task.

## Installation

```bash
cd FilterInterpolation
python setup.py build develop
```

## Features

- Learnable interpolation filters
- CUDA-accelerated convolution operations
- Adaptive upsampling based on input features
- Full gradient support for filter learning

## Usage

```python
import torch
from FilterInterpolation.FilterInterpolation import FilterInterpolationModule

# Create filter interpolation module
filter_interp = FilterInterpolationModule()

# Apply learnable filter-based interpolation
input_features = torch.randn(1, 64, 32, 32).cuda()
filters = torch.randn(1, filter_size, H, W).cuda()  # Learnable filters
upsampled = filter_interp(input_features, filters)
```

## Use Cases

- Adaptive super-resolution
- Context-aware upsampling
- Learned image reconstruction
- Dynamic filter networks

## Build Requirements

- CUDA >= 10.1
- PyTorch >= 1.8
- C++17 compatible compiler
- Supported CUDA architectures: 7.5, 8.9, 9.0
