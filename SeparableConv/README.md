# SeparableConv

Separable convolution implementation for efficient convolutional operations.

## Description

This module implements separable convolution that factorizes standard convolution into depthwise and pointwise operations. This significantly reduces computational cost and number of parameters while maintaining similar representational power.

## Installation

```bash
cd SeparableConv
python setup.py build develop
```

## Features

- CUDA-accelerated separable convolutions
- Depthwise convolution followed by pointwise (1x1) convolution
- Reduced computational complexity compared to standard convolution
- Fully differentiable for backpropagation

## Usage

```python
import torch
from SeparableConv.SeparableConv import SeparableConvModule

# Create separable convolution module
sep_conv = SeparableConvModule(
    in_channels=64,
    out_channels=128,
    kernel_size=3
)

# Apply separable convolution
input_features = torch.randn(1, 64, 128, 128).cuda()
output = sep_conv(input_features)
```

## Use Cases

- Efficient mobile networks (e.g., MobileNet)
- Resource-constrained environments
- Real-time applications
- Reducing model size and computation

## Build Requirements

- CUDA >= 10.1
- PyTorch >= 1.8
- C++17 compatible compiler
- Supported CUDA architectures: 7.5, 8.9, 9.0
