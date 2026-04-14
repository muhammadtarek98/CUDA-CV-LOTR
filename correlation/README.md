# Correlation Layer

CUDA-accelerated correlation layer for computing feature correlations between patches.

## Description

This module implements a correlation layer that is commonly used in optical flow estimation networks (e.g., FlowNet). It computes the correlation between patches in two feature maps, which helps identify correspondences between frames.

## Installation

```bash
cd correlation
python setup.py build develop
```

## Features

- Fast CUDA implementation for correlation computation
- Configurable parameters: pad size, kernel size, max displacement, stride
- Supports forward and backward passes for gradient computation
- Integrated with PyTorch's autograd system

## Usage

```python
import torch
from correlation.correlation import Correlation

# Create correlation layer
corr = Correlation(
    pad_size=4,
    kernel_size=1,
    max_displacement=4,
    stride1=1,
    stride2=1,
    corr_multiply=1
)

# Apply to two feature maps
feature1 = torch.randn(1, 256, 32, 32).cuda()
feature2 = torch.randn(1, 256, 32, 32).cuda()
output = corr(feature1, feature2)
```

## Parameters

- `pad_size`: Padding size for input feature maps
- `kernel_size`: Size of the correlation kernel
- `max_displacement`: Maximum displacement for correlation search
- `stride1`: Stride for the first feature map
- `stride2`: Stride for the second feature map
- `corr_multiply`: Multiplication factor for correlation values

## Build Requirements

- CUDA >= 10.1
- PyTorch >= 1.8
- C++17 compatible compiler
- Supported CUDA architectures: 7.5, 8.9, 9.0
