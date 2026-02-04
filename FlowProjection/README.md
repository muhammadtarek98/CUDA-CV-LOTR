# FlowProjection

Optical flow projection layer for warping features or images according to flow fields.

## Description

This module implements flow projection operations that warp input features or images based on optical flow fields. It's commonly used in video processing, temporal prediction, and optical flow-based networks.

## Installation

```bash
cd FlowProjection
python setup.py build develop
```

## Features

- CUDA-accelerated flow-based warping
- Differentiable operations for end-to-end training
- Supports arbitrary flow fields
- Efficient bilinear interpolation

## Usage

```python
import torch
from FlowProjection.FlowProjection import FlowProjectionModule

# Create flow projection module
projector = FlowProjectionModule(requires_grad=True)

# Apply flow projection
input_image = torch.randn(1, 3, 256, 256).cuda()
flow = torch.randn(1, 2, 256, 256).cuda()  # 2 channels for x,y flow
warped_image = projector(input_image, flow)
```

## Use Cases

- Video frame interpolation
- Optical flow visualization
- Feature warping in video synthesis
- Motion-compensated prediction

## Build Requirements

- CUDA >= 10.1
- PyTorch >= 1.8
- C++17 compatible compiler
- Supported CUDA architectures: 7.5, 8.9, 9.0
