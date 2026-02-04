# DepthFlowProjection

Combined depth and optical flow projection for 3D-aware image warping.

## Description

This module combines depth information with optical flow to perform 3D-aware warping operations. It's particularly useful for video processing tasks that require depth-aware motion compensation and multi-view synthesis.

## Installation

```bash
cd DepthFlowProjection
python setup.py build develop
```

## Features

- CUDA-accelerated depth-aware flow projection
- Combines depth maps and optical flow for accurate warping
- Handles occlusions and depth discontinuities
- Differentiable for end-to-end training

## Usage

```python
import torch
from DepthFlowProjection.DepthFlowProjection import DepthFlowProjectionModule

# Create depth flow projection module
projector = DepthFlowProjectionModule(requires_grad=True)

# Apply projection with depth and flow
input_image = torch.randn(1, 3, 256, 256).cuda()
flow = torch.randn(1, 2, 256, 256).cuda()
depth = torch.randn(1, 1, 256, 256).cuda()

warped_image = projector(input_image, flow, depth)
```

## Use Cases

- Video frame synthesis with depth
- 3D-aware video processing
- Depth-guided frame interpolation
- Novel view synthesis

## Build Requirements

- CUDA >= 10.1
- PyTorch >= 1.8
- C++17 compatible compiler
- Supported CUDA architectures: 7.5, 8.9, 9.0
