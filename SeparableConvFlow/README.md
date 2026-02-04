# SeparableConvFlow

Separable convolution variant designed for optical flow refinement tasks.

## Description

This module implements a specialized version of separable convolution optimized for optical flow processing. It combines efficient depthwise separable convolutions with flow-specific operations for optical flow estimation and refinement.

## Installation

```bash
cd SeparableConvFlow
python setup.py build develop
```

## Features

- Flow-aware separable convolutions
- CUDA-optimized for optical flow processing
- Efficient refinement of flow predictions
- Maintains flow consistency and smoothness

## Usage

```python
import torch
from SeparableConvFlow.SeparableConvFlow import SeparableConvFlowModule

# Create separable conv flow module
sep_conv_flow = SeparableConvFlowModule(
    in_channels=2,  # Flow has 2 channels (x, y)
    out_channels=2,
    kernel_size=3
)

# Apply to optical flow
flow = torch.randn(1, 2, 256, 256).cuda()
refined_flow = sep_conv_flow(flow)
```

## Use Cases

- Optical flow refinement
- Flow estimation networks
- Video frame interpolation
- Motion field processing

## Build Requirements

- CUDA >= 10.1
- PyTorch >= 1.8
- C++17 compatible compiler
- Supported CUDA architectures: 7.5, 8.9, 9.0
