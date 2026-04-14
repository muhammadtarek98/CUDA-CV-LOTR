# Deformable ConvNets with Deformable Attention

Deformable convolution implementation with deformable attention mechanisms.

## Description

This module provides an implementation of deformable convolution networks with additional deformable attention mechanisms. It enables adaptive receptive fields and attention-based feature aggregation for improved spatial modeling.

## Installation

```bash
cd Deformable_ConvNets
python setup.py build develop
```

## Features

- Deformable attention mechanisms
- Adaptive spatial sampling
- Multi-head deformable attention
- CUDA-optimized implementations
- Compatible with PyTorch 1.09 and 1.10+

## Usage

```python
import torch
from Deformable_ConvNets.Deformable_ConvNets import DeformAttn

# Create deformable attention layer
deform_attn = DeformAttn(
    d_model=256,
    n_heads=8,
    n_levels=4,
    n_points=4
).cuda()

# Apply deformable attention
features = torch.randn(1, 256, 64, 64).cuda()
output = deform_attn(features)
```

## Use Cases

- Transformer-based vision models
- Multi-scale feature aggregation
- Object detection with deformable attention
- Dense prediction tasks

## Build Requirements

- CUDA >= 10.1
- PyTorch >= 1.8
- C++17 compatible compiler
- Supported CUDA architectures: 7.5, 8.9, 9.0

## Note

This module includes separate implementations for PyTorch 1.09 and 1.10+ to ensure compatibility.
