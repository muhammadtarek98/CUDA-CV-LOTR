# InterpolationCH

Channel-aware interpolation that performs interpolation with per-channel parameters.

## Description

This module implements channel-aware interpolation operations where each channel can have different interpolation parameters. This provides more flexibility compared to standard interpolation operations that treat all channels uniformly.

## Installation

```bash
cd InterpolationCH
python setup.py build develop
```

## Features

- Per-channel interpolation control
- CUDA-accelerated operations
- Differentiable for end-to-end training
- Flexible channel-wise processing

## Usage

```python
import torch
from InterpolationCH.InterpolationCH import InterpolationChModule

# Create channel-aware interpolation module
interp_ch = InterpolationChModule()

# Apply with per-channel parameters
input_features = torch.randn(1, 64, 32, 32).cuda()
output = interp_ch(input_features)
```

## Use Cases

- Channel-adaptive upsampling
- Multi-channel image processing
- Feature fusion with different scales per channel
- Adaptive super-resolution

## Build Requirements

- CUDA >= 10.1
- PyTorch >= 1.8
- C++17 compatible compiler
- Supported CUDA architectures: 7.5, 8.9, 9.0
