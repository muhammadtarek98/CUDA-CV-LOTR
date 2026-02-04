# ChannelNorm

Channel normalization layer for neural networks.

## Description

This module implements a channel normalization operation that performs normalization across channels independently for each spatial location. It computes the norm of feature values across all channels at each pixel location.

## Installation

```bash
cd ChannelNorm
python setup.py build develop
```

## Features

- CUDA-accelerated channel normalization
- Configurable norm degree (L1, L2, etc.)
- Efficient implementation for processing feature maps
- Full gradient support for backpropagation

## Usage

```python
import torch
from ChannelNorm.ChannelNorm import ChannelNorm

# Create channel norm layer with L2 norm
channel_norm = ChannelNorm(norm_deg=2)

# Apply to feature map
input_features = torch.randn(1, 64, 128, 128).cuda()
output = channel_norm(input_features)

# Output shape: (1, 1, 128, 128) - one norm value per spatial location
```

## Parameters

- `norm_deg`: The degree of the norm (default: 2 for L2 norm)

## Build Requirements

- CUDA >= 10.1
- PyTorch >= 1.8
- C++17 compatible compiler
- Supported CUDA architectures: 7.5, 8.9, 9.0
