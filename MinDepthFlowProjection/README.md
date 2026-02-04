# MinDepthFlowProjection

Minimum depth flow projection variant optimized for multi-frame fusion tasks.

## Description

This module implements a minimum depth flow projection operation that is particularly useful for multi-frame fusion and temporal aggregation tasks. It selects the minimum depth among multiple projected frames to handle occlusions.

## Installation

```bash
cd MinDepthFlowProjection
python setup.py build develop
```

## Features

- CUDA-accelerated minimum depth projection
- Optimized for multi-frame processing
- Handles occlusion through depth comparison
- Efficient memory usage for batch processing

## Usage

```python
import torch
from MinDepthFlowProjection.MinDepthFlowProjection import MinDepthFlowProjectionModule

# Create min depth flow projection module
projector = MinDepthFlowProjectionModule()

# Apply to multiple frames with depth
# The module will select pixels from the frame with minimum depth
warped_output = projector(frames, flows, depths)
```

## Use Cases

- Multi-frame super-resolution
- Video stabilization
- Temporal fusion with occlusion handling
- Multi-view aggregation

## Build Requirements

- CUDA >= 10.1
- PyTorch >= 1.8
- C++17 compatible compiler
- Supported CUDA architectures: 7.5, 8.9, 9.0
