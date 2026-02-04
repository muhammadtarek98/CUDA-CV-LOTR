# Video-to-Video Correlation

Specialized correlation layer for video-to-video synthesis tasks.

## Description

This module implements a correlation operation specifically designed for video-to-video synthesis and temporal feature matching. It enables efficient computation of correspondences between video frames for tasks like video translation and synthesis.

## Installation

```bash
cd vid_2_vid_correlation
python setup.py build develop
```

## Features

- CUDA-optimized for video processing
- Temporal feature matching across frames
- Efficient correlation computation for sequences
- Supports various displacement and stride configurations

## Usage

```python
import torch
from vid_2_vid_correlation.vid_2_vid_correlation import Vid2VidCorrelation

# Create video correlation layer
vid_corr = Vid2VidCorrelation()

# Apply to sequential feature maps
frame_features_t = torch.randn(1, 256, 64, 64).cuda()
frame_features_t1 = torch.randn(1, 256, 64, 64).cuda()
correlation = vid_corr(frame_features_t, frame_features_t1)
```

## Use Cases

- Video-to-video translation
- Temporal feature alignment
- Video synthesis networks
- Frame-to-frame correspondence

## Build Requirements

- CUDA >= 10.1
- PyTorch >= 1.8
- C++17 compatible compiler
- Supported CUDA architectures: 7.5, 8.9, 9.0
