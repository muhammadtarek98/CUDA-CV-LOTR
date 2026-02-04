# DCNv2 (Deformable Convolutional Networks v2)

Deformable Convolutional Networks v2 implementation with deformable convolution and deformable RoI pooling.

## Description

This module implements Deformable Convolutional Networks v2 (DCNv2), which allows convolutional kernels to adapt their sampling locations based on learned offsets. This enables the network to handle geometric transformations and deformations more effectively.

## Installation

```bash
cd DCNv2_latest
python setup.py build develop
```

## Features

- Deformable convolution with learnable offsets
- Deformable RoI pooling for object detection
- Modulated deformable convolution (DCNv2)
- ONNX export support for deployment
- CUDA-accelerated operations

## Usage

```python
import torch
from DCNv2.dcn_v2 import DCN

# Create deformable convolution layer
dcn = DCN(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    deformable_groups=1
).cuda()

# Forward pass - offsets are learned automatically
input_features = torch.randn(1, 64, 128, 128).cuda()
output = dcn(input_features)
```

## Use Cases

- Object detection with deformable features
- Semantic segmentation with irregular shapes
- Video object detection
- Handling geometric transformations in CNNs

## Testing

```bash
cd test
python test_dcn.py
```

## Build Requirements

- CUDA >= 10.1
- PyTorch >= 1.8
- C++17 compatible compiler
- Supported CUDA architectures: 7.5, 8.9, 9.0

## Reference

Based on "Deformable ConvNets v2: More Deformable, Better Results" (CVPR 2019)
