# Alternative CUDA Correlation

Alternative CUDA implementation of correlation operations with optimized performance characteristics.

## Description

This module provides an alternative implementation of the correlation operation using CUDA. The correlation operation is commonly used in optical flow estimation networks to compute feature correlations between patches from two feature maps.

## Installation

```bash
cd alt_cuda_corr
python setup.py build develop
```

## Features

- Optimized CUDA kernels for correlation computation
- PyTorch integration through custom C++ extensions
- Supports various correlation parameters (pad size, kernel size, max displacement, stride)

## Usage

```python
from Alt_Correlation import alt_cuda_corr

# Use the correlation module in your PyTorch models
# The module computes correlation between two feature maps
```

## Build Requirements

- CUDA >= 10.1
- PyTorch >= 1.8
- C++17 compatible compiler
- Supported CUDA architectures: 7.5, 8.9, 9.0
