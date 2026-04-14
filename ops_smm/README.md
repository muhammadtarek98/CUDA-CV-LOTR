# Sparse Matrix Multiplication Operations

Specialized sparse matrix multiplication operations optimized for deep learning workloads.

## Description

This module provides CUDA-optimized sparse matrix multiplication (SMM) operations designed for efficient computation in deep learning applications. It's particularly useful for networks that involve sparse computations or graph neural networks.

## Installation

```bash
cd ops_smm
python setup.py build develop
```

## Features

- CUDA-accelerated sparse matrix operations
- Optimized for deep learning tensor operations
- Memory-efficient sparse representations
- Integrated with PyTorch's autograd system

## Usage

```python
import torch
import smm_cuda

# Use sparse matrix multiplication in your models
# Particularly useful for:
# - Graph neural networks
# - Sparse attention mechanisms
# - Efficient linear transformations
```

## Testing

Run the tests to verify installation:

```bash
cd test
python test_smm.py
```

## Build Requirements

- CUDA >= 10.1
- PyTorch >= 1.8
- C++17 compatible compiler
- Supported CUDA architectures: 7.5, 8.9, 9.0

## Author

WeiLong
