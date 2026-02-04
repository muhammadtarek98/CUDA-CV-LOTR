# Include Directory

Header files for CUDA operations and PyTorch extensions.

## Description

This directory contains all the header files (.h and .cuh) that define the interfaces and declarations for the custom CUDA operations. These headers are used by the source files during compilation.

## Contents

The header files in this directory include:

### Correlation Operations
- `correlation_cuda_kernel.cuh` - CUDA correlation kernel declarations
- `vid2vid_correlation_cuda_kernel.cuh` - Video-to-video correlation kernels

### Projection Operations
- `flowprojection_cuda_kernel.cuh` - Flow projection kernels
- `depthflowprojection_cuda_kernel.cuh` - Depth flow projection kernels
- `mindepthflowprojection_cuda_kernel.cuh` - Minimum depth flow projection kernels

### Interpolation Operations
- `interpolation_cuda_kernel.cuh` - Standard interpolation kernels
- `interpolationch_cuda_kernel.cuh` - Channel-aware interpolation kernels
- `filterinterpolation_cuda_kernel.cuh` - Filter-based interpolation kernels

### Deformable Convolution
- `dcn_v2.h` - DCNv2 main header
- `dcn_v2_im2col_cpu.h` - CPU implementation of im2col for DCN
- `dcn_v2_im2col_cuda.h` - CUDA implementation of im2col for DCN
- `deform_attn_ext.h` - Deformable attention extension
- `deform_attn_cuda_kernel.cuh` - Deformable attention CUDA kernels
- `deform_attn_cuda_pt109.h` - PyTorch 1.09 compatibility
- `deform_attn_cuda_pt110.h` - PyTorch 1.10+ compatibility

### Convolution Operations
- `separableconv_cuda_kernel.cuh` - Separable convolution kernels
- `separableconvflow_cuda_kernel.cuh` - Separable convolution for flow

### Normalization
- `channelnorm_kernel.cuh` - Channel normalization kernels

### Other Operations
- `resample2d_kernel.cuh` - 2D resampling kernels
- `smm_cuda.h` - Sparse matrix multiplication

### LieTorch (Lie Group Operations)
- `lietorch_cpu.h` - CPU implementations for Lie groups
- `lietorch_gpu.h` - GPU implementations for Lie groups
- `so3.h` - SO(3) rotation group
- `rxso3.h` - RxSO(3) rotation with scale
- `se3.h` - SE(3) rigid transformation
- `sim3.h` - Sim(3) similarity transformation

### Utilities
- `common.h` - Common definitions and utilities
- `dispatch.h` - Type dispatch utilities
- `cpu_vision.h` - CPU vision operation declarations
- `cuda_vision.h` - CUDA vision operation declarations

## Usage

These headers are automatically included during the build process when compiling the various modules. They are referenced in the `include_dirs` parameter of the setup.py files:

```python
include_dirs=[os.path.join(ROOT,"../include/")]
```

## Requirements

- CUDA >= 10.1
- C++17 compatible compiler
- PyTorch development headers
