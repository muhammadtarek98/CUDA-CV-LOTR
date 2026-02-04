# Kernels Directory

CUDA kernel implementations for all custom operations.

## Description

This directory contains the CUDA kernel implementations (.cu files) that perform the actual GPU computations for all custom operations in this repository. Each kernel is optimized for parallel execution on NVIDIA GPUs.

## Contents

### Correlation Kernels
- `correlation_cuda_kernel.cu` - Main correlation kernel implementation
- `correlation_kernel.cu` - Alternative correlation implementation
- `altcorr_kernel.cu` - Alternative correlation variant
- `corr_index_kernel.cu` - Correlation indexing operations
- `vid2vid_correlation_cuda_kernel.cu` - Video-to-video correlation

### Projection Kernels
- `flowprojection_cuda_kernel.cu` - Optical flow projection
- `depthflowprojection_cuda_kernel.cu` - Depth-aware flow projection
- `mindepthflowprojection_cuda_kernel.cu` - Minimum depth flow projection

### Interpolation Kernels
- `interpolation_cuda_kernel.cu` - Standard interpolation operations
- `interpolationch_cuda_kernel.cu` - Channel-aware interpolation
- `filterinterpolation_cuda_kernel.cu` - Filter-based interpolation

### Deformable Convolution Kernels
- `dcn_v2_cuda.cu` - Deformable convolution v2 main kernel
- `dcn_v2_im2col_cuda.cu` - Im2col operation for DCNv2
- `dcn_v2_psroi_pooling_cuda.cu` - Position-sensitive RoI pooling
- `deform_attn_cuda_kernel.cu` - Deformable attention mechanism

### Convolution Kernels
- `separableconv_cuda_kernel.cu` - Separable convolution
- `separableconvflow_cuda_kernel.cu` - Separable convolution for flow

### Normalization Kernels
- `channelnorm_kernel.cu` - Channel normalization

### Other Kernels
- `resample2d_kernel.cu` - 2D resampling operations
- `smm_cuda.cu` - Sparse matrix multiplication

### LieTorch Kernels
- `lietorch_gpu.cu` - Main Lie group operations on GPU
- `se3_builder.cu` - SE3 matrix builder
- `se3_inplace_builder.cu` - In-place SE3 operations
- `se3_solver.cu` - SE3 solver kernels

## Kernel Optimization

All kernels in this directory are optimized with:
- Efficient memory access patterns (coalesced reads/writes)
- Shared memory usage for frequently accessed data
- Thread block and grid size optimization
- Support for multiple CUDA architectures (sm_75, sm_89, sm_90)

## Build Configuration

The kernels are compiled with the following NVCC flags:
```
-gencode arch=compute_90,code=sm_90
-gencode arch=compute_75,code=sm_75
-gencode arch=compute_89,code=sm_89
-O3
-Wno-deprecated-gpu-targets
--expt-relaxed-constexpr
```

## Requirements

- CUDA >= 10.1 with nvcc compiler
- Compatible NVIDIA GPU (compute capability 7.5+)
- CUDA Toolkit development libraries

## Performance Notes

These kernels are designed for high performance on modern NVIDIA GPUs. For optimal performance:
- Ensure your GPU compute capability matches the build configuration
- Use batch sizes that are multiples of 32 for better thread utilization
- Profile your application to identify bottlenecks
