# Source Directory

C++ and CUDA source files for PyTorch extensions.

## Description

This directory contains the C++ and CUDA source files (.cpp, .cc, .cu) that implement the PyTorch extension bindings for all custom operations. These files bridge the gap between Python/PyTorch and the CUDA kernels, handling tensor operations and autograd integration.

## Contents

### Correlation Implementations
- `correlation.cpp` - Correlation layer PyTorch bindings
- `correlation_cuda.cc` - CUDA correlation extension interface

### Projection Implementations
- `flowprojection_cuda.cc` - Flow projection extension
- `depthflowprojection_cuda.cc` - Depth flow projection extension
- `mindepthflowprojection_cuda.cc` - Minimum depth flow projection extension

### Interpolation Implementations
- `interpolation_cuda.cc` - Standard interpolation extension
- `interpolationch_cuda.cc` - Channel-aware interpolation extension
- `filterinterpolation_cuda.cc` - Filter-based interpolation extension

### Deformable Convolution Implementations
- `dcn_v2_cpu.cpp` - DCNv2 CPU implementation
- `dcn_v2_im2col_cpu.cpp` - Im2col CPU operations for DCN
- `dcn_v2_psroi_pooling_cpu.cpp` - Position-sensitive RoI pooling (CPU)
- `deform_attn_ext.cpp` - Deformable attention extension interface
- `deform_attn_cuda_pt109.cpp` - Deformable attention for PyTorch 1.09
- `deform_attn_cuda_pt110.cpp` - Deformable attention for PyTorch 1.10+

### Convolution Implementations
- `separableconv_cuda.cc` - Separable convolution extension
- `separableconvflow_cuda.cc` - Separable convolution flow extension

### Normalization Implementations
- `channelnorm_cuda.cc` - Channel normalization extension

### Other Implementations
- `resample2d_cuda.cc` - 2D resampling extension
- `vid2vid_correlation_cuda.cc` - Video-to-video correlation extension

### LieTorch Implementations
- `lietorch.cpp` - Main LieTorch PyTorch bindings
- `lietorch_cpu.cpp` - Lie group CPU implementations

### Utilities
- `vision.cpp` - Vision operations entry point
- `extras.cpp` - Additional utility functions

## Functionality

These source files implement:

1. **PyTorch C++ Extensions**: Using pybind11 to expose C++/CUDA functions to Python
2. **Tensor Operations**: Handle PyTorch tensor inputs/outputs
3. **Autograd Integration**: Implement forward and backward passes for gradient computation
4. **Type Dispatching**: Support different tensor types (float, double, half)
5. **Error Handling**: Validate inputs and provide meaningful error messages
6. **Memory Management**: Efficient tensor allocation and deallocation

## Build Process

These files are compiled into Python extension modules using PyTorch's `cpp_extension` API:

```python
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    ext_modules=[
        CUDAExtension(
            name='module_name',
            sources=['source_file.cc', 'kernel_file.cu'],
            ...
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

## Compilation Flags

C++ files are compiled with:
```
-std=c++17
-w
-D_GLIBCXX_USE_CXX11_ABI=1
```

## Requirements

- PyTorch >= 1.8 with development headers
- CUDA >= 10.1
- C++17 compatible compiler (GCC 7+, Clang 5+)
- pybind11 (provided by PyTorch)

## CPU vs GPU

Some operations have both CPU and GPU implementations:
- CPU implementations are used for debugging and when CUDA is not available
- GPU implementations (CUDA) are used for production and high performance
- The code automatically dispatches to the correct implementation based on tensor location
