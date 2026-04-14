# Custom PyTorch Layers and Operations

A comprehensive collection of custom PyTorch layers and CUDA operations for deep learning, including deformable convolutions, optical flow operations, correlation layers, and Lie group operations for 3D transformations.

## Overview

This repository contains optimized CUDA implementations of various custom neural network operations designed to extend PyTorch's capabilities. These operations are particularly useful for computer vision tasks such as video processing, optical flow estimation, 3D reconstruction, and spatial transformations.

## Requirements

- **CUDA >= 10.1** (with nvcc compiler)
- **PyTorch >= 1.8**
- **CMake >= 3.31**
- **Python 3.x**
- **pybind11**
- **Eigen3** (for Lie group operations)
- **GCC with C++17 support**

### Recommended Development Tools
- For examples: OpenCV, Open3D
- For testing: pytest, numpy

## Installation

### Option 1: Build All Modules with CMake (Recommended)

```bash
# Clone the repository
git clone --recursive https://github.com/muhammadtarek98/custom_pytorch_layers_and_operations.git
cd custom_pytorch_layers_and_operations

# Configure and build
mkdir build && cd build
cmake ..
make

# Install all Python modules
make install_all
```

### Option 2: Install Individual Modules

Each module can be installed independently using its own `setup.py`:

```bash
cd <module_name>
python setup.py build develop
```

For example:
```bash
cd ChannelNorm
python setup.py build develop
```

## Modules

### 1. **ChannelNorm**
Channel normalization layer for neural networks. Performs normalization across channels independently for each spatial location.

**Location:** `ChannelNorm/`

### 2. **correlation**
Correlation layer for computing feature correlations between patches. Commonly used in optical flow estimation networks.

**Location:** `correlation/`

### 3. **alt_cuda_corr**
Alternative CUDA implementation of correlation operations with optimized performance characteristics.

**Location:** `alt_cuda_corr/`

### 4. **vid_2_vid_correlation**
Specialized correlation layer for video-to-video synthesis tasks, enabling efficient temporal feature matching.

**Location:** `vid_2_vid_correlation/`

### 5. **DCNv2_latest**
Deformable Convolutional Networks v2 - implements deformable convolution and deformable RoI pooling operations. Allows convolutional kernels to adapt their sampling locations based on learned offsets.

**Location:** `DCNv2_latest/`

### 6. **Deformable_ConvNets**
Another implementation of deformable convolution networks with deformable attention mechanisms.

**Location:** `Deformable_ConvNets/`

### 7. **lietorch**
Tangent space backpropagation for 3D transformation groups (SO3, SE3, Sim3, RxSO3). Provides differentiable Lie group operations for robotics, SLAM, and 3D vision applications.

**Location:** `lietorch/`  
**Documentation:** See `lietorch/README.md` for detailed usage

**Supported Groups:**
- SO3 (rotation)
- RxSO3 (rotation + scaling)
- SE3 (rotation + translation)
- Sim3 (rotation + translation + scaling)

### 8. **FlowProjection**
Optical flow projection layer that warps features or images according to flow fields.

**Location:** `FlowProjection/`

### 9. **DepthFlowProjection**
Combined depth and optical flow projection for 3D-aware image warping in video processing tasks.

**Location:** `DepthFlowProjection/`

### 10. **MinDepthFlowProjection**
Minimum depth flow projection variant optimized for multi-frame fusion tasks.

**Location:** `MinDepthFlowProjection/`

### 11. **Interpolation**
Custom interpolation operations for feature map upsampling and warping.

**Location:** `Interpolation/`

### 12. **InterpolationCH**
Channel-aware interpolation that performs interpolation with per-channel parameters.

**Location:** `InterpolationCH/`

### 13. **FilterInterpolation**
Filter-based interpolation using learnable convolution kernels for adaptive upsampling.

**Location:** `FilterInterpolation/`

### 14. **SeparableConv**
Separable convolution implementation that factorizes standard convolution into depthwise and pointwise operations for efficiency.

**Location:** `SeparableConv/`

### 15. **SeparableConvFlow**
Separable convolution variant designed specifically for optical flow refinement tasks.

**Location:** `SeparableConvFlow/`

### 16. **Resample2D**
2D resampling operations for spatial transformations and geometric warping.

**Location:** `Resample2D/`

### 17. **ops_smm**
Specialized sparse matrix multiplication operations optimized for deep learning workloads.

**Location:** `ops_smm/`

## Additional Files and Directories

### Core Implementation Files
- **`kernels/`** - CUDA kernel implementations for all operations
- **`src/`** - C++/CUDA source files for PyTorch extensions
- **`include/`** - Header files for the CUDA operations

### Build and Test Files
- **`CMakeLists.txt`** - CMake build configuration for all modules
- **`test_module.py`** - Test script for validating module implementations
- **`.gitignore`** - Git ignore rules for build artifacts

## Usage Example

```python
import torch
from lietorch import SE3
from DepthFlowProjection.DepthFlowProjection import DepthFlowProjectionModule
from correlation.correlation import Correlation

# Example 1: Using LieTorch for 3D transformations
phi = torch.randn(8000, 3, device='cuda', requires_grad=True)
R = SE3.exp(phi)
dR = R[:,None].inv() * R[None,:]
loss = dR.log().norm(dim=-1).sum()
loss.backward()

# Example 2: Using Correlation layer
corr = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1)
out = corr(feature1, feature2)

# Example 3: Using DepthFlowProjection
projector = DepthFlowProjectionModule(requires_grad=True)
warped = projector(image, flow, depth)
```

## Testing

Run the test module to verify installations:

```bash
python test_module.py
```

This will test the forward and backward passes of various modules, comparing CPU and GPU implementations.

## Building for Different CUDA Architectures

The modules are configured to build for CUDA compute capabilities 7.5, 8.9, and 9.0 by default. To build for different architectures, modify the `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` or `TORCH_CUDA_ARCH_LIST` in individual `setup.py` files.

## Citation

If you use the lietorch module, please cite:

```bibtex
@inproceedings{teed2021tangent,
  title={Tangent Space Backpropagation for 3D Transformation Groups},
  author={Teed, Zachary and Deng, Jia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021},
}
```

## Acknowledgments

- LieTorch implementations are adapted from [Sophus](https://github.com/strasdat/Sophus)
- Various operations are based on implementations from optical flow and video synthesis research

## License

Please check individual module directories for specific license information.

## Contributing

Contributions are welcome! Please ensure that:
1. Your code follows the existing style conventions
2. CUDA kernels are properly optimized
3. Both CPU and GPU implementations are tested
4. Documentation is updated accordingly

## Troubleshooting

### Common Issues

1. **CUDA Architecture Mismatch**: Ensure your GPU's compute capability is included in the build configuration
2. **CMake Not Finding CUDA**: Set `CUDA_TOOLKIT_ROOT_DIR` environment variable
3. **PyTorch Version Issues**: Update to PyTorch >= 1.8 for compatibility

### Getting Help

For issues specific to lietorch, see the [lietorch README](lietorch/README.md).
For general issues, please open an issue on GitHub with:
- Your CUDA version (`nvcc --version`)
- Your PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- Complete error messages and stack traces
