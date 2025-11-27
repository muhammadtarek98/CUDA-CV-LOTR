from setuptools import setup, find_packages
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
os.environ["TORCH_CUDA_ARCH_LIST"]="7.5;8.9;9.0"
ROOT = os.path.dirname(os.path.abspath(__file__))
cxx_args=['-std=c++17', '-w','-D_GLIBCXX_USE_CXX11_ABI=1']
nvcc_args=['-gencode', 'arch=compute_90,code=sm_90',
           '-gencode', 'arch=compute_75,code=sm_75',
           '-gencode', 'arch=compute_89,code=sm_89',
           "-O3",'-Wno-deprecated-gpu-targets','--expt-relaxed-constexpr'
           ]
setup(name="DepthFlowProjection",version="0.1.0",
      packages=find_packages(exclude="./DepthFlowProjection"),
      install_requires=["torch"],
      zip_safe=True,include_package_data=True,
      ext_modules=[CUDAExtension(
        name="DepthFlowProjection.depthflowprojection_cuda",
        sources=["../src/depthflowprojection_cuda.cc",
        "../kernels/depthflowprojection_cuda_kernel.cu",],
        include_dirs=[os.path.join(ROOT,"../include/")],
        extra_compile_args={'cxx': cxx_args,'nvcc': nvcc_args})],
      cmdclass={'build_ext': BuildExtension})
