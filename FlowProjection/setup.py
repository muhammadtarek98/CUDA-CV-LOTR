import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
os.environ["TORCH_CUDA_ARCH_LIST"]="7.5;8.9;9.0"
ROOT = os.path.dirname(os.path.abspath(__file__))
cxx_args=['-std=c++17', '-w','-D_GLIBCXX_USE_CXX11_ABI=1']
nvcc_args=['-gencode', 'arch=compute_90,code=sm_90',
           '-gencode', 'arch=compute_75,code=sm_75',
           '-gencode', 'arch=compute_89,code=sm_89',
           "-O3",'-Wno-deprecated-gpu-targets','--expt-relaxed-constexpr'
           ]
setup(include_package_data=True,zip_safe=True,
    name='FlowProjection',
      install_requires=["torch"],packages=find_packages(exclude="./FlowProjection"),
    ext_modules=[CUDAExtension(
        name='FlowProjection.flowprojection_cuda',
        include_dirs=[os.path.join(ROOT,"../include/")],
        sources=['../src/flowprojection_cuda.cc',
            '../kernels/flowprojection_cuda_kernel.cu'],
        extra_compile_args={'cxx': cxx_args, 'nvcc':nvcc_args })
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
