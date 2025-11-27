import os
import cv2
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension,CppExtension

os.environ["TORCH_CUDA_ARCH_LIST"]="7.5;8.9;9.0"
ROOT = os.path.dirname(os.path.abspath(__file__))
cxx_args=['-std=c++17', '-w','-D_GLIBCXX_USE_CXX11_ABI=1']
nvcc_args=['-gencode', 'arch=compute_90,code=sm_90',
           '-gencode', 'arch=compute_75,code=sm_75',
           '-gencode', 'arch=compute_89,code=sm_89',
           "-O3",'-Wno-deprecated-gpu-targets','--expt-relaxed-constexpr'
           ]

setup(name='FilterInterpolation',
      packages=find_packages(exclude="./FilterInterpolation"),
      install_requires=["torch"],
      zip_safe=True,include_package_data=True,
      ext_modules=[CUDAExtension(name="FilterInterpolation.filterinterpolation_cuda",
                                sources=[
                                    '../src/filterinterpolation_cuda.cc',
                                    '../kernels/filterinterpolation_cuda_kernel.cu'],
                                include_dirs=[os.path.join(ROOT,"../include/")],
                                extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})],
      cmdclass={'build_ext': BuildExtension})
