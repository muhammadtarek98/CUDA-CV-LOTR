import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension,CppExtension
os.environ["TORCH_CUDA_ARCH_LIST"]="7.5;8.9;9.0"
ROOT = os.path.dirname(os.path.abspath(__file__))
cxx_args=['-std=c++17', '-w','-D_GLIBCXX_USE_CXX11_ABI=1']
nvcc_args=['-gencode', 'arch=compute_90,code=sm_90',
           '-gencode', 'arch=compute_75,code=sm_75',
           '-gencode', 'arch=compute_89,code=sm_89',
           "-O3",'-Wno-deprecated-gpu-targets','--expt-relaxed-constexpr'
           ]
setup(include_package_data=True, zip_safe=True,
      name='vid_2_vid_correlation', packages=find_packages(exclude="./vid_2_vid_correlation/*"),
      ext_modules=[
        CUDAExtension(name='vid_2_vid_correlation_cuda',
                      sources=['../kernels/correlation_cuda_kernel.cu',
                               '../src/correlation_cuda.cc'],
        include_dirs=[os.path.join(ROOT,"../include/")],
        extra_compile_args={'cxx': cxx_args,'nvcc': nvcc_args})],
      cmdclass={'build_ext': BuildExtension})
