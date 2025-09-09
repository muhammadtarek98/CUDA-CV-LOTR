import os
import torch
from setuptools import setup,find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension,CppExtension
ROOT = os.path.dirname(os.path.abspath(__file__))

cxx_args = ['-std=c++17', '-w','-D_GLIBCXX_USE_CXX11_ABI=1']

nvcc_args = ['-gencode', 'arch=compute_37,code=sm_37','-gencode', 'arch=compute_50,code=sm_50',
                '-gencode', 'arch=compute_52,code=sm_52','-gencode', 'arch=compute_60,code=sm_60',
                '-gencode', 'arch=compute_61,code=sm_61','-gencode', 'arch=compute_75,code=sm_75',
                '-gencode', 'arch=compute_70,code=sm_70','-gencode', 'arch=compute_89,code=sm_89',"-O3",
                '-Wno-deprecated-gpu-targets','--expt-relaxed-constexpr'

]

setup(include_package_data=True,zip_safe=True,install_requires=["torch"],
      packages=find_packages(where="."),
      name='ChannelNorm',package_dir={'ChannelNorm': './' },
      ext_modules=[CUDAExtension(
    name='ChannelNorm.channelnorm_cuda',
    sources= ['../kernels/channelnorm_kernel.cu',
            '../src/channelnorm_cuda.cc'],
    include_dirs=[os.path.join(ROOT,"../include/")],
 extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})],
    cmdclass={ 'build_ext': BuildExtension})
