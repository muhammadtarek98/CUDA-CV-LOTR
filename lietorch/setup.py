from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
ROOT = os.path.dirname(os.path.abspath(__file__))

setup(
    name='lietorch',package_dir={'': './'},
    version='0.2',
    description='Lie Groups for PyTorch',
    author='teedrz',install_requires=["torch"],
    packages=['lietorch'],
    ext_modules=[
        CUDAExtension(name='lietorch_backends',
            include_dirs=[
                os.path.join(ROOT, 'eigen'),
            os.path.join(ROOT,"../include")],
            sources=[
                '../src/lietorch.cpp',
                '../kernels/lietorch_gpu.cu',
                '../src/lietorch_cpu.cpp'],
            extra_compile_args={
                'cxx': ['-std=c++17', '-w','-D_GLIBCXX_USE_CXX11_ABI=1'],
                'nvcc': ["-O3",
                         '-gencode', 'arch=compute_37,code=sm_37',
                         '-gencode', 'arch=compute_50,code=sm_50',
                         '-gencode', 'arch=compute_52,code=sm_52',
                         '-gencode', 'arch=compute_60,code=sm_60',
                         '-gencode', 'arch=compute_61,code=sm_61',
                         '-gencode', 'arch=compute_75,code=sm_75',
                         '-gencode', 'arch=compute_70,code=sm_70',
                         '-gencode', 'arch=compute_89,code=sm_89',
                         '-Wno-deprecated-gpu-targets','--expt-relaxed-constexpr',
                ]
            }),

        CUDAExtension(name='lietorch_extras',
            sources=[
                '../kernels/altcorr_kernel.cu',
                '../kernels/corr_index_kernel.cu',
                '../kernels/se3_builder.cu',
                '../kernels/se3_inplace_builder.cu',
                '../kernels/se3_solver.cu',
                '../src/extras.cpp',
            ],
            extra_compile_args={'cxx': ['-std=c++17', '-w','-D_GLIBCXX_USE_CXX11_ABI=1'],
                'nvcc': [
                '-gencode', 'arch=compute_37,code=sm_37',
                '-gencode', 'arch=compute_50,code=sm_50',
                '-gencode', 'arch=compute_52,code=sm_52',
                '-gencode', 'arch=compute_60,code=sm_60',
                '-gencode', 'arch=compute_61,code=sm_61',
                '-gencode', 'arch=compute_75,code=sm_75',
                '-gencode', 'arch=compute_89,code=sm_89',
                '-gencode', 'arch=compute_70,code=sm_70',
                "-O3",'-Wno-deprecated-gpu-targets','--expt-relaxed-constexpr'
            ]}),
    ],
    cmdclass={ 'build_ext': BuildExtension }
)


