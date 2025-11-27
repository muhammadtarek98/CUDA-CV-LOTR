from setuptools import setup,find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
os.environ["TORCH_CUDA_ARCH_LIST"]="7.5;8.9;9.0"
ROOT = os.path.dirname(os.path.abspath(__file__))
cxx_args=['-std=c++17', '-w','-D_GLIBCXX_USE_CXX11_ABI=1']
nvcc_args=['-gencode', 'arch=compute_90,code=sm_90',
           '-gencode', 'arch=compute_75,code=sm_75',
           '-gencode', 'arch=compute_89,code=sm_89',
           "-O3",'-Wno-deprecated-gpu-targets','--expt-relaxed-constexpr'
           ]

setup(
    name='lietorch',
    install_requires=["torch"],
    packages=find_packages(exclude=("./lietorch","examples",)),
    ext_modules=[
        CUDAExtension(name='lietorch_backends',
            include_dirs=[os.path.join(ROOT, 'eigen'),
            os.path.join(ROOT,"../include")],
            sources=['../src/lietorch.cpp',
                '../kernels/lietorch_gpu.cu',
                '../src/lietorch_cpu.cpp'],
            extra_compile_args={'cxx': cxx_args,'nvcc': nvcc_args}),

        CUDAExtension(name='lietorch_extras',
            sources=['../kernels/altcorr_kernel.cu',
                '../kernels/corr_index_kernel.cu',
                '../kernels/se3_builder.cu',
                '../kernels/se3_inplace_builder.cu',
                '../kernels/se3_solver.cu',
                '../src/extras.cpp',],
            extra_compile_args={'cxx': cxx_args,'nvcc': nvcc_args}),
    ],
    cmdclass={ 'build_ext': BuildExtension }
)


