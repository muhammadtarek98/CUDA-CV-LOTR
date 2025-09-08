from setuptools import setup,find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension,CppExtension
import os
os.environ["TORCH_CUDA_ARCH_LIST"]="7.5;8.9;9.0"
ROOT = os.path.dirname(os.path.abspath(__file__))
nvcc_args=['-gencode', 'arch=compute_90,code=sm_90',
           '-gencode', 'arch=compute_75,code=sm_75',
           '-gencode', 'arch=compute_89,code=sm_89',
           "-O3",'-Wno-deprecated-gpu-targets','--expt-relaxed-constexpr'
           ]
cxx_args=['-std=c++17', '-w','-D_GLIBCXX_USE_CXX11_ABI=1']
setup(include_package_data=True,zip_safe=True,
    name='correlation',install_requires=["torch"],
    ext_modules=[CppExtension(
        name='alt_corr',
        sources=['../src/correlation.cpp'],
        include_dirs=[os.path.join(ROOT,"../include/")],
        extra_compile_args={'cxx': cxx_args,}
    ),
        CUDAExtension(name='alt_cuda_corr',
            sources=['../kernels/correlation_kernel.cu'],
            include_dirs=[os.path.join(ROOT,"../include/")],
        extra_compile_args={'cxx': cxx_args,'nvcc': nvcc_args}),
    ],
    cmdclass={'build_ext': BuildExtension})

