import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension,CppExtension

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.9;9.0"
cxx_args = ['-std=c++17', '-w', '-D_GLIBCXX_USE_CXX11_ABI=1']
nvcc_args = ['-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_75,code=sm_75',
    '-gencode', 'arch=compute_89,code=sm_89',
    '-gencode', 'arch=compute_90,code=sm_90',
    "-O3",'-Wno-deprecated-gpu-targets', '--expt-relaxed-constexpr']
ROOT = os.path.dirname(os.path.abspath(__file__))
setup(include_package_data=True, zip_safe=True,
      name='Deformable_ConvNets', install_requires=["torch"],
      packages=find_packages(exclude="./Deformable_ConvNets"),
      ext_modules=[CUDAExtension(name='Deformable_ConvNets.deform_attn_cuda',
                         package_dir={'Deformable_ConvNets': f'./Deformable_ConvNets'},
                         include_dirs=[os.path.join(ROOT, "../include/")],
                                 sources=['../src/deform_attn_cuda_pt109.cpp',
                                          '../src/deform_attn_cuda_pt110.cpp',
                                          '../src/deform_attn_ext.cpp',
                                          '../kernels/deform_attn_cuda_kernel.cu'],
                                 extra_compile_args={'cxx': cxx_args,
                                                     'nvcc': nvcc_args})],
      cmdclass={'build_ext': BuildExtension})
