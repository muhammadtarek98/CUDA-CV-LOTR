import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension,CppExtension
os.environ["TORCH_CUDA_ARCH_LIST"]="7.5;8.9;9.0"
ROOT = os.path.dirname(os.path.abspath(__file__))
nvcc_args=['-gencode', 'arch=compute_90,code=sm_90',
           '-gencode', 'arch=compute_75,code=sm_75',
           '-gencode', 'arch=compute_89,code=sm_89',
           "-O3",'-Wno-deprecated-gpu-targets','--expt-relaxed-constexpr'
           ]
cxx_args=['-std=c++17', '-w','-D_GLIBCXX_USE_CXX11_ABI=1']
setup(include_package_data=True,zip_safe=True,
    name="DCNv2",packages=find_packages(exclude=("./test/","./DCNv2/")),
    version="0.1",
    description="deformable convolutional networks",
    ext_modules=[
        CppExtension(name="DCNv2",sources=[
            '../src/dcn_v2_cpu.cpp',
            '../src/dcn_v2_im2col_cpu.cpp',
            '../src/dcn_v2_psroi_pooling_cpu.cpp',
            '../src/vision.cpp'],
                     include_dirs=[os.path.join(ROOT,"../include/")],
                     extra_compile_args={'cxx': cxx_args,}),
      CUDAExtension(name='DCNv2_cuda',
                    sources=['../kernels/dcn_v2_cuda.cu',
                             '../kernels/dcn_v2_im2col_cuda.cu',
                             '../kernels/dcn_v2_psroi_pooling_cuda.cu'],
                    include_dirs=[os.path.join(ROOT,"../include/")],
                    extra_compile_args={'cxx': cxx_args,
                                        'nvcc': nvcc_args})
],
    cmdclass={"build_ext": BuildExtension},
)
