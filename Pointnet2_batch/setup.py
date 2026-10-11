import os, torch
from setuptools import setup,find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
os.environ["TORCH_CUDA_ARCH_LIST"]="7.5;8.9;9.0"
ROOT = os.path.dirname(os.path.abspath(__file__))
nvcc_args=['-gencode', 'arch=compute_90,code=sm_90',
           '-gencode', 'arch=compute_75,code=sm_75',
           '-gencode', 'arch=compute_89,code=sm_89',
           "-O3",'-Wno-deprecated-gpu-targets','--expt-relaxed-constexpr'
           ]
cxx_args=['-std=c++17', '-w','-D_GLIBCXX_USE_CXX11_ABI=1']

setup(include_package_data=True,zip_safe=True,install_requires=["torch"],
      packages=find_packages(exclude="./Pointnet2_batch/"),
      name='Pointnet2_batch',
      ext_modules=[CUDAExtension(
          name='Pointnet2_batch.Pointnet2_batch_cuda',
          sources= ["../kernels/Pointnet2_batch_ball_query_gpu.cu", "../kernels/Pointnet2_batch_group_points_gpu.cu",
                    "../kernels/Pointnet2_batch_interpolate_gpu.cu","../kernels/Pointnet2_batch_sampling_gpu.cu",
                    "../src/Pointnet2_batch_ball_query.cpp", "../src/Pointnet2_batch_group_points.cpp",
                    "../src/Pointnet2_batch_interpolate.cpp","../src/Pointnet2_batch_pointnet2_api.cpp","../src/Pointnet2_batch_sampling.cpp"],
          include_dirs=[os.path.join(ROOT,"../include/")],
          extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})],
      cmdclass={ 'build_ext': BuildExtension})
