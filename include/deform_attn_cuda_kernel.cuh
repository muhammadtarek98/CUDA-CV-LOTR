#pragma once
#define CUDA_KERNEL_LOOP(i, n)    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <bits/stdc++.h>
const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;
inline int GET_BLOCKS(const int &N);
template <typename scalar_t> __device__ scalar_t deformable_im2col_bilinear(const scalar_t *bottom_data, const int &data_width,
const int &height, const int &width, scalar_t h, scalar_t w);
template <typename scalar_t> __device__ scalar_t get_gradient_weight(scalar_t argmax_h, scalar_t argmax_w,
const int &h, const int &w, const int &height, const int &width);
template <typename scalar_t> __device__ scalar_t get_coordinate_weight(scalar_t argmax_h, scalar_t argmax_w, const int &height,
const int &width, const scalar_t *im_data,  const int &data_width, const int &bp_dir);

template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel(const int &n, const scalar_t *data_im, const scalar_t *data_offset,
const int &height, const int &width, const int &kernel_h, const int &kernel_w,const int &pad_h, const int &pad_w,
const int &stride_h, const int &stride_w,const int &dilation_h, const int &dilation_w,const int &channel_per_deformable_group,
const int &batch_size,const int &num_channels,const int &deformable_group,const int &height_col, const int &width_col,
scalar_t *data_col);
