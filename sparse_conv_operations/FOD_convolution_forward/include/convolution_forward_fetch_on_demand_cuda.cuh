/*
Please consider citing the following paper when using the code:

@inproceedings{hong2023pcengine,
  title={{Exploiting Hardware Utilization and Adaptive Dataflow for Efficient Sparse Convolution in 3D Point Clouds}},
  author={Hong, Ke and Yu, Zhongming and Dai, Guohao and Yang, Xinhao and Lian, Yaoxiu and Liu, Zehao and Xu, Ningyi and Wang, Yu},
  booktitle={Sixth Conference on Machine Learning and Systems (MLSys)},
  year={2023}
}
*/

#pragma once

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <torch/extension.h>
#include <mma.h>
#if __CUDA_ARCH__ >= 700
#include <cuda/pipeline>
#endif
#define DIV_UP(x, y) ((x) + (y) - 1) / (y)
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp32(const int *__restrict__ kpos,const int *__restrict__ qkpos, const int k_vol, const int c_in,const int c_out,const float *__restrict__ in_f,
    const float *__restrict__ kw, float *out_f,const int *imap, const int *omap) ;


template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_no_fusion_fp32(const int knnz,const int c_in,const int c_out,const float *__restrict__ in_f, const float *__restrict__ kw, float *out_f,const int *imap,
    const int *omap);


template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp32_once(const int *__restrict__ kpos,const int *__restrict__ qkpos, const int k_vol, const int c_in,const int c_out,const float *__restrict__ in_f,
    const float *__restrict__ kw, float *out_f,const int *imap, const int *omap);

template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp32_2(const int *__restrict__ kpos,const int *__restrict__ qkpos, const int k_vol, const int c_in,const int c_out,const float *__restrict__ in_f,
    const float *__restrict__ kw, float *out_f,const int *imap, const int *omap);

template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp32_1(const int *__restrict__ kpos,const int *__restrict__ qkpos, const int k_vol, const int c_in,const int c_out,const float *__restrict__ in_f,
    const float *__restrict__ kw, float *out_f,const int *imap, const int *omap);

at::Tensor conv_forward_fetch_on_demand_cuda(
    at::Tensor in_feat, at::Tensor kernel, 
    at::Tensor neighbor_map, const int sum_nnz, 
    at::Tensor neighbor_address, at::Tensor q_neighbor_address,
    const int output_size, const int qsum_nnz, const bool transpose, 
    const bool allow_tf32, const bool allow_fp16);

at::Tensor conv_forward_fetch_on_demand_no_fusion_cuda(
    at::Tensor in_feat, at::Tensor kernel,
    at::Tensor neighbor_map, at::Tensor neighbor_offset, 
    const int sum_nnz, const int output_size, const bool transpose, 
    const bool allow_tf32, const bool allow_fp16);
