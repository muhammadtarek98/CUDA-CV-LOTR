#pragma once
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <torch/extension.h>
#include <algorithm>
#include <chrono>


#define CONVERT_FLOAT(pointer) (reinterpret_cast<float *>(&(pointer))[0])
#define CONVERT_HALF2(pointer) (reinterpret_cast<half2 *>(&(pointer))[0])
#define CONVERT_INT4(pointer) (reinterpret_cast<int4 *>(&(pointer))[0])
template <typename scalar_t>
__global__ void gather_kernel(const int n_k, const int n_in, const int c,
                              scalar_t *in_feat, scalar_t *out_feat,
                              const int *kmap, const bool transpose);
template <typename scalar_t>
__global__ void scatter_kernel(const int n_in, const int n_out, const int c,
                               scalar_t *in_feat, scalar_t *out_feat,
                               const int *kmap, const bool transpose);

__global__ void scatter_all_kernel_pad_sep_with_mask_half(
    const int n, const int c, const int kernel_volume, half *in_feat,
    half *out_feat, const int *kmap, const int *kmap_sizes,
    const int *cum_kmap_sizes, const int *cum_buffer_sizes,
    const int *input_mask, const int *output_mask, const bool transpose,
    const bool precompute_mid);
template <typename scalar_t>
__global__ void gather_all_kernel_pad_sep_with_mask(
    const int n, const int c, const int kernel_volume, scalar_t *in_feat,
    scalar_t *out_feat, const int *kmap, const int *kmap_sizes,
    const int *cum_kmap_sizes, const int *cum_buffer_sizes,
    const int *input_mask, const int *output_mask, const bool transpose,
    const bool precompute_mid) ;

template <typename scalar_t>
__global__ void scatter_all_kernel_pad_sep(
    const int c, const int kernel_volume, scalar_t *in_feat, scalar_t *out_feat,
    const int *kmap, const int *kmap_sizes, const int *cum_kmap_sizes,
    const int *cum_buffer_sizes, const bool transpose,
    const bool precompute_mid) ;


void group_strategy_generation(
    // inputs
    int kernel_volume, float epsilon, int mm_thresh, int conv_mode,
    at::Tensor neighbor_offset, bool precompute_mid,
    // outputs
    std::vector<std::vector<int>> &groups, std::vector<int> &mm_ops,
    std::vector<int> &group_sizes, at::Tensor cum_buffer_sizes,
    int &buffer_size) ;
at::Tensor conv_forward_gather_scatter_cuda(
    at::Tensor in_feat, at::Tensor kernel, at::Tensor neighbor_map,
    at::Tensor neighbor_offset, at::Tensor input_mask, at::Tensor output_mask,
    const int output_size, const float epsilon, const int mm_thresh,
    const int conv_mode, const bool transpose, at::Tensor buffer);

at::Tensor conv_forward_gather_scatter_cuda_latest(
    at::Tensor in_feat, at::Tensor kernel, at::Tensor neighbor_map,
    at::Tensor neighbor_offset, at::Tensor input_mask, at::Tensor output_mask,
    const int output_size, const float epsilon, const int mm_thresh,
    const int conv_mode, const bool transpose, at::Tensor buffer);

at::Tensor conv_forward_gather_scatter_cuda_fallback(
    at::Tensor in_feat, at::Tensor kernel, at::Tensor neighbor_map,
    const int output_size, const int conv_mode, at::Tensor neighbor_offset,
    const bool transpose);

void conv_backward_gather_scatter_cuda(at::Tensor in_feat, at::Tensor grad_in_feat,
                               at::Tensor grad_out_feat, at::Tensor kernel,
                               at::Tensor grad_kernel, at::Tensor neighbor_map,
                               at::Tensor neighbor_offset,
                               const bool transpose);
