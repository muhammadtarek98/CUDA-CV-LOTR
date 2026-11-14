#pragma once
#include <torch/extension.h>
#include "memory.cuh"
#include <cuda_fp16.h>
#include <stdexcept>
template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
__global__ void __launch_bounds__(32) conv_backward_cuda_setting1_mode1_f16f16f32(int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters, int split_mask_len,
    int reduced_mask_len, int reorder_loc_len, half *__restrict__ A, half *__restrict__ B, int *__restrict__ reduced_mask, int *__restrict__ out_in_map, int *__restrict__ reorder_loc,
    half *__restrict__ C);
__global__ void __launch_bounds__(64) conv_backward_cuda_setting2_mode1_f16f16f32(int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters, int split_mask_len,
    int reduced_mask_len, int reorder_loc_len, half *__restrict__ A, half *__restrict__ B, int *__restrict__ reduced_mask, int *__restrict__ out_in_map, int *__restrict__ reorder_loc,
    half *__restrict__ C);

template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
__global__ void __launch_bounds__(32) conv_backward_cuda_setting1_mode1_tf32tf32f32(int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters, int split_mask_len,
    int reduced_mask_len, int reorder_loc_len, float *__restrict__ A, float *__restrict__ B, int *__restrict__ reduced_mask, int *__restrict__ out_in_map, int *__restrict__ reorder_loc,
    float *__restrict__ C);

__global__ void __launch_bounds__(64) conv_backward_cuda_setting2_mode1_tf32tf32f32(int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters, int split_mask_len,
    int reduced_mask_len, int reorder_loc_len, float *__restrict__ A, float *__restrict__ B, int *__restrict__ reduced_mask, int *__restrict__ out_in_map, int *__restrict__ reorder_loc,
    float *__restrict__ C);
template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
__global__ void __launch_bounds__(32) conv_backward_cuda_setting1_mode1_f32f32f32(int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters, int split_mask_len,
    int reduced_mask_len, int reorder_loc_len, float *__restrict__ A, float *__restrict__ B, int *__restrict__ reduced_mask, int *__restrict__ out_in_map, int *__restrict__ reorder_loc,
    float *__restrict__ C);

__global__ void __launch_bounds__(64) conv_backward_cuda_setting2_mode1_f32f32f32(int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters, int split_mask_len,
    int reduced_mask_len, int reorder_loc_len, float *__restrict__ A, float *__restrict__ B, int *__restrict__ reduced_mask, int *__restrict__ out_in_map, int *__restrict__ reorder_loc,
    float *__restrict__ C);



at::Tensor conv_backward_wgrad_implicit_gemm_sorted_cuda(
                       torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _out_in_map,torch::Tensor _reduced_mask,
                       torch::Tensor _reorder_loc, const int split_k_iters,
                       bool allow_tf32, bool allow_fp16);
