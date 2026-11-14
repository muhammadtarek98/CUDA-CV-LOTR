#pragma once
#include <torch/torch.h>
#include "hashmap_cuda.cuh"
#include <torch/extension.h>
#include <torch/torch.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#define NDim 4
#define MAX_KVOL 27

std::vector<at::Tensor> build_mask_from_kmap(int n_points, int n_out_points,
                                             at::Tensor _kmap,
                                             at::Tensor _kmap_sizes);

std::vector<at::Tensor> build_kernel_map_subm_hashmap(
    hashtable& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride,
    at::Tensor padding, bool to_insert);

std::vector<at::Tensor> build_kernel_map_downsample_hashmap(
    hashtable& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride,
    at::Tensor _padding, bool to_insert);

std::vector<at::Tensor> build_kernel_map_subm_hashmap_int32(
    hashtable32& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride,
    at::Tensor padding, bool to_insert);

std::vector<at::Tensor> build_kernel_map_downsample_hashmap_int32(
    hashtable32& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride,
    at::Tensor _padding, bool to_insert);
template <typename type_int>  // int32_t or int64_t
__host__ __device__ inline type_int transform_coords(int *in_coords,int *coords_min,int *coords_max);
template <typename type_int>  // int32_t or int64_t
__host__ __device__ inline void inverse_transform_coords(type_int *in_coords,int *coords_min,int *coords_max,int *out_coords) ;
template <typename type_hashtable_device_view, typename type_int>  // int32_t or int64_t
__global__ void inverse_transform_coords_and_insert_kernel(type_hashtable_device_view table,int n_points,type_int *in_coords,
    int *coords_min,int *coords_max,int *out_coords);

template <typename type_int, bool odd>  // int32_t or int64_t
__global__ void downsample_grid_kmap_stage1_specialized_fast(int n_points, int kernel_volume, int *in_coords, int *kernel_sizes, int *stride,
    int *padding, int *coords_min, int *coords_max, int *n_out_points,type_int *transformed_coords, type_int *out_in_map);

template <typename type_hashtable_device_view, typename type_int>  // int32_t or int64_t
__global__ void downsample_hashmap_kmap_stage3(type_hashtable_device_view table,int n_points, int n_points_out,
    int kernel_volume,type_int *in_out_in_map,int *out_in_map);

template <typename type_hashtable_device_view, typename type_int>  // int32_t or int64_t
__global__ void subm_hashmap_kmap_stage1(type_hashtable_device_view table,int n_points, int kernel_volume,int *in_coords,
    int *coords_min,int *coords_max, type_int *out_coords);

template <typename type_hashtable_device_view, typename type_int>  // int32_t or int64_t
__global__ void subm_hashmap_kmap_stage2_odd_kernel(type_hashtable_device_view table,int n_points, int kernel_volume,
    int *in_coords, int *coords_min,int *coords_max, int *kernel_sizes,int *out_in_map);
template <typename type_hashtable_device_view, typename type_int>  // int32_t or int64_t
__global__ void subm_hashmap_kmap_stage2_even_kernel(type_hashtable_device_view table,int n_points, int kernel_volume,int *in_coords,
    int *coords_min,int *coords_max, int *kernel_sizes,int *out_in_map) ;

__global__ void get_masks_from_kmap_kernel(int n_points, int n_points_out,int kernel_volume, int *kmap,int *kmap_sizes,
    int *cum_kmap_sizes,int *input_mask, int *output_mask) ;
