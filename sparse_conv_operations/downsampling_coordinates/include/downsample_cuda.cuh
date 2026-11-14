#pragma once
#include <torch/extension.h>
#include <torch/torch.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#define NDim 4
#define Tuple std::vector<int>
__host__ __device__ int get_output_coords(int kernel_volume, int *in_coords,
                                          int *kernel_sizes, int *stride,
                                          int *coords_min,
                                          int *coords_max,
                                          int *padding,
                                          int *out_coords) ;



__global__ void get_output_coords_kernel(int n_points, int n_kernel,
                                         int *in_coords, int *kernel_sizes,
                                         int *stride,
                                         int *coords_min, int *coords_max,
                                         int *padding,
                                         int *n_out_points,
                                         int64_t *transformed_coords);

__global__ void inverse_transform_coords_kernel(int n_points,
                                                int64_t *in_coords,
                                                int *coords_min,
                                                int *coords_max,
                                                int *out_coords) ;


at::Tensor downsample_cuda(at::Tensor _in_coords, at::Tensor _coords_max,
                           at::Tensor _coords_min, at::Tensor _kernel_sizes,
                           at::Tensor _stride, at::Tensor _padding);