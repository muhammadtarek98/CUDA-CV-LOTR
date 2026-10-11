#pragma once
#include<cuda_runtime_api.h>

#include "roipoint_pool3d_kernel.h"
__device__ inline void lidar_to_local_coords(float shift_x, float shift_y, float rot_angle, float &local_x, float &local_y);
__device__ inline int check_pt_in_box3d(const float *pt, const float *box3d, float &local_x, float &local_y);
__global__ void assign_pts_to_box3d(int batch_size, int pts_num, int boxes_num, const float *xyz, const float *boxes3d, int *pts_assign);
__global__ void get_pooled_idx(int batch_size, int pts_num, int boxes_num, int sampled_pts_num,
                               const int *pts_assign, int *pts_idx, int *pooled_empty_flag);
__global__ void roipool3d_forward(int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num,
                                   const float *xyz, const int *pts_idx, const float *pts_feature,
                                   float *pooled_features, int *pooled_empty_flag);

