#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include "roipoint_pool3d_kernel.h"
#include "roipoint_pool3d_kernel.cuh"
void roipool3dLauncher(int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num,
                       const float *xyz, const float *boxes3d, const float *pts_feature, float *pooled_features, int *pooled_empty_flag){

    // printf("batch_size=%d, pts_num=%d, boxes_num=%d\n", batch_size, pts_num, boxes_num);
    int *pts_assign = nullptr;
    cudaMalloc(&pts_assign, batch_size * pts_num * boxes_num * sizeof(int));  // (batch_size, N, M)
    // cudaMemset(&pts_assign, -1, batch_size * pts_num * boxes_num * sizeof(int));

    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num, batch_size);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    assign_pts_to_box3d<<<blocks, threads>>>(batch_size, pts_num, boxes_num, xyz, boxes3d, pts_assign);

    int *pts_idx = NULL;
    cudaMalloc(&pts_idx, batch_size * boxes_num * sampled_pts_num * sizeof(int));  // (batch_size, M, sampled_pts_num)

    dim3 blocks2(DIVUP(boxes_num, THREADS_PER_BLOCK), batch_size);  // blockIdx.x(col), blockIdx.y(row)
    get_pooled_idx<<<blocks2, threads>>>(batch_size, pts_num, boxes_num, sampled_pts_num, pts_assign, pts_idx, pooled_empty_flag);

    dim3 blocks_pool(DIVUP(sampled_pts_num, THREADS_PER_BLOCK), boxes_num, batch_size);
    roipool3d_forward<<<blocks_pool, threads>>>(batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num,
                                                      xyz, pts_idx, pts_feature, pooled_features, pooled_empty_flag);

    cudaFree(pts_assign);
    cudaFree(pts_idx);

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}
int roipool3d_gpu(at::Tensor xyz, at::Tensor boxes3d, at::Tensor pts_feature, at::Tensor pooled_features, at::Tensor pooled_empty_flag){
    // params xyz: (B, N, 3)
    // params boxes3d: (B, M, 7)
    // params pts_feature: (B, N, C)
    // params pooled_features: (B, M, 512, 3+C)
    // params pooled_empty_flag: (B, M)
    CHECK_INPUT(xyz);
    CHECK_INPUT(boxes3d);
    CHECK_INPUT(pts_feature);
    CHECK_INPUT(pooled_features);
    CHECK_INPUT(pooled_empty_flag);

    int batch_size = xyz.size(0);
    int pts_num = xyz.size(1);
    int boxes_num = boxes3d.size(1);
    int feature_in_len = pts_feature.size(2);
    int sampled_pts_num = pooled_features.size(2);


    const float * xyz_data = xyz.data<float>();
    const float * boxes3d_data = boxes3d.data<float>();
    const float * pts_feature_data = pts_feature.data<float>();
    float * pooled_features_data = pooled_features.data<float>();
    int * pooled_empty_flag_data = pooled_empty_flag.data<int>();

    roipool3dLauncher(batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num,
                       xyz_data, boxes3d_data, pts_feature_data, pooled_features_data, pooled_empty_flag_data);

    return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &roipool3d_gpu, "roipool3d forward (CUDA)");
}

