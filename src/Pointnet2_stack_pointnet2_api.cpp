#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "Pointnet2_stack_ball_query_gpu.hpp"
#include "Pointnet2_stack_group_points_gpu.hpp"
#include "Pointnet2_stack_sampling_gpu.hpp"
#include "Pointnet2_stack_interpolate_gpu.hpp"
#include "Pointnet2_stack_voxel_query_gpu.hpp"
#include "Pointnet2_stack_vector_pool_gpu.hpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ball_query_wrapper", &ball_query_wrapper_stack, "ball_query_wrapper_stack");
    m.def("voxel_query_wrapper", &voxel_query_wrapper_stack, "voxel_query_wrapper_stack");

    m.def("farthest_point_sampling_wrapper", &farthest_point_sampling_wrapper, "farthest_point_sampling_wrapper");
    m.def("stack_farthest_point_sampling_wrapper", &stack_farthest_point_sampling_wrapper, "stack_farthest_point_sampling_wrapper");

    m.def("group_points_wrapper", &group_points_wrapper_stack, "group_points_wrapper_stack");
    m.def("group_points_grad_wrapper", &group_points_grad_wrapper_stack, "group_points_grad_wrapper_stack");

    m.def("three_nn_wrapper", &three_nn_wrapper_stack, "three_nn_wrapper_stack");
    m.def("three_interpolate_wrapper", &three_interpolate_wrapper_stack, "three_interpolate_wrapper_stack");
    m.def("three_interpolate_grad_wrapper", &three_interpolate_grad_wrapper_stack, "three_interpolate_grad_wrapper_stack");

    m.def("query_stacked_local_neighbor_idxs_wrapper_stack", &query_stacked_local_neighbor_idxs_wrapper_stack, "query_stacked_local_neighbor_idxs_wrapper_stack");
    m.def("query_three_nn_by_stacked_local_idxs_wrapper_stack", &query_three_nn_by_stacked_local_idxs_wrapper_stack, "query_three_nn_by_stacked_local_idxs_wrapper_stack");

    m.def("vector_pool_wrapper", &vector_pool_wrapper_stack, "vector_pool_grad_wrapper_stack");
    m.def("vector_pool_grad_wrapper", &vector_pool_grad_wrapper_stack, "vector_pool_grad_wrapper_stack");
}
