cmake_minimum_required(VERSION 3.31)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CUDA_ARCHITECTURES "75;89")
set(TORCH_CUDA_ARCH_LIST "7.5;8.9")

project(CUDA_CV_LOTR LANGUAGES C CXX CUDA)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
include(FetchContent)
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Development Interpreter REQUIRED)
find_package(pybind11 QUIET)
find_package(CUDNN REQUIRED)
find_package(CUDSS REQUIRED)
set(USE_CUSPARSELT 1)
set(USE_CUDNN 1)
set(USE_CUDSS 1)
set(USE_CUFILE 1)

find_package(sparsehash QUIET)
if (sparsehash_FOUND)
    message("sparsehash found")
    set(USE_SPARSEHASH TRUE)
else ()
    FetchContent_Declare(
            sparsehash
            GIT_REPOSITORY https://github.com/muhammadtarek98/sparsehash.git
            GIT master
            GIT_SHALLOW FALSE
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sparsehash

    )
    FetchContent_MakeAvailable(sparsehash)
    set(USE_SPARSEHASH TRUE)
endif ()
if (pybind11_FOUND)
    message("pybind11 found")
    set(USE_PYBIND11 TRUE)
else ()
        FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG master
        GIT_SHALLOW FALSE
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11)
    FetchContent_MakeAvailable(pybind11)
        set(USE_PYBIND11 TRUE)
endif ()


find_package(Eigen3 QUIET)
if (Eigen3_FOUND)
    message("Eigen3 found")
    set(USE_Eigen3 TRUE)
else ()
    FetchContent_Declare(
            Eigen3
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG master
            GIT_SHALLOW FALSE
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/Eigen3
    )
    FetchContent_MakeAvailable(Eigen3)
    set(USE_EIGEN3 TRUE)

endif ()
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
set(PYTHON_LIBRARY ${Python3_LIBRARIES})
execute_process(
        COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(Torch REQUIRED PATHS "${TORCH_PREFIX_PATH}")

set(DEFORMABLE_CONVNET_V2 kernels/deform_attn_cuda_kernel.cu src/deform_attn_cuda_pt109.cpp
        src/deform_attn_cuda_pt110.cpp src/deform_attn_ext.cpp)
set(LIETORCH_SRC src/lietorch.cpp src/lietorch_cpu.cpp kernels/lietorch_gpu.cu)
set(LIE_TORCH_EXTRA_SRC src/extras.cpp kernels/altcorr_kernel.cu kernels/corr_index_kernel.cu
        kernels/se3_solver.cu kernels/se3_builder.cu kernels/se3_inplace_builder.cu)
set(INTERPOLATIONCH_SRC src/interpolationch_cuda.cc kernels/interpolationch_cuda_kernel.cu)
set(SEPARABLECONVFLOW_SRC src/separableconvflow_cuda.cc kernels/separableconvflow_cuda_kernel.cu)
set(SEPARABLECONV_SRC src/separableconv_cuda.cc kernels/separableconv_cuda_kernel.cu)
set(MINDEPTHFLOWPROJECTION_SRC src/mindepthflowprojection_cuda.cc kernels/mindepthflowprojection_cuda_kernel.cu)
set(FLOWPROJECTION_SRC src/flowprojection_cuda.cc kernels/flowprojection_cuda_kernel.cu)
set(INTERPOLATION_SRC src/interpolation_cuda.cc kernels/interpolation_cuda_kernel.cu)
set(FILTERINTERPOLATION_SRC src/filterinterpolation_cuda.cc  kernels/filterinterpolation_cuda_kernel.cu)
set(DEPTHFLOWPROJECTION_SRC src/depthflowprojection_cuda.cc kernels/depthflowprojection_cuda_kernel.cu)
set(CORRELATION_SRC src/correlation_cuda.cc kernels/correlation_cuda_kernel.cu)
set(RAFT_CORRELATION_SRC src/correlation.cpp kernels/correlation_kernel.cu)
set(CHANNELNORM_SRC src/channelnorm_cuda.cc kernels/channelnorm_kernel.cu)
set(VID2VID_CORRELATION_SRC src/vid2vid_correlation_cuda.cc kernels/vid2vid_correlation_cuda_kernel.cu)
set(RESAMPLE_2D_SRC src/resample2d_cuda.cc kernels/resample2d_kernel.cu)
set(DCNV2_LATEST src/dcn_v2_cpu.cpp src/dcn_v2_im2col_cpu.cpp src/vision.cpp src/dcn_v2_psroi_pooling_cpu.cpp
    kernels/dcn_v2_psroi_pooling_cuda.cu kernels/dcn_v2_im2col_cuda.cu kernels/dcn_v2_cuda.cu
        kernels/correlation_kernel.cu src/correlation.cpp)
set(OPS_SMM kernels/smm_cuda.cu)
set(MS_DEFORMABLE_ATTN_SRC src/ms_deform_attn_vision.cpp src/ms_deform_attn_cpu.cpp
kernels/ms_deform_attn_cuda.cu)
set(IOU3D_NMS_SRC kernels/iou3d_nms_kernel.cu src/iou3d_cpu.cpp src/iou3d_nms.cpp src/iou3d_nms_api.cpp)
set(POINTNET2_BATCH_SRC kernels/Pointnet2_batch_ball_query_gpu.cu kernels/Pointnet2_batch_group_points_gpu.cu kernels/Pointnet2_batch_interpolate_gpu.cu
        kernels/Pointnet2_batch_sampling_gpu.cu src/Pointnet2_batch_ball_query.cpp src/Pointnet2_batch_group_points.cpp src/Pointnet2_batch_interpolate.cpp
        src/Pointnet2_batch_pointnet2_api.cpp src/Pointnet2_batch_sampling.cpp)

set(POINTNET2_STACK_SRC kernels/Pointnet2_stack_ball_query_gpu.cu kernels/Pointnet2_stack_group_points_gpu.cu kernels/Pointnet2_stack_interpolate_gpu.cu kernels/Pointnet2_stack_sampling_gpu.cu
        kernels/Pointnet2_stack_vector_pool_gpu.cu kernels/Pointnet2_stack_voxel_query_gpu.cu src/Pointnet2_stack_ball_query.cpp src/Pointnet2_stack_group_points.cpp src/Pointnet2_stack_interpolate.cpp
        src/Pointnet2_stack_pointnet2_api.cpp src/Pointnet2_stack_sampling.cpp src/Pointnet2_stack_vector_pool.cpp src/Pointnet2_stack_voxel_query.cpp)

set(ROI_AWARE_POOL_3D_SRC kernels/roiaware_pool3d_kernel.cu src/roiaware_pool3d.cpp)
set(ROI_POINT_POOL3D kernels/roipoint_pool3d_kernel.cu src/roipoint_pool3d.cpp)

set(COORD_HASH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu ./sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/coordinate_hashing/src/hash_cpu.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/coordinate_hashing/kernel/hash_cuda.cu)
set(COORD_COUNT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu ./sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/count_coordinates/kernel/count_cuda.cu ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/count_coordinates/src/count_cpu.cpp)
set(DEVOXELIZE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/devoxelize/kernel/devoxelize_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/devoxelize/src/devoxelize_cpu.cpp)

set(DOWNSAMPLING_COORDS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/downsampling_coordinates/kernel/downsample_cuda.cu
)

set(FOD_CONV_BW_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/FOD_convolution_backward/kernel/convolution_gather_scatter_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/FOD_convolution_backward/src/convolution_gather_scatter_cpu.cpp
)

set(FOD_CONV_FW_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/FOD_convolution_forward/kernel/convolution_forward_fetch_on_demand_cuda.cu
        #${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/FOD_convolution_forward/src/convolution_forward_fetch_on_demand_cuda.cpp
)
set(CONV_GATHER_AND_SCATTER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/gather_scatter_convolution/kernel/convolution_gather_scatter_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/gather_scatter_convolution/src/convolution_gather_scatter_cpu.cpp
)

set(HASH_QUERY_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/hash_query/kernel/hash_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/hash_query/src/hash_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/hash_query/src/query_cpu.cpp
)


set(HASH_TABLE_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/hash_table/kernel/hashmap_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/hash_table/src/hashmap_cpu.cpp
)


set(IMP_GEMM_CONV_BW_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/implicit_gemm_convolution_backward/kernel/convolution_backward_wgrad_implicit_gemm_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/implicit_gemm_convolution_backward/kernel/convolution_backward_wgrad_implicit_gemm_sorted_cuda.cu
)

set(IMP_GEMM_CONV_FW_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/implicit_gemm_convolution_forward/kernel/convolution_forward_implicit_gemm_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/implicit_gemm_convolution_forward/kernel/convolution_forward_implicit_gemm_sorted_cuda.cu
)
set(KERNEL_COORD_HASH_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_coordinate_hashing/kernel/hash_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_coordinate_hashing/src/hash_cpu.cpp
)

set(KERNEL_MAP_EXC_SCAN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_exclusive_scan/kernel/exclusive_scan_cuda.cu
)

set(KERNEL_MAP_REORDERING_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_reordering/kernel/query_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_reordering/kernel/reduce_bitmask_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_reordering/kernel/reorder_map_cuda.cu
)
set(SPARSE_TO_DENSE_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/sparse_to_dense/kernel/sparse_to_dense_cuda.cu
)
set(VOXELIZE_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/voxelize/kernel/voxelize_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/voxelize/src/voxelize_cpu.cpp
)

set(KERNEL_MAP_CONSTRUCT_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_construction/kernel/hashmap_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_construction/kernel/sparsemapping_cuda.cu
)



include_directories(
        ${Python3_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/count_coordinates/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/coordinate_hashing/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/devoxelize/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/downsampling_coordinates/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/FOD_convolution_backward/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/FOD_convolution_forward/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/FOD_convolution_forward/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/hash_query/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/hash_table/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/implicit_gemm_convolution_backward/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/implicit_gemm_convolution_forward/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_coordinate_hashing/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_exclusive_scan/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_reordering/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/sparse_to_dense/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/voxelize/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/sparse_conv_utils/include
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_construction/include)




set(OPS kernel_map_construction voxelize sparse_to_dense kernel_map_exclusive_scan kernel_coordinate_hashing implicit_gemm_convolution_forward implicit_gemm_convolution_backward FOD_convolution_forward FOD_convolution_backward downsampling_coordinates devoxelize count_coordinates coordinate_hashing Deformable_ConvNets correlation lietorch_extra lietorch MinDepthFlowProjection
        SeparableConvFlow SeparableConv  vid_2_vid_correlation ChannelNorm Resample2D
        Interpolation InterpolationCH DepthFlowProjection FlowProjection FilterInterpolation
        DCNv2_latest ops_smm MultiScaleDeformableAttention IOU3D_NMS Pointnet2_batch ROI_aware_pool_3D Pointnet2_stack ROI_point_pool3D)
set_source_files_properties(src/deform_attn_ext.cpp PROPERTIES LANGUAGE CUDA)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_${TORCH_CUDA_ARCH_LIST},code=sm_${TORCH_CUDA_ARCH_LIST}")

add_library(IOU3D_NMS SHARED ${IOU3D_NMS_SRC})
target_include_directories(IOU3D_NMS PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(IOU3D_NMS PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(IOU3D_NMS PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/IOU3D_NMS)


add_library(ROI_aware_pool_3D SHARED ${ROI_AWARE_POOL_3D_SRC})
target_include_directories(ROI_aware_pool_3D PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ROI_aware_pool_3D PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(ROI_aware_pool_3D PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ROI_aware_pool_3D)


add_library(Pointnet2_batch SHARED ${POINTNET2_BATCH_SRC})
target_include_directories(Pointnet2_batch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(Pointnet2_batch PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(Pointnet2_batch PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Pointnet2_batch)


add_library(Pointnet2_stack SHARED ${POINTNET2_STACK_SRC})
target_include_directories(Pointnet2_stack PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(Pointnet2_stack PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(Pointnet2_stack PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Pointnet2_stack)


add_library(ROI_point_pool3D SHARED ${ROI_POINT_POOL3D})
target_include_directories(ROI_point_pool3D PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ROI_point_pool3D PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(ROI_point_pool3D PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ROI_point_pool3D)

add_library(MultiScaleDeformableAttention SHARED ${MS_DEFORMABLE_ATTN_SRC})
target_include_directories(MultiScaleDeformableAttention PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(MultiScaleDeformableAttention PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(MultiScaleDeformableAttention PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/MultiScaleDeformableAttention)


add_library(vid_2_vid_correlation SHARED ${VID2VID_CORRELATION_SRC})
target_include_directories(vid_2_vid_correlation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vid_2_vid_correlation PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(vid_2_vid_correlation PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/vid_2_vid_correlation)

add_library(correlation SHARED ${CORRELATION_SRC})
target_include_directories(correlation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(correlation PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(correlation PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/correlation)


add_library(ChannelNorm SHARED ${CHANNELNORM_SRC})
target_include_directories(ChannelNorm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ChannelNorm PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(ChannelNorm PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ChannelNorm)


add_library(Resample2D SHARED ${RESAMPLE_2D_SRC})
target_include_directories(Resample2D PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(Resample2D PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(Resample2D PROPERTIES SUFFIX ".so"  PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Resample2D)

add_library(Deformable_ConvNets SHARED ${DEFORMABLE_CONVNET_V2})
target_include_directories(Deformable_ConvNets PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(Deformable_ConvNets PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(Deformable_ConvNets PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Deformable_ConvNets)

add_library(DCNv2_latest SHARED ${DCNV2_LATEST})
target_include_directories(DCNv2_latest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(DCNv2_latest PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(DCNv2_latest PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/DCNv2_latest)



add_library(ops_smm SHARED ${OPS_SMM})
target_include_directories(ops_smm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ops_smm PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(ops_smm PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/ops_smm)

add_library(SeparableConvFlow SHARED ${SEPARABLECONVFLOW_SRC})
target_include_directories(SeparableConvFlow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(SeparableConvFlow PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(SeparableConvFlow PROPERTIES
        SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/SeparableConvFlow)


add_library(SeparableConv SHARED ${SEPARABLECONV_SRC})
target_include_directories(SeparableConv PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(SeparableConv PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(SeparableConv PROPERTIES
        SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/SeparableConv)


add_library(MinDepthFlowProjection SHARED ${MINDEPTHFLOWPROJECTION_SRC})
target_include_directories(MinDepthFlowProjection PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(MinDepthFlowProjection PRIVATE correlation  "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module
        Python3::Python)
set_target_properties(MinDepthFlowProjection PROPERTIES
        SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/MinDepthFlowProjection)

add_library(lietorch_extra SHARED ${LIE_TORCH_EXTRA_SRC})
target_include_directories(lietorch_extra PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lietorch_extra PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart Eigen3::Eigen pybind11::module
        Python3::Python correlation)
set_target_properties(lietorch_extra PROPERTIES
        SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lietorch)

add_library(lietorch SHARED ${LIETORCH_SRC})
target_include_directories(lietorch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
target_link_libraries(lietorch PRIVATE "${TORCH_LIBRARIES}" pybind11::module CUDA::cudart Python3::Python
        Eigen3::Eigen lietorch_extra correlation)
set_target_properties(lietorch PROPERTIES
        SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lietorch)


add_library(InterpolationCH SHARED ${INTERPOLATIONCH_SRC})
target_include_directories(InterpolationCH PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(InterpolationCH PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(InterpolationCH PROPERTIES
        SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/InterpolationCH)

add_library(Interpolation SHARED ${INTERPOLATION_SRC})
target_include_directories(Interpolation PRIVATE {CMAKE_CURRENT_SOURCE_DIR}/include )
target_link_libraries(Interpolation PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(Interpolation PROPERTIES
        SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Interpolation)


add_library(FlowProjection SHARED  ${FLOWPROJECTION_SRC})
target_include_directories(FlowProjection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(FlowProjection PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(FlowProjection PROPERTIES
        SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/FlowProjection)

add_library(FilterInterpolation SHARED ${FILTERINTERPOLATION_SRC})
target_include_directories(FilterInterpolation PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(FilterInterpolation PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(FilterInterpolation PROPERTIES
        SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/FilterInterpolation)


add_library(DepthFlowProjection SHARED ${DEPTHFLOWPROJECTION_SRC})
target_include_directories(DepthFlowProjection PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(DepthFlowProjection PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(DepthFlowProjection PROPERTIES
        SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/DepthFlowProjection)




add_library(coordinate_hashing SHARED ${COORD_HASH_SRC})
target_include_directories(coordinate_hashing PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/coordinate_hashing/include)
target_link_libraries(coordinate_hashing PRIVATE ${TORCH_LIBRARIES} pybind11::module CUDA::cudart Python3::Python)
set_target_properties(coordinate_hashing PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/coordinate_hashing)



add_library(count_coordinates SHARED ${COORD_COUNT_SRC})
target_include_directories(count_coordinates PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/count_coordinates/include)
target_link_libraries(count_coordinates PRIVATE ${TORCH_LIBRARIES} pybind11::module CUDA::cudart Python3::Python)
set_target_properties(count_coordinates PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/count_coordinates)

add_library(devoxelize SHARED ${DEVOXELIZE_SRC})
target_include_directories(devoxelize PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/devoxelize/include)
target_link_libraries(devoxelize PRIVATE ${TORCH_LIBRARIES} pybind11::module CUDA::cudart Python3::Python)
set_target_properties(devoxelize PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/devoxelize)

add_library(downsampling_coordinates SHARED ${DOWNSAMPLING_COORDS_SRC})
target_include_directories(downsampling_coordinates PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/downsampling_coordinates/include)
target_link_libraries(downsampling_coordinates PRIVATE ${TORCH_LIBRARIES} pybind11::module CUDA::cudart Python3::Python)
set_target_properties(downsampling_coordinates PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/downsampling_coordinates)


add_library(FOD_convolution_backward SHARED ${FOD_CONV_BW_SRC})
target_include_directories(FOD_convolution_backward PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/FOD_convolution_backward/include)
target_link_libraries(FOD_convolution_backward PRIVATE ${TORCH_LIBRARIES} pybind11::module CUDA::cudart Python3::Python)
set_target_properties(FOD_convolution_backward PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/FOD_convolution_backward)


add_library(FOD_convolution_forward SHARED ${FOD_CONV_FW_SRC})
target_include_directories(FOD_convolution_forward PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/FOD_convolution_forward/include)
target_link_libraries(FOD_convolution_forward PRIVATE ${TORCH_LIBRARIES} pybind11::module CUDA::cudart Python3::Python)
set_target_properties(FOD_convolution_forward PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/FOD_convolution_forward)

add_library(gather_scatter_convolution SHARED ${CONV_GATHER_AND_SCATTER_SRC})
target_include_directories(gather_scatter_convolution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/gather_scatter_convolution/include)
target_link_libraries(gather_scatter_convolution PRIVATE ${TORCH_LIBRARIES} pybind11::module CUDA::cudart Python3::Python)
set_target_properties(gather_scatter_convolution PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/gather_scatter_convolution)

add_library(hash_query SHARED ${HASH_QUERY_SRC})
target_include_directories(hash_query PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/hash_query/include)
target_link_libraries(hash_query PRIVATE ${TORCH_LIBRARIES} pybind11::module CUDA::cudart Python3::Python sparsehash::sparsehash)
set_target_properties(hash_query PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/hash_query)

add_library(hash_table SHARED ${HASH_TABLE_SRC})
target_include_directories(hash_table PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/hash_table/include)
target_link_libraries(hash_table PRIVATE ${TORCH_LIBRARIES}
        pybind11::module CUDA::cudart Python3::Python sparsehash::sparsehash)
set_target_properties(hash_table PROPERTIES SUFFIX ".so" PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/hash_table)


add_library(implicit_gemm_convolution_backward SHARED ${IMP_GEMM_CONV_BW_SRC})
target_include_directories(implicit_gemm_convolution_backward PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/implicit_gemm_convolution_backward/include)
target_link_libraries(implicit_gemm_convolution_backward PRIVATE ${TORCH_LIBRARIES}
        pybind11::module CUDA::cudart Python3::Python)
set_target_properties(implicit_gemm_convolution_backward PROPERTIES SUFFIX ".so" PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/implicit_gemm_convolution_backward)


add_library(implicit_gemm_convolution_forward SHARED ${IMP_GEMM_CONV_FW_SRC})
target_include_directories(implicit_gemm_convolution_forward PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/implicit_gemm_convolution_forward/include)
target_link_libraries(implicit_gemm_convolution_forward PRIVATE ${TORCH_LIBRARIES}
        pybind11::module CUDA::cudart Python3::Python)
set_target_properties(implicit_gemm_convolution_forward PROPERTIES SUFFIX ".so" PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/implicit_gemm_convolution_forward)

add_library(kernel_coordinate_hashing SHARED ${KERNEL_COORD_HASH_SRC})
target_include_directories(kernel_coordinate_hashing PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_coordinate_hashing/include)
target_link_libraries(kernel_coordinate_hashing PRIVATE ${TORCH_LIBRARIES}
        pybind11::module CUDA::cudart Python3::Python)
set_target_properties(kernel_coordinate_hashing PROPERTIES SUFFIX ".so" PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_coordinate_hashing)

add_library(kernel_map_exclusive_scan SHARED ${KERNEL_COORD_HASH_SRC})
target_include_directories(kernel_map_exclusive_scan PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_exclusive_scan/include)
target_link_libraries(kernel_map_exclusive_scan PRIVATE ${TORCH_LIBRARIES}
        pybind11::module CUDA::cudart Python3::Python)
set_target_properties(kernel_map_exclusive_scan PROPERTIES SUFFIX ".so" PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_exclusive_scan)


add_library(kernel_map_reordering SHARED ${KERNEL_MAP_REORDERING_SRC})
target_include_directories(kernel_map_reordering PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_reordering/include)
target_link_libraries(kernel_map_reordering PRIVATE ${TORCH_LIBRARIES}
        pybind11::module CUDA::cudart Python3::Python)
set_target_properties(kernel_map_reordering PROPERTIES SUFFIX ".so" PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_reordering)

add_library(sparse_to_dense SHARED ${SPARSE_TO_DENSE_SRC})
target_include_directories(sparse_to_dense PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/sparse_to_dense/include)
target_link_libraries(sparse_to_dense PRIVATE ${TORCH_LIBRARIES}
        pybind11::module CUDA::cudart Python3::Python)
set_target_properties(sparse_to_dense PROPERTIES SUFFIX ".so" PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/sparse_to_dense)


add_library(voxelize SHARED ${VOXELIZE_SRC})
target_include_directories(voxelize PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/voxelize/include)
target_link_libraries(voxelize PRIVATE ${TORCH_LIBRARIES}
        pybind11::module CUDA::cudart Python3::Python)
set_target_properties(voxelize PROPERTIES SUFFIX ".so" PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/voxelize)

add_library(kernel_map_construction SHARED ${KERNEL_MAP_CONSTRUCT_SRC})
target_include_directories(kernel_map_construction PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_construction/include)
target_link_libraries(kernel_map_construction PRIVATE ${TORCH_LIBRARIES}
        pybind11::module CUDA::cudart Python3::Python)
set_target_properties(kernel_map_construction PROPERTIES SUFFIX ".so" PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_construction)



set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/kernels")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/coordinate_hashing/kernel/")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/count_coordinates/kernel/")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/devoxelize/kernel/")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/FOD_convolution_backward/kernel/")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/FOD_convolution_forward/kernel/")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/hash_query/kernel/")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/gather_scatter_convolution/kernel/")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/hash_table/kernel/")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/implicit_gemm_convolution_backward/kernel/")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/implicit_gemm_convolution_forward/kernel/")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_coordinate_hashing/kernel/")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_exclusive_scan/kernel/")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_reordering/kernel/")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/sparse_to_dense/kernel/")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/voxelize/kernel/")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_construction/kernel/")

foreach (layer ${OPS})
    set_target_properties(${layer} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_custom_target(install_${layer}
    COMMAND python3 setup.py build develop
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${layer}
    COMMENT "Installing ${layer} Python module..."
    )
endforeach ()
add_custom_target(install_all DEPENDS
        install_vid_2_vid_correlation
        install_ChannelNorm
        install_correlation
        install_Resample2D
        install_Deformable_ConvNets
        install_DCNv2_latest
        install_lietorch
        install_MinDepthFlowProjection
        install_Interpolation
        install_InterpolationCH
        install_FlowProjection
        install_ops_smm
        install_DepthFlowProjection
        install_FilterInterpolation
        install_SeparableConv install_SeparableConvFlow
        install_IOU3D_NMS install_ROI_aware_pool_3D
        install_Pointnet2_batch install_Pointnet2_stack
        install_ROI_point_pool3D install_MultiScaleDeformableAttention
        install_coordinate_hashing
        install_devoxelize
        install_downsampling_coordinates
        install_FOD_convolution_backward
        install_gather_scatter_convolution
        install_hash_query
        install_hash_table
        install_implicit_gemm_convolution_backward
        install_implicit_gemm_convolution_forward
        install_kernel_coordinate_hashing
        install_kernel_map_exclusive_scan
        install_kernel_map_reordering
        install_sparse_to_dense
        install_kernel_map_construction
        COMMENT "Installing all Python modules")
