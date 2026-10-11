cmake_minimum_required(VERSION 3.31)
set(CMAKE_CUDA_ARCHITECTURES "75;89")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8")
project(custom_pytorch_layers_and_operations LANGUAGES C CXX CUDA)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Development Interpreter REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
set(PYTHON_LIBRARY ${Python3_LIBRARIES})
execute_process(
        COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(${TORCH_PREFIX_PATH})
find_package(Torch REQUIRED PATHS ${TORCH_PREFIX_PATH})
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



include_directories(
        ${Python3_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/src)


set(OPS Deformable_ConvNets correlation lietorch_extra lietorch MinDepthFlowProjection
        SeparableConvFlow SeparableConv  vid_2_vid_correlation ChannelNorm Resample2D
        Interpolation InterpolationCH DepthFlowProjection FlowProjection FilterInterpolation
        DCNv2_latest ops_smm MultiScaleDeformableAttention IOU3D_NMS Pointnet2_batch ROI_aware_pool_3D Pointnet2_stack ROI_point_pool3D)
set_source_files_properties(src/deform_attn_ext.cpp PROPERTIESLANGUAGE CUDA)
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
target_link_libraries(lietorch_extra PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart ${Eigen3_LIBRARIES} pybind11::module
        Python3::Python correlation)
set_target_properties(lietorch_extra PROPERTIES
        SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lietorch)

add_library(lietorch SHARED ${LIETORCH_SRC})
target_include_directories(lietorch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
target_link_libraries(lietorch PRIVATE "${TORCH_LIBRARIES}" pybind11::module CUDA::cudart Python3::Python
        ${Eigen3_LIBRARIES} lietorch_extra correlation)
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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/kernels")



foreach (layer ${OPS})
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
        COMMENT "Installing all Python modules")
