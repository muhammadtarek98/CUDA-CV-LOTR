set(KERNEL_MAP_CONSTRUCT_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/pybind_cpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_construction/kernel/voxelize_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_construction/src/voxelize_cpu.cpp
        )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_construction/include)
add_library(kernel_map_construction SHARED ${KERNEL_MAP_CONSTRUCT_SRC})
target_include_directories(kernel_map_construction PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_construction/include)
target_link_libraries(kernel_map_construction PRIVATE ${TORCH_LIBRARIES}
        pybind11::module CUDA::cudart Python3::Python)
set_target_properties(kernel_map_construction PROPERTIES SUFFIX ".so" PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_construction)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/sparse_conv_operations/kernel_map_construction/kernel/")
add_custom_target(install_kernel_map_construction)