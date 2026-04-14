cmake_minimum_required(VERSION 3.31)
set(CMAKE_CUDA_ARCHITECTURES "75;89")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8")
project(MultiScaleDeformableAttention LANGUAGES C CXX CUDA)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Development Interpreter REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
execute_process(
        COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(${TORCH_PREFIX_PATH})
find_package(Torch REQUIRED PATHS ${TORCH_PREFIX_PATH})
set(MSDASRC src/cuda/ms_deform_attn_cuda.cu src/cpu/ms_deform_attn_cpu.cpp)
include_directories(
        ${Python3_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu
)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_${TORCH_CUDA_ARCH_LIST},code=sm_${TORCH_CUDA_ARCH_LIST}")


add_library(MultiScaleDeformableAttention SHARED ${MSDASRC})
target_include_directories(MultiScaleDeformableAttention PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(MultiScaleDeformableAttention PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart pybind11::module Python3::Python)
set_target_properties(MultiScaleDeformableAttention PROPERTIES SUFFIX ".so" PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/MultiScaleDeformableAttention)

add_custom_target(install_MultiScaleDeformableAttention
    COMMAND python3 setup.py build install
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Installing MultiScaleDeformableAttention Python module..."
    )