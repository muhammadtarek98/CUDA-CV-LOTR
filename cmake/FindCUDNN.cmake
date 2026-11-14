find_path(CUDNN_INCLUDE_DIR
        NAMES cudnn.h cudnn_version.h
        HINTS
        ${CUDNN_ROOT}/include
        $ENV{CUDNN_ROOT}/include
        ${CUDA_TOOLKIT_ROOT_DIR}/include
        /usr/local/cuda-12.8/include
        /usr/local/cuda/include
        PATH_SUFFIXES include
        DOC "Path to cuDNN include directory")

# Locate cuDNN library file
find_library(CUDNN_LIBRARY
        NAMES cudnn libcudnn
        HINTS
        ${CUDNN_ROOT}/lib
        ${CUDNN_ROOT}/lib64
        $ENV{CUDNN_ROOT}/lib
        $ENV{CUDNN_ROOT}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        /usr/local/cuda-12.8/lib64
        /usr/local/cuda/lib64
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
        DOC "Path to cuDNN library"
)

# Handle REQUIRED and QUIET arguments, setting CUDNN_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN
        REQUIRED_VARS CUDNN_LIBRARY CUDNN_INCLUDE_DIR
)

# Create imported target for CUDNN::cudnn
if(CUDNN_FOUND AND NOT TARGET CUDNN::cudnn)
    add_library(CUDNN::cudnn UNKNOWN IMPORTED)
    set_target_properties(CUDNN::cudnn PROPERTIES
            IMPORTED_LOCATION "${CUDNN_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)