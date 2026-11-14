find_path(CUDSS_INCLUDE_DIR
        NAMES cudss.h
        HINTS
        ${CUDSS_ROOT}/include
        $ENV{CUDSS_ROOT}/include
        ${CUDSS_INCLUDE_PATH}
        $ENV{CUDSS_INCLUDE_PATH}
        PATH_SUFFIXES include
        DOC "Path to cuDSS include directory"
)

find_library(CUDSS_LIBRARY
        NAMES cudss libcudss
        HINTS
        ${CUDSS_ROOT}/lib
        ${CUDSS_ROOT}/lib64
        $ENV{CUDSS_ROOT}/lib
        $ENV{CUDSS_ROOT}/lib64
        ${CUDSS_LIBRARY_PATH}
        $ENV{CUDSS_LIBRARY_PATH}
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
        DOC "Path to cuDSS library"
)

# Handle REQUIRED and QUIET arguments, and set CUDSS_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDSS
        REQUIRED_VARS CUDSS_LIBRARY CUDSS_INCLUDE_DIR
)

# Create an imported target for modern CMake target_link_libraries usage
if(CUDSS_FOUND AND NOT TARGET CUDSS::cudss)
    add_library(CUDSS::cudss UNKNOWN IMPORTED)
    set_target_properties(CUDSS::cudss PROPERTIES
            IMPORTED_LOCATION "${CUDSS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUDSS_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(CUDSS_INCLUDE_DIR CUDSS_LIBRARY)