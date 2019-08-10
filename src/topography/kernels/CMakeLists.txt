set(HEADERS 
   ${AWP_MINI_SOURCE_DIR}/include/awp/definitions.h 
   ${AWP_MINI_SOURCE_DIR}/include/topography/kernels/unoptimized.cuh) 

add_library(unoptimized_kernels
        unoptimized.cu 
        ${HEADERS})

target_include_directories(unoptimized_kernels
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )
