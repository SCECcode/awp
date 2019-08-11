set(HEADERS 
   ${AWP_MINI_SOURCE_DIR}/include/awp/definitions.h 
   )

set(UNOPT_HEADERS 
   ${AWP_MINI_SOURCE_DIR}/include/topography/kernels/unoptimized.cuh
   ) 

set(OPT_HEADERS
    ${AWP_MINI_SOURCE_DIR}/include/topography/opt_topography.cuh     
    ${AWP_MINI_SOURCE_DIR}/include/topography/kernels/optimized.cuh     
    )

add_library(unoptimized_kernels
        unoptimized.cu 
        ${HEADERS} ${UNOPT_HEADERS})

target_include_directories(unoptimized_kernels
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

add_library(optimized_kernels
        optimized.cu 
        ${HEADERS} ${OPT_HEADERS})

target_include_directories(optimized_kernels
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )
