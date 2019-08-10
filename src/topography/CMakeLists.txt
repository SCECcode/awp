add_subdirectory(grid)
add_subdirectory(kernels)

set(HEADERS
    ${AWP_MINI_SOURCE_DIR}/include/awp/definitions.h     
    ${AWP_MINI_SOURCE_DIR}/include/topography/topography.h     
    ${AWP_MINI_SOURCE_DIR}/include/topography/grid/shift.h     
    ${AWP_MINI_SOURCE_DIR}/include/topography/grid/grid_3d.h
    ${AWP_MINI_SOURCE_DIR}/include/topography/topography.cuh     
    ${AWP_MINI_SOURCE_DIR}/include/topography/kernels/unoptimized.cuh     
    ${AWP_MINI_SOURCE_DIR}/include/test/test.h     
    ${AWP_MINI_SOURCE_DIR}/include/topography/metrics/metrics.h
    )

add_library(topography
        topography.c topography.cu   ${HEADERS}
        )


target_include_directories(topography
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
       )

target_link_libraries(topography ${MPI_C_LIBRARIES} grid_3d unoptimized_kernels)
