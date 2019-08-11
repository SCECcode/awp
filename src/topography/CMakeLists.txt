add_subdirectory(kernels)
add_subdirectory(initializations)
add_subdirectory(metrics)
add_subdirectory(readers)
add_subdirectory(geometry)

set(HEADERS
    ${AWP_MINI_SOURCE_DIR}/include/awp/definitions.h     
    ${AWP_MINI_SOURCE_DIR}/include/topography/topography.h     
    ${AWP_MINI_SOURCE_DIR}/include/grid/grid_3d.h
    ${AWP_MINI_SOURCE_DIR}/include/grid/shift.h
    ${AWP_MINI_SOURCE_DIR}/include/topography/geometry.h
    ${AWP_MINI_SOURCE_DIR}/include/topography/geometry/geometry.h
    ${AWP_MINI_SOURCE_DIR}/include/topography/readers/serial_reader.h
    ${AWP_MINI_SOURCE_DIR}/include/test/test.h     
    ${AWP_MINI_SOURCE_DIR}/include/vtk/vtk.h     
    ${AWP_MINI_SOURCE_DIR}/include/topography/metrics/metrics.h
    ${AWP_MINI_SOURCE_DIR}/include/topography/host.h
    )

set(UNOPT_HEADERS
    ${HEADERS}
    ${AWP_MINI_SOURCE_DIR}/include/topography/topography.cuh     
    ${AWP_MINI_SOURCE_DIR}/include/topography/kernels/unoptimized.cuh     
    )

set(OPT_HEADERS
    ${HEADERS}
    ${AWP_MINI_SOURCE_DIR}/include/topography/opt_topography.cuh     
    ${AWP_MINI_SOURCE_DIR}/include/topography/kernels/optimized.cuh     
    )

set(LIBRARIES
    ${MPI_C_LIBRARIES} 
    grid 
    metrics
    geometry
    topography_readers
    vtk
    )

set(UNOPT_LIBRARIES ${LIBRARIES} unoptimized_kernels)
set(OPT_LIBRARIES ${LIBRARIES} optimized_kernels)


# Unoptimized version
add_library(topography
        topography.c topography.cu geometry.c host.c
        ${UNOPT_HEADERS}
        )

target_include_directories(topography
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
       )

target_link_libraries(topography 
        ${UNOPT_LIBRARIES}
        )

# Optimized version
add_library(opt_topography
        topography.c opt_topography.cu geometry.c host.c
        ${OPT_HEADERS}
        )

target_include_directories(opt_topography
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
       )

target_link_libraries(opt_topography 
        ${OPT_LIBRARIES} 
        )
