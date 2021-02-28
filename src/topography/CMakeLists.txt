add_subdirectory(kernels)
add_subdirectory(initializations)
add_subdirectory(metrics)
add_subdirectory(readers)
add_subdirectory(geometry)
add_subdirectory(sources)
add_subdirectory(receivers)
add_subdirectory(functions)

set(HEADERS
    ${AWP_MINI_SOURCE_DIR}/include/awp/definitions.h     
    ${AWP_MINI_SOURCE_DIR}/include/topography/topography.h     
    ${AWP_MINI_SOURCE_DIR}/include/grid/grid_3d.h
    ${AWP_MINI_SOURCE_DIR}/include/grid/shift.h
    ${AWP_MINI_SOURCE_DIR}/include/topography/geometry.h
    ${AWP_MINI_SOURCE_DIR}/include/topography/geometry/geometry.h
    ${AWP_MINI_SOURCE_DIR}/include/topography/mms.cuh
    ${AWP_MINI_SOURCE_DIR}/include/topography/readers/serial_reader.h
    ${AWP_MINI_SOURCE_DIR}/include/test/test.h     
    ${AWP_MINI_SOURCE_DIR}/include/vtk/vtk.h     
    ${AWP_MINI_SOURCE_DIR}/include/topography/metrics/metrics.h
    ${AWP_MINI_SOURCE_DIR}/include/topography/host.h
    )

set(LIBRARIES
    ${MPI_C_LIBRARIES} 
    grid 
    metrics
    geometry
    topography_readers
    vtk
    topography_sources
    topography_receivers
    readers
    error
    nvToolsExt
    nvToolsExt
    functions
    )


add_library(topography
        topography.c velocity.cu stress.cu
        geometry.c host.c grids.c mms.cu
        )

target_link_libraries(topography ${LIBRARIES})

target_include_directories(topography
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
       )
