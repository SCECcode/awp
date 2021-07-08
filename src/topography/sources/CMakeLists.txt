add_library(topography_sources
        source.c sources.c source.cu forces.c
        )

target_include_directories(topography_sources
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
        )

target_link_libraries(topography_sources readers buffers mpi metrics mapping)
