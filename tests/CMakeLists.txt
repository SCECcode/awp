add_subdirectory(topography)
add_subdirectory(buffers)
add_subdirectory(grid)
add_subdirectory(mpi)
add_subdirectory(interpolation)
add_subdirectory(readers)

add_executable(topography_kernels topography_kernels.cu)

target_link_libraries(topography_kernels 
        ${MPI_C_LIBRARIES} 
        topography
        argparse
        mpi
        topography_examples
        )

target_include_directories(topography_kernels
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

#add_test(NAME topography_kernels COMMAND topography_kernels )

