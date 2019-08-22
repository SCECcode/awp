add_subdirectory(topography)
add_subdirectory(buffers)
add_subdirectory(grid)
add_subdirectory(mpi)
add_subdirectory(interpolation)
add_subdirectory(readers)


set(LIBRARIES 
    ${MPI_C_LIBRARIES} 
    argparse
    mpi
    topography_initializations
    functions
    testing
    )

set(UNOPT_LIBRARIES
        ${LIBRARIES}
        topography
        )

set(OPT_LIBRARIES
        ${LIBRARIES}
        opt_topography
        )

# Unoptimized version
add_executable(test_unoptimized_kernels topography_kernels.cu)

target_link_libraries(test_unoptimized_kernels 
        ${UNOPT_LIBRARIES} 
        )

target_include_directories(test_unoptimized_kernels
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )
target_compile_definitions(test_unoptimized_kernels 
        PUBLIC USE_OPTIMIZED_KERNELS=0)

# Optimized version
add_executable(test_optimized_kernels topography_kernels.cu)

target_link_libraries(test_optimized_kernels 
        ${OPT_LIBRARIES} 
        )

target_compile_definitions(test_optimized_kernels 
        PUBLIC USE_OPTIMIZED_KERNELS=1)

# Attenuation test
add_executable(test_attenuation test_attenuation.cu)

target_link_libraries(test_attenuation 
        ${LIBRARIES} 
        awp
        topography
        )

target_include_directories(test_attenuation
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )
