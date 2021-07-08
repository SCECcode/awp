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
        )

# Attenuation test
add_executable(test_attenuation test_attenuation.cu)

target_link_libraries(test_attenuation 
        ${LIBRARIES} 
        awp
        topography
        )

target_include_directories(test_attenuation
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
        )

