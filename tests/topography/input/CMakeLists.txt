add_executable(test_topography_input test_input.c)

target_link_libraries(test_topography_input 
        topography_input
        ${MPI_C_LIBRARIES}
        testing
        error
        )

target_include_directories(test_topography_input
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

add_test(NAME test_topography_input 
        COMMAND 
        ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 --oversubscribe
        test_topography_input)


add_executable(test_topography_version test_version.c)

target_link_libraries(test_topography_version 
        topography_input
        ${MPI_C_LIBRARIES}
        testing
        error
        )

target_include_directories(test_topography_version
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

add_test(NAME test_topography_version 
        COMMAND 
        ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 --oversubscribe
        test_topography_version)

