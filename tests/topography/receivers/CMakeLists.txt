add_executable(test_topography_receivers test_receivers.c)

target_link_libraries(test_topography_receivers 
        topography
        buffers
        ${MPI_C_LIBRARIES}
        testing
        )

target_include_directories(test_topography_receivers
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

add_test(NAME test_topography_receivers COMMAND
        ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 --oversubscribe
        test_topography_receivers)



