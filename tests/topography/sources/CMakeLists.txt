add_executable(test_topography_sources test_sources.c)
add_executable(test_topography_sources_dm test_sources_dm.c)

target_link_libraries(test_topography_sources 
        topography
        buffers
        ${MPI_C_LIBRARIES}
        testing
        )

target_link_libraries(test_topography_sources_dm
        topography
        buffers
        ${MPI_C_LIBRARIES}
        testing
        )

target_include_directories(test_topography_sources
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

target_include_directories(test_topography_sources_dm
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

add_test(NAME test_topography_sources_dm COMMAND
        test_topography_sources_dm)

add_test(NAME test_topography_sources COMMAND
        ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 --oversubscribe
        test_topography_sources
        ${AWP_MINI_SOURCE_DIR}/tests/fixtures/source.txt)


