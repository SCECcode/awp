add_executable(test_topography_serial_reader test_serial_reader.c)

target_link_libraries(test_topography_serial_reader 
        topography_readers
        metrics
        geometry
        grid
        ${MPI_C_LIBRARIES}
        testing
        )

target_include_directories(test_topography_serial_reader
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
        )

add_test(NAME test_topography_serial_reader COMMAND
        ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 6 --oversubscribe
        test_topography_serial_reader)

