# Indexed
add_executable(test_indexed test_indexed.c)

target_link_libraries(test_indexed 
        testing
        ${MPI_C_LIBRARIES} 
        )

target_include_directories(test_indexed
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
        )

add_test(NAME test_indexed COMMAND test_indexed)

# Distribute
add_executable(test_mpi_distribute test_distribute.c)

target_link_libraries(test_mpi_distribute 
        testing
        grid
        mpi
        ${MPI_C_LIBRARIES} 
        )

target_include_directories(test_mpi_distribute
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
        )

add_test(NAME test_mpi_distribute COMMAND 
        ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 --oversubscribe
        test_mpi_distribute)

# IO

add_executable(test_mpi_io test_io.c)

target_link_libraries(test_mpi_io 
        testing
        grid
        mpi
        ${MPI_C_LIBRARIES} 
        )

target_include_directories(test_mpi_io
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
        )

add_test(NAME test_mpi_io COMMAND 
        ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 --oversubscribe
        test_mpi_io)



