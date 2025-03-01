add_executable(test_topography_geometry test_geometry.c)

target_link_libraries(test_topography_geometry 
        metrics
        geometry
        grid
        vtk
        ${MPI_C_LIBRARIES}
        testing
        )

target_include_directories(test_topography_geometry
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
        )

add_test(NAME test_topography_geometry 
        COMMAND 
        ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 --oversubscribe
        test_topography_geometry)

