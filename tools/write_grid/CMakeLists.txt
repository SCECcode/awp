add_executable(write_grid write_grid.c)

target_link_libraries(write_grid 
        ${MPI_C_LIBRARIES} 
        m
        topography_readers
        mapping
        )

target_include_directories(write_grid
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

