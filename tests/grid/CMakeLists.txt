add_executable(test_grid_3d test_grid_3d.c)

target_link_libraries(test_grid_3d 
        testing
        grid
        ${MPI_C_LIBRARIES} 
        )

target_include_directories(test_grid_3d
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

add_test(NAME test_grid_3d COMMAND test_grid_3d)

