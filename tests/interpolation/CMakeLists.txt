# Interpolation
add_executable(test_interpolation test_interpolation.c)

target_link_libraries(test_interpolation 
        testing
        grid
        interpolation
        error
        )

target_include_directories(test_interpolation
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

add_test(NAME test_interpolation COMMAND test_interpolation)

# Interpolation (CUDA)

add_executable(test_interpolationcu test_interpolation.cu)
#
target_link_libraries(test_interpolationcu 
        interpolation
        ${MPI_C_LIBRARIES}
        grid
        testing
        )

target_include_directories(test_interpolationcu
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

add_test(NAME test_interpolationcu COMMAND test_interpolationcu)


# Lagrange

add_executable(test_lagrange test_lagrange.c)

target_link_libraries(test_lagrange 
        testing
        grid
        interpolation
        )

target_include_directories(test_lagrange
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

add_test(NAME test_lagrange COMMAND test_lagrange)
