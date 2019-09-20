# Input
add_executable(test_input test_input.c)

target_link_libraries(test_input 
        testing
        readers
        error
        ${MPI_C_LIBRARIES} 
        )

target_include_directories(test_input
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

add_test(NAME test_input COMMAND test_input
        ${AWP_MINI_SOURCE_DIR}/tests/fixtures/input1.txt)


# Version
add_executable(test_version test_version.c)

target_link_libraries(test_version 
        testing
        readers
        ${MPI_C_LIBRARIES} 
        )

target_include_directories(test_version
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

add_test(NAME test_version COMMAND test_version)
