add_executable(test_accuracy test_accuracy.cu)
target_link_libraries(
        test_accuracy ${LIBRARIES} awp topography)
target_include_directories(test_accuracy
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )
add_test(NAME test_accuracy COMMAND test_accuracy)

add_executable(test_convergence test_convergence.cu)
target_link_libraries(
        test_convergence topography_no_bc awp geometry grid functions)
target_include_directories(test_convergence
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )
    add_test(NAME test_convergence COMMAND test_convergence ${PROJECT_SOURCE_DIR}/tests/topography/accuracy/data/)
