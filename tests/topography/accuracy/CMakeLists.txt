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
        test_convergence topography awp geometry grid)
target_include_directories(test_convergence
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )
add_test(NAME test_convergence COMMAND test_convergence)
