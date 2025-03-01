add_executable(test_metrics test_metrics.c)

target_link_libraries(test_metrics 
        functions
        metrics
        grid
        testing
        )

target_include_directories(test_metrics
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
        )

add_test(NAME test_metrics COMMAND test_metrics)


