add_executable(test_mapping test_mapping.c)

target_link_libraries(test_mapping 
        topography
        )

target_include_directories(test_mapping
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
        )

add_test(NAME test_mapping COMMAND test_mapping)


