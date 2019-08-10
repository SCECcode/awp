add_executable(test_buffer test_buffer.cu)

target_link_libraries(test_buffer 
        testing
        buffers
        ${CUDA_LIBRARIES}
        )

target_include_directories(test_buffer
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

add_test(NAME test_buffer COMMAND test_buffer)



