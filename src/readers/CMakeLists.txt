set(HEADERS
    ${AWP_MINI_SOURCE_DIR}/include/awp/definitions.h
    ${AWP_MINI_SOURCE_DIR}/include/readers/input.h
    ${AWP_MINI_SOURCE_DIR}/include/readers/error.h
    ${AWP_MINI_SOURCE_DIR}/include/readers/version.h
    )

add_library(readers
        input.c version.c error.c
        )

target_include_directories(readers
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

target_link_libraries(readers 
        ${MPI_C_LIBRARIES} 
        )


