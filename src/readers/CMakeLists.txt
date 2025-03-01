set(HEADERS
    ${AWP_SOURCE_DIR}/include/awp/definitions.h
    ${AWP_SOURCE_DIR}/include/readers/input.h
    ${AWP_SOURCE_DIR}/include/readers/version.h
    )

add_library(readers
        input.c version.c
        )

target_include_directories(readers
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
        )

target_link_libraries(readers 
        ${MPI_C_LIBRARIES}
        )


