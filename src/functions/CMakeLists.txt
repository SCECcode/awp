set(HEADERS
    ${AWP_MINI_SOURCE_DIR}/include/functions/functions.h     
    ${AWP_MINI_SOURCE_DIR}/include/grid/grid_3d.h
    )

add_library(functions functions.c ${HEADERS})

target_include_directories(functions
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
       )
target_link_libraries(functions grid)


