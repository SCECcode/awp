set(HEADERS
    ${AWP_SOURCE_DIR}/include/functions/functions.h     
    ${AWP_SOURCE_DIR}/include/functions/random.h     
    ${AWP_SOURCE_DIR}/include/functions/norm.h     
    ${AWP_SOURCE_DIR}/include/grid/grid_3d.h
    )

add_library(functions functions.c random.c norm.c ${HEADERS})

target_include_directories(functions
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
       )
target_link_libraries(functions grid m)


