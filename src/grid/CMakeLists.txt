set(HEADERS
    ${AWP_SOURCE_DIR}/include/grid/grid_3d.h
    ${AWP_SOURCE_DIR}/include/grid/shift.h
    ${AWP_SOURCE_DIR}/include/awp/definitions.h
    )

add_library(grid
        grid_3d.c shift.c ${HEADERS}
        )

target_link_libraries(grid 
        m
        )

target_include_directories(grid
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
        )
