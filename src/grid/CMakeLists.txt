set(HEADERS
    ${AWP_MINI_SOURCE_DIR}/include/grid/grid_3d.h
    ${AWP_MINI_SOURCE_DIR}/include/grid/grid_new.h
    ${AWP_MINI_SOURCE_DIR}/include/grid/shift.h
    ${AWP_MINI_SOURCE_DIR}/include/awp/definitions.h
    )

add_library(grid
        grid_3d.c grid_new.c shift.c ${HEADERS}
        )

target_link_libraries(grid 
        m
        )

target_include_directories(grid
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )
