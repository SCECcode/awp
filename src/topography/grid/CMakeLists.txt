set(HEADERS
    ${AWP_MINI_SOURCE_DIR}/include/topography/grid/grid_3d.h
    ${AWP_MINI_SOURCE_DIR}/include/awp/definitions.h
    )

add_library(grid_3d
        grid_3d.c ${HEADERS}
        )

add_library(shift INTERFACE)

target_include_directories(shift
        INTERFACE
        ${AWP_MINI_SOURCE_DIR}/include/topography/shift
        )

target_include_directories(grid_3d
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )
