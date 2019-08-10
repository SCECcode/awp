set(HEADERS
    ${AWP_MINI_SOURCE_DIR}/include/awp/definitions.h
    ${AWP_MINI_SOURCE_DIR}/include/grid/grid_3d.h
    ${AWP_MINI_SOURCE_DIR}/include/interpolation/error.h
    ${AWP_MINI_SOURCE_DIR}/include/interpolation/interpolation.h
    ${AWP_MINI_SOURCE_DIR}/include/interpolation/interpolation.cuh
    ${AWP_MINI_SOURCE_DIR}/include/interpolation/lagrange.h
    )

add_library(interpolation
        interpolation.c interpolation.cu lagrange.c error.c ${HEADERS}
        )

target_link_libraries(interpolation grid)

target_include_directories(interpolation
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )
