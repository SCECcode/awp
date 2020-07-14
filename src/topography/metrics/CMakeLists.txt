set(HEADERS
    ${AWP_MINI_SOURCE_DIR}/include/awp/definitions.h     
    ${AWP_MINI_SOURCE_DIR}/include/functions/functions.h     
    ${AWP_MINI_SOURCE_DIR}/include/topography/metrics/metrics.h
    ${AWP_MINI_SOURCE_DIR}/include/topography/metrics/kernel.h
    ${AWP_MINI_SOURCE_DIR}/include/topography/metrics/shift.h
    )

add_library(metrics metrics.c kernel.c shift.c)

target_include_directories(metrics
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
       )
target_link_libraries(metrics interpolation functions)


