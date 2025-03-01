set(HEADERS
    ${AWP_SOURCE_DIR}/include/awp/definitions.h     
    ${AWP_SOURCE_DIR}/include/topography/geometry/geometry.h
    ${AWP_SOURCE_DIR}/include/topography/metrics/metrics.h
    )

add_library(geometry geometry.c)

target_include_directories(geometry
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
       )
target_link_libraries(geometry grid functions mapping)

