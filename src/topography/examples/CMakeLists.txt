set(HEADERS
    ${AWP_MINI_SOURCE_DIR}/include/awp/definitions.h     
    ${AWP_MINI_SOURCE_DIR}/include/topography/topography.h     
    ${AWP_MINI_SOURCE_DIR}/include/topography/examples/constant.h     
    )

add_library(topography_examples
        constant.c   ${HEADERS}
        )

target_include_directories(topography_examples
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )
