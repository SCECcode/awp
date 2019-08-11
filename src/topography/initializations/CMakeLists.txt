set(HEADERS
    ${AWP_MINI_SOURCE_DIR}/include/awp/definitions.h     
    ${AWP_MINI_SOURCE_DIR}/include/topography/topography.h     
    ${AWP_MINI_SOURCE_DIR}/include/topography/initializations/constant.h     
    ${AWP_MINI_SOURCE_DIR}/include/topography/initializations/random.h     
    ${AWP_MINI_SOURCE_DIR}/include/topography/initializations/cerjan.h     
    )

add_library(topography_initializations
        constant.c random.c cerjan.c ${HEADERS}
        )

target_include_directories(topography_initializations
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )
