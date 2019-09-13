
add_library(topography_input
        input.c version.c
        )

target_include_directories(topography_input
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

