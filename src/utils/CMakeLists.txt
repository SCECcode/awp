set(HEADERS
    ${AWP_MINI_SOURCE_DIR}/include/utils/copy.h
    )

add_library(utils
        copy.c
        )

target_include_directories(utils
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )
