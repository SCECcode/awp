set(HEADERS
    ${AWP_SOURCE_DIR}/include/utils/copy.h
    )

add_library(utils
        copy.c
        )

target_include_directories(utils
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
        )
