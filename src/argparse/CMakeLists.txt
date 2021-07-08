set(HEADERS
    ${AWP_SOURCE_DIR}/include/argparse/argparse.h
    )

add_library(argparse
        argparse.c ${HEADERS}
        )

target_include_directories(argparse
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
        )

