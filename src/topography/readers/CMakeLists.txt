set(HEADERS
    ${AWP_SOURCE_DIR}/include/awp/error.h     
    ${AWP_SOURCE_DIR}/include/topography/readers/serial_reader.h
    )

add_library(topography_readers serial_reader.c)

target_include_directories(topography_readers
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
       )



