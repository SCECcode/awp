 set(HEADERS
    ${AWP_MINI_SOURCE_DIR}/include/checksum/checksum.h     )
 
add_library(checksum checksum.c md5/md5.c)

target_link_libraries(checksum)

target_include_directories(checksum
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/checksum
        ${AWP_MINI_SOURCE_DIR}/include/checksum/md5
       )
