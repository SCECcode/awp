set(HEADERS
    ${AWP_SOURCE_DIR}/include/mpi/distribute.h
    ${AWP_SOURCE_DIR}/include/mpi/io.h
    ${AWP_SOURCE_DIR}/include/test/test.h
    )

add_library(mpi
        partition.c distribute.c io.c)

target_include_directories(mpi 
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
        )
