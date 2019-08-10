set(HEADERS
    ${AWP_MINI_SOURCE_DIR}/include/mpi/partition.h
    ${AWP_MINI_SOURCE_DIR}/include/mpi/distribute.h
    ${AWP_MINI_SOURCE_DIR}/include/mpi/io.h
    ${AWP_MINI_SOURCE_DIR}/include/test/test.h
    )

add_library(mpi
        partition.c distribute.c io.c)

target_include_directories(mpi 
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )
