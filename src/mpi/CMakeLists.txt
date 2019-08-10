set(HEADERS
    ${AWP_MINI_SOURCE_DIR}/include/mpi/partition.h
    )

add_library(mpi_partition
        partition.c)

target_include_directories(mpi_partition
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )
