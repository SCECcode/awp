cmake_minimum_required(VERSION 3.10)
project(AWP_MINI VERSION 1.0 LANGUAGES C CUDA)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(GCC_COMPILE_FLAGS "-std=c99")
set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${GCC_COMPILE_FLAGS}")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

include(CTest)
enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
