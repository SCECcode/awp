cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(AWP_MINI VERSION 1.0 LANGUAGES C CUDA)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

include(CTest)
enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
