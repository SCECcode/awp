#Specify cmake minimum required version
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
#Project name
project(AWP_MINI VERSION 1.0 LANGUAGES C CUDA)

include(GNUInstallDirs)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_INCLUDE_PATH include)
include(CTest)
enable_testing()
include_directories("${CMAKE_SOURCE_DIR}/include/") 
include_directories("${CMAKE_SOURCE_DIR}/include/hello/") 
include_directories(include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(src)
add_subdirectory(tests)
