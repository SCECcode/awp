cmake_minimum_required(VERSION 3.10)
project(AWP VERSION 1.0 LANGUAGES C CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(GCC_COMPILE_FLAGS "-std=c99 -Wall\
   -Wextra -Wmissing-prototypes -Wstrict-prototypes \
          -Wold-style-definition -Wno-unused-parameter")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O4 -Xcompiler -std=c++11 -use_fast_math -Xptxas=-v -lineinfo")

set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${GCC_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(MPI REQUIRED)


include(CTest)
enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tools)
