cmake_minimum_required(VERSION 3.10)
project(AWP VERSION 1.0 LANGUAGES C CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()
include(CMakePrintHelpers)
#include(FindMPI.cmake)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(GCC_COMPILE_FLAGS "-std=c99 -Wall\
   -Wextra -Wmissing-prototypes -Wstrict-prototypes \
   -Wno-unused-parameter")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O4 -Xcompiler -std=c++17 -use_fast_math -Xptxas=-v -g -lineinfo --allow-unsupported-compiler")
set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${GCC_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17")
#add_compile_definitions(MPICH_SKIP_MPICXX=1)

#### MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    cmake_print_variables(CMAKE_INCLUDE_PATH)
    include_directories(${MY_INCLUDE_PATH})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)


include(CTest)
enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tools)
