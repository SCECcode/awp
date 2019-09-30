cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0074 NEW)
project(AWP_MINI VERSION 1.0 LANGUAGES C CUDA)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(GCC_COMPILE_FLAGS "-std=c99")
if (DEFINED ENV{ARCH})
        set(ARCH $ENV{ARCH})
else()
        set(ARCH sm_70)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${ARCH} -Xptxas=-v -lineinfo")

set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${GCC_COMPILE_FLAGS}")

find_package(MPI REQUIRED)


include(CTest)
enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
