set(CMAKE_CUDA_COMPILER "/sw/summit/cuda/11.7.1/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/sw/summit/gcc/9.3.0-2/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "11.7.99")
set(CMAKE_CUDA_DEVICE_LINKER "/sw/summit/cuda/11.7.1/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/sw/summit/cuda/11.7.1/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "14")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "9.3")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)
set(CMAKE_CUDA_LINKER_DEPFILE_SUPPORTED )

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/sw/summit/cuda/11.7.1")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/sw/summit/cuda/11.7.1")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "11.7.99")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/sw/summit/cuda/11.7.1")

set(CMAKE_CUDA_ARCHITECTURES_ALL "35-real;37-real;50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86-real;87")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "35-real;50-real;60-real;70-real;80")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "70-real")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/sw/summit/cuda/11.7.1/targets/ppc64le-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/sw/summit/cuda/11.7.1/targets/ppc64le-linux/lib/stubs;/sw/summit/cuda/11.7.1/targets/ppc64le-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/sw/summit/spack-envs/summit-plus/opt/gcc-8.5.0/spectrum-mpi-10.4.0.6-20230210-f3ouht4ckff2qogy74bwki5ovljfou36/include;/autofs/nccs-svm1_sw/summit/gcc/9.3.0-2/include/c++/9.3.0;/autofs/nccs-svm1_sw/summit/gcc/9.3.0-2/include/c++/9.3.0/powerpc64le-unknown-linux-gnu;/autofs/nccs-svm1_sw/summit/gcc/9.3.0-2/include/c++/9.3.0/backward;/autofs/nccs-svm1_sw/summit/gcc/9.3.0-2/lib/gcc/powerpc64le-unknown-linux-gnu/9.3.0/include;/autofs/nccs-svm1_sw/summit/gcc/9.3.0-2/lib/gcc/powerpc64le-unknown-linux-gnu/9.3.0/include-fixed;/usr/local/include;/autofs/nccs-svm1_sw/summit/gcc/9.3.0-2/include;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/sw/summit/cuda/11.7.1/targets/ppc64le-linux/lib/stubs;/sw/summit/cuda/11.7.1/targets/ppc64le-linux/lib;/autofs/nccs-svm1_sw/summit/gcc/9.3.0-2/lib/gcc/powerpc64le-unknown-linux-gnu/9.3.0;/autofs/nccs-svm1_sw/summit/gcc/9.3.0-2/lib/gcc;/autofs/nccs-svm1_sw/summit/gcc/9.3.0-2/lib64;/lib64;/usr/lib64;/sw/summit/spack-envs/summit-plus/opt/gcc-8.5.0/spectrum-mpi-10.4.0.6-20230210-f3ouht4ckff2qogy74bwki5ovljfou36/lib;/autofs/nccs-svm1_sw/summit/gcc/9.3.0-2/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")
