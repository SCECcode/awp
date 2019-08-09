# Hello world
This code example demonstrates how to compile C and CUDA code using cmake. The
program simply prints hello world (see below).

## Compiling
Run the following commands:
```
cd build

cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-9.0/bin/nvcc ..

make

```

The executable `hello` is created in the build directory. To remove it and all object files, use `make
clean`.
Note that -DCMAKE_CUDA_COMPILER specifies the folder where the NVIDIA compiler is contained. The current code compiles and runs with CUDA 9.0

## Other useful info
In the source code, all of the cuda calls are checked for errors using the macro
`cudachk` defined in `cuhello.cuh`.


## Example output:
```
CPU: Hello world!
GPU[0]: Hello world!
GPU[1]: Hello world!
GPU[2]: Hello world!
GPU[3]: Hello world!
GPU[4]: Hello world!
GPU[5]: Hello world!
GPU[6]: Hello world!
GPU[7]: Hello world!
GPU[8]: Hello world!
GPU[9]: Hello world!
GPU[10]: Hello world!
GPU[11]: Hello world!
GPU[12]: Hello world!
GPU[13]: Hello world!
GPU[14]: Hello world!
GPU[15]: Hello world!
a[0] = 0
a[1] = 1
a[2] = 2
a[3] = 3
a[4] = 4
a[5] = 5
a[6] = 6
a[7] = 7
a[8] = 8
a[9] = 9
a[10] = 10
a[11] = 11
a[12] = 12
a[13] = 13
a[14] = 14
a[15] = 15
```
