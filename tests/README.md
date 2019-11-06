# Topography kernels
This README explains how to use the test programs produced by
`topography_kernels.cu`.

## Compilation
These instructions pertain to Summit. Load the required modules, CMake, gcc, CUDA
```bash
module load cmake gcc cuda
```
Compile using CMake:
```bash
mkdir release
cd release
cmake ..
make

```

## Usage
There are two main applications :`tests/test_unoptimized_kernels` and
`tests/test_optimized_kernels`. The former application runs a test using the
original, unoptimized version of the compute kernels. The latter version runs
the same test using an optimized version of the compute kernels. You can use the
output form the unoptimized-version to verify that the any modifications to the
optimized kernels produce the same results. 
Use `-h` to display usage for each application.
```
Usage: topography_kernels [options] [[--] args]
   or: topography_kernels [options]

Performance analysis of CUDA compute kernels for AWP.

    -h, --help            show this help message and exit

Options
    -p, --px=<int>        Number of processes in the X-direction
    -q, --py=<int>        Number of processes in the Y-direction
    -x, --nx=<int>        Number of grid points in the X-direction
    -y, --ny=<int>        Number of grid points in the Y-direction
    -z, --nz=<int>        Number of grid points in the Z-direction
    -t, --nt=<int>        Number of iterations to perform
    -o, --output=<str>    Write results to output directory
    -i, --input=<str>     Read results from input directory
```

Use the unoptimized-version to write a reference solution for a given
configuration, 
```
tests/unoptimized_kernels --output=logs  [[--] args]
```
Here, `--output` is assigned a directory for storing the output data. If it does
not exist, it will be created. All of the other required flags are omitted.
Then, run the optimized-version to see that it produces the same results, 
```
tests/optimized_kernels --input=logs  [[--] args]
```
The flag `--input` loads the data produced by unoptimized-version.
The results are the same if, after execution,
```
vx: 0 vy: 0 vz: 0
```
is displayed. This message shows the maximum difference in each component of the particle
velocity field.



