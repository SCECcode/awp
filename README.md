# awp-opt
This repository contains a stripped down version of AWP with topography that is
intended for performance analysis and optimization of the CUDA compute kernels.

## Compilation
These instructions pertain to Summit. Load the required modules, CMake, gcc, CUDA
```bash
module load cmake gcc cuda
```
Compile a release build using CMake
```bash
mkdir release
cd release
cmake -DCMAKE_BUILD_TYPE=Release ..
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

## Optimized kernels

Currently, both the unoptimized kernels and optimized kernels are the same. The
files of interest to start optimizing the kernels are 
* `src/topography/opt_topography.cu` responsible for calling compute kernels
* `src/topography/kernels/optimized.cu` contains compute kernels
* `include/topography/opt_topography.cuh` header file that contains block size
  configuration parameters


## Performance analysis
The script `scripts/profile.sh` can be used to see an example usage and also to
sweep through a list of different grid sizes. This scripts profiles the
application using `nvprof`. To run this script on Summit, submit a job using
```
bsub scripts/submit.lsf
```



