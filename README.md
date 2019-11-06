# AWP

## Dependencies
The following dependencies must be met in order to compile the source code:
* C
* CUDA
* MPI
* CMake

## Installation

To compile the source code on Summit, it is necessary to first load some
modules:
```bash
module load cmake gcc cuda spectrum-mpi
```

Compile AWP using CMake:
```bash
$ mkdir release
$ cd release
$ cmake ..
$ make

```
If the source is successfully compiled, the main executable `pmcl3d` is placed in `release/src/awp`.  

## Tests
For test environments that have *CMake*, *MPI*, *CUDA*-enabled devices available,
testing can be done by calling 
```
make test
```
For verbose output, call *CTest* directly
```
ctest --verbose
```

**Warning**: Many of the tests will fail on Summit. See the
[awp-benchmarks](https://github.com/SCECcode/awp-benchmarks) repository for
running tests on Summit.

