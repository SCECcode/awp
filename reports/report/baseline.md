# Performance analysis and optimization of AWP-TOPO for Nvidia Volta
Our starting point is commit ... in the repository `awp-topo`. Here is a quick overview
of the organization of this project
* `src` : .c, .cu source files 
* `include` : .h, .cuh include files
* `tests` : .c source files for test programs.

The `src` and `include` directories mirror the same structure. The CUDA GPU
kernels reside in `src/topography/kernels` and the host code responsible for
calling these kernels are in `src/topography/*.cu`  In particular, we focus on
the files `src/topography/opt_topography.cu` and
`src/topography/kernels/optimized.cu`. These files are copies of
`src/topography/topography.cu` and `src/topography/kernels/unoptimized.cu`. At
this particular commit, the optimized version is the same as the unoptimized
version. Only the "optimized" files will be modified and the "unoptimized" files
will be used as a reference that we trust is correct.

## Baseline performance
To get a sense of how the unoptimized compute kernels perform, we can run nvprof
for different grid sizes. The reason why we use different grid sizes is to make
sure that we are not optimizing at an extreme end of the spectrum. At too small
sizes, the problem might fit well into a cache and therefore skew the results.
Our optimization efforts should target the area in which we will run the
application. A well optimized application should ideally have a flat region
where the problem size does not influence performance much until we reach the
extreme ends of the spectrum (too small, or too large). The script,
`scripts/analysis.sh` can be used for this purpose.

### PTX Assembly
The compiler can also assist in guiding optimizations. In particular, when the
source code is compiled with the flag "-ptxasinfo" the compiler runs the PTX
assembler that provides
information about how many registers, how much constant, local, and shared
memory is used. For our application, compiling with "-ptaxsinfo" reveals

```
 ptxas info    : Compiling entry function '_Z13dtopo_vel_112PfS_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_ffiiiiiii' for 'sm_70'

ptxas info    : Function properties for _Z13dtopo_vel_112PfS_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_ffiiiiiii

    1008 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads

ptxas info    : Used 80 registers, 604 bytes cmem[0]
```
Local memory is
private memory owned by each thread. This memory space resides in global
memory. Local memory can
either be used for stack space or register spillage.

Each thread can use its own stack space that can for example be used for holding small
data structure function argument, or for book-keeping of recursive functions.

In this case, 1008 bytes of stack frame are placed in local memory. The stack
frame is used by C arrays that can be found in the function `d_topo_vel_112`.
For example,
```C
        const float phzr[6][8] = {
            {0.0000000000000000, 0.8338228784688313, 0.1775123316429260,
             0.1435067013076542, -0.1548419114194114, 0.0000000000000000,
             0.0000000000000000, 0.0000000000000000},
            {0.0000000000000000, 0.1813404047323969, 1.1246711188154426,
             -0.2933634518280757, -0.0126480717197637, 0.0000000000000000,
             0.0000000000000000, 0.0000000000000000},
            {0.0000000000000000, -0.1331142706282399, 0.7930714675884345,
             0.3131998767078508, 0.0268429263319546, 0.0000000000000000,
             0.0000000000000000, 0.0000000000000000},
            {0.0000000000000000, 0.0969078556633046, -0.1539344946680898,
             0.4486491202844389, 0.6768738207821733, -0.0684963020618270,
             0.0000000000000000, 0.0000000000000000},
            {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.0625000000000000, 0.5625000000000000, 0.5625000000000000,
             -0.0625000000000000, 0.0000000000000000},
            {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             0.0000000000000000, -0.0625000000000000, 0.5625000000000000,
             0.5625000000000000, -0.0625000000000000}};
```

If all of the available registers per thread are used up, spill to local memory
occurs. In this case, since the PTX assembly output reports *0 bytes spill
stores* and *0 bytes spill reads* no register spillage has occurred. The
architecture we are working with are `Tesla V100-SXM2-16GB` the report found in
`reports/baseline/dtopo_vel_111.pdf` shows its specifications. This hardware
supports up to 80 registers per thread.


The 604 bytes of `cmem[0]` are constant memory. In this case, what has most likely
happened is that some of the arrays have been directly placed into constant
memory by the compiler. These arrays are
```C
        const float phy[4] = {-0.0625000000000000, 0.5625000000000000,
                              0.5625000000000000, -0.0625000000000000};
        const float phx[4] = {-0.0625000000000000, 0.5625000000000000,
                              0.5625000000000000, -0.0625000000000000};
        const float dhy[4] = {0.0416666666666667, -1.1250000000000000,
                              1.1250000000000000, -0.0416666666666667};
        const float dhx[4] = {0.0416666666666667, -1.1250000000000000,
                              1.1250000000000000, -0.0416666666666667};
        const float px[4] = {-0.0625000000000000, 0.5625000000000000,
                             0.5625000000000000, -0.0625000000000000};
        const float py[4] = {-0.0625000000000000, 0.5625000000000000,
                             0.5625000000000000, -0.0625000000000000};
        const float dx[4] = {0.0416666666666667, -1.1250000000000000,
                             1.1250000000000000, -0.0416666666666667};
        const float dy[4] = {0.0416666666666667, -1.1250000000000000,
                             1.1250000000000000, -0.0416666666666667};
```
 In total, these arrays should at most
consume 128 bytes. Note that these arrays all share the same value except for
two of them, and could
therefore by replaced by two arrays. The computation could be further
optimized by noticing that two of the four values in each array are the same (up
to a sign).  

### Index mapping
In AWP-TOPO, the memory layout is chosen such that z is the fast direction, y is
the second fastest direction, and x is the slowest direction. This is opposite
to the thread block configuration in CUDA; here x is the fastest, and z is the
slowest direction. The initial version of the kernels are unfortunately designed
such that the memory index z maps to z in CUDA and so forth. Since the number of
threads per block in the x, and y-directions are set to 1 it does not matter in
this particular case. However, the kernel `dvel_112`, responsible for the
boundary computation, contains the following conditional:
```C
        const int k = threadIdx.z + blockIdx.z * blockDim.z;
        if (k >= 6) return;

```
The configuration of the number of threads per block is set to `x = 1, y = 1, z
= 64` (see macros `TBX`, `TBY`, and `TBZ` in `topography.cuh`. For this reason,
the CUDA runtime API will launch a block with 64 threads in the z-direction, but
out of those threads, only 7 threads will remain active after reaching the
conditional `if (k >= 6)`. As such, the occupancy is a mere `7/64 = 10 %`.
Since our profiling reports indicate that this poor mapping of the problem to
the CUDA threads is the predominant reason for the kernel's poor performance, it
will be the first problem that we will address.



