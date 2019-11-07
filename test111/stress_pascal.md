# Stress kernel performance analysis and optimization


## Memory traffic 
Number of arrays to READ:
1. 6 stresses
2. 6 memory variables
3. 3 velocities
4. 4 material parameters
5. 4 other parameters
= 23

Number of arrays to WRITE:
1. 6 stresses
2. 6 memory variables
= 12


## Baseline performance
The results presented in these notes were obtained for Pascal.

For the stress kernel, we want to explore two strategies: 
1. Array accesses using pre-computed line and slice sizes
2. Array accesses using macros


The baseline kernel, `dtopo_str_111` uses a combination of macros and
pre-computed indices. The macros are used for accessing velocity components,
whereas the pre-computed indices are used for material properties.


Performance analysis using nvprof.
```
==30525== NVPROF is profiling process 30525, command: ./test111.x 300 350 512 100
==30525== Profiling application: ./test111.x 300 350 512 100
==30525== Profiling result:
            Type  Time      Time     Calls       Avg       Min       Max  Name
 GPU activities:   34.66%  4.09576s       100  40.958ms  40.258ms  42.121ms  void dtopo_str_111_index<int=64, int=8, int=1>
                   33.41%  3.94839s       100  39.484ms  38.745ms  43.033ms  void dtopo_str_111<int=64, int=8, int=1>
                   31.60%  3.73463s       100  37.346ms  36.684ms  38.368ms  void dtopo_str_111_macro<int=64, int=8, int=1>
```



ptxas info reveals that register usage much higher for the macro version compare
to the other two versions:
```
ptxas info    : Compiling entry function '_Z19dtopo_str_111_indexILi64ELi8ELi1EEvPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_PKiS2_iiiiiiiiii' for 'sm_61'
ptxas info    : Function properties for _Z19dtopo_str_111_indexILi64ELi8ELi1EEvPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_PKiS2_iiiiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 78 registers, 688 bytes cmem[0], 28 bytes cmem[2]


ptxas info    : Compiling entry function '_Z13dtopo_str_111ILi64ELi8ELi1EEvPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_PKiS2_iiiiiiiiii' for 'sm_61'
ptxas info    : Function properties for _Z13dtopo_str_111ILi64ELi8ELi1EEvPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_PKiS2_iiiiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 88 registers, 688 bytes cmem[0], 28 bytes cmem[2]

ptxas info    : Compiling entry function '_Z19dtopo_str_111_macroILi64ELi8ELi1EEvPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_PKiS2_iiiiiiiiii' for 'sm_61'
ptxas info    : Function properties for _Z19dtopo_str_111_macroILi64ELi8ELi1EEvPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_PKiS2_iiiiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 100 registers, 688 bytes cmem[0], 28 bytes cmem[2]

```

## Pre-computed indices in macros
In the macro-version, we can modify the array accesses such that line and block
sizes are pre-computed and put into CUDA constant memory. In `stress_macro.cu`
this option is enabled by setting `USE_CONST_ARRAY_ACCESS` to `1`. This option
does affect performance slightly:

1. `USE_CONST_ARRAY_ACCESS 0`

```
==8940== NVPROF is profiling process 8940, command: ./test111.x 300 350 512 100
==8940== Profiling application: ./test111.x 300 350 512 100
==8940== Profiling result:
            Type  Time      Time     Calls       Avg       Min       Max  Name
 GPU activities:   52.17%  4.20285s       100  42.028ms  40.772ms  48.367ms  void dtopo_str_111<int=64, int=8, int=1>
                   47.32%  3.81155s       100  38.115ms  36.608ms  41.578ms  void dtopo_str_111_macro<int=64, int=8, int=1>

```

2.  `USE_CONST_ARRAY_ACCESS 1`

```
==9040== Profiling application: ./test111.x 300 350 512 100
==9040== Profiling result:
            Type  Time      Time     Calls       Avg       Min       Max  Name
 GPU activities:   51.14%  4.17009s       100  41.701ms  40.557ms  45.657ms  void dtopo_str_111<int=64, int=8, int=1>
                   48.34%  3.94116s       100  39.412ms  38.299ms  42.956ms  void dtopo_str_111_macro<int=64, int=8, int=1>
```

For this reason, `USE_CONST_ARRAY_ACCESS 0` will be used in future
investigations.


## Loop unrolling

The second optimization we considered is loop unrolling. We apply this
optimization to both strategies. For the macro version, it is difficult to
unroll the loops without causing spilling. The only configuration that does not
cause spilling is by unrolling in CUDA y using an unroll factor of `nb=2`.

```
ptxas info    : Compiling entry function '_Z26dtopo_str_111_index_unrollILi32ELi1ELi4ELi1ELi2EEvPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_PKiS2_iiiiiiiiii' for 'sm_61'
ptxas info    : Function properties for _Z26dtopo_str_111_index_unrollILi32ELi1ELi4ELi1ELi2EEvPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_PKiS2_iiiiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 168 registers, 688 bytes cmem[0], 28 bytes cmem[2]


ptxas info    : Compiling entry function '_Z26dtopo_str_111_macro_unrollILi32ELi1ELi4ELi1ELi2EEvPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_PKiS2_iiiiiiiiii' for 'sm_61'
ptxas info    : Function properties for _Z26dtopo_str_111_macro_unrollILi32ELi1ELi4ELi1ELi2EEvPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_PKiS2_iiiiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 168 registers, 688 bytes cmem[0], 32 bytes cmem[2]

```


```
==9301== NVPROF is profiling process 9301, command: ./test111.x 300 350 512 100
==9301== Profiling application: ./test111.x 300 350 512 100
==9301== Profiling result:
            Type  Time      Time     Calls       Avg       Min       Max  Name
 GPU activities:   38.93%  4.19563s       100  41.956ms  40.706ms  46.958ms  void dtopo_str_111<int=64, int=8, int=1>
                   30.77%  3.31595s       100  33.160ms  31.768ms  38.061ms  void dtopo_str_111_index_unroll<int=32, int=1, int=4, int=1, int=2>
                   29.91%  3.22305s       100  32.230ms  31.103ms  34.780ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=1, int=2>
```


### Disable L1 cache on Pascal
Compile with `-Xptxas -dlcm=cg` to disable L1 cache.

Without `dlcm=cg`, 
```

==13812== NVPROF is profiling process 13812, command: ./test111.x 300 350 512 100
==13812== Profiling application: ./test111.x 300 350 512 100
==13812== Profiling result:
            Type  Time      Time     Calls       Avg       Min       Max  Name
 GPU activities:   38.82%  4.02261s       100  40.226ms  38.904ms  43.325ms  void dtopo_str_111<int=64, int=8, int=1>
                   30.75%  3.18606s       100  31.861ms  31.091ms  34.948ms  void dtopo_str_111_index_unroll<int=32, int=1, int=4, int=1, int=2>
                   30.05%  3.11308s       100  31.131ms  30.462ms  34.360ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=1, int=2>
                    0.38%  38.903ms        23  1.6914ms  1.6908ms  1.6934ms  fill
```
With `dlcm=cg`, 

```
==13883== NVPROF is profiling process 13883, command: ./test111.x 300 350 512 100
==13883== Profiling application: ./test111.x 300 350 512 100
==13883== Profiling result:
            Type  Time      Time     Calls       Avg       Min       Max  Name
 GPU activities:   38.82%  3.97483s       100  39.748ms  39.024ms  43.402ms  void dtopo_str_111<int=64, int=8, int=1>
                   30.75%  3.14827s       100  31.483ms  31.083ms  34.106ms  void dtopo_str_111_index_unroll<int=32, int=1, int=4, int=1, int=2>
                   30.04%  3.07511s       100  30.751ms  30.419ms  32.648ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=1, int=2>
```

##  Exhaustive parameter space search

We search for the best available thread block configuration and loop unrolling
factors. It turns out that all of the previous experiments have been configured
at the optimal settings for Pascal. After trying more than 170 different
combinations, `void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=1, int=2>` is the fastest


On Volta, the best configuration (for macro unroll) was achieved using <32, 2,
8, 1, 1> which corresponds to no loop unrolling!
```
==15444== NVPROF is profiling process 15444, command: ./str_32_2_8_1_1 300 350 512 100
==15444== Profiling application: ./str_32_2_8_1_1 300 350 512 100
==15444== Profiling result:
            Type  Time      Time     Calls       Avg       Min       Max  Name
 GPU activities:   44.68%  1.80164s       100  18.016ms  17.855ms  18.656ms  void dtopo_str_111<int=64, int=8, int=1>
                   28.05%  1.13113s       100  11.311ms  11.270ms  11.341ms  void dtopo_str_111_index_unroll<int=32, int=1, int=4, int=1, int=2>
                   27.11%  1.09298s       100  10.930ms  10.897ms  11.134ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=8, int=1, int=1>
```

## Plane cycling
The kernel `stress_macro_planes.cu` loops over the slowest dimension and puts
all of the velocity components in a register queue that is cycled. It also uses
loop unrolling for the other two dimensions. However, loop unrolling causes too
much register pressure and results in spilling; at least on Pascal.


```
==21953== NVPROF is profiling process 21953, command: ./test111.x 300 350 512 100
==21953== Profiling application: ./test111.x 300 350 512 100
==21953== Profiling result:
            Type  Time      Time     Calls       Avg       Min       Max  Name
 GPU activities:   53.70%  3.96536s       100  39.654ms  38.794ms  50.799ms  void dtopo_str_111<int=64, int=8, int=1>
                   45.06%  3.32738s       100  33.274ms  32.845ms  37.400ms  void dtopo_str_111_macro_planes<int=64, int=4, int=1, int=1>
```
