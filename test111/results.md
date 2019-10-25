    
# Pascal
```
ptxas info    : 4 bytes gmem
ptxas info    : Compiling entry function '_Z20dtopo_vel_111_unrollILi1ELi2ELi2EEvPfS0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_ffiiiiiii' for 'sm_61'
ptxas info    : Function properties for _Z20dtopo_vel_111_unrollILi1ELi2ELi2EEvPfS0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_ffiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 247 registers, 572 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compiling entry function '_Z20dtopo_vel_111_split2ILi2ELi2ELi4EEvPfS0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_ffiiiiiii' for 'sm_61'
ptxas info    : Function properties for _Z20dtopo_vel_111_split2ILi2ELi2ELi4EEvPfS0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_ffiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 224 registers, 572 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compiling entry function '_Z20dtopo_vel_111_split1ILi2ELi2ELi2EEvPfS0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_ffiiiiiii' for 'sm_61'
ptxas info    : Function properties for _Z20dtopo_vel_111_split1ILi2ELi2ELi2EEvPfS0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_ffiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 240 registers, 572 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compiling entry function '_Z7comparePKfS0_S0_S0_S0_S0_iii' for 'sm_61'
ptxas info    : Function properties for _Z7comparePKfS0_S0_S0_S0_S0_iii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, 380 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compiling entry function '_Z20dtopo_vel_111_blocksPfS_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_ffiiiiiii' for 'sm_61'
ptxas info    : Function properties for _Z20dtopo_vel_111_blocksPfS_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_ffiiiiiii
    184 bytes stack frame, 224 bytes spill stores, 252 bytes spill loads
ptxas info    : Used 64 registers, 572 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compiling entry function '_Z13dtopo_vel_111PfS_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_ffiiiiiii' for 'sm_61'
ptxas info    : Function properties for _Z13dtopo_vel_111PfS_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_ffiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 72 registers, 572 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compiling entry function '_Z17dtopo_vel_111_optPfS_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_ffiiiiiii' for 'sm_61'
ptxas info    : Function properties for _Z17dtopo_vel_111_optPfS_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_ffiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 100 registers, 23352 bytes smem, 572 bytes cmem[0], 8 bytes cmem[2]
```

## Timing
```

                   15.90%  411.86ms        10  41.186ms  41.052ms  41.976ms  void dtopo_vel_111_unroll<int=1, int=2, int=2>
                    9.35%  242.02ms        10  24.202ms  24.088ms  24.963ms  void dtopo_vel_111_split1<int=2, int=2, int=2>
                    5.22%  135.13ms        10  13.513ms  13.487ms  13.552ms  void dtopo_vel_111_split2<int=2, int=2, int=4>
```

## Dram
```
==31156== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "TITAN X "
    Kernel: compare
          1                      dram_read_throughput             Device Memory Read Throughput  299.68GB/s  299.68GB/s  299.68GB/s
          1                     dram_write_throughput            Device Memory Write Throughput  847.05MB/s  847.05MB/s  847.05MB/s
    Kernel: void dtopo_vel_111_unroll<int=1, int=2, int=2>
         10                      dram_read_throughput             Device Memory Read Throughput  181.33GB/s  185.46GB/s  184.79GB/s
         10                     dram_write_throughput            Device Memory Write Throughput  35.638GB/s  36.318GB/s  36.200GB/s
    Kernel: dtopo_vel_111_blocks
         10                      dram_read_throughput             Device Memory Read Throughput  82.137GB/s  82.846GB/s  82.694GB/s
         10                     dram_write_throughput            Device Memory Write Throughput  158.84GB/s  159.79GB/s  159.47GB/s
    Kernel: void dtopo_vel_111_split1<int=2, int=2, int=2>
         10                      dram_read_throughput             Device Memory Read Throughput  198.10GB/s  204.07GB/s  203.35GB/s
         10                     dram_write_throughput            Device Memory Write Throughput  39.796GB/s  41.004GB/s  40.859GB/s
    Kernel: void dtopo_vel_111_split2<int=2, int=2, int=4>
         10                      dram_read_throughput             Device Memory Read Throughput  216.03GB/s  224.93GB/s  223.66GB/s
         10                     dram_write_throughput            Device Memory Write Throughput  35.358GB/s  36.792GB/s  36.593GB/s
```

# Volta
```
ptxas info    : 4 bytes gmem
ptxas info    : Compiling entry function '_Z20dtopo_vel_111_unrollILi1ELi2ELi2EEvPfS0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_ffiiiiiii' for 'sm_70'
ptxas info    : Function properties for _Z20dtopo_vel_111_unrollILi1ELi2ELi2EEvPfS0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_ffiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 255 registers, 604 bytes cmem[0]
ptxas info    : Compiling entry function '_Z20dtopo_vel_111_split2ILi2ELi2ELi4EEvPfS0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_ffiiiiiii' for 'sm_70'
ptxas info    : Function properties for _Z20dtopo_vel_111_split2ILi2ELi2ELi4EEvPfS0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_ffiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 253 registers, 604 bytes cmem[0]
ptxas info    : Compiling entry function '_Z20dtopo_vel_111_split1ILi1ELi2ELi4EEvPfS0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_ffiiiiiii' for 'sm_70'
ptxas info    : Function properties for _Z20dtopo_vel_111_split1ILi1ELi2ELi4EEvPfS0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_ffiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 255 registers, 604 bytes cmem[0]
ptxas info    : Compiling entry function '_Z7comparePKfS0_S0_S0_S0_S0_iii' for 'sm_70'
ptxas info    : Function properties for _Z7comparePKfS0_S0_S0_S0_S0_iii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, 412 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compiling entry function '_Z20dtopo_vel_111_blocksPfS_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_ffiiiiiii' for 'sm_70'
ptxas info    : Function properties for _Z20dtopo_vel_111_blocksPfS_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_ffiiiiiii
    24 bytes stack frame, 24 bytes spill stores, 24 bytes spill loads
ptxas info    : Used 64 registers, 604 bytes cmem[0]
ptxas info    : Compiling entry function '_Z13dtopo_vel_111PfS_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_ffiiiiiii' for 'sm_70'
ptxas info    : Function properties for _Z13dtopo_vel_111PfS_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_ffiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 64 registers, 604 bytes cmem[0]
ptxas info    : Compiling entry function '_Z17dtopo_vel_111_optPfS_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_ffiiiiiii' for 'sm_70'
ptxas info    : Function properties for _Z17dtopo_vel_111_optPfS_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_ffiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

## Timing
```
            Type  Time      Time     Calls       Avg       Min       Max  Name
 GPU activities:   37.78%  207.48ms        10  20.748ms  20.628ms  20.834ms  dtopo_vel_111_blocks
                   28.17%  154.68ms        10  15.468ms  15.435ms  15.482ms  void dtopo_vel_111_unroll<int=1, int=2, int=2>
                   18.97%  104.16ms        10  10.416ms  10.407ms  10.423ms  void dtopo_vel_111_split1<int=1, int=2, int=4>
                    9.62%  52.851ms        10  5.2851ms  5.2796ms  5.2892ms  void dtopo_vel_111_split2<int=2, int=2, int=4>
                    5.07%  27.815ms        29  959.13us  4.0320us  2.3836ms  void gen_sequenced<curandStateXORWOW, float, int, __operator_&__
                    0.34%  1.8808ms         2  940.38us  938.01us  942.75us  generate_seed_pseudo
                    0.05%  264.19us         1  264.19us  264.19us  264.19us  compare
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  [CUDA memcpy DtoH]
```

## Dram
```
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB "
    Kernel: compare
          1                      dram_read_throughput             Device Memory Read Throughput  798.93GB/s  798.93GB/s  798.93GB/s
          1                     dram_write_throughput            Device Memory Write Throughput  5.5919GB/s  5.5919GB/s  5.5919GB/s
    Kernel: void dtopo_vel_111_unroll<int=1, int=2, int=2>
         10                      dram_read_throughput             Device Memory Read Throughput  497.89GB/s  498.81GB/s  498.47GB/s
         10                     dram_write_throughput            Device Memory Write Throughput  94.995GB/s  95.264GB/s  95.192GB/s
    Kernel: dtopo_vel_111_blocks
         10                      dram_read_throughput             Device Memory Read Throughput  306.74GB/s  314.00GB/s  312.83GB/s
         10                     dram_write_throughput            Device Memory Write Throughput  154.49GB/s  157.68GB/s  157.13GB/s
    Kernel: void dtopo_vel_111_split1<int=1, int=2, int=4>
         10                      dram_read_throughput             Device Memory Read Throughput  564.41GB/s  568.70GB/s  568.02GB/s
         10                     dram_write_throughput            Device Memory Write Throughput  93.608GB/s  94.312GB/s  94.202GB/s
    Kernel: void dtopo_vel_111_split2<int=2, int=2, int=4>
         10                      dram_read_throughput             Device Memory Read Throughput  576.88GB/s  581.30GB/s  580.40GB/s
         10                     dram_write_throughput            Device Memory Write Throughput  92.129GB/s  92.838GB/s  92.693GB/s
```

