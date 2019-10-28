    
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
## Dram (all metrics)

```
 Invocations                               Metric Name                          Metric Description         Min         Max         Avg
Device "TITAN X "
    Kernel: compare
          1                      dram_read_throughput               Device Memory Read Throughput  298.33GB/s  298.33GB/s  298.33GB/s
          1                     dram_write_throughput              Device Memory Write Throughput  844.46MB/s  844.46MB/s  844.46MB/s
          1                    dram_read_transactions             Device Memory Read Transactions     6709218     6709218     6709218
          1                   dram_write_transactions            Device Memory Write Transactions       18546       18546       18546
          1                          dram_utilization                   Device Memory Utilization    High 
          1                           dram_read_bytes      Total bytes read from DRAM to L2 cache   214694976   214694976   214694976
          1                          dram_write_bytes   Total bytes written from L2 cache to DRAM      593472      593472      593472
    Kernel: void dtopo_vel_111_unroll<int=1, int=2, int=2>
         10                      dram_read_throughput               Device Memory Read Throughput  178.63GB/s  186.47GB/s  185.27GB/s
         10                     dram_write_throughput              Device Memory Write Throughput  34.949GB/s  36.531GB/s  36.295GB/s
         10                    dram_read_transactions             Device Memory Read Transactions   251920093   252270487   252051133
         10                   dram_write_transactions            Device Memory Write Transactions    49338611    49567790    49378098
         10                          dram_utilization                   Device Memory Utilization     Mid 
         10                           dram_read_bytes      Total bytes read from DRAM to L2 cache  8061442976  8072655584  8065636272
         10                          dram_write_bytes   Total bytes written from L2 cache to DRAM  1578835552  1586169280  1580099164
    Kernel: dtopo_vel_111_blocks
         10                      dram_read_throughput               Device Memory Read Throughput  81.933GB/s  82.750GB/s  82.577GB/s
         10                     dram_write_throughput              Device Memory Write Throughput  158.25GB/s  159.68GB/s  159.22GB/s
         10                    dram_read_transactions             Device Memory Read Transactions   486735411   487928576   487460679
         10                   dram_write_transactions            Device Memory Write Transactions   939181455   941539859   939879492
         10                          dram_utilization                   Device Memory Utilization     Mid 
         10                           dram_read_bytes      Total bytes read from DRAM to L2 cache  1.5576e+10  1.5614e+10  1.5599e+10
         10                          dram_write_bytes   Total bytes written from L2 cache to DRAM  3.0054e+10  3.0129e+10  3.0076e+10
    Kernel: void dtopo_vel_111_split1<int=2, int=2, int=2>
         10                      dram_read_throughput               Device Memory Read Throughput  198.11GB/s  204.68GB/s  203.76GB/s
         10                     dram_write_throughput              Device Memory Write Throughput  39.784GB/s  41.121GB/s  40.961GB/s
         10                    dram_read_transactions             Device Memory Read Transactions   163938144   164214594   164036514
         10                   dram_write_transactions            Device Memory Write Transactions    32954100    33170328    32976379
         10                          dram_utilization                   Device Memory Utilization     Mid 
         10                           dram_read_bytes      Total bytes read from DRAM to L2 cache  5246020608  5254867008  5249168448
         10                          dram_write_bytes   Total bytes written from L2 cache to DRAM  1054531200  1061450496  1055244156
    Kernel: void dtopo_vel_111_split2<int=2, int=2, int=4>
         10                      dram_read_throughput               Device Memory Read Throughput  206.67GB/s  225.75GB/s  223.42GB/s
         10                     dram_write_throughput              Device Memory Write Throughput  33.816GB/s  36.944GB/s  36.574GB/s
         10                    dram_read_transactions             Device Memory Read Transactions   100591269   100689457   100646845
         10                   dram_write_transactions            Device Memory Write Transactions    16468769    16491696    16475825
         10                          dram_utilization                   Device Memory Utilization     Mid 
         10                           dram_read_bytes      Total bytes read from DRAM to L2 cache  3218920608  3222062624  3220699056
         10                          dram_write_bytes   Total bytes written from L2 cache to DRAM   527000608   527734272   527226406
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

## Dram (all metrics)

```
 Invocations                               Metric Name                          Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB "
    Kernel: compare
          1                      dram_read_throughput               Device Memory Read Throughput  797.86GB/s  797.86GB/s  797.86GB/s
          1                     dram_write_throughput              Device Memory Write Throughput  5.3311GB/s  5.3311GB/s  5.3311GB/s
          1                    dram_read_transactions             Device Memory Read Transactions     7077968     7077968     7077968
          1                   dram_write_transactions            Device Memory Write Transactions       47293       47293       47293
          1                          dram_utilization                   Device Memory Utilization    Max 
          1                           dram_read_bytes      Total bytes read from DRAM to L2 cache   226494976   226494976   226494976
          1                          dram_write_bytes   Total bytes written from L2 cache to DRAM     1513376     1513376     1513376
    Kernel: void dtopo_vel_111_unroll<int=1, int=2, int=2>
         10                      dram_read_throughput               Device Memory Read Throughput  492.49GB/s  497.21GB/s  495.96GB/s
         10                     dram_write_throughput              Device Memory Write Throughput  93.990GB/s  94.958GB/s  94.710GB/s
         10                    dram_read_transactions             Device Memory Read Transactions   258487581   258489507   258488517
         10                   dram_write_transactions            Device Memory Write Transactions    49319274    49366672    49361373
         10                          dram_utilization                   Device Memory Utilization    High 
         10                           dram_read_bytes      Total bytes read from DRAM to L2 cache  8271602592  8271664224  8271632569
         10                          dram_write_bytes   Total bytes written from L2 cache to DRAM  1578216768  1579733504  1579563961
    Kernel: dtopo_vel_111_blocks
         10                      dram_read_throughput               Device Memory Read Throughput  303.65GB/s  314.56GB/s  311.06GB/s
         10                     dram_write_throughput              Device Memory Write Throughput  152.45GB/s  157.28GB/s  155.71GB/s
         10                    dram_read_transactions             Device Memory Read Transactions   216391014   216404813   216399830
         10                   dram_write_transactions            Device Memory Write Transactions   108186898   108646164   108324084
         10                          dram_utilization                   Device Memory Utilization     Mid 
         10                           dram_read_bytes      Total bytes read from DRAM to L2 cache  6924512448  6924954016  6924794585
         10                          dram_write_bytes   Total bytes written from L2 cache to DRAM  3461980736  3476677248  3466370688
    Kernel: void dtopo_vel_111_split1<int=1, int=2, int=4>
         10                      dram_read_throughput               Device Memory Read Throughput  563.00GB/s  569.16GB/s  567.54GB/s
         10                     dram_write_throughput              Device Memory Write Throughput  93.376GB/s  94.388GB/s  94.122GB/s
         10                    dram_read_transactions             Device Memory Read Transactions   198598334   198624861   198616967
         10                   dram_write_transactions            Device Memory Write Transactions    32938349    32939244    32938879
         10                          dram_utilization                   Device Memory Utilization    High 
         10                           dram_read_bytes      Total bytes read from DRAM to L2 cache  6355146688  6355995552  6355742944
         10                          dram_write_bytes   Total bytes written from L2 cache to DRAM  1054027168  1054055808  1054044140
    Kernel: void dtopo_vel_111_split2<int=2, int=2, int=4>
         10                      dram_read_throughput               Device Memory Read Throughput  575.41GB/s  581.31GB/s  579.85GB/s
         10                     dram_write_throughput              Device Memory Write Throughput  91.910GB/s  92.851GB/s  92.621GB/s
         10                    dram_read_transactions             Device Memory Read Transactions   103017599   103023409   103021363
         10                   dram_write_transactions            Device Memory Write Transactions    16454765    16456863    16455829
         10                          dram_utilization                   Device Memory Utilization    High 
         10                           dram_read_bytes      Total bytes read from DRAM to L2 cache  3296563168  3296749088  3296683632
         10                          dram_write_bytes   Total bytes written from L2 cache to DRAM   526552480   526619616   526586537
```


Read transactions (unroll) : 0.25 * 10^9
Read transactions (split1 + split2) : 301615933
