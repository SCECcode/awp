# Stress kernel performance 


## Baseline
The baseline kernel is called "dtopo_str_111" and is found in `stress.cu`. This
version contains a mixture of index accesses using pre-computed indices stored
in registers, and macros. In addition, there is a loop over the slowest
dimension.

```
threads ( 64, 8, 1)
__launch_bounds__ (512)
```

```
ptxas info    : Compiling entry function '_Z13dtopo_str_111PfS_S_S_S_S_S_S_S_S_S_S_S_S_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_PKiS1_iiiiiiiiii' for 'sm_61'
ptxas info    : Function properties for _Z13dtopo_str_111PfS_S_S_S_S_S_S_S_S_S_S_S_S_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_PKiS1_iiiiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 90 registers, 688 bytes cmem[0], 28 bytes cmem[2]
```

### Timings (Pascal):
            Type  Time      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.77%  398.32ms        10  39.832ms  39.498ms  40.132ms  dtopo_str_111

At these settings, we are only getting one block per SM. We can change the
launch bounds to `256` to try to get 2 blocks:

```
threads ( 64, 4, 1)
__launch_bounds__ (256)
```

```
ptxas info    : Compiling entry function '_Z13dtopo_str_111PfS_S_S_S_S_S_S_S_S_S_S_S_S_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_PKiS1_iiiiiiiiii' for 'sm_61'
ptxas info    : Function properties for _Z13dtopo_str_111PfS_S_S_S_S_S_S_S_S_S_S_S_S_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_PKiS1_iiiiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 104 registers, 688 bytes cmem[0], 28 bytes cmem[2]
```

            Type  Time      Time     Calls       Avg       Min       Max  Name
  GPU activities:  47.04%  368.48ms        10  36.848ms  35.394ms  39.454ms  dtopo_str_111

## Macro
We remove all `pos`, `pos_ip1` etc and replace with macros. The loop over the
slowest dimension is still left in the code.

```
threads ( 64, 8, 1)
__launch_bounds__ (512)
```
```
ptxas info    : Compiling entry function '_Z19dtopo_str_111_macroPfS_S_S_S_S_S_S_S_S_S_S_S_S_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_PKiS1_iiiiiiiiii' for 'sm_61'
ptxas info    : Function properties for _Z19dtopo_str_111_macroPfS_S_S_S_S_S_S_S_S_S_S_S_S_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_PKiS1_iiiiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 102 registers, 688 bytes cmem[0], 28 bytes cmem[2]
```

```
            Type  Time      Time     Calls       Avg       Min       Max  Name
 GPU activities:   50.02%  3.79059s       100  37.906ms  35.817ms  41.374ms  dtopo_str_111_macro
                   49.43%  3.74567s       100  37.457ms  34.939ms  41.796ms  dtopo_str_111
```
The macro version is slightly slower compared to the baseline, but not by much.

## Macro with unrolling

The macro version with loop unrolling has been implemented. To find the fastest
version, we do a brute force search over the parameters
`<threads.x, threads.y, threads.z, unroll.x, unroll.y>`

Results after 10 iterations: (ordered by performance)

 ```
                 Type  Time      Time     Calls       Avg       Min       Max  Name
                    43.57%  316.94ms        10  31.694ms  31.552ms  31.906ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=1, int=2>
                   44.62%  331.08ms        10  33.108ms  33.028ms  33.286ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=1, int=2>
                   45.02%  336.87ms        10  33.687ms  33.624ms  33.743ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=1, int=2>
                   45.12%  339.23ms        10  33.923ms  33.803ms  34.031ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=1, int=2>
                   45.34%  338.46ms        10  33.846ms  33.774ms  33.885ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=1, int=1>
                   45.44%  342.11ms        10  34.211ms  34.128ms  34.437ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=1, int=1>
                   45.47%  341.95ms        10  34.195ms  34.009ms  34.480ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=1, int=2>
                   45.54%  347.26ms        10  34.726ms  34.675ms  34.825ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=1, int=1>
                   45.94%  348.09ms        10  34.809ms  34.729ms  34.924ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=1, int=2>
                   46.15%  352.30ms        10  35.230ms  35.074ms  35.421ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=1, int=1>
                   46.25%  356.78ms        10  35.678ms  35.395ms  35.876ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=1, int=2>
                   46.38%  355.01ms        10  35.501ms  35.343ms  35.731ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=1, int=1>
                   46.41%  358.40ms        10  35.840ms  35.552ms  36.124ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=1, int=2>
                   46.54%  357.60ms        10  35.760ms  35.647ms  36.000ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=1, int=1>
                   46.56%  359.59ms        10  35.959ms  35.759ms  36.059ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=1, int=2>
                   47.02%  363.97ms        10  36.397ms  36.264ms  36.562ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=1, int=2>
                   47.07%  367.11ms        10  36.711ms  36.552ms  37.020ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=1, int=1>
                   47.09%  366.78ms        10  36.678ms  36.579ms  36.798ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=1, int=1>
 GPU activities:   47.41%  369.06ms        10  36.906ms  36.799ms  37.031ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=1, int=2>
 GPU activities:   47.46%  375.06ms        10  37.506ms  37.297ms  37.681ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=1, int=2>
 GPU activities:   47.62%  373.71ms        10  37.371ms  37.182ms  37.705ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=1, int=1>
 GPU activities:   47.88%  380.00ms        10  38.000ms  37.853ms  38.115ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=1, int=2>
 GPU activities:   48.27%  385.96ms        10  38.596ms  38.515ms  38.743ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=1, int=2>
 GPU activities:   48.34%  382.28ms        10  38.228ms  38.133ms  38.335ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=1, int=1>
 GPU activities:   48.37%  384.59ms        10  38.459ms  38.373ms  38.593ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=1, int=2>
 GPU activities:   48.39%  386.56ms        10  38.656ms  37.991ms  39.315ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=1, int=2>
 GPU activities:   48.60%  388.58ms        10  38.858ms  38.786ms  38.962ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=1, int=1>
 GPU activities:   48.73%  390.67ms        10  39.067ms  38.995ms  39.246ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=1, int=1>
 GPU activities:   48.81%  391.45ms        10  39.145ms  39.119ms  39.212ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=1, int=1>
 GPU activities:   48.99%  395.39ms        10  39.539ms  39.436ms  39.628ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=1, int=2>
 GPU activities:   49.14%  397.71ms        10  39.771ms  39.708ms  39.887ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=1, int=1>
 GPU activities:   49.21%  395.91ms        10  39.591ms  39.536ms  39.714ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=1, int=1>
 GPU activities:   49.63%  404.13ms        10  40.413ms  40.308ms  40.543ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=1, int=1>
 GPU activities:   50.09%  412.61ms        10  41.261ms  41.071ms  41.402ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=2, int=1>
 GPU activities:   50.39%  418.12ms        10  41.812ms  41.035ms  43.431ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=1, int=2>

```
 A rather odd configuration appears to be the fastest
 ```
 threads (32, 1, 4)
 unroll (1, 2)
 ```

### Disable align
We can reduce the amount of memory to load by disabling the alignment setting.
Without alignment

GPU activities:   59.44%  4.27647s       100  42.765ms  41.126ms  45.576ms  dtopo_str_111
                   39.97%  2.87539s       100  28.754ms  28.646ms  28.977ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=1, int=2>

With aligment enabled (`align = 32`)
```
            Type  Time      Time     Calls       Avg       Min       Max  Name
 GPU activities:   53.22%  3.61607s       100  36.161ms  35.044ms  43.440ms  dtopo_str_111
                   45.99%  3.12491s       100  31.249ms  31.012ms  34.838ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=1, int=2>
```
With alignment disabled (`align = 0`)
```
            Type  Time      Time     Calls       Avg       Min       Max  Name
 GPU activities:   58.70%  4.33064s       100  43.306ms  41.505ms  50.575ms  dtopo_str_111
                   40.60%  2.99472s       100  29.947ms  29.768ms  32.673ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=1, int=2>
```
This setting slowsdown the baseline, but speeds-up the unrolled version.

We perform a brute force search to see if there is a better configuration available
when align is disabled.
```
                   41.05%  3.12865s       100  31.287ms  30.178ms  32.207ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=1, int=2>
                   41.10%  3.11504s       100  31.150ms  29.719ms  32.471ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=2, int=1>
                   41.56%  3.18147s       100  31.815ms  30.730ms  33.946ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=2, int=1>
                   41.56%  3.19363s       100  31.936ms  30.548ms  34.248ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=2, int=1>
                   41.74%  3.22496s       100  32.250ms  30.682ms  33.349ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=2, int=1>
                   41.75%  3.21643s       100  32.164ms  31.269ms  32.825ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=1, int=2>
                   42.16%  3.24836s       100  32.484ms  31.643ms  32.945ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=2, int=1>
                   42.19%  3.27553s       100  32.755ms  31.699ms  33.527ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=1, int=2>
                   42.31%  3.28832s       100  32.883ms  31.951ms  33.567ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=1, int=2>
                   42.38%  3.46060s       100  34.606ms  31.846ms  38.406ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=2, int=1>
                   42.42%  3.29400s       100  32.940ms  31.930ms  33.963ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=2, int=1>
                   42.46%  3.30416s       100  33.042ms  31.919ms  33.912ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=2, int=1>
                   42.48%  3.33062s       100  33.306ms  31.993ms  34.094ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=2, int=1>
                   42.52%  3.31082s       100  33.108ms  32.000ms  34.084ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=2, int=1>
                   42.56%  3.32350s       100  33.235ms  31.887ms  34.519ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=2, int=1>
                   42.63%  3.31174s       100  33.117ms  32.183ms  34.025ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=2, int=1>
                   42.71%  3.35151s       100  33.515ms  32.046ms  34.372ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=2, int=1>
                   42.93%  3.29938s       100  32.994ms  32.885ms  33.218ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=2, int=1>
                   43.06%  3.45541s       100  34.554ms  32.453ms  42.640ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=2, int=1>
                   43.43%  3.42675s       100  34.268ms  33.162ms  35.601ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=1, int=1>
                   43.57%  3.46046s       100  34.605ms  33.251ms  37.372ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=2, int=1>
                   43.57%  3.50801s       100  35.080ms  33.349ms  38.190ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=2, int=1>
                   43.63%  3.50208s       100  35.021ms  33.387ms  36.270ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=2, int=1>
                   43.70%  3.49730s       100  34.973ms  33.350ms  36.132ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=1, int=2>
                   43.95%  3.53616s       100  35.362ms  33.844ms  36.361ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=1, int=1>
                   44.21%  3.54593s       100  35.459ms  34.585ms  35.956ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=2, int=1>
                   44.27%  3.60604s       100  36.060ms  34.632ms  37.337ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=2, int=1>
                   44.33%  3.58582s       100  35.858ms  34.814ms  36.764ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=2, int=1>
                   44.37%  3.58065s       100  35.806ms  34.089ms  36.933ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=1, int=2>
                   44.39%  3.59864s       100  35.986ms  34.433ms  36.942ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=1, int=1>
                   44.40%  3.60139s       100  36.014ms  35.166ms  36.678ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=1, int=2>
                   44.42%  3.59309s       100  35.931ms  34.799ms  36.609ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=2, int=1>
                   44.43%  3.57862s       100  35.786ms  34.619ms  37.041ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=1, int=1>
                   44.44%  3.62809s       100  36.281ms  35.026ms  37.081ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=2, int=1>
                   44.47%  3.60966s       100  36.097ms  35.003ms  36.876ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=2, int=1>
                   44.51%  3.57938s       100  35.794ms  35.229ms  36.272ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=1, int=2>
                   44.59%  3.60129s       100  36.013ms  35.358ms  36.382ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=1, int=2>
                   44.61%  3.62163s       100  36.216ms  35.090ms  37.191ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=2, int=1>
                   44.77%  3.62577s       100  36.258ms  35.059ms  37.232ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=1, int=2>
                   44.85%  3.65047s       100  36.505ms  34.795ms  37.503ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=1, int=2>
                   44.91%  3.66992s       100  36.699ms  35.837ms  38.285ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=1, int=2>
                   45.10%  3.67975s       100  36.798ms  36.090ms  37.403ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=1, int=2>
                   45.23%  3.68051s       100  36.805ms  36.057ms  37.166ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=2, int=1>
                   45.45%  3.73073s       100  37.307ms  36.148ms  38.155ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=1, int=1>
                   45.47%  3.75078s       100  37.508ms  36.242ms  38.340ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=1, int=1>
                   45.58%  3.81057s       100  38.106ms  36.655ms  41.873ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=2, int=1>
                   45.59%  3.78087s       100  37.809ms  36.548ms  42.393ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=2, int=1>
                   45.67%  3.75899s       100  37.590ms  36.243ms  38.332ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=1, int=2>
                   45.73%  3.75855s       100  37.586ms  36.792ms  38.006ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=1, int=2>
                   45.74%  3.78195s       100  37.819ms  36.722ms  38.889ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=1, int=1>
                   45.83%  3.79455s       100  37.946ms  36.657ms  38.717ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=1, int=1>
                   45.84%  3.68276s       100  36.828ms  36.762ms  36.897ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=1, int=2>
                   45.87%  3.79418s       100  37.942ms  36.757ms  38.935ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=1, int=1>
                   46.68%  3.92246s       100  39.225ms  38.366ms  39.820ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=1, int=2>
                   46.81%  3.95010s       100  39.501ms  38.751ms  40.073ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=1, int=2>
                   46.88%  3.97753s       100  39.775ms  38.747ms  40.601ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=1, int=1>
                   46.91%  3.96109s       100  39.611ms  38.721ms  40.223ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=1, int=2>
                   47.11%  3.99118s       100  39.912ms  38.949ms  40.489ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=1, int=2>
                   47.44%  4.07029s       100  40.703ms  39.792ms  41.127ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=1, int=2>
                   47.45%  4.03180s       100  40.318ms  39.398ms  40.988ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=1, int=1>
                   47.52%  4.06030s       100  40.603ms  39.597ms  41.305ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=1, int=1>
                   47.67%  4.09343s       100  40.934ms  39.766ms  41.594ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=1, int=1>
                   47.81%  4.10549s       100  41.055ms  39.960ms  41.800ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=1, int=1>
                   47.92%  4.14069s       100  41.407ms  40.156ms  42.227ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=1, int=1>
                   48.01%  4.14857s       100  41.486ms  40.196ms  42.200ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=1, int=1>
                   48.45%  4.18538s       100  41.854ms  41.302ms  42.233ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=1, int=2>
                   48.49%  4.23780s       100  42.378ms  41.545ms  42.784ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=1, int=2>
                   49.30%  4.46610s       100  44.661ms  42.887ms  52.040ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=1, int=2>
 GPU activities:   50.08%  4.52371s       100  45.237ms  44.177ms  47.122ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=1, int=2>
 GPU activities:   50.85%  4.63386s       100  46.339ms  45.220ms  47.009ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=1, int=2>
```


```
             Type  Time      Time     Calls       Avg       Min       Max  Name
 GPU activities:   59.24%  4.32375s       100  43.237ms  42.035ms  46.414ms  dtopo_str_111
                   40.14%  2.92988s       100  29.299ms  29.147ms  30.159ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=2, int=1>
                   40.70%  3.03347s       100  30.335ms  29.929ms  32.592ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=2, int=1>
                   40.63%  2.99602s       100  29.960ms  29.747ms  32.186ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=1, int=2>
```
Hence,

```
threads (32, 1, 8)
unroll (2, 1)
```
Seems to be the best option.

## Indexing

Next, we try to replace all macros with pre-computed index accesses stored in
registers. The kernel found in `stress_index.cu` contains the implementation,
but still has the loop over fastest x in it. Also, it currently uses macros for
the f and g variables.

ptxas shows that the register usage decreases slightly compared to baseline:

```
ptxas info    : Compiling entry function '_Z19dtopo_str_111_indexILi64ELi4ELi1EEvPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_PKiS2_iiiiiiiiii' for 'sm_61'
ptxas info    : Function properties for _Z19dtopo_str_111_indexILi64ELi4ELi1EEvPfS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_PKfS2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_S2_PKiS2_iiiiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 75 registers, 688 bytes cmem[0], 28 bytes cmem[2]

ptxas info    : Compiling entry function '_Z13dtopo_str_111PfS_S_S_S_S_S_S_S_S_S_S_S_S_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_PKiS1_iiiiiiiiii' for 'sm_61'
ptxas info    : Function properties for _Z13dtopo_str_111PfS_S_S_S_S_S_S_S_S_S_S_S_S_S_PKfS1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_S1_PKiS1_iiiiiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 80 registers, 688 bytes cmem[0], 28 bytes cmem[2]

```

![](figures/op_index.png)
*Opcodes for index version*

![](figures/op_macros.png)
*Opcodes for macro version*

The indexed version runs slightly slower compared to the macro version:
```
  GPU activities:   49.81%  4.64623s       100  46.462ms  45.177ms  49.540ms  void dtopo_str_111_index<int=64, int=4, int=1>
                   49.22%  4.59038s       100  45.904ms  44.811ms  50.941ms  dtopo_str_111
```

