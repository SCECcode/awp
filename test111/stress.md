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

Results after 10 iterations: (best has been highlighted)

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
 GPU activities:   50.47%  418.30ms        10  41.830ms  41.635ms  42.074ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=2, int=1>
 GPU activities:   50.55%  420.40ms        10  42.040ms  41.848ms  42.448ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=2, int=1>
 GPU activities:   50.58%  422.97ms        10  42.297ms  41.541ms  42.700ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=1, int=2>
 GPU activities:   50.93%  424.96ms        10  42.496ms  42.402ms  42.655ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=1, int=2>
 GPU activities:   50.93%  425.70ms        10  42.570ms  42.301ms  42.984ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=2, int=1>
 GPU activities:   51.06%  427.78ms        10  42.778ms  42.399ms  43.214ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=2, int=1>
 GPU activities:   51.08%  428.95ms        10  42.895ms  42.550ms  43.280ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=2, int=1>
 GPU activities:   51.12%  432.56ms        10  43.256ms  42.193ms  43.956ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=1, int=2>
 GPU activities:   51.16%  429.45ms        10  42.945ms  42.634ms  43.311ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=2, int=1>
 GPU activities:   51.20%  433.85ms        10  43.385ms  42.491ms  44.013ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=1, int=2>
 GPU activities:   51.33%  438.11ms        10  43.811ms  43.043ms  44.290ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=1, int=2>
 GPU activities:   51.53%  438.62ms        10  43.862ms  42.826ms  46.246ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=2, int=1>
 GPU activities:   51.62%  438.83ms        10  43.883ms  42.723ms  45.951ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=2, int=1>
 GPU activities:   51.88%  445.73ms        10  44.573ms  43.651ms  45.138ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=2, int=1>
 GPU activities:   51.96%  442.30ms        10  44.230ms  44.063ms  44.705ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=2, int=1>
 GPU activities:   52.08%  445.82ms        10  44.582ms  44.425ms  44.961ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=2, int=1>
 GPU activities:   52.08%  446.63ms        10  44.663ms  44.357ms  45.367ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=2, int=1>
 GPU activities:   52.17%  448.01ms        10  44.801ms  43.920ms  45.297ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=2, int=1>
 GPU activities:   52.28%  451.08ms        10  45.108ms  44.556ms  45.523ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=2, int=1>
 GPU activities:   52.29%  452.98ms        10  45.298ms  44.393ms  46.000ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=2, int=1>
 GPU activities:   52.30%  450.25ms        10  45.025ms  44.809ms  45.156ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=2, int=1>
 GPU activities:   52.37%  450.40ms        10  45.040ms  44.812ms  45.325ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=2, int=1>
 GPU activities:   52.41%  451.94ms        10  45.194ms  45.061ms  45.351ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=2, int=1>
 GPU activities:   52.58%  475.67ms        10  47.567ms  46.131ms  50.167ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=1, int=1>
 GPU activities:   52.62%  458.88ms        10  45.888ms  45.656ms  46.487ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=2, int=1>
 GPU activities:   52.85%  460.24ms        10  46.024ms  45.784ms  46.316ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=2, int=1>
 GPU activities:   53.00%  461.63ms        10  46.163ms  45.884ms  46.519ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=2, int=1>
 GPU activities:   53.11%  464.96ms        10  46.496ms  46.306ms  46.758ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=2, int=1>
 GPU activities:   53.13%  468.09ms        10  46.809ms  46.529ms  47.383ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=1, int=2>
 GPU activities:   53.19%  463.91ms        10  46.391ms  45.734ms  46.775ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=1, int=1>
 GPU activities:   53.56%  471.82ms        10  47.182ms  47.089ms  47.286ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=1, int=2>
 GPU activities:   53.59%  472.96ms        10  47.296ms  47.137ms  47.517ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=1, int=2>
 GPU activities:   53.80%  479.29ms        10  47.929ms  47.738ms  48.399ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=1, int=1>
 GPU activities:   54.03%  482.75ms        10  48.275ms  48.000ms  48.506ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=2, int=1>
 GPU activities:   54.05%  484.04ms        10  48.404ms  48.137ms  48.814ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=1, int=2>
 GPU activities:   54.15%  483.77ms        10  48.377ms  47.985ms  48.791ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=1, int=1>
 GPU activities:   54.23%  485.07ms        10  48.507ms  48.417ms  48.673ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=1, int=1>
 GPU activities:   54.32%  488.23ms        10  48.823ms  48.606ms  49.367ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=2, int=1>
 GPU activities:   54.40%  492.23ms        10  49.223ms  48.989ms  49.373ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=2, int=1>
 GPU activities:   54.42%  489.68ms        10  48.968ms  48.791ms  49.199ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=1, int=2>
 GPU activities:   54.57%  492.75ms        10  49.275ms  48.986ms  49.677ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=2, int=1>
 GPU activities:   54.58%  493.21ms        10  49.321ms  48.947ms  49.816ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=1, int=1>
 GPU activities:   54.58%  494.95ms        10  49.495ms  49.130ms  49.852ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=1, int=1>
 GPU activities:   54.73%  498.16ms        10  49.816ms  49.530ms  50.516ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=2, int=1>
 GPU activities:   55.06%  501.40ms        10  50.140ms  50.027ms  50.286ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=1, int=1>
 GPU activities:   55.19%  507.43ms        10  50.743ms  50.377ms  51.109ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=1, int=1>
 GPU activities:   55.39%  509.44ms        10  50.944ms  50.839ms  51.138ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=1, int=1>
 GPU activities:   55.44%  509.47ms        10  50.947ms  50.845ms  51.104ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=1, int=1>
 GPU activities:   55.81%  520.91ms        10  52.091ms  51.806ms  52.456ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=1, int=1>
 GPU activities:   59.25%  594.97ms        10  59.497ms  59.364ms  59.774ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=1, int=4>
 GPU activities:   59.32%  599.29ms        10  59.929ms  59.807ms  60.480ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=1, int=4>
 GPU activities:   59.43%  600.47ms        10  60.047ms  59.734ms  60.661ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=1, int=4>
 GPU activities:   59.44%  598.75ms        10  59.875ms  59.778ms  60.260ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=1, int=4>
 GPU activities:   59.69%  610.71ms        10  61.071ms  60.611ms  61.518ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=1, int=4>
 GPU activities:   59.76%  608.30ms        10  60.830ms  60.572ms  61.459ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=1, int=4>
 GPU activities:   59.92%  621.07ms        10  62.107ms  61.735ms  62.502ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=1, int=4>
 GPU activities:   59.97%  616.51ms        10  61.651ms  61.171ms  62.186ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=1, int=4>
 GPU activities:   60.14%  617.45ms        10  61.745ms  61.499ms  62.517ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=1, int=4>
 GPU activities:   60.27%  621.84ms        10  62.184ms  61.830ms  62.706ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=1, int=4>
 GPU activities:   60.36%  624.34ms        10  62.434ms  62.263ms  62.766ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=1, int=4>
 GPU activities:   60.51%  627.98ms        10  62.798ms  62.406ms  63.452ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=1, int=4>
 GPU activities:   60.74%  639.14ms        10  63.914ms  63.592ms  64.596ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=1, int=4>
 GPU activities:   60.86%  634.49ms        10  63.449ms  63.236ms  63.750ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=1, int=4>
 GPU activities:   61.10%  640.95ms        10  64.095ms  64.017ms  64.558ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=1, int=4>
 GPU activities:   61.12%  646.21ms        10  64.621ms  64.429ms  64.929ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=1, int=4>
 GPU activities:   61.14%  645.69ms        10  64.569ms  64.520ms  64.875ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=1, int=4>
 GPU activities:   61.26%  649.72ms        10  64.972ms  64.875ms  65.570ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=1, int=4>
 GPU activities:   61.31%  647.67ms        10  64.767ms  64.668ms  65.168ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=1, int=4>
 GPU activities:   61.36%  653.29ms        10  65.329ms  64.754ms  66.130ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=1, int=4>
 GPU activities:   61.47%  656.31ms        10  65.631ms  64.997ms  66.469ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=1, int=8>
 GPU activities:   61.48%  653.39ms        10  65.339ms  65.085ms  65.778ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=1, int=8>
 GPU activities:   61.58%  656.93ms        10  65.693ms  65.320ms  66.210ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=1, int=8>
 GPU activities:   61.58%  659.05ms        10  65.905ms  65.618ms  67.361ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=1, int=4>
 GPU activities:   61.61%  659.75ms        10  65.975ms  65.469ms  66.874ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=1, int=8>
 GPU activities:   61.62%  658.55ms        10  65.855ms  65.666ms  66.213ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=1, int=8>
 GPU activities:   61.64%  658.80ms        10  65.880ms  65.773ms  66.252ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=1, int=4>
 GPU activities:   61.71%  658.30ms        10  65.830ms  65.691ms  66.410ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=1, int=8>
 GPU activities:   61.72%  659.83ms        10  65.983ms  65.778ms  66.347ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=1, int=4>
 GPU activities:   61.72%  661.85ms        10  66.185ms  65.888ms  66.897ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=1, int=4>
 GPU activities:   61.74%  657.07ms        10  65.707ms  65.603ms  66.284ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=1, int=8>
 GPU activities:   61.78%  661.29ms        10  66.129ms  65.867ms  66.576ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=1, int=8>
 GPU activities:   62.03%  672.31ms        10  67.231ms  66.993ms  68.056ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=1, int=4>
 GPU activities:   62.07%  672.56ms        10  67.256ms  66.974ms  67.830ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=1, int=4>
 GPU activities:   62.15%  670.86ms        10  67.086ms  66.851ms  67.314ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=1, int=8>
 GPU activities:   62.15%  675.77ms        10  67.577ms  66.644ms  68.490ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=1, int=4>
 GPU activities:   62.19%  675.95ms        10  67.595ms  67.119ms  68.377ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=1, int=8>
 GPU activities:   62.24%  679.07ms        10  67.907ms  67.612ms  68.458ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=1, int=8>
 GPU activities:   62.25%  677.72ms        10  67.772ms  67.463ms  68.088ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=1, int=8>
 GPU activities:   62.27%  675.72ms        10  67.572ms  67.454ms  67.946ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=1, int=8>
 GPU activities:   62.28%  679.40ms        10  67.940ms  67.533ms  70.067ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=1, int=4>
 GPU activities:   62.29%  682.14ms        10  68.214ms  67.368ms  69.147ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=1, int=8>
 GPU activities:   62.35%  678.49ms        10  67.849ms  67.488ms  68.348ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=1, int=8>
 GPU activities:   62.35%  679.25ms        10  67.925ms  67.080ms  72.089ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=1, int=8>
 GPU activities:   62.45%  685.73ms        10  68.573ms  68.265ms  69.106ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=1, int=8>
 GPU activities:   62.46%  686.00ms        10  68.600ms  68.256ms  69.595ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=1, int=8>
 GPU activities:   62.56%  681.20ms        10  68.120ms  68.036ms  68.403ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=1, int=8>
 GPU activities:   62.64%  685.99ms        10  68.599ms  68.420ms  69.409ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=1, int=8>
 GPU activities:   62.66%  685.73ms        10  68.573ms  68.448ms  69.450ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=1, int=8>
 GPU activities:   63.07%  699.45ms        10  69.945ms  69.693ms  70.570ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=1, int=8>
 GPU activities:   63.16%  704.73ms        10  70.473ms  69.882ms  73.121ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=1, int=8>
 GPU activities:   63.20%  703.84ms        10  70.384ms  69.860ms  70.944ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=1, int=8>
 GPU activities:   63.90%  729.79ms        10  72.979ms  72.683ms  73.641ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=1, int=8>
 GPU activities:   64.28%  736.88ms        10  73.688ms  73.495ms  74.491ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=1, int=8>
 GPU activities:   64.65%  755.60ms        10  75.560ms  74.870ms  76.323ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=1, int=8>
 GPU activities:   64.73%  752.98ms        10  75.298ms  75.103ms  76.056ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=1, int=8>
 GPU activities:   67.60%  854.67ms        10  85.467ms  85.241ms  85.855ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=2, int=4>
 GPU activities:   67.74%  861.18ms        10  86.118ms  85.942ms  86.601ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=2, int=4>
 GPU activities:   67.83%  863.18ms        10  86.318ms  86.042ms  86.713ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=2, int=4>
 GPU activities:   67.84%  864.79ms        10  86.479ms  86.339ms  86.785ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=2, int=4>
 GPU activities:   68.03%  870.05ms        10  87.005ms  86.900ms  87.530ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=2, int=4>
 GPU activities:   68.15%  877.23ms        10  87.723ms  87.429ms  88.098ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=2, int=4>
 GPU activities:   68.23%  882.96ms        10  88.296ms  87.988ms  88.848ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=2, int=4>
 GPU activities:   68.25%  885.18ms        10  88.518ms  88.213ms  89.116ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=2, int=4>
 GPU activities:   68.41%  890.57ms        10  89.057ms  88.731ms  89.994ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=2, int=4>
 GPU activities:   68.44%  890.89ms        10  89.089ms  88.843ms  89.342ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=2, int=4>
 GPU activities:   68.53%  891.56ms        10  89.156ms  88.840ms  89.647ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=2, int=4>
 GPU activities:   68.59%  893.15ms        10  89.315ms  89.053ms  89.794ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=2, int=4>
 GPU activities:   68.63%  895.25ms        10  89.525ms  89.109ms  90.005ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=2, int=4>
 GPU activities:   68.63%  897.51ms        10  89.751ms  89.530ms  90.210ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=2, int=4>
 GPU activities:   68.74%  897.58ms        10  89.758ms  89.488ms  91.315ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=2, int=4>
 GPU activities:   68.95%  907.15ms        10  90.715ms  90.423ms  91.113ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=2, int=4>
 GPU activities:   69.00%  918.48ms        10  91.848ms  91.551ms  92.504ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=2, int=4>
 GPU activities:   69.02%  912.59ms        10  91.259ms  91.131ms  91.705ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=2, int=4>
 GPU activities:   69.50%  934.77ms        10  93.477ms  93.316ms  93.766ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=2, int=4>
 GPU activities:   69.55%  938.77ms        10  93.877ms  93.758ms  94.157ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=2, int=4>
 GPU activities:   69.58%  939.90ms        10  93.990ms  93.718ms  94.319ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=2, int=4>
 GPU activities:   69.65%  940.06ms        10  94.006ms  93.812ms  94.445ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=2, int=4>
 GPU activities:   69.67%  940.06ms        10  94.006ms  93.774ms  94.297ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=2, int=4>
 GPU activities:   69.69%  935.43ms        10  93.543ms  93.335ms  94.077ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=2, int=4>
 GPU activities:   69.92%  951.51ms        10  95.151ms  95.045ms  95.339ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=2, int=4>
 GPU activities:   69.96%  955.60ms        10  95.560ms  95.111ms  96.113ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=2, int=4>
 GPU activities:   69.99%  961.93ms        10  96.193ms  96.015ms  96.344ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=2, int=2>
 GPU activities:   70.20%  965.35ms        10  96.535ms  95.945ms  98.898ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=2, int=4>
 GPU activities:   70.22%  961.02ms        10  96.102ms  95.872ms  96.274ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=2, int=2>
 GPU activities:   70.24%  967.90ms        10  96.790ms  96.462ms  97.435ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=2, int=4>
 GPU activities:   70.29%  967.09ms        10  96.709ms  96.464ms  97.063ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=2, int=2>
 GPU activities:   70.66%  988.29ms        10  98.829ms  98.667ms  99.164ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=2, int=2>
 GPU activities:   70.81%  995.07ms        10  99.507ms  99.366ms  99.734ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=2, int=2>
 GPU activities:   70.85%  996.60ms        10  99.660ms  99.261ms  100.33ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=2, int=2>
 GPU activities:   70.88%  997.13ms        10  99.713ms  99.467ms  100.03ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=2, int=2>
 GPU activities:   71.00%  1.00137s        10  100.14ms  99.964ms  100.30ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=2, int=2>
 GPU activities:   71.08%  1.00610s        10  100.61ms  100.37ms  100.97ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=2, int=2>
 GPU activities:   71.11%  1.01102s        10  101.10ms  100.88ms  101.34ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=2, int=2>
 GPU activities:   71.23%  1.01427s        10  101.43ms  101.07ms  101.91ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=2, int=2>
 GPU activities:   71.37%  1.02639s        10  102.64ms  102.34ms  103.18ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=2, int=2>
 GPU activities:   71.45%  1.02297s        10  102.30ms  102.01ms  102.68ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=2, int=2>
 GPU activities:   71.45%  1.02394s        10  102.39ms  102.05ms  102.66ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=2, int=2>
 GPU activities:   71.45%  1.02695s        10  102.69ms  102.35ms  103.32ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=2, int=2>
 GPU activities:   71.59%  1.03932s        10  103.93ms  103.56ms  104.38ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=2, int=2>
 GPU activities:   71.60%  1.02984s        10  102.98ms  102.89ms  103.34ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=2, int=2>
 GPU activities:   72.11%  1.06250s        10  106.25ms  105.93ms  107.56ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=2, int=2>
 GPU activities:   72.11%  1.06319s        10  106.32ms  106.01ms  106.70ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=2, int=2>
 GPU activities:   72.20%  1.05965s        10  105.97ms  105.84ms  106.28ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=2, int=2>
 GPU activities:   72.38%  1.07196s        10  107.20ms  107.06ms  107.50ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=2, int=2>
 GPU activities:   72.90%  1.09929s        10  109.93ms  109.68ms  110.22ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=2, int=2>
 GPU activities:   72.94%  1.10054s        10  110.05ms  109.91ms  110.28ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=2, int=2>
 GPU activities:   73.08%  1.11029s        10  111.03ms  110.75ms  111.28ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=2, int=2>
 GPU activities:   73.10%  1.11538s        10  111.54ms  111.17ms  111.81ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=2, int=2>
 GPU activities:   73.11%  1.11153s        10  111.15ms  110.58ms  111.70ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=2, int=2>
 GPU activities:   73.71%  1.15283s        10  115.28ms  114.88ms  116.03ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=2, int=2>
 GPU activities:   73.84%  1.16411s        10  116.41ms  116.16ms  116.73ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=2, int=2>
 GPU activities:   75.13%  1.24186s        10  124.19ms  123.83ms  124.56ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=4, int=1>
 GPU activities:   75.13%  1.24476s        10  124.48ms  124.22ms  124.86ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=4, int=1>
 GPU activities:   75.25%  1.24126s        10  124.13ms  123.74ms  124.52ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=4, int=1>
 GPU activities:   75.26%  1.24628s        10  124.63ms  124.52ms  125.14ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=4, int=1>
 GPU activities:   75.29%  1.24701s        10  124.70ms  124.45ms  125.02ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=4, int=1>
 GPU activities:   75.35%  1.26126s        10  126.13ms  125.79ms  126.82ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=4, int=1>
 GPU activities:   75.44%  1.25932s        10  125.93ms  125.70ms  126.25ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=4, int=1>
 GPU activities:   75.63%  1.26983s        10  126.98ms  126.75ms  127.79ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=4, int=1>
 GPU activities:   75.73%  1.28972s        10  128.97ms  128.74ms  129.33ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=4, int=1>
 GPU activities:   75.89%  1.29170s        10  129.17ms  128.92ms  129.42ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=4, int=1>
 GPU activities:   75.92%  1.28756s        10  128.76ms  128.48ms  129.16ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=4, int=1>
 GPU activities:   75.99%  1.28779s        10  128.78ms  127.81ms  129.81ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=4, int=1>
 GPU activities:   75.99%  1.30266s        10  130.27ms  129.78ms  130.85ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=4, int=1>
 GPU activities:   76.06%  1.30215s        10  130.21ms  130.04ms  130.42ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=4, int=1>
 GPU activities:   76.07%  1.30329s        10  130.33ms  130.04ms  130.90ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=4, int=1>
 GPU activities:   76.09%  1.30476s        10  130.48ms  130.15ms  130.67ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=4, int=1>
 GPU activities:   76.17%  1.31054s        10  131.05ms  130.68ms  131.54ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=4, int=1>
 GPU activities:   76.27%  1.32363s        10  132.36ms  132.04ms  132.86ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=4, int=1>
 GPU activities:   76.81%  1.35994s        10  135.99ms  135.79ms  136.19ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=4, int=2>
 GPU activities:   76.82%  1.35618s        10  135.62ms  135.43ms  135.82ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=4, int=1>
 GPU activities:   77.06%  1.37669s        10  137.67ms  137.44ms  137.94ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=4, int=2>
 GPU activities:   77.09%  1.38061s        10  138.06ms  137.82ms  138.36ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=4, int=2>
 GPU activities:   77.15%  1.37942s        10  137.94ms  137.55ms  138.08ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=4, int=2>
 GPU activities:   77.17%  1.39183s        10  139.18ms  138.74ms  139.83ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=4, int=2>
 GPU activities:   77.27%  1.39614s        10  139.61ms  139.19ms  140.45ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=4, int=2>
 GPU activities:   77.31%  1.39114s        10  139.11ms  138.99ms  139.29ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=4, int=2>
 GPU activities:   77.36%  1.39983s        10  139.98ms  139.68ms  140.41ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=4, int=2>
 GPU activities:   77.36%  1.40120s        10  140.12ms  138.88ms  144.37ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=4, int=1>
 GPU activities:   77.41%  1.41167s        10  141.17ms  140.84ms  141.85ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=4, int=2>
 GPU activities:   77.53%  1.41477s        10  141.48ms  139.10ms  142.98ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=4, int=1>
 GPU activities:   77.61%  1.42024s        10  142.02ms  138.42ms  143.97ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=4, int=1>
 GPU activities:   77.79%  1.43379s        10  143.38ms  142.43ms  143.79ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=4, int=2>
 GPU activities:   77.83%  1.44226s        10  144.23ms  143.98ms  144.47ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=4, int=2>
 GPU activities:   77.85%  1.43894s        10  143.89ms  143.48ms  144.34ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=4, int=1>
 GPU activities:   77.85%  1.44182s        10  144.18ms  143.81ms  144.65ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=4, int=1>
 GPU activities:   77.89%  1.44383s        10  144.38ms  143.52ms  144.93ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=4, int=2>
 GPU activities:   77.90%  1.43914s        10  143.91ms  143.43ms  144.30ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=4, int=1>
 GPU activities:   77.91%  1.45373s        10  145.37ms  145.11ms  145.76ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=4, int=1>
 GPU activities:   77.96%  1.44270s        10  144.27ms  143.85ms  144.70ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=4, int=2>
 GPU activities:   77.99%  1.44888s        10  144.89ms  144.67ms  145.06ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=4, int=1>
 GPU activities:   78.06%  1.45113s        10  145.11ms  144.90ms  145.34ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=4, int=2>
 GPU activities:   78.11%  1.46096s        10  146.10ms  145.72ms  146.36ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=4, int=2>
 GPU activities:   78.14%  1.45779s        10  145.78ms  145.61ms  145.96ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=4, int=2>
 GPU activities:   78.15%  1.45685s        10  145.68ms  145.27ms  146.25ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=4, int=2>
 GPU activities:   78.17%  1.47198s        10  147.20ms  146.80ms  148.05ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=4, int=2>
 GPU activities:   78.31%  1.50424s        10  150.42ms  148.92ms  154.35ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=4, int=1>
 GPU activities:   79.30%  1.57494s        10  157.49ms  156.99ms  157.86ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=4, int=2>
 GPU activities:   79.40%  1.58175s        10  158.18ms  157.96ms  158.34ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=4, int=2>
 GPU activities:   79.42%  1.57735s        10  157.73ms  156.76ms  158.52ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=4, int=2>
 GPU activities:   79.51%  1.59015s        10  159.02ms  158.67ms  159.41ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=4, int=2>
 GPU activities:   79.52%  1.58836s        10  158.84ms  158.46ms  159.32ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=4, int=2>
 GPU activities:   79.54%  1.59347s        10  159.35ms  159.18ms  159.73ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=4, int=2>
 GPU activities:   79.58%  1.59381s        10  159.38ms  158.02ms  159.88ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=4, int=2>
 GPU activities:   79.67%  1.61665s        10  161.67ms  161.31ms  162.21ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=4, int=2>
 GPU activities:   79.73%  1.61367s        10  161.37ms  160.98ms  161.60ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=4, int=2>
 GPU activities:   79.79%  1.66676s        10  166.68ms  165.57ms  167.48ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=4, int=2>
 GPU activities:   81.42%  1.80881s        10  180.88ms  179.68ms  181.68ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=8, int=1>
 GPU activities:   81.59%  1.81740s        10  181.74ms  181.36ms  182.18ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=8, int=1>
 GPU activities:   81.60%  1.80847s        10  180.85ms  179.75ms  182.38ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=8, int=1>
 GPU activities:   81.65%  1.82903s        10  182.90ms  182.02ms  184.36ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=8, int=1>
 GPU activities:   81.82%  1.84087s        10  184.09ms  183.33ms  184.96ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=8, int=1>
 GPU activities:   81.85%  1.84874s        10  184.87ms  183.57ms  185.93ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=8, int=1>
 GPU activities:   81.86%  1.84987s        10  184.99ms  184.44ms  185.96ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=8, int=1>
 GPU activities:   81.89%  1.85421s        10  185.42ms  184.23ms  186.27ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=8, int=1>
 GPU activities:   81.90%  1.85572s        10  185.57ms  184.93ms  186.08ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=8, int=1>
 GPU activities:   82.57%  1.93953s        10  193.95ms  193.56ms  194.33ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=8, int=1>
 GPU activities:   82.58%  1.95286s        10  195.29ms  194.26ms  197.14ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=8, int=1>
 GPU activities:   82.71%  1.96412s        10  196.41ms  195.82ms  197.08ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=8, int=1>
 GPU activities:   82.82%  1.98987s        10  198.99ms  198.53ms  199.48ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=8, int=1>
 GPU activities:   82.83%  1.96325s        10  196.33ms  195.83ms  197.21ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=8, int=1>
 GPU activities:   82.84%  1.97160s        10  197.16ms  196.52ms  198.02ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=8, int=1>
 GPU activities:   82.88%  1.97595s        10  197.60ms  197.06ms  198.06ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=8, int=1>
 GPU activities:   82.92%  1.98223s        10  198.22ms  197.10ms  199.00ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=8, int=1>
 GPU activities:   83.10%  2.00620s        10  200.62ms  200.21ms  201.79ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=8, int=1>
 GPU activities:   84.58%  2.24666s        10  224.67ms  224.39ms  224.98ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=8, int=1>
 GPU activities:   84.58%  2.25337s        10  225.34ms  224.92ms  225.63ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=8, int=1>
 GPU activities:   84.66%  2.26563s        10  226.56ms  226.29ms  226.95ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=8, int=1>
 GPU activities:   84.73%  2.30153s        10  230.15ms  228.07ms  231.42ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=8, int=1>
 GPU activities:   84.77%  2.28142s        10  228.14ms  227.84ms  228.48ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=8, int=1>
 GPU activities:   84.78%  2.27413s        10  227.41ms  226.79ms  227.79ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=8, int=1>
 GPU activities:   84.83%  2.30578s        10  230.58ms  230.02ms  231.24ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=8, int=1>
 GPU activities:   84.85%  2.28597s        10  228.60ms  227.82ms  229.37ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=8, int=1>
 GPU activities:   84.93%  2.30264s        10  230.26ms  229.67ms  230.72ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=8, int=1>
 GPU activities:   84.96%  2.31514s        10  231.51ms  230.49ms  232.08ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=8, int=1>

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
 GPU activities:   51.21%  4.72383s       100  47.238ms  45.972ms  51.522ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=1, int=2>
 GPU activities:   51.26%  4.69863s       100  46.986ms  45.991ms  47.607ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=1, int=1>
 GPU activities:   51.33%  4.60147s       100  46.015ms  45.829ms  48.835ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=1, int=1>
 GPU activities:   51.51%  4.76327s       100  47.633ms  46.329ms  48.357ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=1, int=1>
 GPU activities:   51.55%  4.78678s       100  47.868ms  46.606ms  48.668ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=1, int=1>
 GPU activities:   51.56%  4.78822s       100  47.882ms  46.796ms  48.533ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=1, int=2>
 GPU activities:   52.04%  4.86875s       100  48.687ms  47.536ms  49.227ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=1, int=1>
 GPU activities:   52.04%  4.88566s       100  48.857ms  47.442ms  51.740ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=1, int=1>
 GPU activities:   52.49%  4.98230s       100  49.823ms  48.458ms  50.736ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=1, int=1>
 GPU activities:   52.78%  5.02966s       100  50.297ms  48.884ms  50.933ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=1, int=1>
 GPU activities:   53.16%  5.04565s       100  50.456ms  49.773ms  50.923ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=1, int=1>
 GPU activities:   53.57%  5.23533s       100  52.353ms  50.969ms  56.250ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=1, int=1>
 GPU activities:   53.93%  5.28462s       100  52.846ms  51.661ms  54.753ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=1, int=1>
 GPU activities:   54.79%  5.47101s       100  54.710ms  53.122ms  59.207ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=1, int=1>
 GPU activities:   57.20%  5.88468s       100  58.847ms  58.433ms  59.771ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=1, int=4>
 GPU activities:   57.38%  5.93010s       100  59.301ms  58.442ms  62.491ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=1, int=4>
 GPU activities:   58.03%  6.10769s       100  61.077ms  59.907ms  63.888ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=1, int=4>
 GPU activities:   58.20%  6.18238s       100  61.824ms  59.985ms  66.291ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=1, int=4>
 GPU activities:   58.29%  6.21401s       100  62.140ms  60.188ms  65.686ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=1, int=4>
 GPU activities:   58.51%  6.26659s       100  62.666ms  60.720ms  65.878ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=1, int=4>
 GPU activities:   58.60%  6.27083s       100  62.708ms  60.766ms  65.440ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=1, int=4>
 GPU activities:   58.72%  6.21547s       100  62.155ms  61.982ms  62.379ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=1, int=4>
 GPU activities:   58.83%  6.42754s       100  64.275ms  61.759ms  69.635ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=1, int=4>
 GPU activities:   58.85%  6.53962s       100  65.396ms  61.703ms  71.201ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=1, int=4>
 GPU activities:   59.09%  6.54567s       100  65.457ms  62.344ms  70.837ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=1, int=4>
 GPU activities:   59.11%  6.39225s       100  63.922ms  62.365ms  65.793ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=1, int=4>
 GPU activities:   59.19%  6.43728s       100  64.373ms  62.474ms  66.177ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=1, int=4>
 GPU activities:   59.25%  6.43540s       100  64.354ms  62.585ms  66.397ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=1, int=4>
 GPU activities:   59.25%  6.45239s       100  64.524ms  63.002ms  66.431ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=1, int=4>
 GPU activities:   59.38%  6.49227s       100  64.923ms  62.880ms  66.836ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=1, int=4>
 GPU activities:   59.38%  6.51156s       100  65.116ms  63.247ms  68.090ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=1, int=4>
 GPU activities:   59.90%  6.59199s       100  65.920ms  64.802ms  67.874ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=1, int=4>
 GPU activities:   60.00%  6.62727s       100  66.273ms  64.801ms  67.760ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=1, int=4>
 GPU activities:   60.02%  6.65892s       100  66.589ms  64.554ms  68.925ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=1, int=8>
 GPU activities:   60.03%  6.72579s       100  67.258ms  65.167ms  75.560ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=1, int=4>
 GPU activities:   60.08%  6.67776s       100  66.778ms  65.013ms  68.561ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=1, int=4>
 GPU activities:   60.13%  6.66439s       100  66.644ms  65.030ms  69.217ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=1, int=8>
 GPU activities:   60.14%  6.68311s       100  66.831ms  64.897ms  68.289ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=1, int=4>
 GPU activities:   60.14%  6.68769s       100  66.877ms  65.129ms  68.953ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=1, int=4>
 GPU activities:   60.19%  6.72047s       100  67.205ms  65.099ms  69.514ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=1, int=4>
 GPU activities:   60.20%  6.68652s       100  66.865ms  65.309ms  68.140ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=1, int=4>
 GPU activities:   60.23%  6.69639s       100  66.964ms  65.398ms  68.607ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=1, int=8>
 GPU activities:   60.27%  6.72337s       100  67.234ms  65.269ms  69.992ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=1, int=8>
 GPU activities:   60.27%  6.75041s       100  67.504ms  65.464ms  69.999ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=1, int=4>
 GPU activities:   60.34%  6.74027s       100  67.403ms  65.708ms  69.967ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=1, int=8>
 GPU activities:   60.38%  6.65460s       100  66.546ms  66.351ms  66.815ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=1, int=8>
 GPU activities:   60.39%  6.79361s       100  67.936ms  65.663ms  69.958ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=1, int=8>
 GPU activities:   60.43%  6.87845s       100  68.785ms  65.955ms  75.696ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=1, int=8>
 GPU activities:   60.45%  6.76596s       100  67.660ms  66.155ms  70.050ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=1, int=8>
 GPU activities:   60.47%  6.77898s       100  67.790ms  66.176ms  68.926ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=1, int=8>
 GPU activities:   60.50%  6.80182s       100  68.018ms  65.510ms  70.068ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=1, int=8>
 GPU activities:   60.54%  6.80434s       100  68.043ms  66.177ms  69.215ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=1, int=8>
 GPU activities:   60.58%  6.80724s       100  68.072ms  65.956ms  70.531ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=1, int=8>
 GPU activities:   60.82%  6.88858s       100  68.886ms  66.690ms  70.487ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=1, int=8>
 GPU activities:   60.93%  6.87995s       100  68.800ms  67.496ms  71.519ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=1, int=4>
 GPU activities:   61.01%  6.91936s       100  69.194ms  67.838ms  70.364ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=1, int=8>
 GPU activities:   61.14%  6.92828s       100  69.283ms  68.247ms  70.893ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=1, int=8>
 GPU activities:   61.15%  7.14470s       100  71.447ms  67.722ms  79.699ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=1, int=4>
 GPU activities:   61.16%  7.02101s       100  70.210ms  68.263ms  75.846ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=1, int=8>
 GPU activities:   61.17%  6.98377s       100  69.838ms  68.124ms  71.901ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=1, int=8>
 GPU activities:   61.22%  6.96775s       100  69.677ms  68.412ms  72.612ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=1, int=8>
 GPU activities:   61.25%  6.97150s       100  69.715ms  68.446ms  71.077ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=1, int=8>
 GPU activities:   61.38%  7.03866s       100  70.387ms  68.819ms  72.475ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=1, int=8>
 GPU activities:   61.56%  7.10362s       100  71.036ms  69.192ms  74.040ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=1, int=8>
 GPU activities:   61.80%  7.16755s       100  71.675ms  69.935ms  75.726ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=1, int=8>
 GPU activities:   62.03%  7.23217s       100  72.322ms  70.609ms  76.252ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=1, int=8>
 GPU activities:   62.16%  7.34275s       100  73.428ms  71.458ms  82.677ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=1, int=8>
 GPU activities:   63.16%  7.65690s       100  76.569ms  74.363ms  82.327ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=1, int=8>
 GPU activities:   63.47%  7.66610s       100  76.661ms  75.549ms  77.998ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=1, int=8>
 GPU activities:   63.60%  7.92721s       100  79.272ms  76.554ms  86.269ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=1, int=8>
 GPU activities:   65.64%  8.43246s       100  84.325ms  83.234ms  86.827ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=2, int=2>
 GPU activities:   65.66%  8.45015s       100  84.502ms  82.768ms  85.801ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=2, int=2>
 GPU activities:   65.77%  8.46572s       100  84.657ms  83.931ms  85.877ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=2, int=2>
 GPU activities:   65.77%  8.48161s       100  84.816ms  83.500ms  87.154ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=2, int=2>
 GPU activities:   65.85%  8.51287s       100  85.129ms  83.738ms  87.082ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=2, int=2>
 GPU activities:   66.08%  8.61640s       100  86.164ms  84.800ms  87.934ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=2, int=2>
 GPU activities:   66.19%  8.65673s       100  86.567ms  85.283ms  88.358ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=2, int=2>
 GPU activities:   66.45%  8.73438s       100  87.344ms  86.410ms  88.720ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=2, int=2>
 GPU activities:   66.45%  8.75738s       100  87.574ms  85.992ms  90.565ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=2, int=2>
 GPU activities:   66.50%  8.77977s       100  87.798ms  86.296ms  89.403ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=2, int=2>
 GPU activities:   66.52%  8.76288s       100  87.629ms  86.519ms  90.015ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=2, int=2>
 GPU activities:   66.52%  8.79273s       100  87.927ms  86.345ms  89.805ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=2, int=2>
 GPU activities:   66.56%  8.81399s       100  88.140ms  86.444ms  91.007ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=2, int=2>
 GPU activities:   66.61%  8.85825s       100  88.583ms  86.650ms  97.445ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=2, int=2>
 GPU activities:   66.64%  8.74709s       100  87.471ms  87.347ms  87.843ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=2, int=2>
 GPU activities:   67.28%  9.17184s       100  91.718ms  89.377ms  94.285ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=2, int=2>
 GPU activities:   67.34%  9.14512s       100  91.451ms  89.481ms  93.391ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=2, int=2>
 GPU activities:   67.42%  9.15952s       100  91.595ms  89.513ms  94.559ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=2, int=2>
 GPU activities:   67.65%  9.23827s       100  92.383ms  91.449ms  94.203ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=2, int=2>
 GPU activities:   67.78%  9.28300s       100  92.830ms  91.951ms  94.672ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=2, int=2>
 GPU activities:   67.81%  9.28964s       100  92.896ms  92.303ms  94.621ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=2, int=2>
 GPU activities:   67.86%  9.30844s       100  93.084ms  92.312ms  95.372ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=2, int=2>
 GPU activities:   67.87%  9.31506s       100  93.151ms  92.314ms  95.233ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=2, int=2>
 GPU activities:   67.96%  9.33571s       100  93.357ms  92.752ms  95.187ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=2, int=2>
 GPU activities:   67.97%  9.33212s       100  93.321ms  92.733ms  95.172ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=2, int=2>
 GPU activities:   68.64%  9.64843s       100  96.484ms  95.327ms  98.158ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=2, int=2>
 GPU activities:   68.76%  9.73249s       100  97.325ms  95.681ms  99.432ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=2, int=2>
 GPU activities:   68.79%  10.2362s       100  102.36ms  96.862ms  114.76ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=2, int=2>
 GPU activities:   70.53%  10.5540s       100  105.54ms  104.28ms  106.79ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=2, int=4>
 GPU activities:   70.59%  10.6429s       100  106.43ms  104.87ms  108.31ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=2, int=4>
 GPU activities:   70.69%  10.6706s       100  106.71ms  105.37ms  108.74ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=2, int=4>
 GPU activities:   70.75%  10.7331s       100  107.33ms  105.51ms  108.81ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=2, int=4>
 GPU activities:   70.95%  10.7548s       100  107.55ms  106.75ms  109.72ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=2, int=4>
 GPU activities:   71.01%  11.1086s       100  111.09ms  108.26ms  116.30ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=2, int=4>
 GPU activities:   71.06%  10.8257s       100  108.26ms  107.55ms  110.32ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=2, int=4>
 GPU activities:   71.10%  10.9041s       100  109.04ms  107.62ms  111.33ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=2, int=4>
 GPU activities:   71.14%  10.9298s       100  109.30ms  108.19ms  111.03ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=2, int=4>
 GPU activities:   71.15%  10.9287s       100  109.29ms  108.20ms  111.24ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=2, int=4>
 GPU activities:   71.19%  10.9553s       100  109.55ms  108.26ms  110.75ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=2, int=4>
 GPU activities:   71.23%  10.9644s       100  109.64ms  108.06ms  111.21ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=2, int=4>
 GPU activities:   71.27%  11.0655s       100  110.65ms  109.09ms  116.20ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=2, int=4>
 GPU activities:   71.28%  11.0338s       100  110.34ms  109.08ms  116.49ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=2, int=4>
 GPU activities:   71.63%  11.0717s       100  110.72ms  110.64ms  111.09ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=2, int=4>
 GPU activities:   71.72%  11.2682s       100  112.68ms  111.12ms  114.35ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=2, int=4>
 GPU activities:   71.81%  11.3161s       100  113.16ms  111.46ms  115.08ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=2, int=4>
 GPU activities:   71.89%  11.3485s       100  113.48ms  111.99ms  116.18ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=2, int=4>
 GPU activities:   71.89%  11.3558s       100  113.56ms  112.57ms  114.69ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=2, int=4>
 GPU activities:   71.93%  11.3657s       100  113.66ms  112.16ms  116.39ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=2, int=4>
 GPU activities:   71.93%  11.3807s       100  113.81ms  112.23ms  115.64ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=2, int=4>
 GPU activities:   71.95%  11.3868s       100  113.87ms  112.86ms  115.65ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=2, int=4>
 GPU activities:   71.98%  11.3714s       100  113.71ms  112.11ms  115.70ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=2, int=4>
 GPU activities:   71.99%  11.3983s       100  113.98ms  112.52ms  116.44ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=2, int=4>
 GPU activities:   72.20%  11.5209s       100  115.21ms  113.85ms  117.23ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=2, int=4>
 GPU activities:   72.64%  11.7096s       100  117.10ms  116.06ms  118.23ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=2, int=4>
 GPU activities:   72.73%  11.9271s       100  119.27ms  116.86ms  129.82ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=2, int=4>
 GPU activities:   72.75%  12.1727s       100  121.73ms  117.22ms  140.36ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=2, int=4>
 GPU activities:   72.92%  11.9267s       100  119.27ms  114.99ms  120.91ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=4, int=1>
 GPU activities:   73.09%  12.0809s       100  120.81ms  118.51ms  122.88ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=4, int=1>
 GPU activities:   73.14%  12.0925s       100  120.93ms  119.12ms  122.77ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=4, int=1>
 GPU activities:   73.19%  12.0899s       100  120.90ms  118.82ms  122.02ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=4, int=1>
 GPU activities:   73.28%  12.2815s       100  122.82ms  120.89ms  127.51ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=4, int=1>
 GPU activities:   73.33%  12.3535s       100  123.54ms  120.67ms  132.64ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=4, int=1>
 GPU activities:   73.36%  12.2082s       100  122.08ms  120.49ms  123.69ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=4, int=1>
 GPU activities:   73.46%  12.4032s       100  124.03ms  121.59ms  133.90ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=4, int=1>
 GPU activities:   73.47%  12.2121s       100  122.12ms  121.04ms  122.75ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=4, int=1>
 GPU activities:   73.70%  12.4846s       100  124.85ms  122.97ms  126.37ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=4, int=1>
 GPU activities:   73.73%  12.4622s       100  124.62ms  122.66ms  126.14ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=4, int=1>
 GPU activities:   73.85%  12.3949s       100  123.95ms  123.63ms  124.57ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=4, int=1>
 GPU activities:   73.87%  12.5176s       100  125.18ms  123.35ms  126.38ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=4, int=1>
 GPU activities:   73.93%  12.5591s       100  125.59ms  123.47ms  128.05ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=4, int=1>
 GPU activities:   73.99%  12.6450s       100  126.45ms  124.06ms  128.02ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=4, int=1>
 GPU activities:   74.02%  12.7055s       100  127.05ms  124.66ms  128.70ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=4, int=1>
 GPU activities:   74.08%  12.7549s       100  127.55ms  125.31ms  129.62ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=4, int=1>
 GPU activities:   74.20%  12.7895s       100  127.90ms  125.64ms  129.91ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=4, int=1>
 GPU activities:   74.31%  12.8138s       100  128.14ms  127.59ms  129.90ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=4, int=1>
 GPU activities:   74.46%  12.9944s       100  129.94ms  127.87ms  136.85ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=4, int=1>
 GPU activities:   74.50%  13.0025s       100  130.03ms  128.59ms  136.09ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=4, int=1>
 GPU activities:   74.61%  13.0143s       100  130.14ms  128.03ms  133.80ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=4, int=1>
 GPU activities:   74.68%  13.0564s       100  130.56ms  128.88ms  131.75ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=4, int=2>
 GPU activities:   74.86%  13.1872s       100  131.87ms  130.28ms  133.32ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=4, int=2>
 GPU activities:   74.88%  13.3035s       100  133.04ms  129.98ms  140.01ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=4, int=2>
 GPU activities:   74.89%  13.1498s       100  131.50ms  130.08ms  132.13ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=4, int=2>
 GPU activities:   74.92%  13.2143s       100  132.14ms  130.62ms  133.22ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=4, int=2>
 GPU activities:   75.45%  13.6269s       100  136.27ms  134.84ms  137.93ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=4, int=2>
 GPU activities:   75.53%  13.6532s       100  136.53ms  135.29ms  137.87ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=4, int=2>
 GPU activities:   75.54%  13.6647s       100  136.65ms  134.76ms  137.62ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=4, int=1>
 GPU activities:   75.54%  13.6697s       100  136.70ms  135.12ms  137.67ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=4, int=2>
 GPU activities:   75.55%  13.6738s       100  136.74ms  135.06ms  137.90ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=4, int=1>
 GPU activities:   75.55%  13.7043s       100  137.04ms  135.61ms  138.37ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=4, int=2>
 GPU activities:   75.55%  13.7792s       100  137.79ms  135.50ms  143.30ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=4, int=2>
 GPU activities:   75.56%  13.6672s       100  136.67ms  134.35ms  137.81ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=4, int=1>
 GPU activities:   75.56%  13.7146s       100  137.15ms  136.14ms  138.28ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=4, int=2>
 GPU activities:   75.68%  13.8203s       100  138.20ms  136.81ms  139.50ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=4, int=2>
 GPU activities:   75.69%  13.8516s       100  138.52ms  136.41ms  143.46ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=4, int=2>
 GPU activities:   75.70%  13.8971s       100  138.97ms  136.68ms  146.48ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=4, int=1>
 GPU activities:   75.72%  13.8007s       100  138.01ms  136.46ms  139.33ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=4, int=2>
 GPU activities:   75.72%  13.8015s       100  138.02ms  136.59ms  139.02ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=4, int=1>
 GPU activities:   75.74%  14.1753s       100  141.75ms  137.37ms  154.60ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=4, int=2>
 GPU activities:   75.75%  13.8732s       100  138.73ms  137.26ms  140.68ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=4, int=2>
 GPU activities:   76.06%  14.9921s       100  149.92ms  142.90ms  164.85ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=4, int=1>
 GPU activities:   76.15%  14.0768s       100  140.77ms  140.23ms  141.38ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=4, int=2>
 GPU activities:   76.36%  14.3577s       100  143.58ms  142.01ms  145.23ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=4, int=2>
 GPU activities:   77.11%  14.8806s       100  148.81ms  147.98ms  150.06ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=4, int=2>
 GPU activities:   77.15%  14.9708s       100  149.71ms  148.38ms  158.84ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=4, int=2>
 GPU activities:   77.16%  14.9525s       100  149.53ms  148.36ms  151.41ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=4, int=2>
 GPU activities:   77.27%  15.0353s       100  150.35ms  149.45ms  151.53ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=4, int=2>
 GPU activities:   77.31%  15.0507s       100  150.51ms  149.33ms  151.62ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=4, int=2>
 GPU activities:   77.32%  15.0314s       100  150.31ms  149.17ms  151.22ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=4, int=2>
 GPU activities:   77.35%  15.1182s       100  151.18ms  149.90ms  152.91ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=4, int=2>
 GPU activities:   77.38%  15.0858s       100  150.86ms  149.66ms  151.85ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=4, int=2>
 GPU activities:   77.66%  16.3730s       100  163.73ms  158.69ms  174.78ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=4, int=2>
 GPU activities:   78.00%  15.7697s       100  157.70ms  156.41ms  159.74ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=4, int=2>
 GPU activities:   80.36%  18.0880s       100  180.88ms  176.47ms  185.30ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=8, int=8, int=1>
 GPU activities:   80.54%  18.4175s       100  184.17ms  179.42ms  190.17ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=4, int=8, int=1>
 GPU activities:   80.59%  18.3855s       100  183.86ms  179.23ms  188.95ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=4, int=8, int=1>
 GPU activities:   80.61%  18.3722s       100  183.72ms  179.95ms  188.74ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=4, int=8, int=1>
 GPU activities:   80.88%  18.6778s       100  186.78ms  181.80ms  191.10ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=8, int=8, int=1>
 GPU activities:   81.01%  18.9526s       100  189.53ms  188.34ms  191.36ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=2, int=8, int=1>
 GPU activities:   81.02%  19.1940s       100  191.94ms  185.00ms  208.76ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=8, int=8, int=1>
 GPU activities:   81.09%  19.5333s       100  195.33ms  186.32ms  211.01ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=4, int=8, int=1>
 GPU activities:   81.20%  19.3283s       100  193.28ms  188.39ms  200.79ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=4, int=8, int=1>
 GPU activities:   81.26%  19.3682s       100  193.68ms  186.54ms  205.08ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=4, int=8, int=1>
 GPU activities:   81.34%  19.2998s       100  193.00ms  191.38ms  195.23ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=2, int=8, int=1>
 GPU activities:   81.41%  19.4177s       100  194.18ms  192.02ms  197.17ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=2, int=8, int=1>
 GPU activities:   81.44%  19.4455s       100  194.45ms  192.41ms  196.00ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=2, int=8, int=1>
 GPU activities:   81.54%  19.5550s       100  195.55ms  192.81ms  197.95ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=2, int=8, int=1>
 GPU activities:   81.55%  19.6005s       100  196.01ms  192.63ms  199.47ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=2, int=8, int=1>
 GPU activities:   81.60%  19.6812s       100  196.81ms  194.38ms  199.26ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=2, int=8, int=1>
 GPU activities:   81.99%  20.0445s       100  200.44ms  196.84ms  204.50ms  void dtopo_str_111_macro_unroll<int=16, int=1, int=2, int=8, int=1>
 GPU activities:   82.04%  20.2785s       100  202.79ms  196.68ms  209.02ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=2, int=8, int=1>
 GPU activities:   83.18%  22.9911s       100  229.91ms  218.74ms  250.45ms  void dtopo_str_111_macro_unroll<int=16, int=8, int=1, int=8, int=1>
 GPU activities:   83.45%  22.2105s       100  222.10ms  221.46ms  222.96ms  void dtopo_str_111_macro_unroll<int=32, int=8, int=1, int=8, int=1>
 GPU activities:   83.45%  22.3511s       100  223.51ms  222.75ms  224.46ms  void dtopo_str_111_macro_unroll<int=16, int=2, int=1, int=8, int=1>
 GPU activities:   83.47%  22.4166s       100  224.17ms  222.38ms  226.13ms  void dtopo_str_111_macro_unroll<int=16, int=4, int=1, int=8, int=1>
 GPU activities:   83.49%  22.3127s       100  223.13ms  222.29ms  224.05ms  void dtopo_str_111_macro_unroll<int=32, int=4, int=1, int=8, int=1>
 GPU activities:   83.53%  22.3844s       100  223.84ms  223.02ms  224.84ms  void dtopo_str_111_macro_unroll<int=64, int=4, int=1, int=8, int=1>
 GPU activities:   83.55%  22.5603s       100  225.60ms  223.14ms  238.60ms  void dtopo_str_111_macro_unroll<int=32, int=2, int=1, int=8, int=1>
 GPU activities:   83.59%  22.4884s       100  224.88ms  224.25ms  225.75ms  void dtopo_str_111_macro_unroll<int=64, int=2, int=1, int=8, int=1>
 GPU activities:   83.62%  22.6106s       100  226.11ms  225.33ms  227.01ms  void dtopo_str_111_macro_unroll<int=64, int=1, int=1, int=8, int=1>
 GPU activities:   83.70%  22.6656s       100  226.66ms  225.98ms  227.40ms  void dtopo_str_111_macro_unroll<int=32, int=1, int=1, int=8, int=1>
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



