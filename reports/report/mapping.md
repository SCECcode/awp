# Mapping
As previously discussed, our first optimization is to fix the mapping of the
CUDA thread indices to the chosen memory layout. The current mapping is the
following:
```c
int i = threadIdx.x;
int j = threadIdx.y;
int k = threadIdx.z;
int pos = k + j * line + i * slice;
```
This mapping must be changed to:

```c
int i = threadIdx.z;
int j = threadIdx.y;
int k = threadIdx.x;
int pos = k + j * line + i * slice;
```

While these changes can be performed by hand, we will regenerate the kernels
with proper mappings using the kernel generator. If we take a look at
`dtopo_vel_112` this is what we currently have:
```C
  const int i = threadIdx.x + blockIdx.x * blockDim.x + bi;
  if (i >= nx)
    return;
  if (i >= ei)
    return;
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ny)
    return;
  if (j >= ej)
    return;
  const int k = threadIdx.z + blockIdx.z * blockDim.z;
  if (k >= 6)
    return;
```
Git commit #   changes this to:
```

```
