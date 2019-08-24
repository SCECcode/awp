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
```c
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
Git commit: `a14a025b8b46b308bc3efba23ccab52f745fdd20` changes this to:
```c
  const int i = threadIdx.z + blockIdx.z * blockDim.z + bi;
  if (i >= nx)
    return;
  if (i >= ei)
    return;
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ny)
    return;
  if (j >= ej)
    return;
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= 6)
    return;
```
In addition to this change, the launch configuration needs to be adjusted for
all kernels. For example, the launch configuration for the interior kernel
`dtopo_vel_111` used to be:
```c
        dim3 block (TBX, TBY, TBZ);
        dim3 grid ((T->velocity_grid_interior.x+TBX-1)/TBX, 
                   (T->velocity_grid_interior.y+TBY-1)/TBY,
                   (T->velocity_grid_interior.z+TBZ-1)/TBZ);

```
We change it to:
```c
        dim3 block (VEL_INT_X, VEL_INT_Y, VEL_INT_Z);
        int grid_x = T->velocity_bounds_right[1] - T->velocity_bounds_left[0];
        int grid_y = T->velocity_bounds_back[0] - T->velocity_bounds_front[1];
        int grid_z = T->velocity_grid_interior.z;
        dim3 grid((grid_z + VEL_INT_X - 1) / VEL_INT_X,
                  (grid_y + VEL_INT_Y - 1) / VEL_INT_Y,
                  (grid_x + VEL_INT_Z - 1) / VEL_INT_Z);

```
and for the boundary kernel `dtopo_vel_112`, we change it to:
```c
        dim3 block (VEL_BND_X, VEL_BND_Y, VEL_BND_Z);
        int grid_x = T->velocity_bounds_right[1] - T->velocity_bounds_left[0];
        int grid_y = T->velocity_bounds_front[1] - T->velocity_bounds_front[0];
        int grid_z = T->velocity_grid_interior.z;
        dim3 grid((grid_z + VEL_BND_X - 1) / VEL_BND_X,
                  (grid_y + VEL_BND_Y - 1) / VEL_BND_Y,
                  (grid_x + VEL_BND_Z - 1) / VEL_BND_Z);
```
The number of blocks in the CUDA grid is now adjusted to match the size of each
compute region. Before, the same number of grid blocks were requested despite
having compute regions of different sizes.

After this optimization, nvprof reveals that the boundary kernel `dtopo_vel_112`
is no longer taking the longest execution time:

**Baseline**

```
            Type  Time      Time     Calls       Avg       Min       Max  Name
 GPU activities:   56.78%  9.41535s       300  31.384ms  26.998ms  42.701ms  dtopo_vel_112
                   40.55%  6.72360s       300  22.412ms  2.1199ms  81.573ms  dtopo_vel_111
```

**Index mapping optimization**
```
            Type  Time      Time     Calls       Avg       Min       Max  Name
 GPU activities:   78.64%  5.98085s       300  19.936ms  1.1826ms  74.440ms  dtopo_vel_111
                   11.84%  900.20ms       300  3.0006ms  380.75us  8.3974ms  dtopo_vel_112

```
The time taken by `dtopo_vel_112` has been reduced by a factor of 10!
