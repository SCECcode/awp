#include <topography/kernels/optimized.cuh>
#include <topography/kernels/optimized_launch_config.cuh>

__global__ void
dtopo_str_110(float *__restrict__ u1, float *__restrict__ u2,
              float *__restrict__ u3, const float *__restrict__ dcrjx,
              const float *__restrict__ dcrjy, const float *__restrict__ dcrjz,
              const float *__restrict__ f1_1, const float *__restrict__ f1_2,
              const float *__restrict__ f1_c, const float *__restrict__ f2_1,
              const float *__restrict__ f2_2, const float *__restrict__ f2_c,
              const float *__restrict__ f_1, const float *__restrict__ f_2,
              const float *__restrict__ f_c, const float *__restrict__ g,
              const float *__restrict__ g3, const float *__restrict__ g3_c,
              const float *__restrict__ g_c, const float *__restrict__ lami,
              const float *__restrict__ mui, const float *__restrict__ s11,
              const float *__restrict__ s12, const float *__restrict__ s13,
              const float *__restrict__ s22, const float *__restrict__ s23,
              const float *__restrict__ s33, const float a, const float nu,
              const int nx, const int ny, const int nz, const int bi,
              const int bj, const int ei, const int ej) {
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ngsl + ny)
    return;
  if (j >= ej)
    return;
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= 6)
    return;
  for (int i = bi; i < ei; ++i) {
    text
  }
}

__global__ void
dtopo_str_111(float *__restrict__ u1, float *__restrict__ u2,
              float *__restrict__ u3, const float *__restrict__ dcrjx,
              const float *__restrict__ dcrjy, const float *__restrict__ dcrjz,
              const float *__restrict__ f1_1, const float *__restrict__ f1_2,
              const float *__restrict__ f1_c, const float *__restrict__ f2_1,
              const float *__restrict__ f2_2, const float *__restrict__ f2_c,
              const float *__restrict__ f_1, const float *__restrict__ f_2,
              const float *__restrict__ f_c, const float *__restrict__ g,
              const float *__restrict__ g3, const float *__restrict__ g3_c,
              const float *__restrict__ g_c, const float *__restrict__ lami,
              const float *__restrict__ mui, const float *__restrict__ s11,
              const float *__restrict__ s12, const float *__restrict__ s13,
              const float *__restrict__ s22, const float *__restrict__ s23,
              const float *__restrict__ s33, const float a, const float nu,
              const int nx, const int ny, const int nz, const int bi,
              const int bj, const int ei, const int ej) {
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ngsl + ny)
    return;
  if (j >= ej)
    return;
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= nz - 12)
    return;
  for (int i = bi; i < ei; ++i) {
    text
  }
}

__global__ void
dtopo_str_112(float *__restrict__ u1, float *__restrict__ u2,
              float *__restrict__ u3, const float *__restrict__ dcrjx,
              const float *__restrict__ dcrjy, const float *__restrict__ dcrjz,
              const float *__restrict__ f1_1, const float *__restrict__ f1_2,
              const float *__restrict__ f1_c, const float *__restrict__ f2_1,
              const float *__restrict__ f2_2, const float *__restrict__ f2_c,
              const float *__restrict__ f_1, const float *__restrict__ f_2,
              const float *__restrict__ f_c, const float *__restrict__ g,
              const float *__restrict__ g3, const float *__restrict__ g3_c,
              const float *__restrict__ g_c, const float *__restrict__ lami,
              const float *__restrict__ mui, const float *__restrict__ s11,
              const float *__restrict__ s12, const float *__restrict__ s13,
              const float *__restrict__ s22, const float *__restrict__ s23,
              const float *__restrict__ s33, const float a, const float nu,
              const int nx, const int ny, const int nz, const int bi,
              const int bj, const int ei, const int ej) {
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ngsl + ny)
    return;
  if (j >= ej)
    return;
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= 6)
    return;
  for (int i = bi; i < ei; ++i) {
    text
  }
}

__global__ void dtopo_init_material_111(float *__restrict__ lami,
                                        float *__restrict__ mui,
                                        float *__restrict__ rho, const int nx,
                                        const int ny, const int nz) {
  const int i = threadIdx.z + blockIdx.z * blockDim.z;
  if (i >= nx)
    return;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (j >= ny)
    return;
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= nz)
    return;
#define _lami(i, j, k) lami[(i)*ny * nz + (j)*nz + (k)]
#define _mui(i, j, k) mui[(i)*ny * nz + (j)*nz + (k)]
#define _rho(i, j, k) rho[(i)*ny * nz + (j)*nz + (k)]
  _rho(i, j, k) = 1.0;
  _lami(i, j, k) = 1.0;
  _mui(i, j, k) = 1.0;
#undef _lami
#undef _mui
#undef _rho
}
