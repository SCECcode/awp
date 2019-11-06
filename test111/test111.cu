#include <stdio.h>
#include <curand.h>
#include <cuda_profiler_api.h>

//-----------------------------------------------------------------------------
// Configuration for Stress macro kernel (stress_macro.cu)

// Threads in x, y, z
#ifndef STRM_TX
#define STRM_TX 64
#endif

#ifndef STRM_TY
#define STRM_TY 8
#endif

#ifndef STRM_TZ
#define STRM_TZ 1
#endif

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Configuration for Stress macro unroll kernel (stress_macro_unroll.cu)

// Threads in x, y, z
#ifndef STRMU_TX
#define STRMU_TX 32
#endif

#ifndef STRMU_TY
#define STRMU_TY 1
#endif

#ifndef STRMU_TZ
#define STRMU_TZ 8
#endif

// Unroll factor in CUDA x
#ifndef STRMU_NA
#define STRMU_NA 1
#endif

// Unroll factor in CUDA y
#ifndef STRMU_NB
#define STRMU_NB 1
#endif

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Configuration for Stress index kernel (stress_index.cu)

// Threads in x, y, z
#ifndef STRI_TX
#define STRI_TX 64
#endif      
            
#ifndef STRI_TY
#define STRI_TY 8
#endif      
            
#ifndef STRI_TZ
#define STRI_TZ 1
#endif

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Configuration for Stress index with loop unrolling kernel
// (stress_index_unroll.cu)

// Threads in x, y, z
#ifndef STRIU_TX
#define STRIU_TX 32
#endif      
            
#ifndef STRIU_TY
#define STRIU_TY 1
#endif      
            
#ifndef STRIU_TZ
#define STRIU_TZ 4
#endif

// Unroll factor in CUDA x
#ifndef STRIU_NA
#define STRIU_NA 1
#endif

// Unroll factor in CUDA y
#ifndef STRIU_NB
#define STRIU_NB 2
#endif

//-----------------------------------------------------------------------------


// Enable / Disable correctness test
#define TEST 1

//-----------------------------------------------------------------------------
// Velocity kernel optimizations to choose from
#ifndef USE_ORIGINAL_VEL
#define USE_ORIGINAL_VEL 0
#endif
#ifndef USE_SHARED_VEL
#define USE_SHARED_VEL 0
#endif
#ifndef USE_DM_VEL
#define USE_DM_VEL 0
#endif
#ifndef USE_SPLIT_VEL
#define USE_SPLIT_VEL 0
#endif
#ifndef USE_UNROLL_VEL
#define USE_UNROLL_VEL 0
#endif
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Stress kernel optimizations to choose from
#ifndef USE_STRESS_ORIGINAL
#define USE_STRESS_ORIGINAL 1
#endif

#ifndef USE_STRESS_MACRO
#define USE_STRESS_MACRO 1
#endif

#ifndef USE_STRESS_MACRO_UNROLL
#define USE_STRESS_MACRO_UNROLL 0
#endif

#ifndef USE_STRESS_INDEX
#define USE_STRESS_INDEX 0
#endif

#ifndef USE_STRESS_INDEX_UNROLL
#define USE_STRESS_INDEX_UNROLL 0
#endif
//-----------------------------------------------------------------------------

#define align 32
#define ngsl 4
#define ngsl2 8

#define RADIUSZ 3

__device__ int err;
__device__ int nan_err;
#define PRINTERR 0

// Turning __restrict__ on or off...
#define RSTRCT __restrict__
 
#ifndef CUCHK
#ifndef NDEBUG
#define CUCHK(call) {                                                         \
  cudaError_t err = call;                                                     \
  if( cudaSuccess != err) {                                                   \
  fprintf(stderr, "CUDA error in %s:%i %s(): %s.\n",                          \
          __FILE__, __LINE__, __func__, cudaGetErrorString(err) );            \
  fflush(stderr);                                                             \
  exit(EXIT_FAILURE);                                                         \
  }                                                                           \
}                                                                             
#else
#define CUCHK(call) {}
#endif
#endif

// *****************************************************************************
// *****************************************************************************
// *****************************************************************************

#define VEL_111_TILE_Z 32
#define VEL_111_TILE_Y 8

__launch_bounds__ (VEL_111_TILE_Z*VEL_111_TILE_Y)
__global__ void dtopo_vel_111_opt (float *__restrict__ u1, float *__restrict__ u2, float *__restrict__ u3,
                                   const float *__restrict__ dcrjx, const float *__restrict__ dcrjy,
                                   const float *__restrict__ dcrjz, const float *__restrict__ f,
                                   const float *__restrict__ f1_1, const float *__restrict__ f1_2,
                                   const float *__restrict__ f1_c, const float *__restrict__ f2_1,
                                   const float *__restrict__ f2_2, const float *__restrict__ f2_c,
                                   const float *__restrict__ f_1, const float *__restrict__ f_2,
                                   const float *__restrict__ f_c, const float *__restrict__ g,
                                   const float *__restrict__ g3, const float *__restrict__ g3_c,
                                   const float *__restrict__ g_c, const float *__restrict__ rho,
                                   const float *__restrict__ s11, const float *__restrict__ s12,
                                   const float *__restrict__ s13, const float *__restrict__ s22,
                                   const float *__restrict__ s23, const float *__restrict__ s33,
                                   const float a, const float nu, const int nx, const int ny, const int nz,
                                   const int bi, const int bj, const int ei, const int ej) {

  const float phz2[2] = {0.5000000000000000, 0.5000000000000000};
  const float phy2[2] = {0.5000000000000000, 0.5000000000000000};
  const float phx2[2] = {0.5000000000000000, 0.5000000000000000};
  const float dhpz4[7] = {-0.0026041666666667, 0.0937500000000000, -0.6796875000000000, 0.0000000000000000,
                          0.6796875000000000,  -0.0937500000000000, 0.0026041666666667};
  const float phx4[4] = {-0.0625000000000000, 0.5625000000000000, 0.5625000000000000, -0.0625000000000000};
  const float phy4[4] = {-0.0625000000000000, 0.5625000000000000, 0.5625000000000000, -0.0625000000000000};
  const float dhy4[4] = {0.0416666666666667, -1.1250000000000000, 1.1250000000000000, -0.0416666666666667};
  const float dhx4[4] = {0.0416666666666667, -1.1250000000000000, 1.1250000000000000, -0.0416666666666667};
  const float dhz4[4] = {0.0416666666666667, -1.1250000000000000, 1.1250000000000000, -0.0416666666666667};
  const float px4[4] = {-0.0625000000000000, 0.5625000000000000, 0.5625000000000000, -0.0625000000000000};
  const float py4[4] = {-0.0625000000000000, 0.5625000000000000, 0.5625000000000000, -0.0625000000000000};
  const float dx4[4] = {0.0416666666666667, -1.1250000000000000, 1.1250000000000000, -0.0416666666666667};
  const float dy4[4] = {0.0416666666666667, -1.1250000000000000, 1.1250000000000000, -0.0416666666666667};
  const float dphz4[7] = {-0.0026041666666667, 0.0937500000000000, -0.6796875000000000, 0.0000000000000000,
                          0.6796875000000000,  -0.0937500000000000, 0.0026041666666667};
  const float dz4[4] = {0.0416666666666667, -1.1250000000000000, 1.1250000000000000, -0.0416666666666667};

#define TILE_Z VEL_111_TILE_Z
#define TILE_Y VEL_111_TILE_Y
  
#define _rho(i, j, k) rho[((i + ngsl + 2) * (2 * ngsl + ny + 4) + j + ngsl + 2) * (2 * align + nz) + k + align]
#define _s11(i, j, k) s11[((i + ngsl + 2) * (2 * ngsl + ny + 4) + j + ngsl + 2) * (2 * align + nz) + k + align]
#define _s12(i, j, k) s12[((i + ngsl + 2) * (2 * ngsl + ny + 4) + j + ngsl + 2) * (2 * align + nz) + k + align]
#define _s13(i, j, k) s13[((i + ngsl + 2) * (2 * ngsl + ny + 4) + j + ngsl + 2) * (2 * align + nz) + k + align]
#define _s22(i, j, k) s22[((i + ngsl + 2) * (2 * ngsl + ny + 4) + j + ngsl + 2) * (2 * align + nz) + k + align]
#define _s23(i, j, k) s23[((i + ngsl + 2) * (2 * ngsl + ny + 4) + j + ngsl + 2) * (2 * align + nz) + k + align]
#define _s33(i, j, k) s33[((i + ngsl + 2) * (2 * ngsl + ny + 4) + j + ngsl + 2) * (2 * align + nz) + k + align]
#define  _u1(i, j, k)  u1[((i + ngsl + 2) * (2 * ngsl + ny + 4) + j + ngsl + 2) * (2 * align + nz) + k + align]
#define  _u2(i, j, k)  u2[((i + ngsl + 2) * (2 * ngsl + ny + 4) + j + ngsl + 2) * (2 * align + nz) + k + align]
#define  _u3(i, j, k)  u3[((i + ngsl + 2) * (2 * ngsl + ny + 4) + j + ngsl + 2) * (2 * align + nz) + k + align]

#define   _f(i, j)   f[(i + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + j + align + ngsl + 2]
#define _f_1(i, j) f_1[(i + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + j + align + ngsl + 2]
#define _f_2(i, j) f_2[(i + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + j + align + ngsl + 2]
#define _f_c(i, j) f_c[(i + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + j + align + ngsl + 2]

#define _f1_1(i, j) f1_1[(i + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + j + align + ngsl + 2]
#define _f2_1(i, j) f2_1[(i + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + j + align + ngsl + 2]

#define _f1_2(i, j) f1_2[(i + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + j + align + ngsl + 2]
#define _f2_2(i, j) f2_2[(i + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + j + align + ngsl + 2]

#define _f1_c(i, j) f1_c[(i + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + j + align + ngsl + 2]
#define _f2_c(i, j) f2_c[(i + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + j + align + ngsl + 2]

#define    _g(k)    g[k + align]
#define  _g_c(k)  g_c[k + align]
#define   _g3(k)   g3[k + align]
#define _g3_c(k) g3_c[k + align]

#define _dcrjy(j) dcrjy[j + ngsl + 2]
#define _dcrjx(i) dcrjx[i + ngsl + 2]
#define _dcrjz(k) dcrjz[k + align]

  // Mapping the Z dimension on threadIdx.x. NX is the slowest-varying dimension
  int tz = threadIdx.x;
  int ty = threadIdx.y;
  int k = blockIdx.x * TILE_Z + tz + 6;
  int j = blockIdx.y * TILE_Y + ty + bj;

  // Only active threads will write data
  int active = k < nz && j < ej;
  
  // Shared memory arrays needed for :
  // S11(i-2:i+1, j,       k-3:k+3) : 3D array with 4 plans in X
  // S12(i-1:i+2, j-2:j+1, k-3:k+3) : 3D array with 4 plans in X
  // S13(i-1,i+2, j,       k-3:k+3) : 3D array with 4 plans in X
  // S22(i      , j-1:j+2, k-3:k+3) : 2D array
  // S23(i      , j-2:j+1, k-3:k+3) : 2D array
  // S33(i      , j,       k-3:k+3) : 2D array
  // RHO(i-1:i  , j-1:j,   k-1:k  ) : 3D array with 2 plans in X
  
  __shared__ float s_s11 [4][TILE_Y  ][TILE_Z+6];
  __shared__ float s_s12 [4][TILE_Y+3][TILE_Z+6];
  __shared__ float s_s13 [4][TILE_Y  ][TILE_Z+6];
  __shared__ float s_s22    [TILE_Y+3][TILE_Z+6];
  __shared__ float s_s23    [TILE_Y+3][TILE_Z+6];
  __shared__ float s_s33    [TILE_Y  ][TILE_Z+6];
  __shared__ float s_rho [2][TILE_Y+1][TILE_Z+1];

  float g3_z = _g3(k);
  float g3c_z = _g3_c(k);

  float gc_zm3 = _g_c(k-3);
  float gc_zm2 = _g_c(k-2);
  float gc_zm1 = _g_c(k-1);
  float gc_z   = _g_c(k  );
  float gc_zp1 = _g_c(k+1);
  float gc_zp2 = _g_c(k+2);
  float gc_zp3 = _g_c(k+3);

  float g_zm3 = _g(k-3);
  float g_zm2 = _g(k-2);
  float g_zm1 = _g(k-1);
  float g_z   = _g(k  );
  float g_zp1 = _g(k+1);
  float g_zp2 = _g(k+2);
  float g_zp3 = _g(k+3);

  // Prime the first plans of shared memory for the 3D shared memory arrays
  // S11(i-2:i+1, j,       k-3:k+3) : Loading at (i+1, j,       k-3:k+3) = 2 regions
  // S12(i-1:i+2, j-2:j+1, k-3:k+3) : Loading at (i+2, j-2:j+1, k-3:k+3) = 4 regions
  // S13(i-1,i+2, j,       k-3:k+3) : Loading at (i+2, j,       k-3:k+3) = 2 regions
  // RHO(i-1:i  , j-1:j,   k-1:k  ) : Loading at (i,   j-1:j,   k-1:k  ) = 4 regions
  for (int i=0; i<3; i++) {
    // Region 1 = low Y(0:7), low Z (0:31)
    s_s11[i][ty][tz] = _s11(bi+i-2, j,   k-3);
    s_s12[i][ty][tz] = _s12(bi+i-1, j-2, k-3);
    s_s13[i][ty][tz] = _s13(bi+i-1, j,   k-);
    // Region 2 = low Y(0:7) high Z (32:37)
    if (tz < 6) {
      s_s11[i][ty][tz+TILE_Z] = _s11(bi+i-2, j,   k-3+TILE_Z);
      s_s12[i][ty][tz+TILE_Z] = _s12(bi+i-1, j-2, k-3+TILE_Z);
      s_s13[i][ty][tz+TILE_Z] = _s13(bi+i-1, j,   k-3+TILE_Z);
    }
    // Region 3 = high Y(8:11) low Z(0:31)
    if (ty < 4)
      s_s12[i][ty+TILE_Y][tz] = _s12(bi+i-1, j-2+TILE_Y, k-3);
    // Region 4 = high Y(8:11) high Z(32:37)
    if (ty < 4 && tz < 6)
      s_s12[i][ty+TILE_Y][tz+TILE_Z] = _s12(bi+i-1, j-2+TILE_Y, k-3+TILE_Z);
  }
  // Loading density, at plan i=-1
  s_rho[0][ty][tz] = _rho(bi-1, j-1, k-1);
  if (tz == 0)
    s_rho[0][ty][TILE_Z+tz] = _rho(bi-1, j-1, k-1+TILE_Z);
  if (ty == 0)
    s_rho[0][ty+TILE_Y][tz] = _rho(bi-1, j-1+TILE_Y, k-1);
  if (ty == 0 && tz == 0 )
    s_rho[0][ty+TILE_Y][tz+TILE_Z] = _rho(bi-1, j-1+TILE_Y, k-1+TILE_Z);

  // Indices to rotate on the 4 plans of shared memory for S11, S12, S13, and 2 plans for rho
  int p4x0=0, p4x1=1, p4x2=2, p4x3=3;
  int p2x0=0, p2x1=1;

  // Register queues for f, f_1, f_c
  float f_im1, f_ij, f_ip1, f_ip2;
  f_im1 = _f(bi-1, j);
  f_ij  = _f(bi,   j);
  f_ip1 = _f(bi+1, j);
  float f_1_im1, f_1_ij, f_1_ip1, f_1_ip2;
  f_1_im1 = _f_1(bi-1, j);
  f_1_ij  = _f_1(bi,   j);
  f_1_ip1 = _f_1(bi+1, j);
  float f_c_im2, f_c_im1, f_c_ij, f_c_ip1;
  f_c_im2 = _f_c(bi-2, j);
  f_c_im1 = _f_c(bi-1, j);
  f_c_ij  = _f_c(bi,   j);

  float dcrjy_y = _dcrjy(j);
  float dcrjz_z = _dcrjz(k);

  // ******************************************************************************
  // Loop on the X dimension from bi to ei
  for (int i=bi; i<ei; i++) {
    
    __syncthreads();
    // Load new values for this IX
    float u1val = _u1(i, j, k);
    float u2val = _u2(i, j, k);
    float u3val = _u3(i, j, k);

    f_ip2   = _f  (i+2, j);
    f_1_ip2 = _f_1(i+2, j);
    f_c_ip1 = _f_c(i+1, j);
    
    float f_2_jm2, f_2_jm1, f_2_ij, f_2_jp1;
    f_2_jm2 = _f_2(i, j-2);
    f_2_jm1 = _f_2(i, j-1);
    f_2_ij  = _f_2(i, j  );
    f_2_jp1 = _f_2(i, j+1);

    float f_jm2, f_jm1, f_jp1;
    f_jm2 = _f(i, j-2);
    f_jm1 = _f(i, j-1);
    f_jp1 = _f(i, j+1);

    float f_c_jm1, f_c_jp1, f_c_jp2;
    f_c_jm1 = _f_c(i, j-1);
    f_c_jp1 = _f_c(i, j+1);
    f_c_jp2 = _f_c(i, j+2);

    // ******************************************************************************
    // Load new stress and density in shared memory

    // S11(i-2:i+1, j,       k-3:k+3) : Loading at (i+1, j,       k-3:k+3) = 2 regions
    // S12(i-1:i+2, j-2:j+1, k-3:k+3) : Loading at (i+2, j-2:j+1, k-3:k+3) = 4 regions
    // S13(i-1,i+2, j,       k-3:k+3) : Loading at (i+2, j,       k-3:k+3) = 2 regions
    // S22(i      , j-1:j+2, k-3:k+3) : Loading at (i,   j-1:j+2, k-3:k+3) = 4 regions
    // S23(i      , j-2:j+1, k-3:k+3) : Loading at (i,   j-2:j+1, k-3:k+3) = 4 regions
    // S33(i      , j,       k-3:k+3) : Loading at (i,   j,       k-3:k+3) = 2 regions
    // RHO(i-1:i  , j-1:j,   k-1:k  ) : Loading at (i,   j-1:j,   k-1:k  ) = 4 regions

    // Only testing boundary counditions for arrays accesses at i+2

    // Region 1 = low Y(0:7), low Z (0:31)
    s_s11[p4x3][ty][tz] = _s11(i+1, j,   k-3);
    if (j - 2 < ny + ngsl + 2)
      s_s12[p4x3][ty][tz] = _s12(i+2, j-2, k-3);
    if (j < ny + ngsl + 2)
      s_s13[p4x3][ty][tz] = _s13(i+2, j,   k-3);
    s_s22      [ty][tz] = _s22(i,   j-1, k-3);
    s_s23      [ty][tz] = _s23(i,   j-2, k-3);
    s_s33      [ty][tz] = _s33(i,   j-1, k-3);;
    s_rho[p2x1][ty][tz] = _rho(i,   j-1, k-1);
    // Region 2 = low Y(0:7) high Z (32:37)
    if (tz < 6) {
      s_s11[p4x3][ty][TILE_Z+tz] = _s11(i+1, j,   k-3+TILE_Z);
      if (j - 2 < ny + ngsl + 2 && k - 3 + TILE_Z < nz)
        s_s12[p4x3][ty][TILE_Z+tz] = _s12(i+2, j-2, k-3+TILE_Z);
      if (j < ny + ngsl + 2 && k - 3 + TILE_Z < nz)
        s_s13[p4x3][ty][TILE_Z+tz] = _s13(i+2, j,   k-3+TILE_Z);
      s_s22      [ty][TILE_Z+tz] = _s22(i,   j-1, k-3+TILE_Z);
      s_s23      [ty][TILE_Z+tz] = _s23(i,   j-2, k-3+TILE_Z);
      s_s33      [ty][TILE_Z+tz] = _s33(i,   j-1, k-3+TILE_Z);
    }
    if (tz == 0)
      s_rho[p2x1][ty][TILE_Z+tz] = _rho(i, j-1, k-1+TILE_Z);
    // Region 3 = high Y(8:11) low Z(0:31)
    if (ty < 4) {
      if (j - 2 + TILE_Y < ny + ngsl + 2 && k - 3 < nz)
        s_s12[p4x3][TILE_Y+ty][tz] = _s12(i+2, j-2+TILE_Y, k-3);
      s_s22      [TILE_Y+ty][tz] = _s22(i  , j-1+TILE_Y, k-3);
      s_s23      [TILE_Y+ty][tz] = _s23(i  , j-2+TILE_Y, k-3);
    }
    if (ty == 0)
      s_rho[p2x1][TILE_Y+ty][tz] = _rho(i, j-1+TILE_Y, k-1);

    // Region 4 = high Y(8:11) high Z(32:37)
    if (ty < 4 && tz < 6) {
      if (j - 2 + TILE_Y < ny + ngsl + 2 && k - 3 + TILE_Z < nz)
        s_s12[p4x3][TILE_Y+ty][TILE_Z+tz] = _s12(i+2, j-2+TILE_Y, k-3+TILE_Z);
      s_s22      [TILE_Y+ty][TILE_Z+tz] = _s22(i  , j-1+TILE_Y, k-3+TILE_Z);
      s_s23      [TILE_Y+ty][TILE_Z+tz] = _s23(i  , j-2+TILE_Y, k-3+TILE_Z);
    }
    if (ty == 0 && tz == 0 )
      s_rho[p2x1][TILE_Y+ty][TILE_Z+tz] = _rho(i, j-1+TILE_Y, k-1+TILE_Z);

    __syncthreads();

    // ******************************************************************************

    float rho1 =
             phz2[0] * (phy2[1] * s_rho[p2x1][ty+1][tz  ] + phy2[0] * s_rho[p2x1][ty][tz  ]) +
             phz2[1] * (phy2[1] * s_rho[p2x1][ty+1][tz+1] + phy2[0] * s_rho[p2x1][ty][tz+1]);
    float rho2 =
             phz2[0] * (phx2[1] * s_rho[p2x1][ty+1][tz  ] + phx2[0] * s_rho[p2x0][ty+1][tz  ]) +
             phz2[1] * (phx2[1] * s_rho[p2x1][ty+1][tz+1] + phx2[0] * s_rho[p2x0][ty+1][tz+1]);
    float rho3 =
             phy2[1] * (phx2[1] * s_rho[p2x1][ty+1][tz+1] + phx2[0] * s_rho[p2x0][ty+1][tz+1]) +
             phy2[0] * (phx2[1] * s_rho[p2x1][ty  ][tz+1] + phx2[0] * s_rho[p2x0][ty  ][tz+1]);
    float Ai1 = nu / (f_1_ij * g3c_z * rho1);
    float Ai2 = nu / (f_2_ij * g3c_z * rho2);
    float Ai3 = nu / (f_c_ij * g3_z  * rho3);

    float f_dcrj = _dcrjx(i) * dcrjy_y * dcrjz_z;

    float f1_1_ij, f2_1_ij;
    if (active) {
      f1_1_ij = _f1_1 (i, j);
      f2_1_ij = _f2_1 (i, j);
    }
    
    u1val = (a * u1val + Ai1 * (dhx4[0] * f_c_im2 * g3c_z * s_s11[p4x0][ty][tz+3] +
                                dhx4[1] * f_c_im1 * g3c_z * s_s11[p4x1][ty][tz+3] +
                                dhx4[2] * f_c_ij  * g3c_z * s_s11[p4x2][ty][tz+3]+
                                dhx4[3] * f_c_ip1 * g3c_z * s_s11[p4x3][ty][tz+3] +
                                dhy4[0] * f_jm2   * g3c_z * s_s12[p4x1][ty  ][tz] +
                                dhy4[1] * f_jm1   * g3c_z * s_s12[p4x1][ty+1][tz] +
                                dhy4[2] * f_ij    * g3c_z * s_s12[p4x1][ty+2][tz] +
                                dhy4[3] * f_jp1   * g3c_z * s_s12[p4x1][ty+3][tz] +
                                dhz4[0] * s_s13[p4x1][ty][tz+1] + dhz4[1] * s_s13[p4x1][ty][tz+2] +
                                dhz4[2] * s_s13[p4x1][ty][tz+3] + dhz4[3] * s_s13[p4x1][ty][tz+4] -
                                f1_1_ij * (dhpz4[0] * gc_zm3 * (phx4[0] * s_s11[p4x0][ty][tz  ] +
                                                               phx4[1] * s_s11[p4x1][ty][tz  ] +
                                                               phx4[2] * s_s11[p4x2][ty][tz  ] +
                                                               phx4[3] * s_s11[p4x3][ty][tz  ]) +
                                          dhpz4[1] * gc_zm2 * (phx4[0] * s_s11[p4x0][ty][tz+1] +
                                                               phx4[1] * s_s11[p4x1][ty][tz+1] +
                                                               phx4[2] * s_s11[p4x2][ty][tz+1] +
                                                               phx4[3] * s_s11[p4x3][ty][tz+1]) +
                                          dhpz4[2] * gc_zm1 * (phx4[0] * s_s11[p4x0][ty][tz+2] +
                                                               phx4[1] * s_s11[p4x1][ty][tz+2] +
                                                               phx4[2] * s_s11[p4x2][ty][tz+2] +
                                                               phx4[3] * s_s11[p4x3][ty][tz+2]) +
                                          dhpz4[3] * gc_z   * (phx4[0] * s_s11[p4x0][ty][tz+3] +
                                                               phx4[1] * s_s11[p4x1][ty][tz+3] +
                                                               phx4[2] * s_s11[p4x2][ty][tz+3] +
                                                               phx4[3] * s_s11[p4x3][ty][tz+3]) +
                                          dhpz4[4] * gc_zp1 * (phx4[0] * s_s11[p4x0][ty][tz+4] +
                                                               phx4[1] * s_s11[p4x1][ty][tz+4] +
                                                               phx4[2] * s_s11[p4x2][ty][tz+4] +
                                                               phx4[3] * s_s11[p4x3][ty][tz+4]) +
                                          dhpz4[5] * gc_zp2 * (phx4[0] * s_s11[p4x0][ty][tz+5] +
                                                               phx4[1] * s_s11[p4x1][ty][tz+5] +
                                                               phx4[2] * s_s11[p4x2][ty][tz+5] +
                                                               phx4[3] * s_s11[p4x3][ty][tz+5]) +
                                          dhpz4[6] * gc_zp3 * (phx4[0] * s_s11[p4x0][ty][tz+6] +
                                                               phx4[1] * s_s11[p4x1][ty][tz+6] +
                                                               phx4[2] * s_s11[p4x2][ty][tz+6] +
                                                               phx4[3] * s_s11[p4x3][ty][tz+6])) -
                                f2_1_ij * (dhpz4[0] * gc_zm3 * (phy4[0] * s_s12[p4x1][ty  ][tz] +
                                                                phy4[1] * s_s12[p4x1][ty+1][tz] +
                                                                phy4[2] * s_s12[p4x1][ty+2][tz] +
                                                                phy4[3] * s_s12[p4x1][ty+3][tz]) +
                                           dhpz4[1] * gc_zm2 * (phy4[0] * s_s12[p4x1][ty  ][tz] +
                                                                phy4[1] * s_s12[p4x1][ty+1][tz] +
                                                                phy4[2] * s_s12[p4x1][ty+2][tz] +
                                                                phy4[3] * s_s12[p4x1][ty+3][tz]) +
                                           dhpz4[2] * gc_zm1 * (phy4[0] * s_s12[p4x1][ty  ][tz] +
                                                                phy4[1] * s_s12[p4x1][ty+1][tz] +
                                                                phy4[2] * s_s12[p4x1][ty+2][tz] +
                                                                phy4[3] * s_s12[p4x1][ty+3][tz]) +
                                           dhpz4[3] * gc_z   * (phy4[0] * s_s12[p4x1][ty  ][tz] +
                                                                phy4[1] * s_s12[p4x1][ty+1][tz] +
                                                                phy4[2] * s_s12[p4x1][ty+2][tz] +
                                                                phy4[3] * s_s12[p4x1][ty+3][tz]) +
                                           dhpz4[4] * gc_zp1 * (phy4[0] * s_s12[p4x1][ty  ][tz] +
                                                                phy4[1] * s_s12[p4x1][ty+1][tz] +
                                                                phy4[2] * s_s12[p4x1][ty+2][tz] +
                                                                phy4[3] * s_s12[p4x1][ty+3][tz]) +
                                           dhpz4[5] * gc_zp2 * (phy4[0] * s_s12[p4x1][ty  ][tz] +
                                                                phy4[1] * s_s12[p4x1][ty+1][tz] +
                                                                phy4[2] * s_s12[p4x1][ty+2][tz] +
                                                                phy4[3] * s_s12[p4x1][ty+3][tz]) +
                                           dhpz4[6] * gc_zp3 * (phy4[0] * s_s12[p4x1][ty  ][tz] +
                                                                phy4[1] * s_s12[p4x1][ty+1][tz] +
                                                                phy4[2] * s_s12[p4x1][ty+2][tz] +
                                                                phy4[3] * s_s12[p4x1][ty+3][tz])))) * f_dcrj;

    float f1_2_ij, f2_2_ij;
    if (active) {
      f1_2_ij = _f1_2 (i, j);
      f2_2_ij = _f2_2 (i, j);
    }
   
    u2val = (a * u2val + Ai2 * (dhz4[0] * s_s23[ty+2][tz+1] + dhz4[1] * s_s23[ty+2][tz+2] +
                                dhz4[2] * s_s23[ty+2][tz+3] + dhz4[3] * s_s23[ty+2][tz+4] +
                                dx4[0] * f_im1 * g3c_z * s_s12[p4x0][ty+2][tz+3] +
                                dx4[1] * f_ij  * g3c_z * s_s12[p4x1][ty+2][tz+3] +
                                dx4[2] * f_ip1 * g3c_z * s_s12[p4x2][ty+2][tz+3] +
                                dx4[3] * f_ip2 * g3c_z * s_s12[p4x3][ty+2][tz+3] +
                                dy4[0] * f_c_jm1 * g3c_z * s_s22[ty  ][tz+3] +
                                dy4[1] * f_c_ij  * g3c_z * s_s22[ty+1][tz+3] +
                                dy4[2] * f_c_jp1 * g3c_z * s_s22[ty+2][tz+3] +
                                dy4[3] * f_c_jp2 * g3c_z * s_s22[ty+3][tz+3] -
                                f1_2_ij * (dhpz4[0] * gc_zm3 * (px4[0] * s_s12[p4x0][ty+2][tz  ] +
                                                                    px4[1] * s_s12[p4x1][ty+2][tz  ] +
                                                                    px4[2] * s_s12[p4x2][ty+2][tz  ] +
                                                                    px4[3] * s_s12[p4x3][ty+2][tz  ]) +
                                               dhpz4[1] * gc_zm2 * (px4[0] * s_s12[p4x0][ty+2][tz+1] +
                                                                    px4[1] * s_s12[p4x1][ty+2][tz+1] +
                                                                    px4[2] * s_s12[p4x2][ty+2][tz+1] +
                                                                    px4[3] * s_s12[p4x3][ty+2][tz+1]) +
                                               dhpz4[2] * gc_zm1 * (px4[0] * s_s12[p4x0][ty+2][tz+2] +
                                                                    px4[1] * s_s12[p4x1][ty+2][tz+2] +
                                                                    px4[2] * s_s12[p4x2][ty+2][tz+2] +
                                                                    px4[3] * s_s12[p4x3][ty+2][tz+2]) +
                                               dhpz4[3] * gc_z   * (px4[0] * s_s12[p4x0][ty+2][tz+3] +
                                                                    px4[1] * s_s12[p4x1][ty+2][tz+3] +
                                                                    px4[2] * s_s12[p4x2][ty+2][tz+3] +
                                                                    px4[3] * s_s12[p4x3][ty+2][tz+3]) +
                                               dhpz4[4] * gc_zp1 * (px4[0] * s_s12[p4x0][ty+2][tz+4] +
                                                                    px4[1] * s_s12[p4x1][ty+2][tz+4] +
                                                                    px4[2] * s_s12[p4x2][ty+2][tz+4] +
                                                                    px4[3] * s_s12[p4x3][ty+2][tz+4]) +
                                               dhpz4[5] * gc_zp2 * (px4[0] * s_s12[p4x0][ty+2][tz+5] +
                                                                    px4[1] * s_s12[p4x1][ty+2][tz+5] +
                                                                    px4[2] * s_s12[p4x2][ty+2][tz+5] +
                                                                    px4[3] * s_s12[p4x3][ty+2][tz+5]) +
                                               dhpz4[6] * gc_zp3 * (px4[0] * s_s12[p4x0][ty+2][tz+6] +
                                                                    px4[1] * s_s12[p4x1][ty+2][tz+6] +
                                                                    px4[2] * s_s12[p4x2][ty+2][tz+6] +
                                                                    px4[3] * s_s12[p4x3][ty+2][tz+6])) -
                                f2_2_ij * (dhpz4[0] * gc_zm3 * (py4[1] * s_s22[ty+1][tz  ] +
                                                                    py4[0] * s_s22[ty  ][tz  ] +
                                                                    py4[2] * s_s22[ty+2][tz  ] +
                                                                    py4[3] * s_s22[ty+3][tz  ]) +
                                               dhpz4[1] * gc_zm2 * (py4[1] * s_s22[ty+1][tz+1] +
                                                                    py4[0] * s_s22[ty  ][tz+1] +
                                                                    py4[2] * s_s22[ty+2][tz+1] +
                                                                    py4[3] * s_s22[ty+3][tz+1]) +
                                               dhpz4[2] * gc_zm1 * (py4[1] * s_s22[ty+1][tz+2] +
                                                                    py4[0] * s_s22[ty  ][tz+2] +
                                                                    py4[2] * s_s22[ty+2][tz+2] +
                                                                    py4[3] * s_s22[ty+3][tz+2]) +
                                               dhpz4[3] * gc_z   * (py4[1] * s_s22[ty+1][tz+3] +
                                                                    py4[0] * s_s22[ty  ][tz+3] +
                                                                    py4[2] * s_s22[ty+2][tz+3] +
                                                                    py4[3] * s_s22[ty+3][tz+3]) +
                                               dhpz4[4] * gc_zp1 * (py4[1] * s_s22[ty+1][tz+4] +
                                                                    py4[0] * s_s22[ty  ][tz+4] +
                                                                    py4[2] * s_s22[ty+2][tz+4] +
                                                                    py4[3] * s_s22[ty+3][tz+4]) +
                                               dhpz4[5] * gc_zp2 * (py4[1] * s_s22[ty+1][tz+5] +
                                                                    py4[0] * s_s22[ty  ][tz+5] +
                                                                    py4[2] * s_s22[ty+2][tz+5] +
                                                                    py4[3] * s_s22[ty+3][tz+5]) +
                                               dhpz4[6] * gc_zp3 * (py4[1] * s_s22[ty+1][tz+6] +
                                                                    py4[0] * s_s22[ty  ][tz+6] +
                                                                    py4[2] * s_s22[ty+2][tz+6] +
                                                                    py4[3] * s_s22[ty+3][tz+6])))) * f_dcrj;
    float f1_c_ij, f2_c_ij;
    if (active) {
      f1_c_ij = _f1_c (i, j);
      f2_c_ij = _f2_c (i, j);
    }
    u3val = (a * u3val + Ai3 * (dhy4[0] * f_2_jm2 * g3_z * s_s23[ty  ][tz+3] +
                                dhy4[1] * f_2_jm1 * g3_z * s_s23[ty+1][tz+3] +
                                dhy4[2] * f_2_ij  * g3_z * s_s23[ty+2][tz+3] +
                                dhy4[3] * f_2_jp1 * g3_z * s_s23[ty+3][tz+3] +
                                dx4[0] * f_1_im1 * g3_z * s_s13[p4x0][ty][tz+3] +
                                dx4[1] * f_1_ij  * g3_z * s_s13[p4x1][ty][tz+3] +
                                dx4[2] * f_1_ip1 * g3_z * s_s13[p4x2][ty][tz+3] +
                                dx4[3] * f_1_ip2 * g3_z * s_s13[p4x3][ty][tz+3] +
                                dz4[0] * s_s33[ty][tz+2] + dz4[1] * s_s33[ty][tz+3] +
                                dz4[2] * s_s33[ty][tz+4] + dz4[3] * s_s33[ty][tz+5] -
                                f1_c_ij * (dphz4[0] * g_zm3 * (px4[0] * s_s13[p4x0][ty][tz  ] +
                                                                   px4[1] * s_s13[p4x1][ty][tz  ] +
                                                                   px4[2] * s_s13[p4x2][ty][tz  ] +
                                                                   px4[3] * s_s13[p4x3][ty][tz  ]) +
                                               dphz4[1] * g_zm2 * (px4[0] * s_s13[p4x0][ty][tz+1] +
                                                                   px4[1] * s_s13[p4x1][ty][tz+1] +
                                                                   px4[2] * s_s13[p4x2][ty][tz+1] +
                                                                   px4[3] * s_s13[p4x3][ty][tz+1]) +
                                               dphz4[2] * g_zm1 * (px4[0] * s_s13[p4x0][ty][tz+2] +
                                                                   px4[1] * s_s13[p4x1][ty][tz+2] +
                                                                   px4[2] * s_s13[p4x2][ty][tz+2] +
                                                                   px4[3] * s_s13[p4x3][ty][tz+2]) +
                                               dphz4[3] * g_z   * (px4[0] * s_s13[p4x0][ty][tz+3] +
                                                                   px4[1] * s_s13[p4x1][ty][tz+3] +
                                                                   px4[2] * s_s13[p4x2][ty][tz+3] +
                                                                   px4[3] * s_s13[p4x3][ty][tz+3]) +
                                               dphz4[4] * g_zp1 * (px4[0] * s_s13[p4x0][ty][tz+4] +
                                                                   px4[1] * s_s13[p4x1][ty][tz+4] +
                                                                   px4[2] * s_s13[p4x2][ty][tz+4] +
                                                                   px4[3] * s_s13[p4x3][ty][tz+4]) +
                                               dphz4[5] * g_zp2 * (px4[0] * s_s13[p4x0][ty][tz+5] +
                                                                   px4[1] * s_s13[p4x1][ty][tz+5] +
                                                                   px4[2] * s_s13[p4x2][ty][tz+5] +
                                                                   px4[3] * s_s13[p4x3][ty][tz+5]) +
                                               dphz4[6] * g_zp3 * (px4[0] * s_s13[p4x0][ty][tz+6] +
                                                                   px4[1] * s_s13[p4x1][ty][tz+6] +
                                                                   px4[2] * s_s13[p4x2][ty][tz+6] +
                                                                   px4[3] * s_s13[p4x3][ty][tz+6])) -
                                f2_c_ij * (dphz4[0] * g_zm3 * (phy4[0] * s_s23[ty  ][tz  ] +
                                                                   phy4[1] * s_s23[ty+1][tz  ] +
                                                                   phy4[2] * s_s23[ty+2][tz  ] +
                                                                   phy4[3] * s_s23[ty+3][tz  ]) +
                                               dphz4[1] * g_zm2 * (phy4[0] * s_s23[ty  ][tz+1] +
                                                                   phy4[1] * s_s23[ty+1][tz+1] +
                                                                   phy4[2] * s_s23[ty+2][tz+1] +
                                                                   phy4[3] * s_s23[ty+3][tz+1]) +
                                               dphz4[2] * g_zm1 * (phy4[0] * s_s23[ty  ][tz+2] +
                                                                   phy4[1] * s_s23[ty+1][tz+2] +
                                                                   phy4[2] * s_s23[ty+2][tz+2] +
                                                                   phy4[3] * s_s23[ty+3][tz+2]) +
                                               dphz4[3] * g_z   * (phy4[0] * s_s23[ty  ][tz+3] +
                                                                   phy4[1] * s_s23[ty+1][tz+3] +
                                                                   phy4[2] * s_s23[ty+2][tz+3] +
                                                                   phy4[3] * s_s23[ty+3][tz+3]) +
                                               dphz4[4] * g_zp1 * (phy4[0] * s_s23[ty  ][tz+4] +
                                                                   phy4[1] * s_s23[ty+1][tz+4] +
                                                                   phy4[2] * s_s23[ty+2][tz+4] +
                                                                   phy4[3] * s_s23[ty+3][tz+4]) +
                                               dphz4[5] * g_zp2 * (phy4[0] * s_s23[ty  ][tz+5] +
                                                                   phy4[1] * s_s23[ty+1][tz+5] +
                                                                   phy4[2] * s_s23[ty+2][tz+5] +
                                                                   phy4[3] * s_s23[ty+3][tz+5]) +
                                               dphz4[6] * g_zp3 * (phy4[0] * s_s23[ty  ][tz+6] +
                                                                   phy4[1] * s_s23[ty+1][tz+6] +
                                                                   phy4[2] * s_s23[ty+2][tz+6] +
                                                                   phy4[3] * s_s23[ty+3][tz+6])))) * f_dcrj;


    if (active) {
      _u1(i, j, k) = u1val;
      _u2(i, j, k) = u2val;
      _u3(i, j, k) = u3val;
    }
    
    // Rotate register queues
    f_im1 = f_ij;
    f_ij  = f_ip1;
    f_ip1 = f_ip2;

    f_1_im1 = f_1_ij;
    f_1_ij  = f_1_ip1;
    f_1_ip1 = f_1_ip2;

    f_c_im2 = f_c_im1;
    f_c_im1 = f_c_ij;
    f_c_ij  = f_c_ip1;

    // Rotate 3D shared memory plan indices
    int tmp = p4x0;
    p4x0 = p4x1; 
    p4x1 = p4x2;
    p4x2 = p4x3;
    p4x3 = tmp;
    tmp = p2x0;
    p2x0 = p2x1;
    p2x1 = tmp;

  } // End ix loop

#undef TILE_Z
#undef TILE_Y  
#undef _rho
#undef _g3_c
#undef _f_1
#undef _f_2
#undef _f_c
#undef _g3
#undef _dcrjx
#undef _dcrjz
#undef _dcrjy
#undef _s11
#undef _f
#undef _f2_1
#undef _f1_1
#undef _s13
#undef _g_c
#undef _u1
#undef _s12
#undef _u2
#undef _s23
#undef _f1_2
#undef _f2_2
#undef _s22
#undef _u3
#undef _f1_c
#undef _f2_c
#undef _g
#undef _s33
}

// *****************************************************************************
// *****************************************************************************
// *****************************************************************************

__global__ void dtopo_vel_111(
    float *u1, float *u2, float *u3, const float *dcrjx, const float *dcrjy,
    const float *dcrjz, const float *f, const float *f1_1, const float *f1_2,
    const float *f1_c, const float *f2_1, const float *f2_2, const float *f2_c,
    const float *f_1, const float *f_2, const float *f_c, const float *g,
    const float *g3, const float *g3_c, const float *g_c, const float *rho,
    const float *s11, const float *s12, const float *s13, const float *s22,
    const float *s23, const float *s33, const float a, const float nu,
    const int nx, const int ny, const int nz, const int bi, const int bj,
    const int ei, const int ej) {
        const float phz[4] = {-0.0625000000000000, 0.5625000000000000,
                              0.5625000000000000, -0.0625000000000000};
        const float phy[4] = {-0.0625000000000000, 0.5625000000000000,
                              0.5625000000000000, -0.0625000000000000};
        const float phx[4] = {-0.0625000000000000, 0.5625000000000000,
                              0.5625000000000000, -0.0625000000000000};
        const float dhpz[7] = {-0.0026041666666667, 0.0937500000000000,
                               -0.6796875000000000, 0.0000000000000000,
                               0.6796875000000000,  -0.0937500000000000,
                               0.0026041666666667};
        const float dhy[4] = {0.0416666666666667, -1.1250000000000000,
                              1.1250000000000000, -0.0416666666666667};
        const float dhx[4] = {0.0416666666666667, -1.1250000000000000,
                              1.1250000000000000, -0.0416666666666667};
        const float dhz[4] = {0.0416666666666667, -1.1250000000000000,
                              1.1250000000000000, -0.0416666666666667};
        const float px[4] = {-0.0625000000000000, 0.5625000000000000,
                             0.5625000000000000, -0.0625000000000000};
        const float py[4] = {-0.0625000000000000, 0.5625000000000000,
                             0.5625000000000000, -0.0625000000000000};
        const float dx[4] = {0.0416666666666667, -1.1250000000000000,
                             1.1250000000000000, -0.0416666666666667};
        const float dy[4] = {0.0416666666666667, -1.1250000000000000,
                             1.1250000000000000, -0.0416666666666667};
        const float dphz[7] = {-0.0026041666666667, 0.0937500000000000,
                               -0.6796875000000000, 0.0000000000000000,
                               0.6796875000000000,  -0.0937500000000000,
                               0.0026041666666667};
        const float dz[4] = {0.0416666666666667, -1.1250000000000000,
                             1.1250000000000000, -0.0416666666666667};
        const int i = threadIdx.x + blockIdx.x * blockDim.x + bi;
        if (i >= nx) return;
        if (i >= ei) return;
        const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
        if (j >= ny) return;
        if (j >= ej) return;
        const int k = threadIdx.z + blockIdx.z * blockDim.z;
        if (k >= nz - 12) return;
#define _rho(i, j, k)                                                   \
        rho[(k) + align +                                               \
            (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
            (2 * align + nz) * ((j) + ngsl + 2)]
#define _g3_c(k) g3_c[(k) + align]
#define _f_1(i, j)               \
        f_1[(j) + align + ngsl + \
            ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f_2(i, j)               \
        f_2[(j) + align + ngsl + \
            ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f_c(i, j)               \
        f_c[(j) + align + ngsl + \
            ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3(k) g3[(k) + align]
#define _dcrjx(i) dcrjx[(i) + ngsl + 2]
#define _dcrjz(k) dcrjz[(k) + align]
#define _dcrjy(j) dcrjy[(j) + ngsl + 2]
#define _s11(i, j, k)                                                   \
        s11[(k) + align +                                               \
            (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
            (2 * align + nz) * ((j) + ngsl + 2)]
#define _f(i, j)               \
        f[(j) + align + ngsl + \
          ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_1(i, j)               \
        f2_1[(j) + align + ngsl + \
             ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f1_1(i, j)               \
        f1_1[(j) + align + ngsl + \
             ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s13(i, j, k)                                                   \
        s13[(k) + align +                                               \
            (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
            (2 * align + nz) * ((j) + ngsl + 2)]
#define _g_c(k) g_c[(k) + align]
#define _u1(i, j, k)                                                   \
        u1[(k) + align +                                               \
           (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
           (2 * align + nz) * ((j) + ngsl + 2)]
#define _s12(i, j, k)                                                   \
        s12[(k) + align +                                               \
            (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
            (2 * align + nz) * ((j) + ngsl + 2)]
#define _u2(i, j, k)                                                   \
        u2[(k) + align +                                               \
           (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
           (2 * align + nz) * ((j) + ngsl + 2)]
#define _s23(i, j, k)                                                   \
        s23[(k) + align +                                               \
            (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
            (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_2(i, j)               \
        f1_2[(j) + align + ngsl + \
             ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_2(i, j)               \
        f2_2[(j) + align + ngsl + \
             ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s22(i, j, k)                                                   \
        s22[(k) + align +                                               \
            (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
            (2 * align + nz) * ((j) + ngsl + 2)]
#define _u3(i, j, k)                                                   \
        u3[(k) + align +                                               \
           (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
           (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_c(i, j)               \
        f1_c[(j) + align + ngsl + \
             ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_c(i, j)               \
        f2_c[(j) + align + ngsl + \
             ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g(k) g[(k) + align]
#define _s33(i, j, k)                                                   \
        s33[(k) + align +                                               \
            (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
            (2 * align + nz) * ((j) + ngsl + 2)]
        float rho1 = phz[0] * (phy[2] * _rho(i, j, k + 4) +
                               phy[0] * _rho(i, j - 2, k + 4) +
                               phy[1] * _rho(i, j - 1, k + 4) +
                               phy[3] * _rho(i, j + 1, k + 4)) +
                     phz[1] * (phy[2] * _rho(i, j, k + 5) +
                               phy[0] * _rho(i, j - 2, k + 5) +
                               phy[1] * _rho(i, j - 1, k + 5) +
                               phy[3] * _rho(i, j + 1, k + 5)) +
                     phz[2] * (phy[2] * _rho(i, j, k + 6) +
                               phy[0] * _rho(i, j - 2, k + 6) +
                               phy[1] * _rho(i, j - 1, k + 6) +
                               phy[3] * _rho(i, j + 1, k + 6)) +
                     phz[3] * (phy[2] * _rho(i, j, k + 7) +
                               phy[0] * _rho(i, j - 2, k + 7) +
                               phy[1] * _rho(i, j - 1, k + 7) +
                               phy[3] * _rho(i, j + 1, k + 7));
        float rho2 = phz[0] * (phx[2] * _rho(i, j, k + 4) +
                               phx[0] * _rho(i - 2, j, k + 4) +
                               phx[1] * _rho(i - 1, j, k + 4) +
                               phx[3] * _rho(i + 1, j, k + 4)) +
                     phz[1] * (phx[2] * _rho(i, j, k + 5) +
                               phx[0] * _rho(i - 2, j, k + 5) +
                               phx[1] * _rho(i - 1, j, k + 5) +
                               phx[3] * _rho(i + 1, j, k + 5)) +
                     phz[2] * (phx[2] * _rho(i, j, k + 6) +
                               phx[0] * _rho(i - 2, j, k + 6) +
                               phx[1] * _rho(i - 1, j, k + 6) +
                               phx[3] * _rho(i + 1, j, k + 6)) +
                     phz[3] * (phx[2] * _rho(i, j, k + 7) +
                               phx[0] * _rho(i - 2, j, k + 7) +
                               phx[1] * _rho(i - 1, j, k + 7) +
                               phx[3] * _rho(i + 1, j, k + 7));
        float rho3 = phy[2] * (phx[2] * _rho(i, j, k + 6) +
                               phx[0] * _rho(i - 2, j, k + 6) +
                               phx[1] * _rho(i - 1, j, k + 6) +
                               phx[3] * _rho(i + 1, j, k + 6)) +
                     phy[0] * (phx[2] * _rho(i, j - 2, k + 6) +
                               phx[0] * _rho(i - 2, j - 2, k + 6) +
                               phx[1] * _rho(i - 1, j - 2, k + 6) +
                               phx[3] * _rho(i + 1, j - 2, k + 6)) +
                     phy[1] * (phx[2] * _rho(i, j - 1, k + 6) +
                               phx[0] * _rho(i - 2, j - 1, k + 6) +
                               phx[1] * _rho(i - 1, j - 1, k + 6) +
                               phx[3] * _rho(i + 1, j - 1, k + 6)) +
                     phy[3] * (phx[2] * _rho(i, j + 1, k + 6) +
                               phx[0] * _rho(i - 2, j + 1, k + 6) +
                               phx[1] * _rho(i - 1, j + 1, k + 6) +
                               phx[3] * _rho(i + 1, j + 1, k + 6));
        float Ai1 = _f_1(i, j) * _g3_c(k + 6) * rho1;
        Ai1 = nu * 1.0 / Ai1;
        float Ai2 = _f_2(i, j) * _g3_c(k + 6) * rho2;
        Ai2 = nu * 1.0 / Ai2;
        float Ai3 = _f_c(i, j) * _g3(k + 6) * rho3;
        Ai3 = nu * 1.0 / Ai3;
        float f_dcrj = _dcrjx(i) * _dcrjy(j) * _dcrjz(k + 6);
        _u1(i, j, k + 6) =
            (a * _u1(i, j, k + 6) +
             Ai1 *
                 (dhx[2] * _f_c(i, j) * _g3_c(k + 6) * _s11(i, j, k + 6) +
                  dhx[0] * _f_c(i - 2, j) * _g3_c(k + 6) *
                      _s11(i - 2, j, k + 6) +
                  dhx[1] * _f_c(i - 1, j) * _g3_c(k + 6) *
                      _s11(i - 1, j, k + 6) +
                  dhx[3] * _f_c(i + 1, j) * _g3_c(k + 6) *
                      _s11(i + 1, j, k + 6) +
                  dhy[2] * _f(i, j) * _g3_c(k + 6) * _s12(i, j, k + 6) +
                  dhy[0] * _f(i, j - 2) * _g3_c(k + 6) * _s12(i, j - 2, k + 6) +
                  dhy[1] * _f(i, j - 1) * _g3_c(k + 6) * _s12(i, j - 1, k + 6) +
                  dhy[3] * _f(i, j + 1) * _g3_c(k + 6) * _s12(i, j + 1, k + 6) +
                  dhz[0] * _s13(i, j, k + 4) + dhz[1] * _s13(i, j, k + 5) +
                  dhz[2] * _s13(i, j, k + 6) + dhz[3] * _s13(i, j, k + 7) -
                  _f1_1(i, j) * (dhpz[0] * _g_c(k + 3) *
                                     (phx[2] * _s11(i, j, k + 3) +
                                      phx[0] * _s11(i - 2, j, k + 3) +
                                      phx[1] * _s11(i - 1, j, k + 3) +
                                      phx[3] * _s11(i + 1, j, k + 3)) +
                                 dhpz[1] * _g_c(k + 4) *
                                     (phx[2] * _s11(i, j, k + 4) +
                                      phx[0] * _s11(i - 2, j, k + 4) +
                                      phx[1] * _s11(i - 1, j, k + 4) +
                                      phx[3] * _s11(i + 1, j, k + 4)) +
                                 dhpz[2] * _g_c(k + 5) *
                                     (phx[2] * _s11(i, j, k + 5) +
                                      phx[0] * _s11(i - 2, j, k + 5) +
                                      phx[1] * _s11(i - 1, j, k + 5) +
                                      phx[3] * _s11(i + 1, j, k + 5)) +
                                 dhpz[3] * _g_c(k + 6) *
                                     (phx[2] * _s11(i, j, k + 6) +
                                      phx[0] * _s11(i - 2, j, k + 6) +
                                      phx[1] * _s11(i - 1, j, k + 6) +
                                      phx[3] * _s11(i + 1, j, k + 6)) +
                                 dhpz[4] * _g_c(k + 7) *
                                     (phx[2] * _s11(i, j, k + 7) +
                                      phx[0] * _s11(i - 2, j, k + 7) +
                                      phx[1] * _s11(i - 1, j, k + 7) +
                                      phx[3] * _s11(i + 1, j, k + 7)) +
                                 dhpz[5] * _g_c(k + 8) *
                                     (phx[2] * _s11(i, j, k + 8) +
                                      phx[0] * _s11(i - 2, j, k + 8) +
                                      phx[1] * _s11(i - 1, j, k + 8) +
                                      phx[3] * _s11(i + 1, j, k + 8)) +
                                 dhpz[6] * _g_c(k + 9) *
                                     (phx[2] * _s11(i, j, k + 9) +
                                      phx[0] * _s11(i - 2, j, k + 9) +
                                      phx[1] * _s11(i - 1, j, k + 9) +
                                      phx[3] * _s11(i + 1, j, k + 9))) -
                  _f2_1(i, j) * (dhpz[0] * _g_c(k + 3) *
                                     (phy[2] * _s12(i, j, k + 3) +
                                      phy[0] * _s12(i, j - 2, k + 3) +
                                      phy[1] * _s12(i, j - 1, k + 3) +
                                      phy[3] * _s12(i, j + 1, k + 3)) +
                                 dhpz[1] * _g_c(k + 4) *
                                     (phy[2] * _s12(i, j, k + 4) +
                                      phy[0] * _s12(i, j - 2, k + 4) +
                                      phy[1] * _s12(i, j - 1, k + 4) +
                                      phy[3] * _s12(i, j + 1, k + 4)) +
                                 dhpz[2] * _g_c(k + 5) *
                                     (phy[2] * _s12(i, j, k + 5) +
                                      phy[0] * _s12(i, j - 2, k + 5) +
                                      phy[1] * _s12(i, j - 1, k + 5) +
                                      phy[3] * _s12(i, j + 1, k + 5)) +
                                 dhpz[3] * _g_c(k + 6) *
                                     (phy[2] * _s12(i, j, k + 6) +
                                      phy[0] * _s12(i, j - 2, k + 6) +
                                      phy[1] * _s12(i, j - 1, k + 6) +
                                      phy[3] * _s12(i, j + 1, k + 6)) +
                                 dhpz[4] * _g_c(k + 7) *
                                     (phy[2] * _s12(i, j, k + 7) +
                                      phy[0] * _s12(i, j - 2, k + 7) +
                                      phy[1] * _s12(i, j - 1, k + 7) +
                                      phy[3] * _s12(i, j + 1, k + 7)) +
                                 dhpz[5] * _g_c(k + 8) *
                                     (phy[2] * _s12(i, j, k + 8) +
                                      phy[0] * _s12(i, j - 2, k + 8) +
                                      phy[1] * _s12(i, j - 1, k + 8) +
                                      phy[3] * _s12(i, j + 1, k + 8)) +
                                 dhpz[6] * _g_c(k + 9) *
                                     (phy[2] * _s12(i, j, k + 9) +
                                      phy[0] * _s12(i, j - 2, k + 9) +
                                      phy[1] * _s12(i, j - 1, k + 9) +
                                      phy[3] * _s12(i, j + 1, k + 9))))) *
            f_dcrj;
        _u2(i, j, k + 6) =
            (a * _u2(i, j, k + 6) +
             Ai2 *
                 (dhz[0] * _s23(i, j, k + 4) + dhz[1] * _s23(i, j, k + 5) +
                  dhz[2] * _s23(i, j, k + 6) + dhz[3] * _s23(i, j, k + 7) +
                  dx[1] * _f(i, j) * _g3_c(k + 6) * _s12(i, j, k + 6) +
                  dx[0] * _f(i - 1, j) * _g3_c(k + 6) * _s12(i - 1, j, k + 6) +
                  dx[2] * _f(i + 1, j) * _g3_c(k + 6) * _s12(i + 1, j, k + 6) +
                  dx[3] * _f(i + 2, j) * _g3_c(k + 6) * _s12(i + 2, j, k + 6) +
                  dy[1] * _f_c(i, j) * _g3_c(k + 6) * _s22(i, j, k + 6) +
                  dy[0] * _f_c(i, j - 1) * _g3_c(k + 6) *
                      _s22(i, j - 1, k + 6) +
                  dy[2] * _f_c(i, j + 1) * _g3_c(k + 6) *
                      _s22(i, j + 1, k + 6) +
                  dy[3] * _f_c(i, j + 2) * _g3_c(k + 6) *
                      _s22(i, j + 2, k + 6) -
                  _f1_2(i, j) * (dhpz[0] * _g_c(k + 3) *
                                     (px[1] * _s12(i, j, k + 3) +
                                      px[0] * _s12(i - 1, j, k + 3) +
                                      px[2] * _s12(i + 1, j, k + 3) +
                                      px[3] * _s12(i + 2, j, k + 3)) +
                                 dhpz[1] * _g_c(k + 4) *
                                     (px[1] * _s12(i, j, k + 4) +
                                      px[0] * _s12(i - 1, j, k + 4) +
                                      px[2] * _s12(i + 1, j, k + 4) +
                                      px[3] * _s12(i + 2, j, k + 4)) +
                                 dhpz[2] * _g_c(k + 5) *
                                     (px[1] * _s12(i, j, k + 5) +
                                      px[0] * _s12(i - 1, j, k + 5) +
                                      px[2] * _s12(i + 1, j, k + 5) +
                                      px[3] * _s12(i + 2, j, k + 5)) +
                                 dhpz[3] * _g_c(k + 6) *
                                     (px[1] * _s12(i, j, k + 6) +
                                      px[0] * _s12(i - 1, j, k + 6) +
                                      px[2] * _s12(i + 1, j, k + 6) +
                                      px[3] * _s12(i + 2, j, k + 6)) +
                                 dhpz[4] * _g_c(k + 7) *
                                     (px[1] * _s12(i, j, k + 7) +
                                      px[0] * _s12(i - 1, j, k + 7) +
                                      px[2] * _s12(i + 1, j, k + 7) +
                                      px[3] * _s12(i + 2, j, k + 7)) +
                                 dhpz[5] * _g_c(k + 8) *
                                     (px[1] * _s12(i, j, k + 8) +
                                      px[0] * _s12(i - 1, j, k + 8) +
                                      px[2] * _s12(i + 1, j, k + 8) +
                                      px[3] * _s12(i + 2, j, k + 8)) +
                                 dhpz[6] * _g_c(k + 9) *
                                     (px[1] * _s12(i, j, k + 9) +
                                      px[0] * _s12(i - 1, j, k + 9) +
                                      px[2] * _s12(i + 1, j, k + 9) +
                                      px[3] * _s12(i + 2, j, k + 9))) -
                  _f2_2(i, j) * (dhpz[0] * _g_c(k + 3) *
                                     (py[1] * _s22(i, j, k + 3) +
                                      py[0] * _s22(i, j - 1, k + 3) +
                                      py[2] * _s22(i, j + 1, k + 3) +
                                      py[3] * _s22(i, j + 2, k + 3)) +
                                 dhpz[1] * _g_c(k + 4) *
                                     (py[1] * _s22(i, j, k + 4) +
                                      py[0] * _s22(i, j - 1, k + 4) +
                                      py[2] * _s22(i, j + 1, k + 4) +
                                      py[3] * _s22(i, j + 2, k + 4)) +
                                 dhpz[2] * _g_c(k + 5) *
                                     (py[1] * _s22(i, j, k + 5) +
                                      py[0] * _s22(i, j - 1, k + 5) +
                                      py[2] * _s22(i, j + 1, k + 5) +
                                      py[3] * _s22(i, j + 2, k + 5)) +
                                 dhpz[3] * _g_c(k + 6) *
                                     (py[1] * _s22(i, j, k + 6) +
                                      py[0] * _s22(i, j - 1, k + 6) +
                                      py[2] * _s22(i, j + 1, k + 6) +
                                      py[3] * _s22(i, j + 2, k + 6)) +
                                 dhpz[4] * _g_c(k + 7) *
                                     (py[1] * _s22(i, j, k + 7) +
                                      py[0] * _s22(i, j - 1, k + 7) +
                                      py[2] * _s22(i, j + 1, k + 7) +
                                      py[3] * _s22(i, j + 2, k + 7)) +
                                 dhpz[5] * _g_c(k + 8) *
                                     (py[1] * _s22(i, j, k + 8) +
                                      py[0] * _s22(i, j - 1, k + 8) +
                                      py[2] * _s22(i, j + 1, k + 8) +
                                      py[3] * _s22(i, j + 2, k + 8)) +
                                 dhpz[6] * _g_c(k + 9) *
                                     (py[1] * _s22(i, j, k + 9) +
                                      py[0] * _s22(i, j - 1, k + 9) +
                                      py[2] * _s22(i, j + 1, k + 9) +
                                      py[3] * _s22(i, j + 2, k + 9))))) *
            f_dcrj;
        _u3(i, j, k + 6) =
            (a * _u3(i, j, k + 6) +
             Ai3 *
                 (dhy[2] * _f_2(i, j) * _g3(k + 6) * _s23(i, j, k + 6) +
                  dhy[0] * _f_2(i, j - 2) * _g3(k + 6) * _s23(i, j - 2, k + 6) +
                  dhy[1] * _f_2(i, j - 1) * _g3(k + 6) * _s23(i, j - 1, k + 6) +
                  dhy[3] * _f_2(i, j + 1) * _g3(k + 6) * _s23(i, j + 1, k + 6) +
                  dx[1] * _f_1(i, j) * _g3(k + 6) * _s13(i, j, k + 6) +
                  dx[0] * _f_1(i - 1, j) * _g3(k + 6) * _s13(i - 1, j, k + 6) +
                  dx[2] * _f_1(i + 1, j) * _g3(k + 6) * _s13(i + 1, j, k + 6) +
                  dx[3] * _f_1(i + 2, j) * _g3(k + 6) * _s13(i + 2, j, k + 6) +
                  dz[0] * _s33(i, j, k + 5) + dz[1] * _s33(i, j, k + 6) +
                  dz[2] * _s33(i, j, k + 7) + dz[3] * _s33(i, j, k + 8) -
                  _f1_c(i, j) * (dphz[0] * _g(k + 3) *
                                     (px[1] * _s13(i, j, k + 3) +
                                      px[0] * _s13(i - 1, j, k + 3) +
                                      px[2] * _s13(i + 1, j, k + 3) +
                                      px[3] * _s13(i + 2, j, k + 3)) +
                                 dphz[1] * _g(k + 4) *
                                     (px[1] * _s13(i, j, k + 4) +
                                      px[0] * _s13(i - 1, j, k + 4) +
                                      px[2] * _s13(i + 1, j, k + 4) +
                                      px[3] * _s13(i + 2, j, k + 4)) +
                                 dphz[2] * _g(k + 5) *
                                     (px[1] * _s13(i, j, k + 5) +
                                      px[0] * _s13(i - 1, j, k + 5) +
                                      px[2] * _s13(i + 1, j, k + 5) +
                                      px[3] * _s13(i + 2, j, k + 5)) +
                                 dphz[3] * _g(k + 6) *
                                     (px[1] * _s13(i, j, k + 6) +
                                      px[0] * _s13(i - 1, j, k + 6) +
                                      px[2] * _s13(i + 1, j, k + 6) +
                                      px[3] * _s13(i + 2, j, k + 6)) +
                                 dphz[4] * _g(k + 7) *
                                     (px[1] * _s13(i, j, k + 7) +
                                      px[0] * _s13(i - 1, j, k + 7) +
                                      px[2] * _s13(i + 1, j, k + 7) +
                                      px[3] * _s13(i + 2, j, k + 7)) +
                                 dphz[5] * _g(k + 8) *
                                     (px[1] * _s13(i, j, k + 8) +
                                      px[0] * _s13(i - 1, j, k + 8) +
                                      px[2] * _s13(i + 1, j, k + 8) +
                                      px[3] * _s13(i + 2, j, k + 8)) +
                                 dphz[6] * _g(k + 9) *
                                     (px[1] * _s13(i, j, k + 9) +
                                      px[0] * _s13(i - 1, j, k + 9) +
                                      px[2] * _s13(i + 1, j, k + 9) +
                                      px[3] * _s13(i + 2, j, k + 9))) -
                  _f2_c(i, j) * (dphz[0] * _g(k + 3) *
                                     (phy[2] * _s23(i, j, k + 3) +
                                      phy[0] * _s23(i, j - 2, k + 3) +
                                      phy[1] * _s23(i, j - 1, k + 3) +
                                      phy[3] * _s23(i, j + 1, k + 3)) +
                                 dphz[1] * _g(k + 4) *
                                     (phy[2] * _s23(i, j, k + 4) +
                                      phy[0] * _s23(i, j - 2, k + 4) +
                                      phy[1] * _s23(i, j - 1, k + 4) +
                                      phy[3] * _s23(i, j + 1, k + 4)) +
                                 dphz[2] * _g(k + 5) *
                                     (phy[2] * _s23(i, j, k + 5) +
                                      phy[0] * _s23(i, j - 2, k + 5) +
                                      phy[1] * _s23(i, j - 1, k + 5) +
                                      phy[3] * _s23(i, j + 1, k + 5)) +
                                 dphz[3] * _g(k + 6) *
                                     (phy[2] * _s23(i, j, k + 6) +
                                      phy[0] * _s23(i, j - 2, k + 6) +
                                      phy[1] * _s23(i, j - 1, k + 6) +
                                      phy[3] * _s23(i, j + 1, k + 6)) +
                                 dphz[4] * _g(k + 7) *
                                     (phy[2] * _s23(i, j, k + 7) +
                                      phy[0] * _s23(i, j - 2, k + 7) +
                                      phy[1] * _s23(i, j - 1, k + 7) +
                                      phy[3] * _s23(i, j + 1, k + 7)) +
                                 dphz[5] * _g(k + 8) *
                                     (phy[2] * _s23(i, j, k + 8) +
                                      phy[0] * _s23(i, j - 2, k + 8) +
                                      phy[1] * _s23(i, j - 1, k + 8) +
                                      phy[3] * _s23(i, j + 1, k + 8)) +
                                 dphz[6] * _g(k + 9) *
                                     (phy[2] * _s23(i, j, k + 9) +
                                      phy[0] * _s23(i, j - 2, k + 9) +
                                      phy[1] * _s23(i, j - 1, k + 9) +
                                      phy[3] * _s23(i, j + 1, k + 9))))) *
            f_dcrj;
#undef _rho
#undef _g3_c
#undef _f_1
#undef _f_2
#undef _f_c
#undef _g3
#undef _dcrjx
#undef _dcrjz
#undef _dcrjy
#undef _s11
#undef _f
#undef _f2_1
#undef _f1_1
#undef _s13
#undef _g_c
#undef _u1
#undef _s12
#undef _u2
#undef _s23
#undef _f1_2
#undef _f2_2
#undef _s22
#undef _u3
#undef _f1_c
#undef _f2_c
#undef _g
#undef _s33
}

// *****************************************************************************
// *****************************************************************************
// *****************************************************************************


__launch_bounds__ (1024)
__global__ void dtopo_vel_111_blocks(
    float *RSTRCT u1, float *RSTRCT u2, float *RSTRCT u3, const float *RSTRCT dcrjx, const float *RSTRCT dcrjy,
    const float *RSTRCT dcrjz, const float *RSTRCT f, const float *RSTRCT f1_1, const float *RSTRCT f1_2,
    const float *RSTRCT f1_c, const float *RSTRCT f2_1, const float *RSTRCT f2_2, const float *RSTRCT f2_c,
    const float *RSTRCT f_1, const float *RSTRCT f_2, const float *RSTRCT f_c, const float *RSTRCT g,
    const float *RSTRCT g3, const float *RSTRCT g3_c, const float *RSTRCT g_c, const float *RSTRCT rho,
    const float *RSTRCT s11, const float *RSTRCT s12, const float *RSTRCT s13, const float *RSTRCT s22,
    const float *RSTRCT s23, const float *RSTRCT s33, const float a, const float nu,
    const int nx, const int ny, const int nz, const int bi, const int bj,
    const int ei, const int ej) {
        const float phy[4] = {-0.0625000000000000, 0.5625000000000000,
                              0.5625000000000000, -0.0625000000000000};
        const float phx[4] = {-0.0625000000000000, 0.5625000000000000,
                              0.5625000000000000, -0.0625000000000000};
        const float dhpz[7] = {-0.0026041666666667, 0.0937500000000000,
                               -0.6796875000000000, 0.0000000000000000,
                               0.6796875000000000,  -0.0937500000000000,
                               0.0026041666666667};
        const float dhy[4] = {0.0416666666666667, -1.1250000000000000,
                              1.1250000000000000, -0.0416666666666667};
        const float dhx[4] = {0.0416666666666667, -1.1250000000000000,
                              1.1250000000000000, -0.0416666666666667};
        const float dhz[4] = {0.0416666666666667, -1.1250000000000000,
                              1.1250000000000000, -0.0416666666666667};
        const float px[4] = {-0.0625000000000000, 0.5625000000000000,
                             0.5625000000000000, -0.0625000000000000};
        const float py[4] = {-0.0625000000000000, 0.5625000000000000,
                             0.5625000000000000, -0.0625000000000000};
        const float dx[4] = {0.0416666666666667, -1.1250000000000000,
                             1.1250000000000000, -0.0416666666666667};
        const float dy[4] = {0.0416666666666667, -1.1250000000000000,
                             1.1250000000000000, -0.0416666666666667};
        const float dphz[7] = {-0.0026041666666667, 0.0937500000000000,
                               -0.6796875000000000, 0.0000000000000000,
                               0.6796875000000000,  -0.0937500000000000,
                               0.0026041666666667};
        const float dz[4] = {0.0416666666666667, -1.1250000000000000,
                             1.1250000000000000, -0.0416666666666667};
        const int k = threadIdx.x + blockIdx.x * blockDim.x;
        if (k >= nz - 12) return;
        const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
        if (j >= ny) return;
        if (j >= ej) return;
        const int i = threadIdx.z + blockIdx.z * blockDim.z + bi;
        if (i >= nx) return;
        if (i >= ei) return;
#define _rho(i, j, k)                                                   \
        rho[(k) + align +                                               \
            (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
            (2 * align + nz) * ((j) + ngsl + 2)]
#define _g3_c(k) g3_c[(k) + align]
#define _f_1(i, j)               \
        f_1[(j) + align + ngsl + \
            ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f_2(i, j)               \
        f_2[(j) + align + ngsl + \
            ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f_c(i, j)               \
        f_c[(j) + align + ngsl + \
            ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3(k) g3[(k) + align]
#define _dcrjx(i) dcrjx[(i) + ngsl + 2]
#define _dcrjz(k) dcrjz[(k) + align]
#define _dcrjy(j) dcrjy[(j) + ngsl + 2]
#define _s11(i, j, k)                                                   \
        s11[(k) + align +                                               \
            (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
            (2 * align + nz) * ((j) + ngsl + 2)]
#define _f(i, j)               \
        f[(j) + align + ngsl + \
          ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_1(i, j)               \
        f2_1[(j) + align + ngsl + \
             ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f1_1(i, j)               \
        f1_1[(j) + align + ngsl + \
             ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s13(i, j, k)                                                   \
        s13[(k) + align +                                               \
            (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
            (2 * align + nz) * ((j) + ngsl + 2)]
#define _g_c(k) g_c[(k) + align]
#define _u1(i, j, k)                                                   \
        u1[(k) + align +                                               \
           (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
           (2 * align + nz) * ((j) + ngsl + 2)]
#define _s12(i, j, k)                                                   \
        s12[(k) + align +                                               \
            (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
            (2 * align + nz) * ((j) + ngsl + 2)]
#define _u2(i, j, k)                                                   \
        u2[(k) + align +                                               \
           (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
           (2 * align + nz) * ((j) + ngsl + 2)]
#define _s23(i, j, k)                                                   \
        s23[(k) + align +                                               \
            (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
            (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_2(i, j)               \
        f1_2[(j) + align + ngsl + \
             ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_2(i, j)               \
        f2_2[(j) + align + ngsl + \
             ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s22(i, j, k)                                                   \
        s22[(k) + align +                                               \
            (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
            (2 * align + nz) * ((j) + ngsl + 2)]
#define _u3(i, j, k)                                                   \
        u3[(k) + align +                                               \
           (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
           (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_c(i, j)               \
        f1_c[(j) + align + ngsl + \
             ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_c(i, j)               \
        f2_c[(j) + align + ngsl + \
             ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g(k) g[(k) + align]
#define _s33(i, j, k)                                                   \
        s33[(k) + align +                                               \
            (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
            (2 * align + nz) * ((j) + ngsl + 2)]
        float c = 0.25f;
        float rho1 = c * (_rho(i, j, k + 5) + _rho(i, j - 1, k + 5)) +
                     c * (_rho(i, j, k + 6) + _rho(i, j - 1, k + 6));
        float rho2 = c * (_rho(i, j, k + 5) + _rho(i - 1, j, k + 5)) +
                     c * (_rho(i, j, k + 6) + _rho(i - 1, j, k + 6));
        float rho3 = c * (_rho(i, j, k + 6) + _rho(i - 1, j, k + 6)) +
                     c * (_rho(i, j - 1, k + 6) + _rho(i - 1, j - 1, k + 6));
        float Ai1 = _f_1(i, j) * _g3_c(k + 6) * rho1;
        Ai1 = nu * 1.0 / Ai1;
        float Ai2 = _f_2(i, j) * _g3_c(k + 6) * rho2;
        Ai2 = nu * 1.0 / Ai2;
        float Ai3 = _f_c(i, j) * _g3(k + 6) * rho3;
        Ai3 = nu * 1.0 / Ai3;
        float f_dcrj = _dcrjx(i) * _dcrjy(j) * _dcrjz(k + 6);
        _u1(i, j, k + 6) =
            (a * _u1(i, j, k + 6) +
             Ai1 *
                 (dhx[2] * _f_c(i, j) * _g3_c(k + 6) * _s11(i, j, k + 6) +
                  dhx[0] * _f_c(i - 2, j) * _g3_c(k + 6) *
                      _s11(i - 2, j, k + 6) +
                  dhx[1] * _f_c(i - 1, j) * _g3_c(k + 6) *
                      _s11(i - 1, j, k + 6) +
                  dhx[3] * _f_c(i + 1, j) * _g3_c(k + 6) *
                      _s11(i + 1, j, k + 6) +
                  dhy[2] * _f(i, j) * _g3_c(k + 6) * _s12(i, j, k + 6) +
                  dhy[0] * _f(i, j - 2) * _g3_c(k + 6) * _s12(i, j - 2, k + 6) +
                  dhy[1] * _f(i, j - 1) * _g3_c(k + 6) * _s12(i, j - 1, k + 6) +
                  dhy[3] * _f(i, j + 1) * _g3_c(k + 6) * _s12(i, j + 1, k + 6) +
                  dhz[0] * _s13(i, j, k + 4) + dhz[1] * _s13(i, j, k + 5) +
                  dhz[2] * _s13(i, j, k + 6) + dhz[3] * _s13(i, j, k + 7) -
                  _f1_1(i, j) * (dhpz[0] * _g_c(k + 3) *
                                     (phx[2] * _s11(i, j, k + 3) +
                                      phx[0] * _s11(i - 2, j, k + 3) +
                                      phx[1] * _s11(i - 1, j, k + 3) +
                                      phx[3] * _s11(i + 1, j, k + 3)) +
                                 dhpz[1] * _g_c(k + 4) *
                                     (phx[2] * _s11(i, j, k + 4) +
                                      phx[0] * _s11(i - 2, j, k + 4) +
                                      phx[1] * _s11(i - 1, j, k + 4) +
                                      phx[3] * _s11(i + 1, j, k + 4)) +
                                 dhpz[2] * _g_c(k + 5) *
                                     (phx[2] * _s11(i, j, k + 5) +
                                      phx[0] * _s11(i - 2, j, k + 5) +
                                      phx[1] * _s11(i - 1, j, k + 5) +
                                      phx[3] * _s11(i + 1, j, k + 5)) +
                                 dhpz[3] * _g_c(k + 6) *
                                     (phx[2] * _s11(i, j, k + 6) +
                                      phx[0] * _s11(i - 2, j, k + 6) +
                                      phx[1] * _s11(i - 1, j, k + 6) +
                                      phx[3] * _s11(i + 1, j, k + 6)) +
                                 dhpz[4] * _g_c(k + 7) *
                                     (phx[2] * _s11(i, j, k + 7) +
                                      phx[0] * _s11(i - 2, j, k + 7) +
                                      phx[1] * _s11(i - 1, j, k + 7) +
                                      phx[3] * _s11(i + 1, j, k + 7)) +
                                 dhpz[5] * _g_c(k + 8) *
                                     (phx[2] * _s11(i, j, k + 8) +
                                      phx[0] * _s11(i - 2, j, k + 8) +
                                      phx[1] * _s11(i - 1, j, k + 8) +
                                      phx[3] * _s11(i + 1, j, k + 8)) +
                                 dhpz[6] * _g_c(k + 9) *
                                     (phx[2] * _s11(i, j, k + 9) +
                                      phx[0] * _s11(i - 2, j, k + 9) +
                                      phx[1] * _s11(i - 1, j, k + 9) +
                                      phx[3] * _s11(i + 1, j, k + 9))) -
                  _f2_1(i, j) * (dhpz[0] * _g_c(k + 3) *
                                     (phy[2] * _s12(i, j, k + 3) +
                                      phy[0] * _s12(i, j - 2, k + 3) +
                                      phy[1] * _s12(i, j - 1, k + 3) +
                                      phy[3] * _s12(i, j + 1, k + 3)) +
                                 dhpz[1] * _g_c(k + 4) *
                                     (phy[2] * _s12(i, j, k + 4) +
                                      phy[0] * _s12(i, j - 2, k + 4) +
                                      phy[1] * _s12(i, j - 1, k + 4) +
                                      phy[3] * _s12(i, j + 1, k + 4)) +
                                 dhpz[2] * _g_c(k + 5) *
                                     (phy[2] * _s12(i, j, k + 5) +
                                      phy[0] * _s12(i, j - 2, k + 5) +
                                      phy[1] * _s12(i, j - 1, k + 5) +
                                      phy[3] * _s12(i, j + 1, k + 5)) +
                                 dhpz[3] * _g_c(k + 6) *
                                     (phy[2] * _s12(i, j, k + 6) +
                                      phy[0] * _s12(i, j - 2, k + 6) +
                                      phy[1] * _s12(i, j - 1, k + 6) +
                                      phy[3] * _s12(i, j + 1, k + 6)) +
                                 dhpz[4] * _g_c(k + 7) *
                                     (phy[2] * _s12(i, j, k + 7) +
                                      phy[0] * _s12(i, j - 2, k + 7) +
                                      phy[1] * _s12(i, j - 1, k + 7) +
                                      phy[3] * _s12(i, j + 1, k + 7)) +
                                 dhpz[5] * _g_c(k + 8) *
                                     (phy[2] * _s12(i, j, k + 8) +
                                      phy[0] * _s12(i, j - 2, k + 8) +
                                      phy[1] * _s12(i, j - 1, k + 8) +
                                      phy[3] * _s12(i, j + 1, k + 8)) +
                                 dhpz[6] * _g_c(k + 9) *
                                     (phy[2] * _s12(i, j, k + 9) +
                                      phy[0] * _s12(i, j - 2, k + 9) +
                                      phy[1] * _s12(i, j - 1, k + 9) +
                                      phy[3] * _s12(i, j + 1, k + 9))))) *
            f_dcrj;
        _u2(i, j, k + 6) =
            (a * _u2(i, j, k + 6) +
             Ai2 *
                 (dhz[0] * _s23(i, j, k + 4) + dhz[1] * _s23(i, j, k + 5) +
                  dhz[2] * _s23(i, j, k + 6) + dhz[3] * _s23(i, j, k + 7) +
                  dx[1] * _f(i, j) * _g3_c(k + 6) * _s12(i, j, k + 6) +
                  dx[0] * _f(i - 1, j) * _g3_c(k + 6) * _s12(i - 1, j, k + 6) +
                  dx[2] * _f(i + 1, j) * _g3_c(k + 6) * _s12(i + 1, j, k + 6) +
                  dx[3] * _f(i + 2, j) * _g3_c(k + 6) * _s12(i + 2, j, k + 6) +
                  dy[1] * _f_c(i, j) * _g3_c(k + 6) * _s22(i, j, k + 6) +
                  dy[0] * _f_c(i, j - 1) * _g3_c(k + 6) *
                      _s22(i, j - 1, k + 6) +
                  dy[2] * _f_c(i, j + 1) * _g3_c(k + 6) *
                      _s22(i, j + 1, k + 6) +
                  dy[3] * _f_c(i, j + 2) * _g3_c(k + 6) *
                      _s22(i, j + 2, k + 6) -
                  _f1_2(i, j) * (dhpz[0] * _g_c(k + 3) *
                                     (px[1] * _s12(i, j, k + 3) +
                                      px[0] * _s12(i - 1, j, k + 3) +
                                      px[2] * _s12(i + 1, j, k + 3) +
                                      px[3] * _s12(i + 2, j, k + 3)) +
                                 dhpz[1] * _g_c(k + 4) *
                                     (px[1] * _s12(i, j, k + 4) +
                                      px[0] * _s12(i - 1, j, k + 4) +
                                      px[2] * _s12(i + 1, j, k + 4) +
                                      px[3] * _s12(i + 2, j, k + 4)) +
                                 dhpz[2] * _g_c(k + 5) *
                                     (px[1] * _s12(i, j, k + 5) +
                                      px[0] * _s12(i - 1, j, k + 5) +
                                      px[2] * _s12(i + 1, j, k + 5) +
                                      px[3] * _s12(i + 2, j, k + 5)) +
                                 dhpz[3] * _g_c(k + 6) *
                                     (px[1] * _s12(i, j, k + 6) +
                                      px[0] * _s12(i - 1, j, k + 6) +
                                      px[2] * _s12(i + 1, j, k + 6) +
                                      px[3] * _s12(i + 2, j, k + 6)) +
                                 dhpz[4] * _g_c(k + 7) *
                                     (px[1] * _s12(i, j, k + 7) +
                                      px[0] * _s12(i - 1, j, k + 7) +
                                      px[2] * _s12(i + 1, j, k + 7) +
                                      px[3] * _s12(i + 2, j, k + 7)) +
                                 dhpz[5] * _g_c(k + 8) *
                                     (px[1] * _s12(i, j, k + 8) +
                                      px[0] * _s12(i - 1, j, k + 8) +
                                      px[2] * _s12(i + 1, j, k + 8) +
                                      px[3] * _s12(i + 2, j, k + 8)) +
                                 dhpz[6] * _g_c(k + 9) *
                                     (px[1] * _s12(i, j, k + 9) +
                                      px[0] * _s12(i - 1, j, k + 9) +
                                      px[2] * _s12(i + 1, j, k + 9) +
                                      px[3] * _s12(i + 2, j, k + 9))) -
                  _f2_2(i, j) * (dhpz[0] * _g_c(k + 3) *
                                     (py[1] * _s22(i, j, k + 3) +
                                      py[0] * _s22(i, j - 1, k + 3) +
                                      py[2] * _s22(i, j + 1, k + 3) +
                                      py[3] * _s22(i, j + 2, k + 3)) +
                                 dhpz[1] * _g_c(k + 4) *
                                     (py[1] * _s22(i, j, k + 4) +
                                      py[0] * _s22(i, j - 1, k + 4) +
                                      py[2] * _s22(i, j + 1, k + 4) +
                                      py[3] * _s22(i, j + 2, k + 4)) +
                                 dhpz[2] * _g_c(k + 5) *
                                     (py[1] * _s22(i, j, k + 5) +
                                      py[0] * _s22(i, j - 1, k + 5) +
                                      py[2] * _s22(i, j + 1, k + 5) +
                                      py[3] * _s22(i, j + 2, k + 5)) +
                                 dhpz[3] * _g_c(k + 6) *
                                     (py[1] * _s22(i, j, k + 6) +
                                      py[0] * _s22(i, j - 1, k + 6) +
                                      py[2] * _s22(i, j + 1, k + 6) +
                                      py[3] * _s22(i, j + 2, k + 6)) +
                                 dhpz[4] * _g_c(k + 7) *
                                     (py[1] * _s22(i, j, k + 7) +
                                      py[0] * _s22(i, j - 1, k + 7) +
                                      py[2] * _s22(i, j + 1, k + 7) +
                                      py[3] * _s22(i, j + 2, k + 7)) +
                                 dhpz[5] * _g_c(k + 8) *
                                     (py[1] * _s22(i, j, k + 8) +
                                      py[0] * _s22(i, j - 1, k + 8) +
                                      py[2] * _s22(i, j + 1, k + 8) +
                                      py[3] * _s22(i, j + 2, k + 8)) +
                                 dhpz[6] * _g_c(k + 9) *
                                     (py[1] * _s22(i, j, k + 9) +
                                      py[0] * _s22(i, j - 1, k + 9) +
                                      py[2] * _s22(i, j + 1, k + 9) +
                                      py[3] * _s22(i, j + 2, k + 9))))) *
            f_dcrj;
        _u3(i, j, k + 6) =
            (a * _u3(i, j, k + 6) +
             Ai3 *
                 (dhy[2] * _f_2(i, j) * _g3(k + 6) * _s23(i, j, k + 6) +
                  dhy[0] * _f_2(i, j - 2) * _g3(k + 6) * _s23(i, j - 2, k + 6) +
                  dhy[1] * _f_2(i, j - 1) * _g3(k + 6) * _s23(i, j - 1, k + 6) +
                  dhy[3] * _f_2(i, j + 1) * _g3(k + 6) * _s23(i, j + 1, k + 6) +
                  dx[1] * _f_1(i, j) * _g3(k + 6) * _s13(i, j, k + 6) +
                  dx[0] * _f_1(i - 1, j) * _g3(k + 6) * _s13(i - 1, j, k + 6) +
                  dx[2] * _f_1(i + 1, j) * _g3(k + 6) * _s13(i + 1, j, k + 6) +
                  dx[3] * _f_1(i + 2, j) * _g3(k + 6) * _s13(i + 2, j, k + 6) +
                  dz[0] * _s33(i, j, k + 5) + dz[1] * _s33(i, j, k + 6) +
                  dz[2] * _s33(i, j, k + 7) + dz[3] * _s33(i, j, k + 8) -
                  _f1_c(i, j) * (dphz[0] * _g(k + 3) *
                                     (px[1] * _s13(i, j, k + 3) +
                                      px[0] * _s13(i - 1, j, k + 3) +
                                      px[2] * _s13(i + 1, j, k + 3) +
                                      px[3] * _s13(i + 2, j, k + 3)) +
                                 dphz[1] * _g(k + 4) *
                                     (px[1] * _s13(i, j, k + 4) +
                                      px[0] * _s13(i - 1, j, k + 4) +
                                      px[2] * _s13(i + 1, j, k + 4) +
                                      px[3] * _s13(i + 2, j, k + 4)) +
                                 dphz[2] * _g(k + 5) *
                                     (px[1] * _s13(i, j, k + 5) +
                                      px[0] * _s13(i - 1, j, k + 5) +
                                      px[2] * _s13(i + 1, j, k + 5) +
                                      px[3] * _s13(i + 2, j, k + 5)) +
                                 dphz[3] * _g(k + 6) *
                                     (px[1] * _s13(i, j, k + 6) +
                                      px[0] * _s13(i - 1, j, k + 6) +
                                      px[2] * _s13(i + 1, j, k + 6) +
                                      px[3] * _s13(i + 2, j, k + 6)) +
                                 dphz[4] * _g(k + 7) *
                                     (px[1] * _s13(i, j, k + 7) +
                                      px[0] * _s13(i - 1, j, k + 7) +
                                      px[2] * _s13(i + 1, j, k + 7) +
                                      px[3] * _s13(i + 2, j, k + 7)) +
                                 dphz[5] * _g(k + 8) *
                                     (px[1] * _s13(i, j, k + 8) +
                                      px[0] * _s13(i - 1, j, k + 8) +
                                      px[2] * _s13(i + 1, j, k + 8) +
                                      px[3] * _s13(i + 2, j, k + 8)) +
                                 dphz[6] * _g(k + 9) *
                                     (px[1] * _s13(i, j, k + 9) +
                                      px[0] * _s13(i - 1, j, k + 9) +
                                      px[2] * _s13(i + 1, j, k + 9) +
                                      px[3] * _s13(i + 2, j, k + 9))) -
                  _f2_c(i, j) * (dphz[0] * _g(k + 3) *
                                     (phy[2] * _s23(i, j, k + 3) +
                                      phy[0] * _s23(i, j - 2, k + 3) +
                                      phy[1] * _s23(i, j - 1, k + 3) +
                                      phy[3] * _s23(i, j + 1, k + 3)) +
                                 dphz[1] * _g(k + 4) *
                                     (phy[2] * _s23(i, j, k + 4) +
                                      phy[0] * _s23(i, j - 2, k + 4) +
                                      phy[1] * _s23(i, j - 1, k + 4) +
                                      phy[3] * _s23(i, j + 1, k + 4)) +
                                 dphz[2] * _g(k + 5) *
                                     (phy[2] * _s23(i, j, k + 5) +
                                      phy[0] * _s23(i, j - 2, k + 5) +
                                      phy[1] * _s23(i, j - 1, k + 5) +
                                      phy[3] * _s23(i, j + 1, k + 5)) +
                                 dphz[3] * _g(k + 6) *
                                     (phy[2] * _s23(i, j, k + 6) +
                                      phy[0] * _s23(i, j - 2, k + 6) +
                                      phy[1] * _s23(i, j - 1, k + 6) +
                                      phy[3] * _s23(i, j + 1, k + 6)) +
                                 dphz[4] * _g(k + 7) *
                                     (phy[2] * _s23(i, j, k + 7) +
                                      phy[0] * _s23(i, j - 2, k + 7) +
                                      phy[1] * _s23(i, j - 1, k + 7) +
                                      phy[3] * _s23(i, j + 1, k + 7)) +
                                 dphz[5] * _g(k + 8) *
                                     (phy[2] * _s23(i, j, k + 8) +
                                      phy[0] * _s23(i, j - 2, k + 8) +
                                      phy[1] * _s23(i, j - 1, k + 8) +
                                      phy[3] * _s23(i, j + 1, k + 8)) +
                                 dphz[6] * _g(k + 9) *
                                     (phy[2] * _s23(i, j, k + 9) +
                                      phy[0] * _s23(i, j - 2, k + 9) +
                                      phy[1] * _s23(i, j - 1, k + 9) +
                                      phy[3] * _s23(i, j + 1, k + 9))))) *
            f_dcrj;
#undef _rho
#undef _g3_c
#undef _f_1
#undef _f_2
#undef _f_c
#undef _g3
#undef _dcrjx
#undef _dcrjz
#undef _dcrjy
#undef _s11
#undef _f
#undef _f2_1
#undef _f1_1
#undef _s13
#undef _g_c
#undef _u1
#undef _s12
#undef _u2
#undef _s23
#undef _f1_2
#undef _f2_2
#undef _s22
#undef _u3
#undef _f1_c
#undef _f2_c
#undef _g
#undef _s33
}

#define _f(field, i, j, k)                                                   \
        field[(k) + align +                                               \
           (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
           (2 * align + nz) * ((j) + ngsl + 2)]

__global__ void fill(float *RSTRCT u1, int seed,
                        int nx, int ny, int nz)
{
        const int k = threadIdx.x + blockIdx.x * blockDim.x;
        const int j = threadIdx.y + blockIdx.y * blockDim.y;
        const int i = threadIdx.z + blockIdx.z * blockDim.z;

        if (i >= nx || j >= ny || k >= nz)
                return;

        _f(u1, i, j, k) =
            0.1 + ((512 * i + 1024 * k + 2047 * j + seed) % (65776 - 1)) *
                      1.0f / 65776.0f;
}

__global__ void fill1(float *RSTRCT u1, int seed, int n)
{
        const int k = threadIdx.x + blockIdx.x * blockDim.x;

        if (k >= n)
                return;

        u1[k] =
            0.1 + ((1024 * k + seed ) % (65776 - 1)) * 1.0f / 65776.0f;
}

__global__ void fill2(float *RSTRCT u1, int seed, int nx, int ny)
{
        const int k = threadIdx.x + blockIdx.x * blockDim.x;
        const int j = threadIdx.y + blockIdx.y * blockDim.y;

        if (k >= ny || j >= nx)
                return;

        u1[k + j * ny] =
            0.1 + ((1024 * k + 2047 * j + seed) % (65776 - 1)) * 1.0f / 65776.0f;
}

template <typename T>
__global__ void set_const(T *RSTRCT u1, T value, int nx, int ny,
                            int nz) {
        const int k = threadIdx.x + blockIdx.x * blockDim.x;
        const int j = threadIdx.y + blockIdx.y * blockDim.y;
        const int i = threadIdx.z + blockIdx.z * blockDim.z;

        if (i >= nx || j >= ny || k >= nz)
                return;

        _f(u1, i, j, k) = value;
}

__global__ void compare(const float *RSTRCT u1,
                        const float *RSTRCT u2,
                        const float *RSTRCT u3,
                        const float *RSTRCT v1,
                        const float *RSTRCT v2,
                        const float *RSTRCT v3,
                        int nx, int ny, int nz)
{
        const int k = threadIdx.x + blockIdx.x * blockDim.x;
        const int j = threadIdx.y + blockIdx.y * blockDim.y;
        const int i = threadIdx.z + blockIdx.z * blockDim.z;
        if (k >= nz) return;
        if (j >= ny) return;
        if (i >= nx) return;

        if (fabs(_f(u1, i, j, k) - _f(v1, i, j, k)) > 1e-6 ||
            fabs(_f(u2, i, j, k) - _f(v2, i, j, k)) > 1e-6 ||
            fabs(_f(u3, i, j, k) - _f(v3, i, j, k)) > 1e-6) {
                err = -1;
#if PRINTERR
                printf("%d %d %d | %f %f | %f %f | %f %f \n", i, j, k, 
                                _f(u1, i, j, k), 
                                _f(v1, i, j, k), 
                                _f(u2, i, j, k), 
                                _f(v2, i, j, k), 
                                _f(u3, i, j, k), 
                                _f(v3, i, j, k));
#endif
        }
}

__global__ void chknan(const float *RSTRCT u1,
                        int nx, int ny, int nz)
{
        const int k = threadIdx.x + blockIdx.x * blockDim.x;
        const int j = threadIdx.y + blockIdx.y * blockDim.y;
        const int i = threadIdx.z + blockIdx.z * blockDim.z;
        if (k >= nz) return;
        if (j >= ny) return;
        if (i >= nx) return;

        if (isnan(_f(u1, i, j, k)))
                nan_err = -1;
}

#undef _f

#include "split.cu"
#include "unroll.cu"
#include "dm.cu"
#include "dm_unroll.cu"
#include "stress.cu"
#include "stress_macro.cu"
#include "stress_macro_unroll.cu"
#include "stress_index.cu"
#include "stress_index_unroll.cu"

#undef RSTRCT
// *****************************************************************************
// *****************************************************************************
// *****************************************************************************

int getLeadPad (int pad) {
  if (pad % 32)
    return (32 - pad % 32);
  return 0;
}

// *****************************************************************************

template<typename T>
void gpuPaddedAlloc(T* &arr, int ldimx, int ldimy, int ldimz, int pad) {
  int leadpad = getLeadPad (pad);
  size_t size = (ldimx * ldimy * ldimz + leadpad) * sizeof (T);
  T *ptr;
  CUCHK(cudaMalloc ((void**)&ptr, size));
  arr = ptr + leadpad;
}

// *****************************************************************************

template<typename T>
void gpuPaddedFree(T* &arr, int pad) {
  int leadpad = getLeadPad (pad);
  arr -= leadpad;
  CUCHK(cudaFree(arr));
  arr = NULL;
}


// *****************************************************************************
// *****************************************************************************
// *****************************************************************************

int main (int argc, char **argv) {
  int nx = 512;
  int ny = 512;
  int nz = 512;
  int nt = 100;
  // time step normalized by cell volume
  curandGenerator_t gen1, gen2;
  if (argc == 5) {
    nx = atoi (argv[1]);
    ny = atoi (argv[2]);
    nz = atoi (argv[3]);
    nt = atoi (argv[4]);
  }
  else if (argc != 1) {
    printf ("Usage : %s nx ny nz nt\n", argv[0]);
    return -1;
  }

  int ldimz = nz + 2 * align;
  int ldimy = ny + 2 * ngsl + 4;
  int ldimx = nx + 2 * ngsl + 4;
 
  printf ("Running (NX, NY, NZ) = (%d, %d, %d) -> (%d, %d, %d) x %d iterations\n",
          nx, ny, nz, ldimx, ldimy, ldimz, nt);

  int rankx = 0;
  int ranky = 0;

  set_constants(1.0, 1e-3, nx, ny, nz);

  // 3D arrays, with padding
  int pad = 0;
  float *lam, *mu, *qp, *coeff, *qs, *d_vx1, *d_vx2, *d_wwo;
  int *d_ww;
  float *rho, *u1, *u2, *u3, *s11, *s12, *s13, *s22, *s23, *s33;
  float *r1, *r2, *r3, *r4, *r5, *r6;
  // 3D arrays for testing
  // When testing is disabled these arrays will point to their corresponding
  // arrays, i.e.
  // v = u, t = s, p = r
  float *v1, *v2, *v3, *t11, *t12, *t13, *t22, *t23, *t33, *p1, *p2, *p3, *p4,
      *p5, *p6;
  gpuPaddedAlloc (rho, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (u1,  ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (u2,  ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (u3,  ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (s11, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (s12, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (s13, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (s22, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (s23, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (s33, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (r1, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (r2, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (r3, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (r4, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (r5, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (r6, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (lam, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (mu, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (qp, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (coeff, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (qs, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (d_vx1, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (d_vx2, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (d_wwo, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (d_ww, ldimx, ldimy, ldimz, pad);

#if TEST
  gpuPaddedAlloc (v1,  ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (v2,  ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (v3,  ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (t11, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (t12, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (t13, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (t22, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (t23, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (t33, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (p1, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (p2, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (p3, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (p4, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (p5, ldimx, ldimy, ldimz, pad);
  gpuPaddedAlloc (p6, ldimx, ldimy, ldimz, pad);
#else
  v1 = u1;
  v2 = u2;
  v3 = u3;
  t11 = s11;
  t22 = s22;
  t33 = s33;
  t12 = s12;
  t13 = s13;
  t23 = s23;
  p1 = r1;
  p2 = r2;
  p3 = r3;
  p4 = r4;
  p5 = r5;
  p6 = r6;
#endif

  /* Generate n floats on device */
  int n = ldimx * ldimy * ldimz; 
  {
  dim3 threads (512, 1, 1);
  dim3 blocks ((nz - 1) / threads.x + 1, 
               (ny - 1) / threads.y + 1,
               (nx - 1) / threads.z + 1);
  fill<<<blocks, threads>>>(u1, 11, nx, ny, nz);
  fill<<<blocks, threads>>>(u2, 12, nx, ny, nz);
  fill<<<blocks, threads>>>(u3, 13, nx, ny, nz);
  fill<<<blocks, threads>>>(s11, 14, nx, ny, nz);
  fill<<<blocks, threads>>>(s22, 15, nx, ny, nz);
  fill<<<blocks, threads>>>(s33, 16, nx, ny, nz);
  fill<<<blocks, threads>>>(s12, 17, nx, ny, nz);
  fill<<<blocks, threads>>>(s13, 18, nx, ny, nz);
  fill<<<blocks, threads>>>(s23, 19, nx, ny, nz);
  fill<<<blocks, threads>>>(r1, 31, nx, ny, nz);
  fill<<<blocks, threads>>>(r2, 32, nx, ny, nz);
  fill<<<blocks, threads>>>(r3, 33, nx, ny, nz);
  fill<<<blocks, threads>>>(r4, 34, nx, ny, nz);
  fill<<<blocks, threads>>>(r5, 35, nx, ny, nz);
  fill<<<blocks, threads>>>(r6, 36, nx, ny, nz);
  fill<<<blocks, threads>>>(rho, 10, nx, ny, nz);
  fill<<<blocks, threads>>>(lam, 10, nx, ny, nz);
  fill<<<blocks, threads>>>(mu, 10, nx, ny, nz);
  fill<<<blocks, threads>>>(qp, 41, nx, ny, nz);
  fill<<<blocks, threads>>>(qs, 42, nx, ny, nz);
  fill<<<blocks, threads>>>(d_wwo, 10, nx, ny, nz);
  fill<<<blocks, threads>>>(d_vx1, 10, nx, ny, nz);
  fill<<<blocks, threads>>>(d_vx2, 10, nx, ny, nz);

#if TEST
  fill<<<blocks, threads>>>(v1, 11, nx, ny, nz);
  fill<<<blocks, threads>>>(v2, 12, nx, ny, nz);
  fill<<<blocks, threads>>>(v3, 13, nx, ny, nz);
  fill<<<blocks, threads>>>(t11, 14, nx, ny, nz);
  fill<<<blocks, threads>>>(t22, 15, nx, ny, nz);
  fill<<<blocks, threads>>>(t33, 16, nx, ny, nz);
  fill<<<blocks, threads>>>(t12, 17, nx, ny, nz);
  fill<<<blocks, threads>>>(t13, 18, nx, ny, nz);
  fill<<<blocks, threads>>>(t23, 19, nx, ny, nz);
  fill<<<blocks, threads>>>(p1, 31, nx, ny, nz);
  fill<<<blocks, threads>>>(p2, 32, nx, ny, nz);
  fill<<<blocks, threads>>>(p3, 33, nx, ny, nz);
  fill<<<blocks, threads>>>(p4, 34, nx, ny, nz);
  fill<<<blocks, threads>>>(p5, 35, nx, ny, nz);
  fill<<<blocks, threads>>>(p6, 36, nx, ny, nz);
#endif
  set_const<<<blocks, threads>>>(d_ww, 1, nx, ny, nz);
  //set_const<<<blocks, threads>>>(qp, 0.0f, nx, ny, nz);
  //set_const<<<blocks, threads>>>(qs, 0.0f, nx, ny, nz);
  }

  // 2D arrays
  float *f, *f_1, *f_2, *f_c, *f1_1, *f2_1, *f1_2, *f2_2, *f1_c, *f2_c;
  size_t size = ldimx * (2 * align + ldimy) * sizeof (float);
  if (cudaMalloc ((void**)&f, size) != cudaSuccess ||
      cudaMalloc ((void**)&f_1, size) != cudaSuccess ||
      cudaMalloc ((void**)&f_2, size) != cudaSuccess ||
      cudaMalloc ((void**)&f_c, size) != cudaSuccess ||
      cudaMalloc ((void**)&f1_1, size) != cudaSuccess ||
      cudaMalloc ((void**)&f2_1, size) != cudaSuccess ||
      cudaMalloc ((void**)&f1_2, size) != cudaSuccess ||
      cudaMalloc ((void**)&f2_2, size) != cudaSuccess ||
      cudaMalloc ((void**)&f1_c, size) != cudaSuccess ||
      cudaMalloc ((void**)&f2_c, size) != cudaSuccess) {
    printf ("CudaMalloc 2D failed\n");
    return -1;
  }
  {

  dim3 threads (64, 8, 1);
  int ldimy2 = ldimy + 2 * align;
  dim3 blocks((ldimy2 - 1) / threads.x + 1, (ldimx - 1) / threads.y, 1);

  fill2<<<blocks, threads>>>(f, 18, ldimx, ldimy2);
  fill2<<<blocks, threads>>>(f_1, 19, ldimx, ldimy2);
  fill2<<<blocks, threads>>>(f_2, 20, ldimx, ldimy2);
  fill2<<<blocks, threads>>>(f_c, 20, ldimx, ldimy2);
  fill2<<<blocks, threads>>>(f1_1, 19, ldimx, ldimy2);
  fill2<<<blocks, threads>>>(f1_2, 21, ldimx, ldimy2);
  fill2<<<blocks, threads>>>(f1_c, 22, ldimx, ldimy2);
  fill2<<<blocks, threads>>>(f2_1, 23, ldimx, ldimy2);
  fill2<<<blocks, threads>>>(f2_2, 24, ldimx, ldimy2);
  fill2<<<blocks, threads>>>(f2_c, 25, ldimx, ldimy2);

  }

  n = ldimx * (2 * align + ldimy);

  // 1D arrays
  float *g, *g_c, *g3, *g3_c, *dcrjx, *dcrjy, *dcrjz;
  if (cudaMalloc ((void**)&g,    ldimz * sizeof (float)) != cudaSuccess ||
      cudaMalloc ((void**)&g_c,  ldimz * sizeof (float)) != cudaSuccess ||
      cudaMalloc ((void**)&g3,   ldimz * sizeof (float)) != cudaSuccess ||
      cudaMalloc ((void**)&g3_c, ldimz * sizeof (float)) != cudaSuccess ||
      cudaMalloc ((void**)&dcrjx, ldimx * sizeof (float)) != cudaSuccess ||
      cudaMalloc ((void**)&dcrjy, ldimy * sizeof (float)) != cudaSuccess ||
      cudaMalloc ((void**)&dcrjz, ldimz * sizeof (float)) != cudaSuccess) {
    printf ("CudaMalloc 1D failed\n");
    return -1;
  }

  {
  dim3 threads (256, 1, 1);
  dim3 blocks ((ldimx - 1) / threads.x + 1, 1, 1);

  fill1<<<blocks, threads>>>(dcrjx, 31, ldimx);
  }

  {
  dim3 threads (256, 1, 1);
  dim3 blocks ((ldimy - 1) / threads.x + 1, 1, 1);
  fill1<<<blocks, threads>>>(dcrjy, 32, ldimy);
  }
  
  {
  dim3 threads (256, 1, 1);
  dim3 blocks ((ldimz - 1) / threads.x + 1, 1, 1);

  fill1<<<blocks, threads>>>(g, 27, ldimz);
  fill1<<<blocks, threads>>>(g_c, 28, ldimz);
  fill1<<<blocks, threads>>>(g3, 29, ldimz);
  fill1<<<blocks, threads>>>(g3_c, 30, ldimz);
  fill1<<<blocks, threads>>>(dcrjz, 33, ldimz);
  }

      
  
  CUCHK(cudaDeviceSynchronize());
 
  cudaProfilerStart();

  for (int iter=0; iter<nt; iter++) {

//-----------------------------------------------------------------------------
#if USE_ORIGINAL_VEL
    // Original
    {
      dim3 threads (1, 1, 64);
      dim3 blocks (nx, ny, (nz-7)/64+1);
      dtopo_vel_111<<<blocks,threads>>> (u1, u2, u3,
                                         dcrjx, dcrjy, dcrjz,
                                         f, f1_1, f1_2, f1_c,
                                         f2_1, f2_2, f2_c,
                                         f_1, f_2, f_c,
                                         g, g3, g3_c, g_c,
                                         rho, s11, s12, s13, s22, s23, s33,
                                         1.0f, 1.0f, nx, ny, nz,
                                         0, 0, nx-1, ny-1);
    }

    // Original with 3D block
    {
      dim3 threads (64, 4, 4);
      dim3 blocks ((nz-7)/threads.x+1, (ny-1)/threads.y+1, (nx-1)/threads.z+1);
      dtopo_vel_111_blocks<<<blocks,threads>>> (u1, u2, u3,
                                                dcrjx, dcrjy, dcrjz,
                                                f, f1_1, f1_2, f1_c,
                                                f2_1, f2_2, f2_c,
                                                f_1, f_2, f_c,
                                                g, g3, g3_c, g_c,
                                                rho, s11, s12, s13, s22, s23, s33,
                                                1.0f, 1.0f, nx, ny, nz,
                                                0, 0, nx-1, ny-1);
    }

#endif
//-----------------------------------------------------------------------------

#if USE_SHARED_VEL
//-----------------------------------------------------------------------------

    // Optimized code
    //{
    //  dim3 threads (VEL_111_TILE_Z, VEL_111_TILE_Y, 1);
    //  dim3 blocks ((nz-7)/threads.x+1, (ny-1)/threads.y+1, 1);
    //  dtopo_vel_111_opt<<<blocks,threads>>> (u1, u2, u3,
    //                                         dcrjx, dcrjy, dcrjz,
    //                                         f, f1_1, f1_2, f1_c,
    //                                         f2_1, f2_2, f2_c,
    //                                         f_1, f_2, f_c,
    //                                         g, g3, g3_c, g_c,
    //                                         rho, s11, s12, s13, s22, s23, s33,
    //                                         1.0f, 1.0f, nx, ny, nz,
    //                                         0, 0, nx-1, ny-1);
    //}
#endif


#if USE_DM_VEL
//-----------------------------------------------------------------------------
// Original compatible with DM and with 3D block    
    {
      dim3 threads (64, 2, 2);
      dim3 blocks ((nz-7)/threads.x+1, (ny-1)/threads.y+1, (nx-1)/threads.z+1);
      dtopo_vel_111_dm<<<blocks,threads>>> (u1, u2, u3,
                                            dcrjx, dcrjy, dcrjz,
                                            f, f1_1, f1_2, f1_c,
                                            f2_1, f2_2, f2_c,
                                            f_1, f_2, f_c,
                                            g, g3, g3_c, g_c,
                                            rho, s11, s12, s13, s22, s23, s33,
                                            1.0f, 1.0f, nx, ny, nz,
                                            0, 0, nx-1, ny-1);
    }

// Unrolled DM version
    {
#if sm_61
#define nq 2
#define nr 2
      dim3 threads (64, 2, 2);
      dim3 blocks ((nz-7)/(nr*threads.x)+1, 
                   (ny-1)/(nq*threads.y)+1,
                   (nx-1)/threads.z+1);
#else
#define nq 2
#define nr 4
      dim3 threads (32, 2, 2);
      dim3 blocks ((nz-7)/(nr*threads.x)+1, 
                   (ny-1)/(nq*threads.y)+1,
                   (nx-1)/threads.z+1);
#endif
      dtopo_vel_111_dm_unroll<nq, nr><<<blocks,threads>>> (v1, v2, v3,
                                                dcrjx, dcrjy, dcrjz,
                                                f, f1_1, f1_2, f1_c,
                                                f2_1, f2_2, f2_c,
                                                f_1, f_2, f_c,
                                                g, g3, g3_c, g_c,
                                                rho, s11, s12, s13, s22, s23, s33,
                                                1.0f, 1.0f, nx, ny, nz,
                                                0, 0, nx-1, ny-1);

        if (iter == 0) { 

                if (cudaDeviceSynchronize() != cudaSuccess) {
                  printf ("Kernels failed\n");
                }

                compare<<<blocks, threads>>>(u1, u2, u3, v1, v2, v3, nx, ny,
                                             nz);

                int _err = 0;
                cudaMemcpyFromSymbol(&_err, err, sizeof(_err), 0,
                                     cudaMemcpyDeviceToHost);
                if (_err) {
                        printf("Correctness check failed\n");
                        //return -1;
                }
        }
#undef np
#undef nq
#undef nr
    }
//-----------------------------------------------------------------------------
#endif

//-----------------------------------------------------------------------------
// Split versions
#if USE_SPLIT_VEL

    {

#if sm_61
      dim3 threads (64, 2, 2);
#define np 2
#define nq 2
#define nr 2
      dim3 blocks ((nz-7)/(nr*threads.x)+1, 
                   (ny-1)/(nq*threads.y)+1,
                   (nx-1)/(np*threads.z)+1);
#else
      dim3 threads (32, 2, 2);
#define np 1
#define nq 2
#define nr 4
      dim3 blocks ((nz-7)/(nr*threads.x)+1, 
                   (ny-1)/(nq*threads.y)+1,
                   (nx-1)/(np*threads.z)+1);

#endif


      dtopo_vel_111_split1<np, nq, nr><<<blocks,threads>>> (v1, v2, v3,
                                                dcrjx, dcrjy, dcrjz,
                                                f, f1_1, f1_2, f1_c,
                                                f2_1, f2_2, f2_c,
                                                f_1, f_2, f_c,
                                                g, g3, g3_c, g_c,
                                                rho, s11, s12, s13, s22, s23, s33,
                                                1.0f, 1.0f, nx, ny, nz,
                                                0, 0, nx-1, ny-1);
#undef np
#undef nq
#undef nr
    }

    {
#if sm_61
#define np 2
#define nq 2
#define nr 4
      dim3 threads (64, 2, 2);
      dim3 blocks ((nz-7)/(nr*threads.x)+1, 
                   (ny-1)/(nq*threads.y)+1,
                   (nx-1)/(np*threads.z)+1);
#else
#define np 2
#define nq 2
#define nr 4
      dim3 threads (32, 2, 2);
      dim3 blocks ((nz-7)/(nr*threads.x)+1, 
                   (ny-1)/(nq*threads.y)+1,
                   (nx-1)/(np*threads.z)+1);
#endif
      dtopo_vel_111_split2<np, nq, nr><<<blocks,threads>>> (v1, v2, v3,
                                                dcrjx, dcrjy, dcrjz,
                                                f, f1_1, f1_2, f1_c,
                                                f2_1, f2_2, f2_c,
                                                f_1, f_2, f_c,
                                                g, g3, g3_c, g_c,
                                                rho, s11, s12, s13, s22, s23, s33,
                                                1.0f, 1.0f, nx, ny, nz,
                                                0, 0, nx-1, ny-1);

        if (iter == 0) { 

                if (cudaDeviceSynchronize() != cudaSuccess) {
                  printf ("Kernels failed\n");
                }

                compare<<<blocks, threads>>>(u1, u2, u3, v1, v2, v3, nx, ny,
                                             nz);

                int _err = 0;
                cudaMemcpyFromSymbol(&_err, err, sizeof(_err), 0,
                                     cudaMemcpyDeviceToHost);
                if (_err) {
                        printf("Correctness check failed\n");
                        //return -1;
                }
        }
#undef np
#undef nq
#undef nr
    }

#endif
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Unrolled versions
#if USE_UNROLL_VEL
//    {
//#if sm_61
//#define np 1
//#define nq 2
//#define nr 2
//      dim3 threads (64, 2, 2);
//#else
//#define np 1
//#define nq 2
//#define nr 2
//      dim3 threads (32, 2, 2);
//#endif
//      dim3 blocks ((nz-7)/(nr*threads.x)+1, 
//                   (ny-1)/(nq*threads.y)+1,
//                   (nx-1)/(np*threads.z)+1);
//      dtopo_vel_111_unroll2<np, nq, nr><<<blocks,threads>>> (u1, u2, u3,
//                                                dcrjx, dcrjy, dcrjz,
//                                                f, f1_1, f1_2, f1_c,
//                                                f2_1, f2_2, f2_c,
//                                                f_1, f_2, f_c,
//                                                g, g3, g3_c, g_c,
//                                                rho, s11, s12, s13, s22, s23, s33,
//                                                1.0f, 1.0f, nx, ny, nz,
//                                                0, 0, nx-1, ny-1);
//#undef np
//#undef nq
//#undef nr
//    }

    {
#if sm_61
#define np 1
#define nq 2
#define nr 2
      dim3 threads (64, 2, 2);
#else
#define np 1
#define nq 2
#define nr 2
      dim3 threads (32, 2, 2);
#endif
      dim3 blocks ((nz-7)/(nr*threads.x)+1, 
                   (ny-1)/(nq*threads.y)+1,
                   (nx-1)/(np*threads.z)+1);
      dtopo_vel_111_unroll<np, nq, nr><<<blocks, threads>>>(
          u1, u2, u3, dcrjx, dcrjy, dcrjz, f, f1_1, f1_2, f1_c, f2_1, f2_2,
          f2_c, f_1, f_2, f_c, g, g3, g3_c, g_c, rho, s11, s12, s13, s22, s23,
          s33, 1.0f, 1.0f, nx, ny, nz, 0, 0, nx - 1, ny - 1);
#undef np
#undef nq
#undef nr
    }
#endif
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Original stress kernel
#if USE_STRESS_ORIGINAL
{
        dim3 threads (64, 4, 1);
        dim3 blocks ((nz-4)/(threads.x)+1, 
                     (ny-1)/(threads.y)+1,
                     1);
        dtopo_str_111<<<blocks, threads>>>(
            s11, s22, s33, s12, s13, s23, r1, r2, r3, r4, r5, r6, u1, u2, u3, f,
            f1_1, f1_2, f1_c, f2_1, f2_2, f2_c, f_1, f_2, f_c, g, g3, g3_c, g_c,
            lam, mu, qp, coeff, qs, dcrjx, dcrjy, dcrjz, d_vx1, d_vx2, d_ww,
            d_wwo, nx, ny, nz, rankx, ranky, nz, 8, nx - 8, 8, ny - 8);

  CUCHK(cudaDeviceSynchronize());
}
#endif
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Stress kernel that accesses all arrays using macros
#if USE_STRESS_MACRO
{
        dim3 threads (STRM_TX, STRM_TY, STRM_TZ);
        dim3 blocks ((nz-4)/(threads.x)+1, 
                     (ny-1)/(threads.y)+1,
                     1);
        dtopo_str_111_macro<<<blocks, threads>>>(
            t11, t22, t33, t12, t13, t23, p1, p2, p3, p4, p5, p6, u1, u2, u3, f,
            f1_1, f1_2, f1_c, f2_1, f2_2, f2_c, f_1, f_2, f_c, g, g3, g3_c, g_c,
            lam, mu, qp, coeff, qs, dcrjx, dcrjy, dcrjz, d_vx1, d_vx2, d_ww,
            d_wwo, nx, ny, nz, rankx, ranky, nz, 8, nx - 8, 8, ny - 8);

        
        if (iter == 0) { 

                if (cudaDeviceSynchronize() != cudaSuccess) {
                  printf ("Kernels failed\n");
                }

                printf("Running error check\n");

                dim3 threads (64, 8, 1);
                dim3 blocks ((nz-7)/(threads.x)+1, 
                             (ny-1)/(threads.y)+1,
                             (nx-1)/(threads.z)+1);

                compare<<<blocks, threads>>>(s11, s22, s33, t11, t22, t33, nx,
                                             ny, nz);
                compare<<<blocks, threads>>>(s12, s13, s23, t12, t13, t23, nx,
                                             ny, nz);
                compare<<<blocks, threads>>>(r1, r2, r3, p1, p2, p3, nx, ny,
                                             nz);
                compare<<<blocks, threads>>>(r4, r5, r6, p4, p5, p6, nx, ny,
                                             nz);

                int _err = 0;
                cudaMemcpyFromSymbol(&_err, err, sizeof(_err), 0,
                                     cudaMemcpyDeviceToHost);
                if (_err) {
                        printf("Correctness check failed\n");
                }
                printf("done.\n");
        }
}
#endif

#if USE_STRESS_MACRO_UNROLL
{
#define na STRMU_NA 
#define nb STRMU_NB 
        dim3 threads (STRMU_TX, STRMU_TY, STRMU_TZ);
        dim3 blocks ((nz-4)/(na * threads.x)+1, 
                     (ny-1)/(nb * threads.y)+1,
                     1);
        dtopo_str_111_macro_unroll<STRMU_TX, STRMU_TY, STRMU_TZ, na, nb><<<blocks, threads>>>(
            t11, t22, t33, t12, t13, t23, p1, p2, p3, p4, p5, p6, u1, u2, u3, f,
            f1_1, f1_2, f1_c, f2_1, f2_2, f2_c, f_1, f_2, f_c, g, g3, g3_c, g_c,
            lam, mu, qp, coeff, qs, dcrjx, dcrjy, dcrjz, d_vx1, d_vx2, d_ww,
            d_wwo, nx, ny, nz, rankx, ranky, nz, 8, nx - 8, 8, ny - 8);

        
        if (iter == 0) { 

                if (cudaDeviceSynchronize() != cudaSuccess) {
                  printf ("Kernels failed\n");
                }

                printf("Running error check\n");

                dim3 threads (64, 8, 1);
                dim3 blocks ((nz-7)/(threads.x)+1, 
                             (ny-1)/(threads.y)+1,
                             (nx-1)/(threads.z)+1);

                float *vars[12] = {t11, t22, t33, t12, t13, t23,
                                   r1,  r2,  r3,  r4,  r5,  r6};
                for (int pt = 0; pt < 12; ++pt) {
                        chknan<<<blocks, threads>>>(vars[pt], nx, ny, nz);
                }
                compare<<<blocks, threads>>>(s11, s22, s33, t11, t22, t33, nx,
                                             ny, nz);
                compare<<<blocks, threads>>>(s12, s13, s23, t12, t13, t23, nx,
                                             ny, nz);
                compare<<<blocks, threads>>>(r1, r2, r3, p1, p2, p3, nx, ny,
                                             nz);
                compare<<<blocks, threads>>>(r4, r5, r6, p4, p5, p6, nx, ny,
                                             nz);

                int _err = 0;
                cudaMemcpyFromSymbol(&_err, err, sizeof(_err), 0,
                                     cudaMemcpyDeviceToHost);

                int _nan_err = 0;
                cudaMemcpyFromSymbol(&_nan_err, nan_err, sizeof(_nan_err), 0,
                                     cudaMemcpyDeviceToHost);
                if (_err) {
                        printf("Correctness check failed\n");
                }

                if (_nan_err) {
                        printf("Error: nan detected\n");
                }
                printf("done.\n");
        }
}
#endif
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#if USE_STRESS_INDEX
{
        dim3 threads (STRI_TX, STRI_TY, STRI_TZ);
        dim3 blocks ((nz-4)/(threads.x)+1, 
                     (ny-1)/(threads.y)+1,
                     1);
        dtopo_str_111_index<STRI_TX, STRI_TY, STRI_TZ><<<blocks, threads>>>(
            t11, t22, t33, t12, t13, t23, p1, p2, p3, p4, p5, p6, u1, u2, u3, f,
            f1_1, f1_2, f1_c, f2_1, f2_2, f2_c, f_1, f_2, f_c, g, g3, g3_c, g_c,
            lam, mu, qp, coeff, qs, dcrjx, dcrjy, dcrjz, d_vx1, d_vx2, d_ww,
            d_wwo, nx, ny, nz, rankx, ranky, nz, 8, nx - 8, 8, ny - 8);

        
        if (iter == 0) { 

                if (cudaDeviceSynchronize() != cudaSuccess) {
                  printf ("Kernels failed\n");
                }

                printf("Running error check\n");

                dim3 threads (64, 8, 1);
                dim3 blocks ((nz-7)/(threads.x)+1, 
                             (ny-1)/(threads.y)+1,
                             (nx-1)/(threads.z)+1);

                float *vars[12] = {t11, t22, t33, t12, t13, t23,
                                   r1,  r2,  r3,  r4,  r5,  r6};
                for (int pt = 0; pt < 12; ++pt) {
                        chknan<<<blocks, threads>>>(vars[pt], nx, ny, nz);
                }
                compare<<<blocks, threads>>>(s11, s22, s33, t11, t22, t33, nx,
                                             ny, nz);
                compare<<<blocks, threads>>>(s12, s13, s23, t12, t13, t23, nx,
                                             ny, nz);
                compare<<<blocks, threads>>>(r1, r2, r3, p1, p2, p3, nx, ny,
                                             nz);
                compare<<<blocks, threads>>>(r4, r5, r6, p4, p5, p6, nx, ny,
                                             nz);

                int _err = 0;
                cudaMemcpyFromSymbol(&_err, err, sizeof(_err), 0,
                                     cudaMemcpyDeviceToHost);

                int _nan_err = 0;
                cudaMemcpyFromSymbol(&_nan_err, nan_err, sizeof(_nan_err), 0,
                                     cudaMemcpyDeviceToHost);
                if (_err) {
                        printf("Correctness check failed\n");
                }

                if (_nan_err) {
                        printf("Error: nan detected\n");
                }
                printf("done.\n");
        }
}
#endif

#if USE_STRESS_INDEX_UNROLL
{
#define na STRIU_NA 
#define nb STRIU_NB 
        dim3 threads (STRIU_TX, STRIU_TY, STRIU_TZ);
        dim3 blocks ((nz-4)/(na*threads.x)+1, 
                     (ny-1)/(nb*threads.y)+1,
                     (nx-1)/(threads.z)+1);
        dtopo_str_111_index_unroll<STRIU_TX, STRIU_TY, STRIU_TZ, na, nb><<<blocks, threads>>>(
            t11, t22, t33, t12, t13, t23, p1, p2, p3, p4, p5, p6, u1, u2, u3, f,
            f1_1, f1_2, f1_c, f2_1, f2_2, f2_c, f_1, f_2, f_c, g, g3, g3_c, g_c,
            lam, mu, qp, coeff, qs, dcrjx, dcrjy, dcrjz, d_vx1, d_vx2, d_ww,
            d_wwo, nx, ny, nz, rankx, ranky, nz, 8, nx - 8, 8, ny - 8);

        
        if (iter == 0) { 

                if (cudaDeviceSynchronize() != cudaSuccess) {
                  printf ("Kernels failed\n");
                }

                printf("Running error check\n");

                dim3 threads (64, 8, 1);
                dim3 blocks ((nz-7)/(threads.x)+1, 
                             (ny-1)/(threads.y)+1,
                             (nx-1)/(threads.z)+1);

                float *vars[12] = {t11, t22, t33, t12, t13, t23,
                                   r1,  r2,  r3,  r4,  r5,  r6};
                for (int pt = 0; pt < 12; ++pt) {
                        chknan<<<blocks, threads>>>(vars[pt], nx, ny, nz);
                }
                compare<<<blocks, threads>>>(s11, s22, s33, t11, t22, t33, nx,
                                             ny, nz);
                compare<<<blocks, threads>>>(s12, s13, s23, t12, t13, t23, nx,
                                             ny, nz);
                compare<<<blocks, threads>>>(r1, r2, r3, p1, p2, p3, nx, ny,
                                             nz);
                compare<<<blocks, threads>>>(r4, r5, r6, p4, p5, p6, nx, ny,
                                             nz);

                int _err = 0;
                cudaMemcpyFromSymbol(&_err, err, sizeof(_err), 0,
                                     cudaMemcpyDeviceToHost);

                int _nan_err = 0;
                cudaMemcpyFromSymbol(&_nan_err, nan_err, sizeof(_nan_err), 0,
                                     cudaMemcpyDeviceToHost);
                if (_err) {
                        printf("Correctness check failed\n");
                }

                if (_nan_err) {
                        printf("Error: nan detected\n");
                }

                printf("done.\n");
        }
}
#endif


  }

  CUCHK(cudaDeviceSynchronize());

  cudaProfilerStop();
  return 0;
}

