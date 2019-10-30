__launch_bounds__ (256)
__global__ void dtopo_vel_111_dm(
        float *RSTRCT u1, float *RSTRCT u2, float *RSTRCT u3,
        const float *RSTRCT dcrjx, const float *RSTRCT dcrjy,
        const float *RSTRCT dcrjz, const float *RSTRCT f,
        const float *RSTRCT f1_1, const float *RSTRCT f1_2,
        const float *RSTRCT f1_c, const float *RSTRCT f2_1,
        const float *RSTRCT f2_2, const float *RSTRCT f2_c,
        const float *RSTRCT f_1, const float *RSTRCT f_2,
        const float *RSTRCT f_c, const float *RSTRCT g,
        const float *RSTRCT g3, const float *RSTRCT g3_c,
        const float *RSTRCT g_c, const float *RSTRCT rho,
        const float *RSTRCT s11, const float *RSTRCT s12,
        const float *RSTRCT s13, const float *RSTRCT s22,
        const float *RSTRCT s23, const float *RSTRCT s33,
        const float a, const float nu, const int nx, const int ny, const int nz,
        const int bi, const int bj, const int ei, const int ej) {
  const float dhpz4[7] = {-0.0026041666666667, 0.0937500000000000,
                          -0.6796875000000000, 0.0000000000000000,
                          0.6796875000000000,  -0.0937500000000000,
                          0.0026041666666667};
  const float phx4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float phy4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float dhy4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const float dhx4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const float dhz4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const float px4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float py4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float dx4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float dy4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float dphz4[7] = {-0.0026041666666667, 0.0937500000000000,
                          -0.6796875000000000, 0.0000000000000000,
                          0.6796875000000000,  -0.0937500000000000,
                          0.0026041666666667};
  const float dz4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  int dm_offset = 3;
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
  if (k < dm_offset)
    return;
  if (k >= nz - 6)
    return;
#define _dcrjx(i) dcrjx[(i) + ngsl + 2]
#define _dcrjy(j) dcrjy[(j) + ngsl + 2]
#define _dcrjz(k) dcrjz[(k) + align]
#define _f(i, j)                                                               \
  f[(j) + align + ngsl + ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f1_1(i, j)                                                            \
  f1_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f1_2(i, j)                                                            \
  f1_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f1_c(i, j)                                                            \
  f1_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_1(i, j)                                                            \
  f2_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_2(i, j)                                                            \
  f2_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_c(i, j)                                                            \
  f2_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f_1(i, j)                                                             \
  f_1[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f_2(i, j)                                                             \
  f_2[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f_c(i, j)                                                             \
  f_c[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g(k) g[(k) + align]
#define _g3(k) g3[(k) + align]
#define _g3_c(k) g3_c[(k) + align]
#define _g_c(k) g_c[(k) + align]
#define _rho(i, j, k)                                                          \
  rho[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s11(i, j, k)                                                          \
  s11[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s12(i, j, k)                                                          \
  s12[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s13(i, j, k)                                                          \
  s13[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s22(i, j, k)                                                          \
  s22[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s23(i, j, k)                                                          \
  s23[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s33(i, j, k)                                                          \
  s33[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _u1(i, j, k)                                                           \
  u1[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _u2(i, j, k)                                                           \
  u2[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _u3(i, j, k)                                                           \
  u3[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
  float rho1 = 0.25 * (_rho(i, j, k - 1) + _rho(i, j - 1, k - 1)) +
               0.25 * (_rho(i, j, k) + _rho(i, j - 1, k));
  float rho2 = 0.25 * (_rho(i, j, k - 1) + _rho(i - 1, j, k - 1)) +
               0.25 * (_rho(i, j, k) + _rho(i - 1, j, k));
  float rho3 = 0.25 * (_rho(i, j, k) + _rho(i - 1, j, k)) +
               0.25 * (_rho(i, j - 1, k) + _rho(i - 1, j - 1, k));

  float Ai1 = _f_1(i, j) * _g3_c(k) * rho1;
  Ai1 = nu * 1.0 / Ai1;
  float Ai2 = _f_2(i, j) * _g3_c(k) * rho2;
  Ai2 = nu * 1.0 / Ai2;
  float Ai3 = _f_c(i, j) * _g3(k) * rho3;
  Ai3 = nu * 1.0 / Ai3;
  float f_dcrj = _dcrjx(i) * _dcrjy(j) * _dcrjz(k);
  _u1(i, j, k) =
      (a * _u1(i, j, k) +
       Ai1 * (dhx4[2] * _f_c(i, j) * _g3_c(k) * _s11(i, j, k) +
              dhx4[0] * _f_c(i - 2, j) * _g3_c(k) * _s11(i - 2, j, k) +
              dhx4[1] * _f_c(i - 1, j) * _g3_c(k) * _s11(i - 1, j, k) +
              dhx4[3] * _f_c(i + 1, j) * _g3_c(k) * _s11(i + 1, j, k) +
              dhy4[2] * _f(i, j) * _g3_c(k) * _s12(i, j, k) +
              dhy4[0] * _f(i, j - 2) * _g3_c(k) * _s12(i, j - 2, k) +
              dhy4[1] * _f(i, j - 1) * _g3_c(k) * _s12(i, j - 1, k) +
              dhy4[3] * _f(i, j + 1) * _g3_c(k) * _s12(i, j + 1, k) +
              dhz4[2] * _s13(i, j, k) + dhz4[0] * _s13(i, j, k - 2) +
              dhz4[1] * _s13(i, j, k - 1) + dhz4[3] * _s13(i, j, k + 1) -
              _f1_1(i, j) *
                  (dhpz4[3] * _g_c(k) *
                       (phx4[2] * _s11(i, j, k) + phx4[0] * _s11(i - 2, j, k) +
                        phx4[1] * _s11(i - 1, j, k) +
                        phx4[3] * _s11(i + 1, j, k)) +
                   dhpz4[0] * _g_c(k - 3) *
                       (phx4[2] * _s11(i, j, k - 3) +
                        phx4[0] * _s11(i - 2, j, k - 3) +
                        phx4[1] * _s11(i - 1, j, k - 3) +
                        phx4[3] * _s11(i + 1, j, k - 3)) +
                   dhpz4[1] * _g_c(k - 2) *
                       (phx4[2] * _s11(i, j, k - 2) +
                        phx4[0] * _s11(i - 2, j, k - 2) +
                        phx4[1] * _s11(i - 1, j, k - 2) +
                        phx4[3] * _s11(i + 1, j, k - 2)) +
                   dhpz4[2] * _g_c(k - 1) *
                       (phx4[2] * _s11(i, j, k - 1) +
                        phx4[0] * _s11(i - 2, j, k - 1) +
                        phx4[1] * _s11(i - 1, j, k - 1) +
                        phx4[3] * _s11(i + 1, j, k - 1)) +
                   dhpz4[4] * _g_c(k + 1) *
                       (phx4[2] * _s11(i, j, k + 1) +
                        phx4[0] * _s11(i - 2, j, k + 1) +
                        phx4[1] * _s11(i - 1, j, k + 1) +
                        phx4[3] * _s11(i + 1, j, k + 1)) +
                   dhpz4[5] * _g_c(k + 2) *
                       (phx4[2] * _s11(i, j, k + 2) +
                        phx4[0] * _s11(i - 2, j, k + 2) +
                        phx4[1] * _s11(i - 1, j, k + 2) +
                        phx4[3] * _s11(i + 1, j, k + 2)) +
                   dhpz4[6] * _g_c(k + 3) *
                       (phx4[2] * _s11(i, j, k + 3) +
                        phx4[0] * _s11(i - 2, j, k + 3) +
                        phx4[1] * _s11(i - 1, j, k + 3) +
                        phx4[3] * _s11(i + 1, j, k + 3))) -
              _f2_1(i, j) *
                  (dhpz4[3] * _g_c(k) *
                       (phy4[2] * _s12(i, j, k) + phy4[0] * _s12(i, j - 2, k) +
                        phy4[1] * _s12(i, j - 1, k) +
                        phy4[3] * _s12(i, j + 1, k)) +
                   dhpz4[0] * _g_c(k - 3) *
                       (phy4[2] * _s12(i, j, k - 3) +
                        phy4[0] * _s12(i, j - 2, k - 3) +
                        phy4[1] * _s12(i, j - 1, k - 3) +
                        phy4[3] * _s12(i, j + 1, k - 3)) +
                   dhpz4[1] * _g_c(k - 2) *
                       (phy4[2] * _s12(i, j, k - 2) +
                        phy4[0] * _s12(i, j - 2, k - 2) +
                        phy4[1] * _s12(i, j - 1, k - 2) +
                        phy4[3] * _s12(i, j + 1, k - 2)) +
                   dhpz4[2] * _g_c(k - 1) *
                       (phy4[2] * _s12(i, j, k - 1) +
                        phy4[0] * _s12(i, j - 2, k - 1) +
                        phy4[1] * _s12(i, j - 1, k - 1) +
                        phy4[3] * _s12(i, j + 1, k - 1)) +
                   dhpz4[4] * _g_c(k + 1) *
                       (phy4[2] * _s12(i, j, k + 1) +
                        phy4[0] * _s12(i, j - 2, k + 1) +
                        phy4[1] * _s12(i, j - 1, k + 1) +
                        phy4[3] * _s12(i, j + 1, k + 1)) +
                   dhpz4[5] * _g_c(k + 2) *
                       (phy4[2] * _s12(i, j, k + 2) +
                        phy4[0] * _s12(i, j - 2, k + 2) +
                        phy4[1] * _s12(i, j - 1, k + 2) +
                        phy4[3] * _s12(i, j + 1, k + 2)) +
                   dhpz4[6] * _g_c(k + 3) *
                       (phy4[2] * _s12(i, j, k + 3) +
                        phy4[0] * _s12(i, j - 2, k + 3) +
                        phy4[1] * _s12(i, j - 1, k + 3) +
                        phy4[3] * _s12(i, j + 1, k + 3))))) *
      f_dcrj;
  _u2(i, j, k) =
      (a * _u2(i, j, k) +
       Ai2 *
           (dhz4[2] * _s23(i, j, k) + dhz4[0] * _s23(i, j, k - 2) +
            dhz4[1] * _s23(i, j, k - 1) + dhz4[3] * _s23(i, j, k + 1) +
            dx4[1] * _f(i, j) * _g3_c(k) * _s12(i, j, k) +
            dx4[0] * _f(i - 1, j) * _g3_c(k) * _s12(i - 1, j, k) +
            dx4[2] * _f(i + 1, j) * _g3_c(k) * _s12(i + 1, j, k) +
            dx4[3] * _f(i + 2, j) * _g3_c(k) * _s12(i + 2, j, k) +
            dy4[1] * _f_c(i, j) * _g3_c(k) * _s22(i, j, k) +
            dy4[0] * _f_c(i, j - 1) * _g3_c(k) * _s22(i, j - 1, k) +
            dy4[2] * _f_c(i, j + 1) * _g3_c(k) * _s22(i, j + 1, k) +
            dy4[3] * _f_c(i, j + 2) * _g3_c(k) * _s22(i, j + 2, k) -
            _f1_2(i, j) *
                (dhpz4[3] * _g_c(k) *
                     (px4[1] * _s12(i, j, k) + px4[0] * _s12(i - 1, j, k) +
                      px4[2] * _s12(i + 1, j, k) + px4[3] * _s12(i + 2, j, k)) +
                 dhpz4[0] * _g_c(k - 3) *
                     (px4[1] * _s12(i, j, k - 3) +
                      px4[0] * _s12(i - 1, j, k - 3) +
                      px4[2] * _s12(i + 1, j, k - 3) +
                      px4[3] * _s12(i + 2, j, k - 3)) +
                 dhpz4[1] * _g_c(k - 2) *
                     (px4[1] * _s12(i, j, k - 2) +
                      px4[0] * _s12(i - 1, j, k - 2) +
                      px4[2] * _s12(i + 1, j, k - 2) +
                      px4[3] * _s12(i + 2, j, k - 2)) +
                 dhpz4[2] * _g_c(k - 1) *
                     (px4[1] * _s12(i, j, k - 1) +
                      px4[0] * _s12(i - 1, j, k - 1) +
                      px4[2] * _s12(i + 1, j, k - 1) +
                      px4[3] * _s12(i + 2, j, k - 1)) +
                 dhpz4[4] * _g_c(k + 1) *
                     (px4[1] * _s12(i, j, k + 1) +
                      px4[0] * _s12(i - 1, j, k + 1) +
                      px4[2] * _s12(i + 1, j, k + 1) +
                      px4[3] * _s12(i + 2, j, k + 1)) +
                 dhpz4[5] * _g_c(k + 2) *
                     (px4[1] * _s12(i, j, k + 2) +
                      px4[0] * _s12(i - 1, j, k + 2) +
                      px4[2] * _s12(i + 1, j, k + 2) +
                      px4[3] * _s12(i + 2, j, k + 2)) +
                 dhpz4[6] * _g_c(k + 3) *
                     (px4[1] * _s12(i, j, k + 3) +
                      px4[0] * _s12(i - 1, j, k + 3) +
                      px4[2] * _s12(i + 1, j, k + 3) +
                      px4[3] * _s12(i + 2, j, k + 3))) -
            _f2_2(i, j) *
                (dhpz4[3] * _g_c(k) *
                     (py4[1] * _s22(i, j, k) + py4[0] * _s22(i, j - 1, k) +
                      py4[2] * _s22(i, j + 1, k) + py4[3] * _s22(i, j + 2, k)) +
                 dhpz4[0] * _g_c(k - 3) *
                     (py4[1] * _s22(i, j, k - 3) +
                      py4[0] * _s22(i, j - 1, k - 3) +
                      py4[2] * _s22(i, j + 1, k - 3) +
                      py4[3] * _s22(i, j + 2, k - 3)) +
                 dhpz4[1] * _g_c(k - 2) *
                     (py4[1] * _s22(i, j, k - 2) +
                      py4[0] * _s22(i, j - 1, k - 2) +
                      py4[2] * _s22(i, j + 1, k - 2) +
                      py4[3] * _s22(i, j + 2, k - 2)) +
                 dhpz4[2] * _g_c(k - 1) *
                     (py4[1] * _s22(i, j, k - 1) +
                      py4[0] * _s22(i, j - 1, k - 1) +
                      py4[2] * _s22(i, j + 1, k - 1) +
                      py4[3] * _s22(i, j + 2, k - 1)) +
                 dhpz4[4] * _g_c(k + 1) *
                     (py4[1] * _s22(i, j, k + 1) +
                      py4[0] * _s22(i, j - 1, k + 1) +
                      py4[2] * _s22(i, j + 1, k + 1) +
                      py4[3] * _s22(i, j + 2, k + 1)) +
                 dhpz4[5] * _g_c(k + 2) *
                     (py4[1] * _s22(i, j, k + 2) +
                      py4[0] * _s22(i, j - 1, k + 2) +
                      py4[2] * _s22(i, j + 1, k + 2) +
                      py4[3] * _s22(i, j + 2, k + 2)) +
                 dhpz4[6] * _g_c(k + 3) *
                     (py4[1] * _s22(i, j, k + 3) +
                      py4[0] * _s22(i, j - 1, k + 3) +
                      py4[2] * _s22(i, j + 1, k + 3) +
                      py4[3] * _s22(i, j + 2, k + 3))))) *
      f_dcrj;
  _u3(i, j, k) =
      (a * _u3(i, j, k) +
       Ai3 *
           (dhy4[2] * _f_2(i, j) * _g3(k) * _s23(i, j, k) +
            dhy4[0] * _f_2(i, j - 2) * _g3(k) * _s23(i, j - 2, k) +
            dhy4[1] * _f_2(i, j - 1) * _g3(k) * _s23(i, j - 1, k) +
            dhy4[3] * _f_2(i, j + 1) * _g3(k) * _s23(i, j + 1, k) +
            dx4[1] * _f_1(i, j) * _g3(k) * _s13(i, j, k) +
            dx4[0] * _f_1(i - 1, j) * _g3(k) * _s13(i - 1, j, k) +
            dx4[2] * _f_1(i + 1, j) * _g3(k) * _s13(i + 1, j, k) +
            dx4[3] * _f_1(i + 2, j) * _g3(k) * _s13(i + 2, j, k) +
            dz4[1] * _s33(i, j, k) + dz4[0] * _s33(i, j, k - 1) +
            dz4[2] * _s33(i, j, k + 1) + dz4[3] * _s33(i, j, k + 2) -
            _f1_c(i, j) *
                (dphz4[3] * _g(k) *
                     (px4[1] * _s13(i, j, k) + px4[0] * _s13(i - 1, j, k) +
                      px4[2] * _s13(i + 1, j, k) + px4[3] * _s13(i + 2, j, k)) +
                 dphz4[0] * _g(k - 3) *
                     (px4[1] * _s13(i, j, k - 3) +
                      px4[0] * _s13(i - 1, j, k - 3) +
                      px4[2] * _s13(i + 1, j, k - 3) +
                      px4[3] * _s13(i + 2, j, k - 3)) +
                 dphz4[1] * _g(k - 2) *
                     (px4[1] * _s13(i, j, k - 2) +
                      px4[0] * _s13(i - 1, j, k - 2) +
                      px4[2] * _s13(i + 1, j, k - 2) +
                      px4[3] * _s13(i + 2, j, k - 2)) +
                 dphz4[2] * _g(k - 1) *
                     (px4[1] * _s13(i, j, k - 1) +
                      px4[0] * _s13(i - 1, j, k - 1) +
                      px4[2] * _s13(i + 1, j, k - 1) +
                      px4[3] * _s13(i + 2, j, k - 1)) +
                 dphz4[4] * _g(k + 1) *
                     (px4[1] * _s13(i, j, k + 1) +
                      px4[0] * _s13(i - 1, j, k + 1) +
                      px4[2] * _s13(i + 1, j, k + 1) +
                      px4[3] * _s13(i + 2, j, k + 1)) +
                 dphz4[5] * _g(k + 2) *
                     (px4[1] * _s13(i, j, k + 2) +
                      px4[0] * _s13(i - 1, j, k + 2) +
                      px4[2] * _s13(i + 1, j, k + 2) +
                      px4[3] * _s13(i + 2, j, k + 2)) +
                 dphz4[6] * _g(k + 3) *
                     (px4[1] * _s13(i, j, k + 3) +
                      px4[0] * _s13(i - 1, j, k + 3) +
                      px4[2] * _s13(i + 1, j, k + 3) +
                      px4[3] * _s13(i + 2, j, k + 3))) -
            _f2_c(i, j) *
                (dphz4[3] * _g(k) *
                     (phy4[2] * _s23(i, j, k) + phy4[0] * _s23(i, j - 2, k) +
                      phy4[1] * _s23(i, j - 1, k) +
                      phy4[3] * _s23(i, j + 1, k)) +
                 dphz4[0] * _g(k - 3) *
                     (phy4[2] * _s23(i, j, k - 3) +
                      phy4[0] * _s23(i, j - 2, k - 3) +
                      phy4[1] * _s23(i, j - 1, k - 3) +
                      phy4[3] * _s23(i, j + 1, k - 3)) +
                 dphz4[1] * _g(k - 2) *
                     (phy4[2] * _s23(i, j, k - 2) +
                      phy4[0] * _s23(i, j - 2, k - 2) +
                      phy4[1] * _s23(i, j - 1, k - 2) +
                      phy4[3] * _s23(i, j + 1, k - 2)) +
                 dphz4[2] * _g(k - 1) *
                     (phy4[2] * _s23(i, j, k - 1) +
                      phy4[0] * _s23(i, j - 2, k - 1) +
                      phy4[1] * _s23(i, j - 1, k - 1) +
                      phy4[3] * _s23(i, j + 1, k - 1)) +
                 dphz4[4] * _g(k + 1) *
                     (phy4[2] * _s23(i, j, k + 1) +
                      phy4[0] * _s23(i, j - 2, k + 1) +
                      phy4[1] * _s23(i, j - 1, k + 1) +
                      phy4[3] * _s23(i, j + 1, k + 1)) +
                 dphz4[5] * _g(k + 2) *
                     (phy4[2] * _s23(i, j, k + 2) +
                      phy4[0] * _s23(i, j - 2, k + 2) +
                      phy4[1] * _s23(i, j - 1, k + 2) +
                      phy4[3] * _s23(i, j + 1, k + 2)) +
                 dphz4[6] * _g(k + 3) *
                     (phy4[2] * _s23(i, j, k + 3) +
                      phy4[0] * _s23(i, j - 2, k + 3) +
                      phy4[1] * _s23(i, j - 1, k + 3) +
                      phy4[3] * _s23(i, j + 1, k + 3))))) *
      f_dcrj;
#undef _dcrjx
#undef _dcrjy
#undef _dcrjz
#undef _f
#undef _f1_1
#undef _f1_2
#undef _f1_c
#undef _f2_1
#undef _f2_2
#undef _f2_c
#undef _f_1
#undef _f_2
#undef _f_c
#undef _g
#undef _g3
#undef _g3_c
#undef _g_c
#undef _rho
#undef _s11
#undef _s12
#undef _s13
#undef _s22
#undef _s23
#undef _s33
#undef _u1
#undef _u2
#undef _u3
}


