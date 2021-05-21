#define RSTRCT __restrict__
template <int nq, int nr>
#if sm_61
__launch_bounds__ (256)
#else
__launch_bounds__ (128)
#endif
__global__ void dtopo_vel_111_unroll(
        _prec *RSTRCT u1, _prec *RSTRCT u2, _prec *RSTRCT u3,
        const _prec *RSTRCT dcrjx, const _prec *RSTRCT dcrjy,
        const _prec *RSTRCT dcrjz, const _prec *RSTRCT f,
        const _prec *RSTRCT f1_1, const _prec *RSTRCT f1_2,
        const _prec *RSTRCT f1_c, const _prec *RSTRCT f2_1,
        const _prec *RSTRCT f2_2, const _prec *RSTRCT f2_c,
        const _prec *RSTRCT f_1, const _prec *RSTRCT f_2,
        const _prec *RSTRCT f_c, const _prec *RSTRCT g,
        const _prec *RSTRCT g3, const _prec *RSTRCT g3_c,
        const _prec *RSTRCT g_c, const _prec *RSTRCT rho,
        const _prec *RSTRCT s11, const _prec *RSTRCT s12,
        const _prec *RSTRCT s13, const _prec *RSTRCT s22,
        const _prec *RSTRCT s23, const _prec *RSTRCT s33,
        const _prec a, const _prec nu, const int nx, const int ny, const int nz,
        const int bi, const int bj, const int ei, const int ej) {
  const _prec dhpz4[7] = {-0.0026041666666667, 0.0937500000000000,
                          -0.6796875000000000, 0.0000000000000000,
                          0.6796875000000000,  -0.0937500000000000,
                          0.0026041666666667};
  const _prec phx4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const _prec phy4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const _prec dhy4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const _prec dhx4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const _prec dhz4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const _prec px4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const _prec py4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const _prec dx4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const _prec dy4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const _prec dphz4[7] = {-0.0026041666666667, 0.0937500000000000,
                          -0.6796875000000000, 0.0000000000000000,
                          0.6796875000000000,  -0.0937500000000000,
                          0.0026041666666667};
  const _prec dz4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  int dm_offset = 3;
  const int i = threadIdx.z + blockIdx.z * blockDim.z + bi;
  if (i >= nx)
    return;
  if (i >= ei)
    return;
  const int j = nq * threadIdx.y + nq * blockIdx.y * blockDim.y + bj;
  if (j >= ny)
    return;
  if (j >= ej)
    return;

  const int k = nr * threadIdx.x + nr * blockIdx.x * blockDim.x;
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

  _prec v1[nq][nr];
  _prec v2[nq][nr];
  _prec v3[nq][nr];
#pragma unroll
        for (int q = 0; q < nq; ++q) {
#pragma unroll
        for (int r = 0; r < nr; ++r) {


  _prec c = 0.25f;
  _prec rho1 = c * (_rho(i, j + q, k + r - 1) + _rho(i, j + q - 1, k + r - 1)) +
               c * (_rho(i, j + q, k + r) + _rho(i, j + q - 1, k + r));
  _prec rho2 = c * (_rho(i, j + q, k + r - 1) + _rho(i + 1, j + q, k + r - 1)) +
               c * (_rho(i, j + q, k + r) + _rho(i + 1, j + q, k + r));
  _prec rho3 = c * (_rho(i, j + q, k + r) + _rho(i + 1, j + q, k + r)) +
               c * (_rho(i, j + q - 1, k + r) + _rho(i + 1, j + q - 1, k + r));

  _prec Ai1 = _f_1(i, j + q) * _g3_c(k + r) * rho1;
  Ai1 = nu * 1.0 / Ai1;
  _prec Ai2 = _f_2(i, j + q) * _g3_c(k + r) * rho2;
  Ai2 = nu * 1.0 / Ai2;
  _prec Ai3 = _f_c(i, j + q) * _g3(k + r) * rho3;
  Ai3 = nu * 1.0 / Ai3;
  _prec f_dcrj = _dcrjx(i) * _dcrjy(j + q) * _dcrjz(k + r);
  v1[q][r] =
      (a * _u1(i, j + q, k + r) +
       Ai1 * (dhx4[2] * _f_c(i, j + q) * _g3_c(k + r) * _s11(i, j + q, k + r) +
              dhx4[0] * _f_c(i - 2, j + q) * _g3_c(k + r) * _s11(i - 2, j + q, k + r) +
              dhx4[1] * _f_c(i - 1, j + q) * _g3_c(k + r) * _s11(i - 1, j + q, k + r) +
              dhx4[3] * _f_c(i + 1, j + q) * _g3_c(k + r) * _s11(i + 1, j + q, k + r) +
              dhy4[2] * _f(i, j + q) * _g3_c(k + r) * _s12(i, j + q, k + r) +
              dhy4[0] * _f(i, j + q - 2) * _g3_c(k + r) * _s12(i, j + q - 2, k + r) +
              dhy4[1] * _f(i, j + q - 1) * _g3_c(k + r) * _s12(i, j + q - 1, k + r) +
              dhy4[3] * _f(i, j + q + 1) * _g3_c(k + r) * _s12(i, j + q + 1, k + r) +
              dhz4[2] * _s13(i, j + q, k + r) + dhz4[0] * _s13(i, j + q, k + r - 2) +
              dhz4[1] * _s13(i, j + q, k + r - 1) + dhz4[3] * _s13(i, j + q, k + r + 1) -
              _f1_1(i, j + q) *
                  (dhpz4[3] * _g_c(k + r) *
                       (phx4[2] * _s11(i, j + q, k + r) + phx4[0] * _s11(i - 2, j + q, k + r) +
                        phx4[1] * _s11(i - 1, j + q, k + r) +
                        phx4[3] * _s11(i + 1, j + q, k + r)) +
                   dhpz4[0] * _g_c(k + r - 3) *
                       (phx4[2] * _s11(i, j + q, k + r - 3) +
                        phx4[0] * _s11(i - 2, j + q, k + r - 3) +
                        phx4[1] * _s11(i - 1, j + q, k + r - 3) +
                        phx4[3] * _s11(i + 1, j + q, k + r - 3)) +
                   dhpz4[1] * _g_c(k + r - 2) *
                       (phx4[2] * _s11(i, j + q, k + r - 2) +
                        phx4[0] * _s11(i - 2, j + q, k + r - 2) +
                        phx4[1] * _s11(i - 1, j + q, k + r - 2) +
                        phx4[3] * _s11(i + 1, j + q, k + r - 2)) +
                   dhpz4[2] * _g_c(k + r - 1) *
                       (phx4[2] * _s11(i, j + q, k + r - 1) +
                        phx4[0] * _s11(i - 2, j + q, k + r - 1) +
                        phx4[1] * _s11(i - 1, j + q, k + r - 1) +
                        phx4[3] * _s11(i + 1, j + q, k + r - 1)) +
                   dhpz4[4] * _g_c(k + r + 1) *
                       (phx4[2] * _s11(i, j + q, k + r + 1) +
                        phx4[0] * _s11(i - 2, j + q, k + r + 1) +
                        phx4[1] * _s11(i - 1, j + q, k + r + 1) +
                        phx4[3] * _s11(i + 1, j + q, k + r + 1)) +
                   dhpz4[5] * _g_c(k + r + 2) *
                       (phx4[2] * _s11(i, j + q, k + r + 2) +
                        phx4[0] * _s11(i - 2, j + q, k + r + 2) +
                        phx4[1] * _s11(i - 1, j + q, k + r + 2) +
                        phx4[3] * _s11(i + 1, j + q, k + r + 2)) +
                   dhpz4[6] * _g_c(k + r + 3) *
                       (phx4[2] * _s11(i, j + q, k + r + 3) +
                        phx4[0] * _s11(i - 2, j + q, k + r + 3) +
                        phx4[1] * _s11(i - 1, j + q, k + r + 3) +
                        phx4[3] * _s11(i + 1, j + q, k + r + 3))) -
              _f2_1(i, j + q) *
                  (dhpz4[3] * _g_c(k + r) *
                       (phy4[2] * _s12(i, j + q, k + r) + phy4[0] * _s12(i, j + q - 2, k + r) +
                        phy4[1] * _s12(i, j + q - 1, k + r) +
                        phy4[3] * _s12(i, j + q + 1, k + r)) +
                   dhpz4[0] * _g_c(k + r - 3) *
                       (phy4[2] * _s12(i, j + q, k + r - 3) +
                        phy4[0] * _s12(i, j + q - 2, k + r - 3) +
                        phy4[1] * _s12(i, j + q - 1, k + r - 3) +
                        phy4[3] * _s12(i, j + q + 1, k + r - 3)) +
                   dhpz4[1] * _g_c(k + r - 2) *
                       (phy4[2] * _s12(i, j + q, k + r - 2) +
                        phy4[0] * _s12(i, j + q - 2, k + r - 2) +
                        phy4[1] * _s12(i, j + q - 1, k + r - 2) +
                        phy4[3] * _s12(i, j + q + 1, k + r - 2)) +
                   dhpz4[2] * _g_c(k + r - 1) *
                       (phy4[2] * _s12(i, j + q, k + r - 1) +
                        phy4[0] * _s12(i, j + q - 2, k + r - 1) +
                        phy4[1] * _s12(i, j + q - 1, k + r - 1) +
                        phy4[3] * _s12(i, j + q + 1, k + r - 1)) +
                   dhpz4[4] * _g_c(k + r + 1) *
                       (phy4[2] * _s12(i, j + q, k + r + 1) +
                        phy4[0] * _s12(i, j + q - 2, k + r + 1) +
                        phy4[1] * _s12(i, j + q - 1, k + r + 1) +
                        phy4[3] * _s12(i, j + q + 1, k + r + 1)) +
                   dhpz4[5] * _g_c(k + r + 2) *
                       (phy4[2] * _s12(i, j + q, k + r + 2) +
                        phy4[0] * _s12(i, j + q - 2, k + r + 2) +
                        phy4[1] * _s12(i, j + q - 1, k + r + 2) +
                        phy4[3] * _s12(i, j + q + 1, k + r + 2)) +
                   dhpz4[6] * _g_c(k + r + 3) *
                       (phy4[2] * _s12(i, j + q, k + r + 3) +
                        phy4[0] * _s12(i, j + q - 2, k + r + 3) +
                        phy4[1] * _s12(i, j + q - 1, k + r + 3) +
                        phy4[3] * _s12(i, j + q + 1, k + r + 3))))) *
      f_dcrj;
  v2[q][r] =
      (a * _u2(i, j + q, k + r) +
       Ai2 *
           (dhz4[2] * _s23(i, j + q, k + r) + dhz4[0] * _s23(i, j + q, k + r - 2) +
            dhz4[1] * _s23(i, j + q, k + r - 1) + dhz4[3] * _s23(i, j + q, k + r + 1) +
            dx4[1] * _f(i, j + q) * _g3_c(k + r) * _s12(i, j + q, k + r) +
            dx4[0] * _f(i - 1, j + q) * _g3_c(k + r) * _s12(i - 1, j + q, k + r) +
            dx4[2] * _f(i + 1, j + q) * _g3_c(k + r) * _s12(i + 1, j + q, k + r) +
            dx4[3] * _f(i + 2, j + q) * _g3_c(k + r) * _s12(i + 2, j + q, k + r) +
            dy4[1] * _f_c(i, j + q) * _g3_c(k + r) * _s22(i, j + q, k + r) +
            dy4[0] * _f_c(i, j + q - 1) * _g3_c(k + r) * _s22(i, j + q - 1, k + r) +
            dy4[2] * _f_c(i, j + q + 1) * _g3_c(k + r) * _s22(i, j + q + 1, k + r) +
            dy4[3] * _f_c(i, j + q + 2) * _g3_c(k + r) * _s22(i, j + q + 2, k + r) -
            _f1_2(i, j + q) *
                (dhpz4[3] * _g_c(k + r) *
                     (px4[1] * _s12(i, j + q, k + r) + px4[0] * _s12(i - 1, j + q, k + r) +
                      px4[2] * _s12(i + 1, j + q, k + r) + px4[3] * _s12(i + 2, j + q, k + r)) +
                 dhpz4[0] * _g_c(k + r - 3) *
                     (px4[1] * _s12(i, j + q, k + r - 3) +
                      px4[0] * _s12(i - 1, j + q, k + r - 3) +
                      px4[2] * _s12(i + 1, j + q, k + r - 3) +
                      px4[3] * _s12(i + 2, j + q, k + r - 3)) +
                 dhpz4[1] * _g_c(k + r - 2) *
                     (px4[1] * _s12(i, j + q, k + r - 2) +
                      px4[0] * _s12(i - 1, j + q, k + r - 2) +
                      px4[2] * _s12(i + 1, j + q, k + r - 2) +
                      px4[3] * _s12(i + 2, j + q, k + r - 2)) +
                 dhpz4[2] * _g_c(k + r - 1) *
                     (px4[1] * _s12(i, j + q, k + r - 1) +
                      px4[0] * _s12(i - 1, j + q, k + r - 1) +
                      px4[2] * _s12(i + 1, j + q, k + r - 1) +
                      px4[3] * _s12(i + 2, j + q, k + r - 1)) +
                 dhpz4[4] * _g_c(k + r + 1) *
                     (px4[1] * _s12(i, j + q, k + r + 1) +
                      px4[0] * _s12(i - 1, j + q, k + r + 1) +
                      px4[2] * _s12(i + 1, j + q, k + r + 1) +
                      px4[3] * _s12(i + 2, j + q, k + r + 1)) +
                 dhpz4[5] * _g_c(k + r + 2) *
                     (px4[1] * _s12(i, j + q, k + r + 2) +
                      px4[0] * _s12(i - 1, j + q, k + r + 2) +
                      px4[2] * _s12(i + 1, j + q, k + r + 2) +
                      px4[3] * _s12(i + 2, j + q, k + r + 2)) +
                 dhpz4[6] * _g_c(k + r + 3) *
                     (px4[1] * _s12(i, j + q, k + r + 3) +
                      px4[0] * _s12(i - 1, j + q, k + r + 3) +
                      px4[2] * _s12(i + 1, j + q, k + r + 3) +
                      px4[3] * _s12(i + 2, j + q, k + r + 3))) -
            _f2_2(i, j + q) *
                (dhpz4[3] * _g_c(k + r) *
                     (py4[1] * _s22(i, j + q, k + r) + py4[0] * _s22(i, j + q - 1, k + r) +
                      py4[2] * _s22(i, j + q + 1, k + r) + py4[3] * _s22(i, j + q + 2, k + r)) +
                 dhpz4[0] * _g_c(k + r - 3) *
                     (py4[1] * _s22(i, j + q, k + r - 3) +
                      py4[0] * _s22(i, j + q - 1, k + r - 3) +
                      py4[2] * _s22(i, j + q + 1, k + r - 3) +
                      py4[3] * _s22(i, j + q + 2, k + r - 3)) +
                 dhpz4[1] * _g_c(k + r - 2) *
                     (py4[1] * _s22(i, j + q, k + r - 2) +
                      py4[0] * _s22(i, j + q - 1, k + r - 2) +
                      py4[2] * _s22(i, j + q + 1, k + r - 2) +
                      py4[3] * _s22(i, j + q + 2, k + r - 2)) +
                 dhpz4[2] * _g_c(k + r - 1) *
                     (py4[1] * _s22(i, j + q, k + r - 1) +
                      py4[0] * _s22(i, j + q - 1, k + r - 1) +
                      py4[2] * _s22(i, j + q + 1, k + r - 1) +
                      py4[3] * _s22(i, j + q + 2, k + r - 1)) +
                 dhpz4[4] * _g_c(k + r + 1) *
                     (py4[1] * _s22(i, j + q, k + r + 1) +
                      py4[0] * _s22(i, j + q - 1, k + r + 1) +
                      py4[2] * _s22(i, j + q + 1, k + r + 1) +
                      py4[3] * _s22(i, j + q + 2, k + r + 1)) +
                 dhpz4[5] * _g_c(k + r + 2) *
                     (py4[1] * _s22(i, j + q, k + r + 2) +
                      py4[0] * _s22(i, j + q - 1, k + r + 2) +
                      py4[2] * _s22(i, j + q + 1, k + r + 2) +
                      py4[3] * _s22(i, j + q + 2, k + r + 2)) +
                 dhpz4[6] * _g_c(k + r + 3) *
                     (py4[1] * _s22(i, j + q, k + r + 3) +
                      py4[0] * _s22(i, j + q - 1, k + r + 3) +
                      py4[2] * _s22(i, j + q + 1, k + r + 3) +
                      py4[3] * _s22(i, j + q + 2, k + r + 3))))) *
      f_dcrj;
  v3[q][r] =
      (a * _u3(i, j + q, k + r) +
       Ai3 *
           (dhy4[2] * _f_2(i, j + q) * _g3(k + r) * _s23(i, j + q, k + r) +
            dhy4[0] * _f_2(i, j + q - 2) * _g3(k + r) * _s23(i, j + q - 2, k + r) +
            dhy4[1] * _f_2(i, j + q - 1) * _g3(k + r) * _s23(i, j + q - 1, k + r) +
            dhy4[3] * _f_2(i, j + q + 1) * _g3(k + r) * _s23(i, j + q + 1, k + r) +
            dx4[1] * _f_1(i, j + q) * _g3(k + r) * _s13(i, j + q, k + r) +
            dx4[0] * _f_1(i - 1, j + q) * _g3(k + r) * _s13(i - 1, j + q, k + r) +
            dx4[2] * _f_1(i + 1, j + q) * _g3(k + r) * _s13(i + 1, j + q, k + r) +
            dx4[3] * _f_1(i + 2, j + q) * _g3(k + r) * _s13(i + 2, j + q, k + r) +
            dz4[1] * _s33(i, j + q, k + r) + dz4[0] * _s33(i, j + q, k + r - 1) +
            dz4[2] * _s33(i, j + q, k + r + 1) + dz4[3] * _s33(i, j + q, k + r + 2) -
            _f1_c(i, j + q) *
                (dphz4[3] * _g(k + r) *
                     (px4[1] * _s13(i, j + q, k + r) + px4[0] * _s13(i - 1, j + q, k + r) +
                      px4[2] * _s13(i + 1, j + q, k + r) + px4[3] * _s13(i + 2, j + q, k + r)) +
                 dphz4[0] * _g(k + r - 3) *
                     (px4[1] * _s13(i, j + q, k + r - 3) +
                      px4[0] * _s13(i - 1, j + q, k + r - 3) +
                      px4[2] * _s13(i + 1, j + q, k + r - 3) +
                      px4[3] * _s13(i + 2, j + q, k + r - 3)) +
                 dphz4[1] * _g(k + r - 2) *
                     (px4[1] * _s13(i, j + q, k + r - 2) +
                      px4[0] * _s13(i - 1, j + q, k + r - 2) +
                      px4[2] * _s13(i + 1, j + q, k + r - 2) +
                      px4[3] * _s13(i + 2, j + q, k + r - 2)) +
                 dphz4[2] * _g(k + r - 1) *
                     (px4[1] * _s13(i, j + q, k + r - 1) +
                      px4[0] * _s13(i - 1, j + q, k + r - 1) +
                      px4[2] * _s13(i + 1, j + q, k + r - 1) +
                      px4[3] * _s13(i + 2, j + q, k + r - 1)) +
                 dphz4[4] * _g(k + r + 1) *
                     (px4[1] * _s13(i, j + q, k + r + 1) +
                      px4[0] * _s13(i - 1, j + q, k + r + 1) +
                      px4[2] * _s13(i + 1, j + q, k + r + 1) +
                      px4[3] * _s13(i + 2, j + q, k + r + 1)) +
                 dphz4[5] * _g(k + r + 2) *
                     (px4[1] * _s13(i, j + q, k + r + 2) +
                      px4[0] * _s13(i - 1, j + q, k + r + 2) +
                      px4[2] * _s13(i + 1, j + q, k + r + 2) +
                      px4[3] * _s13(i + 2, j + q, k + r + 2)) +
                 dphz4[6] * _g(k + r + 3) *
                     (px4[1] * _s13(i, j + q, k + r + 3) +
                      px4[0] * _s13(i - 1, j + q, k + r + 3) +
                      px4[2] * _s13(i + 1, j + q, k + r + 3) +
                      px4[3] * _s13(i + 2, j + q, k + r + 3))) -
            _f2_c(i, j + q) *
                (dphz4[3] * _g(k + r) *
                     (phy4[2] * _s23(i, j + q, k + r) + phy4[0] * _s23(i, j + q - 2, k + r) +
                      phy4[1] * _s23(i, j + q - 1, k + r) +
                      phy4[3] * _s23(i, j + q + 1, k + r)) +
                 dphz4[0] * _g(k + r - 3) *
                     (phy4[2] * _s23(i, j + q, k + r - 3) +
                      phy4[0] * _s23(i, j + q - 2, k + r - 3) +
                      phy4[1] * _s23(i, j + q - 1, k + r - 3) +
                      phy4[3] * _s23(i, j + q + 1, k + r - 3)) +
                 dphz4[1] * _g(k + r - 2) *
                     (phy4[2] * _s23(i, j + q, k + r - 2) +
                      phy4[0] * _s23(i, j + q - 2, k + r - 2) +
                      phy4[1] * _s23(i, j + q - 1, k + r - 2) +
                      phy4[3] * _s23(i, j + q + 1, k + r - 2)) +
                 dphz4[2] * _g(k + r - 1) *
                     (phy4[2] * _s23(i, j + q, k + r - 1) +
                      phy4[0] * _s23(i, j + q - 2, k + r - 1) +
                      phy4[1] * _s23(i, j + q - 1, k + r - 1) +
                      phy4[3] * _s23(i, j + q + 1, k + r - 1)) +
                 dphz4[4] * _g(k + r + 1) *
                     (phy4[2] * _s23(i, j + q, k + r + 1) +
                      phy4[0] * _s23(i, j + q - 2, k + r + 1) +
                      phy4[1] * _s23(i, j + q - 1, k + r + 1) +
                      phy4[3] * _s23(i, j + q + 1, k + r + 1)) +
                 dphz4[5] * _g(k + r + 2) *
                     (phy4[2] * _s23(i, j + q, k + r + 2) +
                      phy4[0] * _s23(i, j + q - 2, k + r + 2) +
                      phy4[1] * _s23(i, j + q - 1, k + r + 2) +
                      phy4[3] * _s23(i, j + q + 1, k + r + 2)) +
                 dphz4[6] * _g(k + r + 3) *
                     (phy4[2] * _s23(i, j + q, k + r + 3) +
                      phy4[0] * _s23(i, j + q - 2, k + r + 3) +
                      phy4[1] * _s23(i, j + q - 1, k + r + 3) +
                      phy4[3] * _s23(i, j + q + 1, k + r + 3))))) *
      f_dcrj;
        }
        }

#pragma unroll
        for (int q = 0; q < nq; ++q) {
#pragma unroll
        for (int r = 0; r < nr; ++r) {
                if (k + r < dm_offset) continue;
                if (k + r >= nz - 6) continue;
                if (j + q >= ej) return;
                _u1(i, j + q, k + r) = v1[q][r];
                _u2(i, j + q, k + r) = v2[q][r];
                _u3(i, j + q, k + r) = v3[q][r];
        }
        }
        
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


#undef RSTRCT


