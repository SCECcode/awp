template <int np, int nq, int nr>
#if sm_61
__launch_bounds__ (256)
#else
__launch_bounds__ (128)
#endif
__global__ void dtopo_vel_111_split1(
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
        const int k = nr * threadIdx.x + nr * blockIdx.x * blockDim.x;
        if (k >= nz - 12) return;
        const int j = nq * threadIdx.y + nq * blockIdx.y * blockDim.y + bj;
        if (j >= ny) return;
        if (j >= ej) return;
        const int i = np * threadIdx.z + np * blockIdx.z * blockDim.z + bi;
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


        float v1[np][nq][nr];
        float v2[np][nq][nr];
#pragma unroll
        for (int p = 0; p < np; ++p) {
#pragma unroll
        for (int q = 0; q < nq; ++q) {
#pragma unroll
        for (int r = 0; r < nr; ++r) {
        float c = 0.25f;
        float rho1 = c * (_rho(i + p, j + q, k + r + 5) + _rho(i + p, j + q - 1, k + r + 5)) +
                     c * (_rho(i + p, j + q, k + r + 6) + _rho(i + p, j + q - 1, k + r + 6));
        float rho2 = c * (_rho(i + p, j + q, k + r + 5) + _rho(i + p - 1, j + q, k + r + 5)) +
                     c * (_rho(i + p, j + q, k + r + 6) + _rho(i + p - 1, j + q, k + r + 6));
        float rho3 = c * (_rho(i + p, j + q, k + r + 6) + _rho(i + p - 1, j + q, k + r + 6)) +
                     c * (_rho(i + p, j + q - 1, k + r + 6) + _rho(i + p - 1, j + q - 1, k + r + 6));

        float Ai1 = _f_1(i + p, j + q) * _g3_c(k + r + 6) * rho1;
        Ai1 = nu * 1.0 / Ai1;
        float Ai2 = _f_2(i + p, j + q) * _g3_c(k + r + 6) * rho2;
        Ai2 = nu * 1.0 / Ai2;
        float Ai3 = _f_c(i + p, j + q) * _g3(k + r + 6) * rho3;
        Ai3 = nu * 1.0 / Ai3;
        float f_dcrj = _dcrjx(i + p) * _dcrjy(j + q) * _dcrjz(k + r + 6);
        v1[p][q][r] =
            (a * _u1(i + p, j + q, k + r + 6) +
             Ai1 *
                 (dhx[2] * _f_c(i + p, j + q) * _g3_c(k + r + 6) * _s11(i + p, j + q, k + r + 6) +
                  dhx[0] * _f_c(i + p - 2, j + q) * _g3_c(k + r + 6) *
                      _s11(i + p - 2, j + q, k + r + 6) +
                  dhx[1] * _f_c(i + p - 1, j + q) * _g3_c(k + r + 6) *
                      _s11(i + p - 1, j + q, k + r + 6) +
                  dhx[3] * _f_c(i + p + 1, j + q) * _g3_c(k + r + 6) *
                      _s11(i + p + 1, j + q, k + r + 6) +
                  dhy[2] * _f(i + p, j + q) * _g3_c(k + r + 6) * _s12(i + p, j + q, k + r + 6) +
                  dhy[0] * _f(i + p, j + q - 2) * _g3_c(k + r + 6) * _s12(i + p, j + q - 2, k + r + 6) +
                  dhy[1] * _f(i + p, j + q - 1) * _g3_c(k + r + 6) * _s12(i + p, j + q - 1, k + r + 6) +
                  dhy[3] * _f(i + p, j + q + 1) * _g3_c(k + r + 6) * _s12(i + p, j + q + 1, k + r + 6) +
                  dhz[0] * _s13(i + p, j + q, k + r + 4) + dhz[1] * _s13(i + p, j + q, k + r + 5) +
                  dhz[2] * _s13(i + p, j + q, k + r + 6) + dhz[3] * _s13(i + p, j + q, k + r + 7) -
                  _f1_1(i + p, j + q) * (dhpz[0] * _g_c(k + r + 3) *
                                     (phx[2] * _s11(i + p, j + q, k + r + 3) +
                                      phx[0] * _s11(i + p - 2, j + q, k + r + 3) +
                                      phx[1] * _s11(i + p - 1, j + q, k + r + 3) +
                                      phx[3] * _s11(i + p + 1, j + q, k + r + 3)) +
                                 dhpz[1] * _g_c(k + r + 4) *
                                     (phx[2] * _s11(i + p, j + q, k + r + 4) +
                                      phx[0] * _s11(i + p - 2, j + q, k + r + 4) +
                                      phx[1] * _s11(i + p - 1, j + q, k + r + 4) +
                                      phx[3] * _s11(i + p + 1, j + q, k + r + 4)) +
                                 dhpz[2] * _g_c(k + r + 5) *
                                     (phx[2] * _s11(i + p, j + q, k + r + 5) +
                                      phx[0] * _s11(i + p - 2, j + q, k + r + 5) +
                                      phx[1] * _s11(i + p - 1, j + q, k + r + 5) +
                                      phx[3] * _s11(i + p + 1, j + q, k + r + 5)) +
                                 dhpz[3] * _g_c(k + r + 6) *
                                     (phx[2] * _s11(i + p, j + q, k + r + 6) +
                                      phx[0] * _s11(i + p - 2, j + q, k + r + 6) +
                                      phx[1] * _s11(i + p - 1, j + q, k + r + 6) +
                                      phx[3] * _s11(i + p + 1, j + q, k + r + 6)) +
                                 dhpz[4] * _g_c(k + r + 7) *
                                     (phx[2] * _s11(i + p, j + q, k + r + 7) +
                                      phx[0] * _s11(i + p - 2, j + q, k + r + 7) +
                                      phx[1] * _s11(i + p - 1, j + q, k + r + 7) +
                                      phx[3] * _s11(i + p + 1, j + q, k + r + 7)) +
                                 dhpz[5] * _g_c(k + r + 8) *
                                     (phx[2] * _s11(i + p, j + q, k + r + 8) +
                                      phx[0] * _s11(i + p - 2, j + q, k + r + 8) +
                                      phx[1] * _s11(i + p - 1, j + q, k + r + 8) +
                                      phx[3] * _s11(i + p + 1, j + q, k + r + 8)) +
                                 dhpz[6] * _g_c(k + r + 9) *
                                     (phx[2] * _s11(i + p, j + q, k + r + 9) +
                                      phx[0] * _s11(i + p - 2, j + q, k + r + 9) +
                                      phx[1] * _s11(i + p - 1, j + q, k + r + 9) +
                                      phx[3] * _s11(i + p + 1, j + q, k + r + 9))) -
                  _f2_1(i + p, j + q) * (dhpz[0] * _g_c(k + r + 3) *
                                     (phy[2] * _s12(i + p, j + q, k + r + 3) +
                                      phy[0] * _s12(i + p, j + q - 2, k + r + 3) +
                                      phy[1] * _s12(i + p, j + q - 1, k + r + 3) +
                                      phy[3] * _s12(i + p, j + q + 1, k + r + 3)) +
                                 dhpz[1] * _g_c(k + r + 4) *
                                     (phy[2] * _s12(i + p, j + q, k + r + 4) +
                                      phy[0] * _s12(i + p, j + q - 2, k + r + 4) +
                                      phy[1] * _s12(i + p, j + q - 1, k + r + 4) +
                                      phy[3] * _s12(i + p, j + q + 1, k + r + 4)) +
                                 dhpz[2] * _g_c(k + r + 5) *
                                     (phy[2] * _s12(i + p, j + q, k + r + 5) +
                                      phy[0] * _s12(i + p, j + q - 2, k + r + 5) +
                                      phy[1] * _s12(i + p, j + q - 1, k + r + 5) +
                                      phy[3] * _s12(i + p, j + q + 1, k + r + 5)) +
                                 dhpz[3] * _g_c(k + r + 6) *
                                     (phy[2] * _s12(i + p, j + q, k + r + 6) +
                                      phy[0] * _s12(i + p, j + q - 2, k + r + 6) +
                                      phy[1] * _s12(i + p, j + q - 1, k + r + 6) +
                                      phy[3] * _s12(i + p, j + q + 1, k + r + 6)) +
                                 dhpz[4] * _g_c(k + r + 7) *
                                     (phy[2] * _s12(i + p, j + q, k + r + 7) +
                                      phy[0] * _s12(i + p, j + q - 2, k + r + 7) +
                                      phy[1] * _s12(i + p, j + q - 1, k + r + 7) +
                                      phy[3] * _s12(i + p, j + q + 1, k + r + 7)) +
                                 dhpz[5] * _g_c(k + r + 8) *
                                     (phy[2] * _s12(i + p, j + q, k + r + 8) +
                                      phy[0] * _s12(i + p, j + q - 2, k + r + 8) +
                                      phy[1] * _s12(i + p, j + q - 1, k + r + 8) +
                                      phy[3] * _s12(i + p, j + q + 1, k + r + 8)) +
                                 dhpz[6] * _g_c(k + r + 9) *
                                     (phy[2] * _s12(i + p, j + q, k + r + 9) +
                                      phy[0] * _s12(i + p, j + q - 2, k + r + 9) +
                                      phy[1] * _s12(i + p, j + q - 1, k + r + 9) +
                                      phy[3] * _s12(i + p, j + q + 1, k + r + 9))))) *
            f_dcrj;
        v2[p][q][r] =
            (a * _u2(i + p, j + q, k + r + 6) +
             Ai2 *
                 (dhz[0] * _s23(i + p, j + q, k + r + 4) + dhz[1] * _s23(i + p, j + q, k + r + 5) +
                  dhz[2] * _s23(i + p, j + q, k + r + 6) + dhz[3] * _s23(i + p, j + q, k + r + 7) +
                  dx[1] * _f(i + p, j + q) * _g3_c(k + r + 6) * _s12(i + p, j + q, k + r + 6) +
                  dx[0] * _f(i + p - 1, j + q) * _g3_c(k + r + 6) * _s12(i + p - 1, j + q, k + r + 6) +
                  dx[2] * _f(i + p + 1, j + q) * _g3_c(k + r + 6) * _s12(i + p + 1, j + q, k + r + 6) +
                  dx[3] * _f(i + p + 2, j + q) * _g3_c(k + r + 6) * _s12(i + p + 2, j + q, k + r + 6) +
                  dy[1] * _f_c(i + p, j + q) * _g3_c(k + r + 6) * _s22(i + p, j + q, k + r + 6) +
                  dy[0] * _f_c(i + p, j + q - 1) * _g3_c(k + r + 6) *
                      _s22(i + p, j + q - 1, k + r + 6) +
                  dy[2] * _f_c(i + p, j + q + 1) * _g3_c(k + r + 6) *
                      _s22(i + p, j + q + 1, k + r + 6) +
                  dy[3] * _f_c(i + p, j + q + 2) * _g3_c(k + r + 6) *
                      _s22(i + p, j + q + 2, k + r + 6) -
                  _f1_2(i + p, j + q) * (dhpz[0] * _g_c(k + r + 3) *
                                     (px[1] * _s12(i + p, j + q, k + r + 3) +
                                      px[0] * _s12(i + p - 1, j + q, k + r + 3) +
                                      px[2] * _s12(i + p + 1, j + q, k + r + 3) +
                                      px[3] * _s12(i + p + 2, j + q, k + r + 3)) +
                                 dhpz[1] * _g_c(k + r + 4) *
                                     (px[1] * _s12(i + p, j + q, k + r + 4) +
                                      px[0] * _s12(i + p - 1, j + q, k + r + 4) +
                                      px[2] * _s12(i + p + 1, j + q, k + r + 4) +
                                      px[3] * _s12(i + p + 2, j + q, k + r + 4)) +
                                 dhpz[2] * _g_c(k + r + 5) *
                                     (px[1] * _s12(i + p, j + q, k + r + 5) +
                                      px[0] * _s12(i + p - 1, j + q, k + r + 5) +
                                      px[2] * _s12(i + p + 1, j + q, k + r + 5) +
                                      px[3] * _s12(i + p + 2, j + q, k + r + 5)) +
                                 dhpz[3] * _g_c(k + r + 6) *
                                     (px[1] * _s12(i + p, j + q, k + r + 6) +
                                      px[0] * _s12(i + p - 1, j + q, k + r + 6) +
                                      px[2] * _s12(i + p + 1, j + q, k + r + 6) +
                                      px[3] * _s12(i + p + 2, j + q, k + r + 6)) +
                                 dhpz[4] * _g_c(k + r + 7) *
                                     (px[1] * _s12(i + p, j + q, k + r + 7) +
                                      px[0] * _s12(i + p - 1, j + q, k + r + 7) +
                                      px[2] * _s12(i + p + 1, j + q, k + r + 7) +
                                      px[3] * _s12(i + p + 2, j + q, k + r + 7)) +
                                 dhpz[5] * _g_c(k + r + 8) *
                                     (px[1] * _s12(i + p, j + q, k + r + 8) +
                                      px[0] * _s12(i + p - 1, j + q, k + r + 8) +
                                      px[2] * _s12(i + p + 1, j + q, k + r + 8) +
                                      px[3] * _s12(i + p + 2, j + q, k + r + 8)) +
                                 dhpz[6] * _g_c(k + r + 9) *
                                     (px[1] * _s12(i + p, j + q, k + r + 9) +
                                      px[0] * _s12(i + p - 1, j + q, k + r + 9) +
                                      px[2] * _s12(i + p + 1, j + q, k + r + 9) +
                                      px[3] * _s12(i + p + 2, j + q, k + r + 9))) -
                  _f2_2(i + p, j + q) * (dhpz[0] * _g_c(k + r + 3) *
                                     (py[1] * _s22(i + p, j + q, k + r + 3) +
                                      py[0] * _s22(i + p, j + q - 1, k + r + 3) +
                                      py[2] * _s22(i + p, j + q + 1, k + r + 3) +
                                      py[3] * _s22(i + p, j + q + 2, k + r + 3)) +
                                 dhpz[1] * _g_c(k + r + 4) *
                                     (py[1] * _s22(i + p, j + q, k + r + 4) +
                                      py[0] * _s22(i + p, j + q - 1, k + r + 4) +
                                      py[2] * _s22(i + p, j + q + 1, k + r + 4) +
                                      py[3] * _s22(i + p, j + q + 2, k + r + 4)) +
                                 dhpz[2] * _g_c(k + r + 5) *
                                     (py[1] * _s22(i + p, j + q, k + r + 5) +
                                      py[0] * _s22(i + p, j + q - 1, k + r + 5) +
                                      py[2] * _s22(i + p, j + q + 1, k + r + 5) +
                                      py[3] * _s22(i + p, j + q + 2, k + r + 5)) +
                                 dhpz[3] * _g_c(k + r + 6) *
                                     (py[1] * _s22(i + p, j + q, k + r + 6) +
                                      py[0] * _s22(i + p, j + q - 1, k + r + 6) +
                                      py[2] * _s22(i + p, j + q + 1, k + r + 6) +
                                      py[3] * _s22(i + p, j + q + 2, k + r + 6)) +
                                 dhpz[4] * _g_c(k + r + 7) *
                                     (py[1] * _s22(i + p, j + q, k + r + 7) +
                                      py[0] * _s22(i + p, j + q - 1, k + r + 7) +
                                      py[2] * _s22(i + p, j + q + 1, k + r + 7) +
                                      py[3] * _s22(i + p, j + q + 2, k + r + 7)) +
                                 dhpz[5] * _g_c(k + r + 8) *
                                     (py[1] * _s22(i + p, j + q, k + r + 8) +
                                      py[0] * _s22(i + p, j + q - 1, k + r + 8) +
                                      py[2] * _s22(i + p, j + q + 1, k + r + 8) +
                                      py[3] * _s22(i + p, j + q + 2, k + r + 8)) +
                                 dhpz[6] * _g_c(k + r + 9) *
                                     (py[1] * _s22(i + p, j + q, k + r + 9) +
                                      py[0] * _s22(i + p, j + q - 1, k + r + 9) +
                                      py[2] * _s22(i + p, j + q + 1, k + r + 9) +
                                      py[3] * _s22(i + p, j + q + 2, k + r + 9))))) *
            f_dcrj;
        }
        }
        }

#pragma unroll
        for (int p = 0; p < np; ++p) {
#pragma unroll
        for (int q = 0; q < nq; ++q) {
#pragma unroll
        for (int r = 0; r < nr; ++r) {
                if (k + r  >= nz - 12) continue;
                if (j + q >= ej) return;
                if (i >= ei) return;
                _u1(i + p, j + q, k + r + 6) = v1[p][q][r];
                _u2(i + p, j + q, k + r + 6) = v2[p][q][r];
        }
        }
        }
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

template <int np, int nq, int nr>
#if sm_61
__launch_bounds__ (256)
#else
__launch_bounds__ (128)
#endif
__global__ void dtopo_vel_111_split2(
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
        const int k = nr * threadIdx.x + nr * blockIdx.x * blockDim.x;
        if (k >= nz - 12) return;
        const int j = nq * threadIdx.y + nq * blockIdx.y * blockDim.y + bj;
        if (j >= ny) return;
        if (j >= ej) return;
        const int i = np * threadIdx.z + np * blockIdx.z * blockDim.z + bi;
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


        float v3[np][nq][nr];
#pragma unroll
        for (int p = 0; p < np; ++p) {
#pragma unroll
        for (int q = 0; q < nq; ++q) {
#pragma unroll
        for (int r = 0; r < nr; ++r) {
        float c = 0.25f;
        float rho3 = c * (_rho(i + p, j + q, k + r + 6) + _rho(i + p - 1, j + q, k + r + 6)) +
                     c * (_rho(i + p, j + q - 1, k + r + 6) + _rho(i + p - 1, j + q - 1, k + r + 6));

        float Ai3 = _f_c(i + p, j + q) * _g3(k + r + 6) * rho3;
        Ai3 = nu * 1.0 / Ai3;
        float f_dcrj = _dcrjx(i + p) * _dcrjy(j + q) * _dcrjz(k + r + 6);
        v3[p][q][r] =
            (a * _u3(i + p, j + q, k + r + 6) +
             Ai3 *
                 (dhy[2] * _f_2(i + p, j + q) * _g3(k + r + 6) * _s23(i + p, j + q, k + r + 6) +
                  dhy[0] * _f_2(i + p, j + q - 2) * _g3(k + r + 6) * _s23(i + p, j + q - 2, k + r + 6) +
                  dhy[1] * _f_2(i + p, j + q - 1) * _g3(k + r + 6) * _s23(i + p, j + q - 1, k + r + 6) +
                  dhy[3] * _f_2(i + p, j + q + 1) * _g3(k + r + 6) * _s23(i + p, j + q + 1, k + r + 6) +
                  dx[1] * _f_1(i + p, j + q) * _g3(k + r + 6) * _s13(i + p, j + q, k + r + 6) +
                  dx[0] * _f_1(i + p - 1, j + q) * _g3(k + r + 6) * _s13(i + p - 1, j + q, k + r + 6) +
                  dx[2] * _f_1(i + p + 1, j + q) * _g3(k + r + 6) * _s13(i + p + 1, j + q, k + r + 6) +
                  dx[3] * _f_1(i + p + 2, j + q) * _g3(k + r + 6) * _s13(i + p + 2, j + q, k + r + 6) +
                  dz[0] * _s33(i + p, j + q, k + r + 5) + dz[1] * _s33(i + p, j + q, k + r + 6) +
                  dz[2] * _s33(i + p, j + q, k + r + 7) + dz[3] * _s33(i + p, j + q, k + r + 8) -
                  _f1_c(i + p, j + q) * (dphz[0] * _g(k + r + 3) *
                                     (px[1] * _s13(i + p, j + q, k + r + 3) +
                                      px[0] * _s13(i + p - 1, j + q, k + r + 3) +
                                      px[2] * _s13(i + p + 1, j + q, k + r + 3) +
                                      px[3] * _s13(i + p + 2, j + q, k + r + 3)) +
                                 dphz[1] * _g(k + r + 4) *
                                     (px[1] * _s13(i + p, j + q, k + r + 4) +
                                      px[0] * _s13(i + p - 1, j + q, k + r + 4) +
                                      px[2] * _s13(i + p + 1, j + q, k + r + 4) +
                                      px[3] * _s13(i + p + 2, j + q, k + r + 4)) +
                                 dphz[2] * _g(k + r + 5) *
                                     (px[1] * _s13(i + p, j + q, k + r + 5) +
                                      px[0] * _s13(i + p - 1, j + q, k + r + 5) +
                                      px[2] * _s13(i + p + 1, j + q, k + r + 5) +
                                      px[3] * _s13(i + p + 2, j + q, k + r + 5)) +
                                 dphz[3] * _g(k + r + 6) *
                                     (px[1] * _s13(i + p, j + q, k + r + 6) +
                                      px[0] * _s13(i + p - 1, j + q, k + r + 6) +
                                      px[2] * _s13(i + p + 1, j + q, k + r + 6) +
                                      px[3] * _s13(i + p + 2, j + q, k + r + 6)) +
                                 dphz[4] * _g(k + r + 7) *
                                     (px[1] * _s13(i + p, j + q, k + r + 7) +
                                      px[0] * _s13(i + p - 1, j + q, k + r + 7) +
                                      px[2] * _s13(i + p + 1, j + q, k + r + 7) +
                                      px[3] * _s13(i + p + 2, j + q, k + r + 7)) +
                                 dphz[5] * _g(k + r + 8) *
                                     (px[1] * _s13(i + p, j + q, k + r + 8) +
                                      px[0] * _s13(i + p - 1, j + q, k + r + 8) +
                                      px[2] * _s13(i + p + 1, j + q, k + r + 8) +
                                      px[3] * _s13(i + p + 2, j + q, k + r + 8)) +
                                 dphz[6] * _g(k + r + 9) *
                                     (px[1] * _s13(i + p, j + q, k + r + 9) +
                                      px[0] * _s13(i + p - 1, j + q, k + r + 9) +
                                      px[2] * _s13(i + p + 1, j + q, k + r + 9) +
                                      px[3] * _s13(i + p + 2, j + q, k + r + 9))) -
                  _f2_c(i + p, j + q) * (dphz[0] * _g(k + r + 3) *
                                     (phy[2] * _s23(i + p, j + q, k + r + 3) +
                                      phy[0] * _s23(i + p, j + q - 2, k + r + 3) +
                                      phy[1] * _s23(i + p, j + q - 1, k + r + 3) +
                                      phy[3] * _s23(i + p, j + q + 1, k + r + 3)) +
                                 dphz[1] * _g(k + r + 4) *
                                     (phy[2] * _s23(i + p, j + q, k + r + 4) +
                                      phy[0] * _s23(i + p, j + q - 2, k + r + 4) +
                                      phy[1] * _s23(i + p, j + q - 1, k + r + 4) +
                                      phy[3] * _s23(i + p, j + q + 1, k + r + 4)) +
                                 dphz[2] * _g(k + r + 5) *
                                     (phy[2] * _s23(i + p, j + q, k + r + 5) +
                                      phy[0] * _s23(i + p, j + q - 2, k + r + 5) +
                                      phy[1] * _s23(i + p, j + q - 1, k + r + 5) +
                                      phy[3] * _s23(i + p, j + q + 1, k + r + 5)) +
                                 dphz[3] * _g(k + r + 6) *
                                     (phy[2] * _s23(i + p, j + q, k + r + 6) +
                                      phy[0] * _s23(i + p, j + q - 2, k + r + 6) +
                                      phy[1] * _s23(i + p, j + q - 1, k + r + 6) +
                                      phy[3] * _s23(i + p, j + q + 1, k + r + 6)) +
                                 dphz[4] * _g(k + r + 7) *
                                     (phy[2] * _s23(i + p, j + q, k + r + 7) +
                                      phy[0] * _s23(i + p, j + q - 2, k + r + 7) +
                                      phy[1] * _s23(i + p, j + q - 1, k + r + 7) +
                                      phy[3] * _s23(i + p, j + q + 1, k + r + 7)) +
                                 dphz[5] * _g(k + r + 8) *
                                     (phy[2] * _s23(i + p, j + q, k + r + 8) +
                                      phy[0] * _s23(i + p, j + q - 2, k + r + 8) +
                                      phy[1] * _s23(i + p, j + q - 1, k + r + 8) +
                                      phy[3] * _s23(i + p, j + q + 1, k + r + 8)) +
                                 dphz[6] * _g(k + r + 9) *
                                     (phy[2] * _s23(i + p, j + q, k + r + 9) +
                                      phy[0] * _s23(i + p, j + q - 2, k + r + 9) +
                                      phy[1] * _s23(i + p, j + q - 1, k + r + 9) +
                                      phy[3] * _s23(i + p, j + q + 1, k + r + 9))))) *
            f_dcrj;
        }
        }
        }

#pragma unroll
        for (int p = 0; p < np; ++p) {
#pragma unroll
        for (int q = 0; q < nq; ++q) {
#pragma unroll
        for (int r = 0; r < nr; ++r) {
                if (k + r  >= nz - 12) continue;
                if (j + q >= ej) return;
                if (i >= ei) return;
                _u3(i + p, j + q, k + r + 6) = v3[p][q][r];
        }
        }
        }
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


