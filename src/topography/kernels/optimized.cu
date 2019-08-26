#include <topography/kernels/optimized.cuh>

__global__ void
dtopo_vel_110(float *__restrict__ u1, float *__restrict__ u2,
              float *__restrict__ u3, const float *__restrict__ dcrjx,
              const float *__restrict__ dcrjy, const float *__restrict__ dcrjz,
              const float *__restrict__ f, const float *__restrict__ f1_1,
              const float *__restrict__ f1_2, const float *__restrict__ f1_c,
              const float *__restrict__ f2_1, const float *__restrict__ f2_2,
              const float *__restrict__ f2_c, const float *__restrict__ f_1,
              const float *__restrict__ f_2, const float *__restrict__ f_c,
              const float *__restrict__ g, const float *__restrict__ g3,
              const float *__restrict__ g3_c, const float *__restrict__ g_c,
              const float *__restrict__ rho, const float *__restrict__ s11,
              const float *__restrict__ s12, const float *__restrict__ s13,
              const float *__restrict__ s22, const float *__restrict__ s23,
              const float *__restrict__ s33, const float a, const float nu,
              const int nx, const int ny, const int nz, const int bi,
              const int bj, const int ei, const int ej) {
  const float ph2l[6][7] = {
      {1.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.5000000000000000, 0.5000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.5000000000000000, 0.5000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.5000000000000000,
       0.5000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.5000000000000000, 0.5000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.5000000000000000, 0.5000000000000000,
       0.0000000000000000}};
  const float ph2[2] = {0.5000000000000000, 0.5000000000000000};
  const float dhp4l[6][9] = {
      {-1.4276800979942257, 0.2875185051606178, 2.0072491465276454,
       -0.8773816261307504, 0.0075022330101095, 0.0027918394266035,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {-0.8139439685257414, -0.1273679143938725, 1.1932750007455708,
       -0.1475120181828087, -0.1125814499297686, 0.0081303502866204,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {-0.1639182541610305, -0.3113839909089031, 0.0536007135209480,
       0.3910958927076031, 0.0401741813821989, -0.0095685425408165,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {-0.0171478318814576, 0.0916600077207278, -0.7187220404622644,
       0.1434031863528334, 0.5827389738506837, -0.0847863081664324,
       0.0028540125859095, 0.0000000000000000, 0.0000000000000000},
      {0.0579176640853654, -0.0022069616616207, -0.0108792602269819,
       -0.6803612607837533, 0.0530169938441241, 0.6736586580761996,
       -0.0937500000000000, 0.0026041666666667, 0.0000000000000000},
      {-0.0020323834153791, -0.0002106933140862, 0.0013351454085978,
       0.0938400881871787, -0.6816971139746001, 0.0002232904416222,
       0.6796875000000000, -0.0937500000000000, 0.0026041666666667}};
  const float ph4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float dh4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float dh4l[6][7] = {
      {-1.4511412472637157, 1.8534237417911470, -0.3534237417911469,
       -0.0488587527362844, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {-0.8577143189081458, 0.5731429567244373, 0.4268570432755628,
       -0.1422856810918542, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {-0.1674548505882877, -0.4976354482351368, 0.4976354482351368,
       0.1674548505882877, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.1027061113405124, -0.2624541326469860, -0.8288742701021167,
       1.0342864927831414, -0.0456642013745513, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0416666666666667,
       -1.1250000000000000, 1.1250000000000000, -0.0416666666666667,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0416666666666667, -1.1250000000000000, 1.1250000000000000,
       -0.0416666666666667}};
  const float p4[4] = {-0.0625000000000000, 0.5625000000000000,
                       0.5625000000000000, -0.0625000000000000};
  const float d4[4] = {0.0416666666666667, -1.1250000000000000,
                       1.1250000000000000, -0.0416666666666667};
  const float dph4l[6][9] = {
      {-1.3764648947859957, 1.8523239861274134, -0.5524268681758195,
       0.0537413571133823, 0.0228264197210198, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {-0.4428256655817484, 0.0574614517751293, 0.2022259589759502,
       0.1944663890497050, -0.0113281342190362, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.3360140866060757, -1.2113298407847195, 0.3111668377093505,
       0.6714462506479003, -0.1111440843153523, 0.0038467501367455,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {-0.0338560531369653, 0.0409943223643901, -0.5284757132923059,
       -0.0115571196122084, 0.6162252315536445, -0.0857115441015996,
       0.0023808762250444, 0.0000000000000000, 0.0000000000000000},
      {0.0040378273193044, -0.0064139372778371, 0.0890062133451850,
       -0.6749219241340761, -0.0002498459192428, 0.6796875000000000,
       -0.0937500000000000, 0.0026041666666667, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, -0.0026041666666667,
       0.0937500000000000, -0.6796875000000000, 0.0000000000000000,
       0.6796875000000000, -0.0937500000000000, 0.0026041666666667}};
  const float d4l[6][8] = {
      {-1.7779989465546748, 1.3337480247900155, 0.7775013168066564,
       -0.3332503950419969, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {-0.4410217341392059, -0.1730842484889890, 0.4487228323259926,
       0.1653831503022022, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.1798793213882701, -0.2757257254150788, -0.9597948548284453,
       1.1171892610431817, -0.0615480021879277, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0153911381507088, 0.0568851455503591, -0.1998976464597171,
       -0.8628231468598346, 1.0285385292191949, -0.0380940196007109,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0416666666666667, -1.1250000000000000, 1.1250000000000000,
       -0.0416666666666667, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0416666666666667, -1.1250000000000000,
       1.1250000000000000, -0.0416666666666667}};
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
#define _rho(i, j, k)                                                          \
  rho[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f_1(i, j)                                                             \
  f_1[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3_c(k) g3_c[(k) + align]
#define _f_2(i, j)                                                             \
  f_2[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3(k) g3[(k) + align]
#define _f_c(i, j)                                                             \
  f_c[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _dcrjz(k) dcrjz[(k) + align]
#define _dcrjx(i) dcrjx[(i) + ngsl + 2]
#define _dcrjy(j) dcrjy[(j) + ngsl + 2]
#define _g_c(k) g_c[(k) + align]
#define _s11(i, j, k)                                                          \
  s11[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_1(i, j)                                                            \
  f2_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _u1(i, j, k)                                                           \
  u1[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_1(i, j)                                                            \
  f1_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s12(i, j, k)                                                          \
  s12[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f(i, j)                                                               \
  f[(j) + align + ngsl + ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s13(i, j, k)                                                          \
  s13[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_2(i, j)                                                            \
  f1_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s22(i, j, k)                                                          \
  s22[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s23(i, j, k)                                                          \
  s23[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _u2(i, j, k)                                                           \
  u2[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_2(i, j)                                                            \
  f2_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_c(i, j)                                                            \
  f2_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g(k) g[(k) + align]
#define _f1_c(i, j)                                                            \
  f1_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s33(i, j, k)                                                          \
  s33[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _u3(i, j, k)                                                           \
  u3[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
  float rho1 =
      ph2l[k][0] * (ph2[1] * _rho(i, j, 0) + ph2[0] * _rho(i, j - 1, 0)) +
      ph2l[k][1] * (ph2[1] * _rho(i, j, 1) + ph2[0] * _rho(i, j - 1, 1)) +
      ph2l[k][2] * (ph2[1] * _rho(i, j, 2) + ph2[0] * _rho(i, j - 1, 2)) +
      ph2l[k][3] * (ph2[1] * _rho(i, j, 3) + ph2[0] * _rho(i, j - 1, 3)) +
      ph2l[k][4] * (ph2[1] * _rho(i, j, 4) + ph2[0] * _rho(i, j - 1, 4)) +
      ph2l[k][5] * (ph2[1] * _rho(i, j, 5) + ph2[0] * _rho(i, j - 1, 5)) +
      ph2l[k][6] * (ph2[1] * _rho(i, j, 6) + ph2[0] * _rho(i, j - 1, 6));
  float rho2 =
      ph2l[k][0] * (ph2[1] * _rho(i, j, 0) + ph2[0] * _rho(i - 1, j, 0)) +
      ph2l[k][1] * (ph2[1] * _rho(i, j, 1) + ph2[0] * _rho(i - 1, j, 1)) +
      ph2l[k][2] * (ph2[1] * _rho(i, j, 2) + ph2[0] * _rho(i - 1, j, 2)) +
      ph2l[k][3] * (ph2[1] * _rho(i, j, 3) + ph2[0] * _rho(i - 1, j, 3)) +
      ph2l[k][4] * (ph2[1] * _rho(i, j, 4) + ph2[0] * _rho(i - 1, j, 4)) +
      ph2l[k][5] * (ph2[1] * _rho(i, j, 5) + ph2[0] * _rho(i - 1, j, 5)) +
      ph2l[k][6] * (ph2[1] * _rho(i, j, 6) + ph2[0] * _rho(i - 1, j, 6));
  float rho3 =
      ph2[1] * (ph2[1] * _rho(i, j, k) + ph2[0] * _rho(i - 1, j, k)) +
      ph2[0] * (ph2[1] * _rho(i, j - 1, k) + ph2[0] * _rho(i - 1, j - 1, k));
  float Ai1 = _f_1(i, j) * _g3_c(k) * rho1;
  Ai1 = nu * 1.0 / Ai1;
  float Ai2 = _f_2(i, j) * _g3_c(k) * rho2;
  Ai2 = nu * 1.0 / Ai2;
  float Ai3 = _f_c(i, j) * _g3(k) * rho3;
  Ai3 = nu * 1.0 / Ai3;
  float f_dcrj = _dcrjx(i) * _dcrjy(j) * _dcrjz(k);
  _u1(i, j, k) =
      (a * _u1(i, j, k) +
       Ai1 *
           (dh4[2] * _f(i, j) * _g3_c(k) * _s12(i, j, k) +
            dh4[0] * _f(i, j - 2) * _g3_c(k) * _s12(i, j - 2, k) +
            dh4[1] * _f(i, j - 1) * _g3_c(k) * _s12(i, j - 1, k) +
            dh4[3] * _f(i, j + 1) * _g3_c(k) * _s12(i, j + 1, k) +
            dh4[2] * _f_c(i, j) * _g3_c(k) * _s11(i, j, k) +
            dh4[0] * _f_c(i - 2, j) * _g3_c(k) * _s11(i - 2, j, k) +
            dh4[1] * _f_c(i - 1, j) * _g3_c(k) * _s11(i - 1, j, k) +
            dh4[3] * _f_c(i + 1, j) * _g3_c(k) * _s11(i + 1, j, k) +
            dh4l[k][0] * _s13(i, j, 0) + dh4l[k][1] * _s13(i, j, 1) +
            dh4l[k][2] * _s13(i, j, 2) + dh4l[k][3] * _s13(i, j, 3) +
            dh4l[k][4] * _s13(i, j, 4) + dh4l[k][5] * _s13(i, j, 5) +
            dh4l[k][6] * _s13(i, j, 6) -
            _f1_1(i, j) *
                (dhp4l[k][0] * _g_c(0) *
                     (ph4[2] * _s11(i, j, 0) + ph4[0] * _s11(i - 2, j, 0) +
                      ph4[1] * _s11(i - 1, j, 0) + ph4[3] * _s11(i + 1, j, 0)) +
                 dhp4l[k][1] * _g_c(1) *
                     (ph4[2] * _s11(i, j, 1) + ph4[0] * _s11(i - 2, j, 1) +
                      ph4[1] * _s11(i - 1, j, 1) + ph4[3] * _s11(i + 1, j, 1)) +
                 dhp4l[k][2] * _g_c(2) *
                     (ph4[2] * _s11(i, j, 2) + ph4[0] * _s11(i - 2, j, 2) +
                      ph4[1] * _s11(i - 1, j, 2) + ph4[3] * _s11(i + 1, j, 2)) +
                 dhp4l[k][3] * _g_c(3) *
                     (ph4[2] * _s11(i, j, 3) + ph4[0] * _s11(i - 2, j, 3) +
                      ph4[1] * _s11(i - 1, j, 3) + ph4[3] * _s11(i + 1, j, 3)) +
                 dhp4l[k][4] * _g_c(4) *
                     (ph4[2] * _s11(i, j, 4) + ph4[0] * _s11(i - 2, j, 4) +
                      ph4[1] * _s11(i - 1, j, 4) + ph4[3] * _s11(i + 1, j, 4)) +
                 dhp4l[k][5] * _g_c(5) *
                     (ph4[2] * _s11(i, j, 5) + ph4[0] * _s11(i - 2, j, 5) +
                      ph4[1] * _s11(i - 1, j, 5) + ph4[3] * _s11(i + 1, j, 5)) +
                 dhp4l[k][6] * _g_c(6) *
                     (ph4[2] * _s11(i, j, 6) + ph4[0] * _s11(i - 2, j, 6) +
                      ph4[1] * _s11(i - 1, j, 6) + ph4[3] * _s11(i + 1, j, 6)) +
                 dhp4l[k][7] * _g_c(7) *
                     (ph4[2] * _s11(i, j, 7) + ph4[0] * _s11(i - 2, j, 7) +
                      ph4[1] * _s11(i - 1, j, 7) + ph4[3] * _s11(i + 1, j, 7)) +
                 dhp4l[k][8] * _g_c(8) *
                     (ph4[2] * _s11(i, j, 8) + ph4[0] * _s11(i - 2, j, 8) +
                      ph4[1] * _s11(i - 1, j, 8) +
                      ph4[3] * _s11(i + 1, j, 8))) -
            _f2_1(i, j) *
                (dhp4l[k][0] * _g_c(0) *
                     (ph4[2] * _s12(i, j, 0) + ph4[0] * _s12(i, j - 2, 0) +
                      ph4[1] * _s12(i, j - 1, 0) + ph4[3] * _s12(i, j + 1, 0)) +
                 dhp4l[k][1] * _g_c(1) *
                     (ph4[2] * _s12(i, j, 1) + ph4[0] * _s12(i, j - 2, 1) +
                      ph4[1] * _s12(i, j - 1, 1) + ph4[3] * _s12(i, j + 1, 1)) +
                 dhp4l[k][2] * _g_c(2) *
                     (ph4[2] * _s12(i, j, 2) + ph4[0] * _s12(i, j - 2, 2) +
                      ph4[1] * _s12(i, j - 1, 2) + ph4[3] * _s12(i, j + 1, 2)) +
                 dhp4l[k][3] * _g_c(3) *
                     (ph4[2] * _s12(i, j, 3) + ph4[0] * _s12(i, j - 2, 3) +
                      ph4[1] * _s12(i, j - 1, 3) + ph4[3] * _s12(i, j + 1, 3)) +
                 dhp4l[k][4] * _g_c(4) *
                     (ph4[2] * _s12(i, j, 4) + ph4[0] * _s12(i, j - 2, 4) +
                      ph4[1] * _s12(i, j - 1, 4) + ph4[3] * _s12(i, j + 1, 4)) +
                 dhp4l[k][5] * _g_c(5) *
                     (ph4[2] * _s12(i, j, 5) + ph4[0] * _s12(i, j - 2, 5) +
                      ph4[1] * _s12(i, j - 1, 5) + ph4[3] * _s12(i, j + 1, 5)) +
                 dhp4l[k][6] * _g_c(6) *
                     (ph4[2] * _s12(i, j, 6) + ph4[0] * _s12(i, j - 2, 6) +
                      ph4[1] * _s12(i, j - 1, 6) + ph4[3] * _s12(i, j + 1, 6)) +
                 dhp4l[k][7] * _g_c(7) *
                     (ph4[2] * _s12(i, j, 7) + ph4[0] * _s12(i, j - 2, 7) +
                      ph4[1] * _s12(i, j - 1, 7) + ph4[3] * _s12(i, j + 1, 7)) +
                 dhp4l[k][8] * _g_c(8) *
                     (ph4[2] * _s12(i, j, 8) + ph4[0] * _s12(i, j - 2, 8) +
                      ph4[1] * _s12(i, j - 1, 8) +
                      ph4[3] * _s12(i, j + 1, 8))))) *
      f_dcrj;
  _u2(i, j, k) =
      (a * _u2(i, j, k) +
       Ai2 *
           (d4[1] * _f(i, j) * _g3_c(k) * _s12(i, j, k) +
            d4[0] * _f(i - 1, j) * _g3_c(k) * _s12(i - 1, j, k) +
            d4[2] * _f(i + 1, j) * _g3_c(k) * _s12(i + 1, j, k) +
            d4[3] * _f(i + 2, j) * _g3_c(k) * _s12(i + 2, j, k) +
            d4[1] * _f_c(i, j) * _g3_c(k) * _s22(i, j, k) +
            d4[0] * _f_c(i, j - 1) * _g3_c(k) * _s22(i, j - 1, k) +
            d4[2] * _f_c(i, j + 1) * _g3_c(k) * _s22(i, j + 1, k) +
            d4[3] * _f_c(i, j + 2) * _g3_c(k) * _s22(i, j + 2, k) +
            dh4l[k][0] * _s23(i, j, 0) + dh4l[k][1] * _s23(i, j, 1) +
            dh4l[k][2] * _s23(i, j, 2) + dh4l[k][3] * _s23(i, j, 3) +
            dh4l[k][4] * _s23(i, j, 4) + dh4l[k][5] * _s23(i, j, 5) +
            dh4l[k][6] * _s23(i, j, 6) -
            _f1_2(i, j) *
                (dhp4l[k][0] * _g_c(0) *
                     (p4[1] * _s12(i, j, 0) + p4[0] * _s12(i - 1, j, 0) +
                      p4[2] * _s12(i + 1, j, 0) + p4[3] * _s12(i + 2, j, 0)) +
                 dhp4l[k][1] * _g_c(1) *
                     (p4[1] * _s12(i, j, 1) + p4[0] * _s12(i - 1, j, 1) +
                      p4[2] * _s12(i + 1, j, 1) + p4[3] * _s12(i + 2, j, 1)) +
                 dhp4l[k][2] * _g_c(2) *
                     (p4[1] * _s12(i, j, 2) + p4[0] * _s12(i - 1, j, 2) +
                      p4[2] * _s12(i + 1, j, 2) + p4[3] * _s12(i + 2, j, 2)) +
                 dhp4l[k][3] * _g_c(3) *
                     (p4[1] * _s12(i, j, 3) + p4[0] * _s12(i - 1, j, 3) +
                      p4[2] * _s12(i + 1, j, 3) + p4[3] * _s12(i + 2, j, 3)) +
                 dhp4l[k][4] * _g_c(4) *
                     (p4[1] * _s12(i, j, 4) + p4[0] * _s12(i - 1, j, 4) +
                      p4[2] * _s12(i + 1, j, 4) + p4[3] * _s12(i + 2, j, 4)) +
                 dhp4l[k][5] * _g_c(5) *
                     (p4[1] * _s12(i, j, 5) + p4[0] * _s12(i - 1, j, 5) +
                      p4[2] * _s12(i + 1, j, 5) + p4[3] * _s12(i + 2, j, 5)) +
                 dhp4l[k][6] * _g_c(6) *
                     (p4[1] * _s12(i, j, 6) + p4[0] * _s12(i - 1, j, 6) +
                      p4[2] * _s12(i + 1, j, 6) + p4[3] * _s12(i + 2, j, 6)) +
                 dhp4l[k][7] * _g_c(7) *
                     (p4[1] * _s12(i, j, 7) + p4[0] * _s12(i - 1, j, 7) +
                      p4[2] * _s12(i + 1, j, 7) + p4[3] * _s12(i + 2, j, 7)) +
                 dhp4l[k][8] * _g_c(8) *
                     (p4[1] * _s12(i, j, 8) + p4[0] * _s12(i - 1, j, 8) +
                      p4[2] * _s12(i + 1, j, 8) + p4[3] * _s12(i + 2, j, 8))) -
            _f2_2(i, j) *
                (dhp4l[k][0] * _g_c(0) *
                     (p4[1] * _s22(i, j, 0) + p4[0] * _s22(i, j - 1, 0) +
                      p4[2] * _s22(i, j + 1, 0) + p4[3] * _s22(i, j + 2, 0)) +
                 dhp4l[k][1] * _g_c(1) *
                     (p4[1] * _s22(i, j, 1) + p4[0] * _s22(i, j - 1, 1) +
                      p4[2] * _s22(i, j + 1, 1) + p4[3] * _s22(i, j + 2, 1)) +
                 dhp4l[k][2] * _g_c(2) *
                     (p4[1] * _s22(i, j, 2) + p4[0] * _s22(i, j - 1, 2) +
                      p4[2] * _s22(i, j + 1, 2) + p4[3] * _s22(i, j + 2, 2)) +
                 dhp4l[k][3] * _g_c(3) *
                     (p4[1] * _s22(i, j, 3) + p4[0] * _s22(i, j - 1, 3) +
                      p4[2] * _s22(i, j + 1, 3) + p4[3] * _s22(i, j + 2, 3)) +
                 dhp4l[k][4] * _g_c(4) *
                     (p4[1] * _s22(i, j, 4) + p4[0] * _s22(i, j - 1, 4) +
                      p4[2] * _s22(i, j + 1, 4) + p4[3] * _s22(i, j + 2, 4)) +
                 dhp4l[k][5] * _g_c(5) *
                     (p4[1] * _s22(i, j, 5) + p4[0] * _s22(i, j - 1, 5) +
                      p4[2] * _s22(i, j + 1, 5) + p4[3] * _s22(i, j + 2, 5)) +
                 dhp4l[k][6] * _g_c(6) *
                     (p4[1] * _s22(i, j, 6) + p4[0] * _s22(i, j - 1, 6) +
                      p4[2] * _s22(i, j + 1, 6) + p4[3] * _s22(i, j + 2, 6)) +
                 dhp4l[k][7] * _g_c(7) *
                     (p4[1] * _s22(i, j, 7) + p4[0] * _s22(i, j - 1, 7) +
                      p4[2] * _s22(i, j + 1, 7) + p4[3] * _s22(i, j + 2, 7)) +
                 dhp4l[k][8] * _g_c(8) *
                     (p4[1] * _s22(i, j, 8) + p4[0] * _s22(i, j - 1, 8) +
                      p4[2] * _s22(i, j + 1, 8) +
                      p4[3] * _s22(i, j + 2, 8))))) *
      f_dcrj;
  _u3(i, j, k) =
      (a * _u3(i, j, k) +
       Ai3 *
           (d4[1] * _f_1(i, j) * _g3(k) * _s13(i, j, k) +
            d4[0] * _f_1(i - 1, j) * _g3(k) * _s13(i - 1, j, k) +
            d4[2] * _f_1(i + 1, j) * _g3(k) * _s13(i + 1, j, k) +
            d4[3] * _f_1(i + 2, j) * _g3(k) * _s13(i + 2, j, k) +
            d4l[k][0] * _s33(i, j, 0) + d4l[k][1] * _s33(i, j, 1) +
            d4l[k][2] * _s33(i, j, 2) + d4l[k][3] * _s33(i, j, 3) +
            d4l[k][4] * _s33(i, j, 4) + d4l[k][5] * _s33(i, j, 5) +
            d4l[k][6] * _s33(i, j, 6) + d4l[k][7] * _s33(i, j, 7) +
            dh4[2] * _f_2(i, j) * _g3(k) * _s23(i, j, k) +
            dh4[0] * _f_2(i, j - 2) * _g3(k) * _s23(i, j - 2, k) +
            dh4[1] * _f_2(i, j - 1) * _g3(k) * _s23(i, j - 1, k) +
            dh4[3] * _f_2(i, j + 1) * _g3(k) * _s23(i, j + 1, k) -
            _f1_c(i, j) *
                (dph4l[k][0] * _g(0) *
                     (p4[1] * _s13(i, j, 0) + p4[0] * _s13(i - 1, j, 0) +
                      p4[2] * _s13(i + 1, j, 0) + p4[3] * _s13(i + 2, j, 0)) +
                 dph4l[k][1] * _g(1) *
                     (p4[1] * _s13(i, j, 1) + p4[0] * _s13(i - 1, j, 1) +
                      p4[2] * _s13(i + 1, j, 1) + p4[3] * _s13(i + 2, j, 1)) +
                 dph4l[k][2] * _g(2) *
                     (p4[1] * _s13(i, j, 2) + p4[0] * _s13(i - 1, j, 2) +
                      p4[2] * _s13(i + 1, j, 2) + p4[3] * _s13(i + 2, j, 2)) +
                 dph4l[k][3] * _g(3) *
                     (p4[1] * _s13(i, j, 3) + p4[0] * _s13(i - 1, j, 3) +
                      p4[2] * _s13(i + 1, j, 3) + p4[3] * _s13(i + 2, j, 3)) +
                 dph4l[k][4] * _g(4) *
                     (p4[1] * _s13(i, j, 4) + p4[0] * _s13(i - 1, j, 4) +
                      p4[2] * _s13(i + 1, j, 4) + p4[3] * _s13(i + 2, j, 4)) +
                 dph4l[k][5] * _g(5) *
                     (p4[1] * _s13(i, j, 5) + p4[0] * _s13(i - 1, j, 5) +
                      p4[2] * _s13(i + 1, j, 5) + p4[3] * _s13(i + 2, j, 5)) +
                 dph4l[k][6] * _g(6) *
                     (p4[1] * _s13(i, j, 6) + p4[0] * _s13(i - 1, j, 6) +
                      p4[2] * _s13(i + 1, j, 6) + p4[3] * _s13(i + 2, j, 6)) +
                 dph4l[k][7] * _g(7) *
                     (p4[1] * _s13(i, j, 7) + p4[0] * _s13(i - 1, j, 7) +
                      p4[2] * _s13(i + 1, j, 7) + p4[3] * _s13(i + 2, j, 7)) +
                 dph4l[k][8] * _g(8) *
                     (p4[1] * _s13(i, j, 8) + p4[0] * _s13(i - 1, j, 8) +
                      p4[2] * _s13(i + 1, j, 8) + p4[3] * _s13(i + 2, j, 8))) -
            _f2_c(i, j) *
                (dph4l[k][0] * _g(0) *
                     (ph4[2] * _s23(i, j, 0) + ph4[0] * _s23(i, j - 2, 0) +
                      ph4[1] * _s23(i, j - 1, 0) + ph4[3] * _s23(i, j + 1, 0)) +
                 dph4l[k][1] * _g(1) *
                     (ph4[2] * _s23(i, j, 1) + ph4[0] * _s23(i, j - 2, 1) +
                      ph4[1] * _s23(i, j - 1, 1) + ph4[3] * _s23(i, j + 1, 1)) +
                 dph4l[k][2] * _g(2) *
                     (ph4[2] * _s23(i, j, 2) + ph4[0] * _s23(i, j - 2, 2) +
                      ph4[1] * _s23(i, j - 1, 2) + ph4[3] * _s23(i, j + 1, 2)) +
                 dph4l[k][3] * _g(3) *
                     (ph4[2] * _s23(i, j, 3) + ph4[0] * _s23(i, j - 2, 3) +
                      ph4[1] * _s23(i, j - 1, 3) + ph4[3] * _s23(i, j + 1, 3)) +
                 dph4l[k][4] * _g(4) *
                     (ph4[2] * _s23(i, j, 4) + ph4[0] * _s23(i, j - 2, 4) +
                      ph4[1] * _s23(i, j - 1, 4) + ph4[3] * _s23(i, j + 1, 4)) +
                 dph4l[k][5] * _g(5) *
                     (ph4[2] * _s23(i, j, 5) + ph4[0] * _s23(i, j - 2, 5) +
                      ph4[1] * _s23(i, j - 1, 5) + ph4[3] * _s23(i, j + 1, 5)) +
                 dph4l[k][6] * _g(6) *
                     (ph4[2] * _s23(i, j, 6) + ph4[0] * _s23(i, j - 2, 6) +
                      ph4[1] * _s23(i, j - 1, 6) + ph4[3] * _s23(i, j + 1, 6)) +
                 dph4l[k][7] * _g(7) *
                     (ph4[2] * _s23(i, j, 7) + ph4[0] * _s23(i, j - 2, 7) +
                      ph4[1] * _s23(i, j - 1, 7) + ph4[3] * _s23(i, j + 1, 7)) +
                 dph4l[k][8] * _g(8) *
                     (ph4[2] * _s23(i, j, 8) + ph4[0] * _s23(i, j - 2, 8) +
                      ph4[1] * _s23(i, j - 1, 8) +
                      ph4[3] * _s23(i, j + 1, 8))))) *
      f_dcrj;
#undef _rho
#undef _f_1
#undef _g3_c
#undef _f_2
#undef _g3
#undef _f_c
#undef _dcrjz
#undef _dcrjx
#undef _dcrjy
#undef _g_c
#undef _s11
#undef _f2_1
#undef _u1
#undef _f1_1
#undef _s12
#undef _f
#undef _s13
#undef _f1_2
#undef _s22
#undef _s23
#undef _u2
#undef _f2_2
#undef _f2_c
#undef _g
#undef _f1_c
#undef _s33
#undef _u3
}

__global__ void
dtopo_vel_111(float *__restrict__ u1, float *__restrict__ u2,
              float *__restrict__ u3, const float *__restrict__ dcrjx,
              const float *__restrict__ dcrjy, const float *__restrict__ dcrjz,
              const float *__restrict__ f, const float *__restrict__ f1_1,
              const float *__restrict__ f1_2, const float *__restrict__ f1_c,
              const float *__restrict__ f2_1, const float *__restrict__ f2_2,
              const float *__restrict__ f2_c, const float *__restrict__ f_1,
              const float *__restrict__ f_2, const float *__restrict__ f_c,
              const float *__restrict__ g, const float *__restrict__ g3,
              const float *__restrict__ g3_c, const float *__restrict__ g_c,
              const float *__restrict__ rho, const float *__restrict__ s11,
              const float *__restrict__ s12, const float *__restrict__ s13,
              const float *__restrict__ s22, const float *__restrict__ s23,
              const float *__restrict__ s33, const float a, const float nu,
              const int nx, const int ny, const int nz, const int bi,
              const int bj, const int ei, const int ej) {
  const float ph2[2] = {0.5000000000000000, 0.5000000000000000};
  const float dhp4[7] = {-0.0026041666666667, 0.0937500000000000,
                         -0.6796875000000000, 0.0000000000000000,
                         0.6796875000000000,  -0.0937500000000000,
                         0.0026041666666667};
  const float ph4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float dh4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float p4[4] = {-0.0625000000000000, 0.5625000000000000,
                       0.5625000000000000, -0.0625000000000000};
  const float d4[4] = {0.0416666666666667, -1.1250000000000000,
                       1.1250000000000000, -0.0416666666666667};
  const float dph4[7] = {-0.0026041666666667, 0.0937500000000000,
                         -0.6796875000000000, 0.0000000000000000,
                         0.6796875000000000,  -0.0937500000000000,
                         0.0026041666666667};
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
  if (k >= nz - 12)
    return;
#define _rho(i, j, k)                                                          \
  rho[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f_1(i, j)                                                             \
  f_1[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3_c(k) g3_c[(k) + align]
#define _f_2(i, j)                                                             \
  f_2[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3(k) g3[(k) + align]
#define _f_c(i, j)                                                             \
  f_c[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _dcrjz(k) dcrjz[(k) + align]
#define _dcrjx(i) dcrjx[(i) + ngsl + 2]
#define _dcrjy(j) dcrjy[(j) + ngsl + 2]
#define _g_c(k) g_c[(k) + align]
#define _s11(i, j, k)                                                          \
  s11[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_1(i, j)                                                            \
  f2_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _u1(i, j, k)                                                           \
  u1[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_1(i, j)                                                            \
  f1_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s12(i, j, k)                                                          \
  s12[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f(i, j)                                                               \
  f[(j) + align + ngsl + ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s13(i, j, k)                                                          \
  s13[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_2(i, j)                                                            \
  f1_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s22(i, j, k)                                                          \
  s22[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s23(i, j, k)                                                          \
  s23[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _u2(i, j, k)                                                           \
  u2[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_2(i, j)                                                            \
  f2_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_c(i, j)                                                            \
  f2_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g(k) g[(k) + align]
#define _f1_c(i, j)                                                            \
  f1_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s33(i, j, k)                                                          \
  s33[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _u3(i, j, k)                                                           \
  u3[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
  float rho1 =
      ph2[0] * (ph2[1] * _rho(i, j, k + 5) + ph2[0] * _rho(i, j - 1, k + 5)) +
      ph2[1] * (ph2[1] * _rho(i, j, k + 6) + ph2[0] * _rho(i, j - 1, k + 6));
  float rho2 =
      ph2[0] * (ph2[1] * _rho(i, j, k + 5) + ph2[0] * _rho(i - 1, j, k + 5)) +
      ph2[1] * (ph2[1] * _rho(i, j, k + 6) + ph2[0] * _rho(i - 1, j, k + 6));
  float rho3 =
      ph2[1] * (ph2[1] * _rho(i, j, k + 6) + ph2[0] * _rho(i - 1, j, k + 6)) +
      ph2[0] *
          (ph2[0] * _rho(i, j - 1, k + 6) + ph2[0] * _rho(i - 1, j - 1, k + 6));
  float Ai1 = _f_1(i, j) * _g3_c(k + 6) * rho1;
  Ai1 = nu * 1.0 / Ai1;
  float Ai2 = _f_2(i, j) * _g3_c(k + 6) * rho2;
  Ai2 = nu * 1.0 / Ai2;
  float Ai3 = _f_c(i, j) * _g3(k + 6) * rho3;
  Ai3 = nu * 1.0 / Ai3;
  float f_dcrj = _dcrjx(i) * _dcrjy(j) * _dcrjz(k + 6);
  _u1(i, j, k + 6) =
      (a * _u1(i, j, k + 6) +
       Ai1 * (dh4[2] * _f(i, j) * _g3_c(k + 6) * _s12(i, j, k + 6) +
              dh4[0] * _f(i, j - 2) * _g3_c(k + 6) * _s12(i, j - 2, k + 6) +
              dh4[1] * _f(i, j - 1) * _g3_c(k + 6) * _s12(i, j - 1, k + 6) +
              dh4[3] * _f(i, j + 1) * _g3_c(k + 6) * _s12(i, j + 1, k + 6) +
              dh4[2] * _f_c(i, j) * _g3_c(k + 6) * _s11(i, j, k + 6) +
              dh4[0] * _f_c(i - 2, j) * _g3_c(k + 6) * _s11(i - 2, j, k + 6) +
              dh4[1] * _f_c(i - 1, j) * _g3_c(k + 6) * _s11(i - 1, j, k + 6) +
              dh4[3] * _f_c(i + 1, j) * _g3_c(k + 6) * _s11(i + 1, j, k + 6) +
              dh4[0] * _s13(i, j, k + 4) + dh4[1] * _s13(i, j, k + 5) +
              dh4[2] * _s13(i, j, k + 6) + dh4[3] * _s13(i, j, k + 7) -
              _f1_1(i, j) * (dhp4[0] * _g_c(k + 3) *
                                 (ph4[2] * _s11(i, j, k + 3) +
                                  ph4[0] * _s11(i - 2, j, k + 3) +
                                  ph4[1] * _s11(i - 1, j, k + 3) +
                                  ph4[3] * _s11(i + 1, j, k + 3)) +
                             dhp4[1] * _g_c(k + 4) *
                                 (ph4[2] * _s11(i, j, k + 4) +
                                  ph4[0] * _s11(i - 2, j, k + 4) +
                                  ph4[1] * _s11(i - 1, j, k + 4) +
                                  ph4[3] * _s11(i + 1, j, k + 4)) +
                             dhp4[2] * _g_c(k + 5) *
                                 (ph4[2] * _s11(i, j, k + 5) +
                                  ph4[0] * _s11(i - 2, j, k + 5) +
                                  ph4[1] * _s11(i - 1, j, k + 5) +
                                  ph4[3] * _s11(i + 1, j, k + 5)) +
                             dhp4[3] * _g_c(k + 6) *
                                 (ph4[2] * _s11(i, j, k + 6) +
                                  ph4[0] * _s11(i - 2, j, k + 6) +
                                  ph4[1] * _s11(i - 1, j, k + 6) +
                                  ph4[3] * _s11(i + 1, j, k + 6)) +
                             dhp4[4] * _g_c(k + 7) *
                                 (ph4[2] * _s11(i, j, k + 7) +
                                  ph4[0] * _s11(i - 2, j, k + 7) +
                                  ph4[1] * _s11(i - 1, j, k + 7) +
                                  ph4[3] * _s11(i + 1, j, k + 7)) +
                             dhp4[5] * _g_c(k + 8) *
                                 (ph4[2] * _s11(i, j, k + 8) +
                                  ph4[0] * _s11(i - 2, j, k + 8) +
                                  ph4[1] * _s11(i - 1, j, k + 8) +
                                  ph4[3] * _s11(i + 1, j, k + 8)) +
                             dhp4[6] * _g_c(k + 9) *
                                 (ph4[2] * _s11(i, j, k + 9) +
                                  ph4[0] * _s11(i - 2, j, k + 9) +
                                  ph4[1] * _s11(i - 1, j, k + 9) +
                                  ph4[3] * _s11(i + 1, j, k + 9))) -
              _f2_1(i, j) * (dhp4[0] * _g_c(k + 3) *
                                 (ph4[2] * _s12(i, j, k + 3) +
                                  ph4[0] * _s12(i, j - 2, k + 3) +
                                  ph4[1] * _s12(i, j - 1, k + 3) +
                                  ph4[3] * _s12(i, j + 1, k + 3)) +
                             dhp4[1] * _g_c(k + 4) *
                                 (ph4[2] * _s12(i, j, k + 4) +
                                  ph4[0] * _s12(i, j - 2, k + 4) +
                                  ph4[1] * _s12(i, j - 1, k + 4) +
                                  ph4[3] * _s12(i, j + 1, k + 4)) +
                             dhp4[2] * _g_c(k + 5) *
                                 (ph4[2] * _s12(i, j, k + 5) +
                                  ph4[0] * _s12(i, j - 2, k + 5) +
                                  ph4[1] * _s12(i, j - 1, k + 5) +
                                  ph4[3] * _s12(i, j + 1, k + 5)) +
                             dhp4[3] * _g_c(k + 6) *
                                 (ph4[2] * _s12(i, j, k + 6) +
                                  ph4[0] * _s12(i, j - 2, k + 6) +
                                  ph4[1] * _s12(i, j - 1, k + 6) +
                                  ph4[3] * _s12(i, j + 1, k + 6)) +
                             dhp4[4] * _g_c(k + 7) *
                                 (ph4[2] * _s12(i, j, k + 7) +
                                  ph4[0] * _s12(i, j - 2, k + 7) +
                                  ph4[1] * _s12(i, j - 1, k + 7) +
                                  ph4[3] * _s12(i, j + 1, k + 7)) +
                             dhp4[5] * _g_c(k + 8) *
                                 (ph4[2] * _s12(i, j, k + 8) +
                                  ph4[0] * _s12(i, j - 2, k + 8) +
                                  ph4[1] * _s12(i, j - 1, k + 8) +
                                  ph4[3] * _s12(i, j + 1, k + 8)) +
                             dhp4[6] * _g_c(k + 9) *
                                 (ph4[2] * _s12(i, j, k + 9) +
                                  ph4[0] * _s12(i, j - 2, k + 9) +
                                  ph4[1] * _s12(i, j - 1, k + 9) +
                                  ph4[3] * _s12(i, j + 1, k + 9))))) *
      f_dcrj;
  _u2(i, j, k + 6) =
      (a * _u2(i, j, k + 6) +
       Ai2 * (d4[1] * _f(i, j) * _g3_c(k + 6) * _s12(i, j, k + 6) +
              d4[0] * _f(i - 1, j) * _g3_c(k + 6) * _s12(i - 1, j, k + 6) +
              d4[2] * _f(i + 1, j) * _g3_c(k + 6) * _s12(i + 1, j, k + 6) +
              d4[3] * _f(i + 2, j) * _g3_c(k + 6) * _s12(i + 2, j, k + 6) +
              d4[1] * _f_c(i, j) * _g3_c(k + 6) * _s22(i, j, k + 6) +
              d4[0] * _f_c(i, j - 1) * _g3_c(k + 6) * _s22(i, j - 1, k + 6) +
              d4[2] * _f_c(i, j + 1) * _g3_c(k + 6) * _s22(i, j + 1, k + 6) +
              d4[3] * _f_c(i, j + 2) * _g3_c(k + 6) * _s22(i, j + 2, k + 6) +
              dh4[0] * _s23(i, j, k + 4) + dh4[1] * _s23(i, j, k + 5) +
              dh4[2] * _s23(i, j, k + 6) + dh4[3] * _s23(i, j, k + 7) -
              _f1_2(i, j) * (dhp4[0] * _g_c(k + 3) *
                                 (p4[1] * _s12(i, j, k + 3) +
                                  p4[0] * _s12(i - 1, j, k + 3) +
                                  p4[2] * _s12(i + 1, j, k + 3) +
                                  p4[3] * _s12(i + 2, j, k + 3)) +
                             dhp4[1] * _g_c(k + 4) *
                                 (p4[1] * _s12(i, j, k + 4) +
                                  p4[0] * _s12(i - 1, j, k + 4) +
                                  p4[2] * _s12(i + 1, j, k + 4) +
                                  p4[3] * _s12(i + 2, j, k + 4)) +
                             dhp4[2] * _g_c(k + 5) *
                                 (p4[1] * _s12(i, j, k + 5) +
                                  p4[0] * _s12(i - 1, j, k + 5) +
                                  p4[2] * _s12(i + 1, j, k + 5) +
                                  p4[3] * _s12(i + 2, j, k + 5)) +
                             dhp4[3] * _g_c(k + 6) *
                                 (p4[1] * _s12(i, j, k + 6) +
                                  p4[0] * _s12(i - 1, j, k + 6) +
                                  p4[2] * _s12(i + 1, j, k + 6) +
                                  p4[3] * _s12(i + 2, j, k + 6)) +
                             dhp4[4] * _g_c(k + 7) *
                                 (p4[1] * _s12(i, j, k + 7) +
                                  p4[0] * _s12(i - 1, j, k + 7) +
                                  p4[2] * _s12(i + 1, j, k + 7) +
                                  p4[3] * _s12(i + 2, j, k + 7)) +
                             dhp4[5] * _g_c(k + 8) *
                                 (p4[1] * _s12(i, j, k + 8) +
                                  p4[0] * _s12(i - 1, j, k + 8) +
                                  p4[2] * _s12(i + 1, j, k + 8) +
                                  p4[3] * _s12(i + 2, j, k + 8)) +
                             dhp4[6] * _g_c(k + 9) *
                                 (p4[1] * _s12(i, j, k + 9) +
                                  p4[0] * _s12(i - 1, j, k + 9) +
                                  p4[2] * _s12(i + 1, j, k + 9) +
                                  p4[3] * _s12(i + 2, j, k + 9))) -
              _f2_2(i, j) * (dhp4[0] * _g_c(k + 3) *
                                 (p4[1] * _s22(i, j, k + 3) +
                                  p4[0] * _s22(i, j - 1, k + 3) +
                                  p4[2] * _s22(i, j + 1, k + 3) +
                                  p4[3] * _s22(i, j + 2, k + 3)) +
                             dhp4[1] * _g_c(k + 4) *
                                 (p4[1] * _s22(i, j, k + 4) +
                                  p4[0] * _s22(i, j - 1, k + 4) +
                                  p4[2] * _s22(i, j + 1, k + 4) +
                                  p4[3] * _s22(i, j + 2, k + 4)) +
                             dhp4[2] * _g_c(k + 5) *
                                 (p4[1] * _s22(i, j, k + 5) +
                                  p4[0] * _s22(i, j - 1, k + 5) +
                                  p4[2] * _s22(i, j + 1, k + 5) +
                                  p4[3] * _s22(i, j + 2, k + 5)) +
                             dhp4[3] * _g_c(k + 6) *
                                 (p4[1] * _s22(i, j, k + 6) +
                                  p4[0] * _s22(i, j - 1, k + 6) +
                                  p4[2] * _s22(i, j + 1, k + 6) +
                                  p4[3] * _s22(i, j + 2, k + 6)) +
                             dhp4[4] * _g_c(k + 7) *
                                 (p4[1] * _s22(i, j, k + 7) +
                                  p4[0] * _s22(i, j - 1, k + 7) +
                                  p4[2] * _s22(i, j + 1, k + 7) +
                                  p4[3] * _s22(i, j + 2, k + 7)) +
                             dhp4[5] * _g_c(k + 8) *
                                 (p4[1] * _s22(i, j, k + 8) +
                                  p4[0] * _s22(i, j - 1, k + 8) +
                                  p4[2] * _s22(i, j + 1, k + 8) +
                                  p4[3] * _s22(i, j + 2, k + 8)) +
                             dhp4[6] * _g_c(k + 9) *
                                 (p4[1] * _s22(i, j, k + 9) +
                                  p4[0] * _s22(i, j - 1, k + 9) +
                                  p4[2] * _s22(i, j + 1, k + 9) +
                                  p4[3] * _s22(i, j + 2, k + 9))))) *
      f_dcrj;
  _u3(i, j, k + 6) =
      (a * _u3(i, j, k + 6) +
       Ai3 * (d4[1] * _f_1(i, j) * _g3(k + 6) * _s13(i, j, k + 6) +
              d4[0] * _f_1(i - 1, j) * _g3(k + 6) * _s13(i - 1, j, k + 6) +
              d4[2] * _f_1(i + 1, j) * _g3(k + 6) * _s13(i + 1, j, k + 6) +
              d4[3] * _f_1(i + 2, j) * _g3(k + 6) * _s13(i + 2, j, k + 6) +
              d4[0] * _s33(i, j, k + 5) + d4[1] * _s33(i, j, k + 6) +
              d4[2] * _s33(i, j, k + 7) + d4[3] * _s33(i, j, k + 8) +
              dh4[2] * _f_2(i, j) * _g3(k + 6) * _s23(i, j, k + 6) +
              dh4[0] * _f_2(i, j - 2) * _g3(k + 6) * _s23(i, j - 2, k + 6) +
              dh4[1] * _f_2(i, j - 1) * _g3(k + 6) * _s23(i, j - 1, k + 6) +
              dh4[3] * _f_2(i, j + 1) * _g3(k + 6) * _s23(i, j + 1, k + 6) -
              _f1_c(i, j) * (dph4[0] * _g(k + 3) *
                                 (p4[1] * _s13(i, j, k + 3) +
                                  p4[0] * _s13(i - 1, j, k + 3) +
                                  p4[2] * _s13(i + 1, j, k + 3) +
                                  p4[3] * _s13(i + 2, j, k + 3)) +
                             dph4[1] * _g(k + 4) *
                                 (p4[1] * _s13(i, j, k + 4) +
                                  p4[0] * _s13(i - 1, j, k + 4) +
                                  p4[2] * _s13(i + 1, j, k + 4) +
                                  p4[3] * _s13(i + 2, j, k + 4)) +
                             dph4[2] * _g(k + 5) *
                                 (p4[1] * _s13(i, j, k + 5) +
                                  p4[0] * _s13(i - 1, j, k + 5) +
                                  p4[2] * _s13(i + 1, j, k + 5) +
                                  p4[3] * _s13(i + 2, j, k + 5)) +
                             dph4[3] * _g(k + 6) *
                                 (p4[1] * _s13(i, j, k + 6) +
                                  p4[0] * _s13(i - 1, j, k + 6) +
                                  p4[2] * _s13(i + 1, j, k + 6) +
                                  p4[3] * _s13(i + 2, j, k + 6)) +
                             dph4[4] * _g(k + 7) *
                                 (p4[1] * _s13(i, j, k + 7) +
                                  p4[0] * _s13(i - 1, j, k + 7) +
                                  p4[2] * _s13(i + 1, j, k + 7) +
                                  p4[3] * _s13(i + 2, j, k + 7)) +
                             dph4[5] * _g(k + 8) *
                                 (p4[1] * _s13(i, j, k + 8) +
                                  p4[0] * _s13(i - 1, j, k + 8) +
                                  p4[2] * _s13(i + 1, j, k + 8) +
                                  p4[3] * _s13(i + 2, j, k + 8)) +
                             dph4[6] * _g(k + 9) *
                                 (p4[1] * _s13(i, j, k + 9) +
                                  p4[0] * _s13(i - 1, j, k + 9) +
                                  p4[2] * _s13(i + 1, j, k + 9) +
                                  p4[3] * _s13(i + 2, j, k + 9))) -
              _f2_c(i, j) * (dph4[0] * _g(k + 3) *
                                 (ph4[2] * _s23(i, j, k + 3) +
                                  ph4[0] * _s23(i, j - 2, k + 3) +
                                  ph4[1] * _s23(i, j - 1, k + 3) +
                                  ph4[3] * _s23(i, j + 1, k + 3)) +
                             dph4[1] * _g(k + 4) *
                                 (ph4[2] * _s23(i, j, k + 4) +
                                  ph4[0] * _s23(i, j - 2, k + 4) +
                                  ph4[1] * _s23(i, j - 1, k + 4) +
                                  ph4[3] * _s23(i, j + 1, k + 4)) +
                             dph4[2] * _g(k + 5) *
                                 (ph4[2] * _s23(i, j, k + 5) +
                                  ph4[0] * _s23(i, j - 2, k + 5) +
                                  ph4[1] * _s23(i, j - 1, k + 5) +
                                  ph4[3] * _s23(i, j + 1, k + 5)) +
                             dph4[3] * _g(k + 6) *
                                 (ph4[2] * _s23(i, j, k + 6) +
                                  ph4[0] * _s23(i, j - 2, k + 6) +
                                  ph4[1] * _s23(i, j - 1, k + 6) +
                                  ph4[3] * _s23(i, j + 1, k + 6)) +
                             dph4[4] * _g(k + 7) *
                                 (ph4[2] * _s23(i, j, k + 7) +
                                  ph4[0] * _s23(i, j - 2, k + 7) +
                                  ph4[1] * _s23(i, j - 1, k + 7) +
                                  ph4[3] * _s23(i, j + 1, k + 7)) +
                             dph4[5] * _g(k + 8) *
                                 (ph4[2] * _s23(i, j, k + 8) +
                                  ph4[0] * _s23(i, j - 2, k + 8) +
                                  ph4[1] * _s23(i, j - 1, k + 8) +
                                  ph4[3] * _s23(i, j + 1, k + 8)) +
                             dph4[6] * _g(k + 9) *
                                 (ph4[2] * _s23(i, j, k + 9) +
                                  ph4[0] * _s23(i, j - 2, k + 9) +
                                  ph4[1] * _s23(i, j - 1, k + 9) +
                                  ph4[3] * _s23(i, j + 1, k + 9))))) *
      f_dcrj;
#undef _rho
#undef _f_1
#undef _g3_c
#undef _f_2
#undef _g3
#undef _f_c
#undef _dcrjz
#undef _dcrjx
#undef _dcrjy
#undef _g_c
#undef _s11
#undef _f2_1
#undef _u1
#undef _f1_1
#undef _s12
#undef _f
#undef _s13
#undef _f1_2
#undef _s22
#undef _s23
#undef _u2
#undef _f2_2
#undef _f2_c
#undef _g
#undef _f1_c
#undef _s33
#undef _u3
}

__global__ void
dtopo_vel_112(float *__restrict__ u1, float *__restrict__ u2,
              float *__restrict__ u3, const float *__restrict__ dcrjx,
              const float *__restrict__ dcrjy, const float *__restrict__ dcrjz,
              const float *__restrict__ f, const float *__restrict__ f1_1,
              const float *__restrict__ f1_2, const float *__restrict__ f1_c,
              const float *__restrict__ f2_1, const float *__restrict__ f2_2,
              const float *__restrict__ f2_c, const float *__restrict__ f_1,
              const float *__restrict__ f_2, const float *__restrict__ f_c,
              const float *__restrict__ g, const float *__restrict__ g3,
              const float *__restrict__ g3_c, const float *__restrict__ g_c,
              const float *__restrict__ rho, const float *__restrict__ s11,
              const float *__restrict__ s12, const float *__restrict__ s13,
              const float *__restrict__ s22, const float *__restrict__ s23,
              const float *__restrict__ s33, const float a, const float nu,
              const int nx, const int ny, const int nz, const int bi,
              const int bj, const int ei, const int ej) {
  const float ph2r[6][8] = {
      {0.0000000000000000, 1.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.5000000000000000, 0.5000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.5000000000000000,
       0.5000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.5000000000000000, 0.5000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.5000000000000000, 0.5000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.5000000000000000,
       0.5000000000000000, 0.0000000000000000}};
  const float ph2[2] = {0.5000000000000000, 0.5000000000000000};
  const float dhp4r[6][9] = {
      {-1.5373923010673118, -1.1059180740634813, -0.2134752473866528,
       -0.0352027995732726, -0.0075022330101095, -0.0027918394266035,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.8139439685257414, 0.1273679143938725, -1.1932750007455710,
       0.1475120181828087, 0.1125814499297686, -0.0081303502866204,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.1639182541610305, 0.3113839909089030, -0.0536007135209480,
       -0.3910958927076030, -0.0401741813821989, 0.0095685425408165,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.0171478318814576, -0.0916600077207278, 0.7187220404622645,
       -0.1434031863528334, -0.5827389738506837, 0.0847863081664324,
       -0.0028540125859095, 0.0000000000000000, 0.0000000000000000},
      {-0.0579176640853654, 0.0022069616616207, 0.0108792602269819,
       0.6803612607837533, -0.0530169938441240, -0.6736586580761996,
       0.0937500000000000, -0.0026041666666667, 0.0000000000000000},
      {0.0020323834153791, 0.0002106933140862, -0.0013351454085978,
       -0.0938400881871787, 0.6816971139746001, -0.0002232904416222,
       -0.6796875000000000, 0.0937500000000000, -0.0026041666666667}};
  const float ph4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float dh4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float dh4r[6][8] = {
      {0.0000000000000000, -1.4511412472637157, -1.8534237417911470,
       0.3534237417911469, 0.0488587527362844, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.8577143189081458, -0.5731429567244373,
       -0.4268570432755628, 0.1422856810918542, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.1674548505882877, 0.4976354482351368,
       -0.4976354482351368, -0.1674548505882877, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, -0.1027061113405124, 0.2624541326469860,
       0.8288742701021167, -1.0342864927831414, 0.0456642013745513,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       -0.0416666666666667, 1.1250000000000000, -1.1250000000000000,
       0.0416666666666667, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, -0.0416666666666667, 1.1250000000000000,
       -1.1250000000000000, 0.0416666666666667}};
  const float p4[4] = {-0.0625000000000000, 0.5625000000000000,
                       0.5625000000000000, -0.0625000000000000};
  const float d4[4] = {0.0416666666666667, -1.1250000000000000,
                       1.1250000000000000, -0.0416666666666667};
  const float dph4r[6][9] = {
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, -1.5886075042755421, -2.4835574634505861,
       0.0421173406787286, 0.4968761536590695, -0.0228264197210198,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.4428256655817484, -0.0574614517751294,
       -0.2022259589759502, -0.1944663890497050, 0.0113281342190362,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, -0.3360140866060758, 1.2113298407847195,
       -0.3111668377093505, -0.6714462506479002, 0.1111440843153523,
       -0.0038467501367455, 0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0338560531369653, -0.0409943223643902,
       0.5284757132923059, 0.0115571196122084, -0.6162252315536446,
       0.0857115441015996, -0.0023808762250444, 0.0000000000000000},
      {0.0000000000000000, -0.0040378273193044, 0.0064139372778371,
       -0.0890062133451850, 0.6749219241340761, 0.0002498459192428,
       -0.6796875000000000, 0.0937500000000000, -0.0026041666666667}};
  const float d4r[6][7] = {
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {-1.7779989465546753, -1.3337480247900155, -0.7775013168066564,
       0.3332503950419969, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.4410217341392059, 0.1730842484889890, -0.4487228323259926,
       -0.1653831503022022, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {-0.1798793213882701, 0.2757257254150788, 0.9597948548284453,
       -1.1171892610431817, 0.0615480021879277, 0.0000000000000000,
       0.0000000000000000},
      {-0.0153911381507088, -0.0568851455503591, 0.1998976464597171,
       0.8628231468598346, -1.0285385292191949, 0.0380940196007109,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       -0.0416666666666667, 1.1250000000000000, -1.1250000000000000,
       0.0416666666666667}};
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
#define _rho(i, j, k)                                                          \
  rho[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f_1(i, j)                                                             \
  f_1[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3_c(k) g3_c[(k) + align]
#define _f_2(i, j)                                                             \
  f_2[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3(k) g3[(k) + align]
#define _f_c(i, j)                                                             \
  f_c[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _dcrjz(k) dcrjz[(k) + align]
#define _dcrjx(i) dcrjx[(i) + ngsl + 2]
#define _dcrjy(j) dcrjy[(j) + ngsl + 2]
#define _g_c(k) g_c[(k) + align]
#define _s11(i, j, k)                                                          \
  s11[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_1(i, j)                                                            \
  f2_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _u1(i, j, k)                                                           \
  u1[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_1(i, j)                                                            \
  f1_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s12(i, j, k)                                                          \
  s12[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f(i, j)                                                               \
  f[(j) + align + ngsl + ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s13(i, j, k)                                                          \
  s13[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_2(i, j)                                                            \
  f1_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s22(i, j, k)                                                          \
  s22[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s23(i, j, k)                                                          \
  s23[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _u2(i, j, k)                                                           \
  u2[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_2(i, j)                                                            \
  f2_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_c(i, j)                                                            \
  f2_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g(k) g[(k) + align]
#define _f1_c(i, j)                                                            \
  f1_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s33(i, j, k)                                                          \
  s33[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _u3(i, j, k)                                                           \
  u3[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
  float rho1 =
      ph2r[k][7] *
          (ph2[1] * _rho(i, j, nz - 8) + ph2[0] * _rho(i, j - 1, nz - 8)) +
      ph2r[k][6] *
          (ph2[1] * _rho(i, j, nz - 7) + ph2[0] * _rho(i, j - 1, nz - 7)) +
      ph2r[k][5] *
          (ph2[1] * _rho(i, j, nz - 6) + ph2[0] * _rho(i, j - 1, nz - 6)) +
      ph2r[k][4] *
          (ph2[1] * _rho(i, j, nz - 5) + ph2[0] * _rho(i, j - 1, nz - 5)) +
      ph2r[k][3] *
          (ph2[1] * _rho(i, j, nz - 4) + ph2[0] * _rho(i, j - 1, nz - 4)) +
      ph2r[k][2] *
          (ph2[1] * _rho(i, j, nz - 3) + ph2[0] * _rho(i, j - 1, nz - 3)) +
      ph2r[k][1] *
          (ph2[1] * _rho(i, j, nz - 2) + ph2[0] * _rho(i, j - 1, nz - 2)) +
      ph2r[k][0] *
          (ph2[1] * _rho(i, j, nz - 1) + ph2[0] * _rho(i, j - 1, nz - 1));
  float rho2 =
      ph2r[k][7] *
          (ph2[1] * _rho(i, j, nz - 8) + ph2[0] * _rho(i - 1, j, nz - 8)) +
      ph2r[k][6] *
          (ph2[1] * _rho(i, j, nz - 7) + ph2[0] * _rho(i - 1, j, nz - 7)) +
      ph2r[k][5] *
          (ph2[1] * _rho(i, j, nz - 6) + ph2[0] * _rho(i - 1, j, nz - 6)) +
      ph2r[k][4] *
          (ph2[1] * _rho(i, j, nz - 5) + ph2[0] * _rho(i - 1, j, nz - 5)) +
      ph2r[k][3] *
          (ph2[1] * _rho(i, j, nz - 4) + ph2[0] * _rho(i - 1, j, nz - 4)) +
      ph2r[k][2] *
          (ph2[1] * _rho(i, j, nz - 3) + ph2[0] * _rho(i - 1, j, nz - 3)) +
      ph2r[k][1] *
          (ph2[1] * _rho(i, j, nz - 2) + ph2[0] * _rho(i - 1, j, nz - 2)) +
      ph2r[k][0] *
          (ph2[1] * _rho(i, j, nz - 1) + ph2[0] * _rho(i - 1, j, nz - 1));
  float rho3 = ph2[1] * (ph2[1] * _rho(i, j, nz - 1 - k) +
                         ph2[0] * _rho(i - 1, j, nz - 1 - k)) +
               ph2[0] * (ph2[1] * _rho(i, j - 1, nz - 1 - k) +
                         ph2[0] * _rho(i - 1, j - 1, nz - 1 - k));
  float Ai1 = _f_1(i, j) * _g3_c(nz - 1 - k) * rho1;
  Ai1 = nu * 1.0 / Ai1;
  float Ai2 = _f_2(i, j) * _g3_c(nz - 1 - k) * rho2;
  Ai2 = nu * 1.0 / Ai2;
  float Ai3 = _f_c(i, j) * _g3(nz - 1 - k) * rho3;
  Ai3 = nu * 1.0 / Ai3;
  float f_dcrj = _dcrjx(i) * _dcrjy(j) * _dcrjz(nz - 1 - k);
  _u1(i, j, nz - 1 - k) =
      (a * _u1(i, j, nz - 1 - k) +
       Ai1 *
           (dh4[2] * _f(i, j) * _g3_c(nz - 1 - k) * _s12(i, j, nz - 1 - k) +
            dh4[0] * _f(i, j - 2) * _g3_c(nz - 1 - k) *
                _s12(i, j - 2, nz - 1 - k) +
            dh4[1] * _f(i, j - 1) * _g3_c(nz - 1 - k) *
                _s12(i, j - 1, nz - 1 - k) +
            dh4[3] * _f(i, j + 1) * _g3_c(nz - 1 - k) *
                _s12(i, j + 1, nz - 1 - k) +
            dh4[2] * _f_c(i, j) * _g3_c(nz - 1 - k) * _s11(i, j, nz - 1 - k) +
            dh4[0] * _f_c(i - 2, j) * _g3_c(nz - 1 - k) *
                _s11(i - 2, j, nz - 1 - k) +
            dh4[1] * _f_c(i - 1, j) * _g3_c(nz - 1 - k) *
                _s11(i - 1, j, nz - 1 - k) +
            dh4[3] * _f_c(i + 1, j) * _g3_c(nz - 1 - k) *
                _s11(i + 1, j, nz - 1 - k) +
            dh4r[k][7] * _s13(i, j, nz - 8) + dh4r[k][6] * _s13(i, j, nz - 7) +
            dh4r[k][5] * _s13(i, j, nz - 6) + dh4r[k][4] * _s13(i, j, nz - 5) +
            dh4r[k][3] * _s13(i, j, nz - 4) + dh4r[k][2] * _s13(i, j, nz - 3) +
            dh4r[k][1] * _s13(i, j, nz - 2) + dh4r[k][0] * _s13(i, j, nz - 1) -
            _f1_1(i, j) * (dhp4r[k][8] * _g_c(nz - 9) *
                               (ph4[2] * _s11(i, j, nz - 9) +
                                ph4[0] * _s11(i - 2, j, nz - 9) +
                                ph4[1] * _s11(i - 1, j, nz - 9) +
                                ph4[3] * _s11(i + 1, j, nz - 9)) +
                           dhp4r[k][7] * _g_c(nz - 8) *
                               (ph4[2] * _s11(i, j, nz - 8) +
                                ph4[0] * _s11(i - 2, j, nz - 8) +
                                ph4[1] * _s11(i - 1, j, nz - 8) +
                                ph4[3] * _s11(i + 1, j, nz - 8)) +
                           dhp4r[k][6] * _g_c(nz - 7) *
                               (ph4[2] * _s11(i, j, nz - 7) +
                                ph4[0] * _s11(i - 2, j, nz - 7) +
                                ph4[1] * _s11(i - 1, j, nz - 7) +
                                ph4[3] * _s11(i + 1, j, nz - 7)) +
                           dhp4r[k][5] * _g_c(nz - 6) *
                               (ph4[2] * _s11(i, j, nz - 6) +
                                ph4[0] * _s11(i - 2, j, nz - 6) +
                                ph4[1] * _s11(i - 1, j, nz - 6) +
                                ph4[3] * _s11(i + 1, j, nz - 6)) +
                           dhp4r[k][4] * _g_c(nz - 5) *
                               (ph4[2] * _s11(i, j, nz - 5) +
                                ph4[0] * _s11(i - 2, j, nz - 5) +
                                ph4[1] * _s11(i - 1, j, nz - 5) +
                                ph4[3] * _s11(i + 1, j, nz - 5)) +
                           dhp4r[k][3] * _g_c(nz - 4) *
                               (ph4[2] * _s11(i, j, nz - 4) +
                                ph4[0] * _s11(i - 2, j, nz - 4) +
                                ph4[1] * _s11(i - 1, j, nz - 4) +
                                ph4[3] * _s11(i + 1, j, nz - 4)) +
                           dhp4r[k][2] * _g_c(nz - 3) *
                               (ph4[2] * _s11(i, j, nz - 3) +
                                ph4[0] * _s11(i - 2, j, nz - 3) +
                                ph4[1] * _s11(i - 1, j, nz - 3) +
                                ph4[3] * _s11(i + 1, j, nz - 3)) +
                           dhp4r[k][1] * _g_c(nz - 2) *
                               (ph4[2] * _s11(i, j, nz - 2) +
                                ph4[0] * _s11(i - 2, j, nz - 2) +
                                ph4[1] * _s11(i - 1, j, nz - 2) +
                                ph4[3] * _s11(i + 1, j, nz - 2)) +
                           dhp4r[k][0] * _g_c(nz - 1) *
                               (ph4[2] * _s11(i, j, nz - 1) +
                                ph4[0] * _s11(i - 2, j, nz - 1) +
                                ph4[1] * _s11(i - 1, j, nz - 1) +
                                ph4[3] * _s11(i + 1, j, nz - 1))) -
            _f2_1(i, j) * (dhp4r[k][8] * _g_c(nz - 9) *
                               (ph4[2] * _s12(i, j, nz - 9) +
                                ph4[0] * _s12(i, j - 2, nz - 9) +
                                ph4[1] * _s12(i, j - 1, nz - 9) +
                                ph4[3] * _s12(i, j + 1, nz - 9)) +
                           dhp4r[k][7] * _g_c(nz - 8) *
                               (ph4[2] * _s12(i, j, nz - 8) +
                                ph4[0] * _s12(i, j - 2, nz - 8) +
                                ph4[1] * _s12(i, j - 1, nz - 8) +
                                ph4[3] * _s12(i, j + 1, nz - 8)) +
                           dhp4r[k][6] * _g_c(nz - 7) *
                               (ph4[2] * _s12(i, j, nz - 7) +
                                ph4[0] * _s12(i, j - 2, nz - 7) +
                                ph4[1] * _s12(i, j - 1, nz - 7) +
                                ph4[3] * _s12(i, j + 1, nz - 7)) +
                           dhp4r[k][5] * _g_c(nz - 6) *
                               (ph4[2] * _s12(i, j, nz - 6) +
                                ph4[0] * _s12(i, j - 2, nz - 6) +
                                ph4[1] * _s12(i, j - 1, nz - 6) +
                                ph4[3] * _s12(i, j + 1, nz - 6)) +
                           dhp4r[k][4] * _g_c(nz - 5) *
                               (ph4[2] * _s12(i, j, nz - 5) +
                                ph4[0] * _s12(i, j - 2, nz - 5) +
                                ph4[1] * _s12(i, j - 1, nz - 5) +
                                ph4[3] * _s12(i, j + 1, nz - 5)) +
                           dhp4r[k][3] * _g_c(nz - 4) *
                               (ph4[2] * _s12(i, j, nz - 4) +
                                ph4[0] * _s12(i, j - 2, nz - 4) +
                                ph4[1] * _s12(i, j - 1, nz - 4) +
                                ph4[3] * _s12(i, j + 1, nz - 4)) +
                           dhp4r[k][2] * _g_c(nz - 3) *
                               (ph4[2] * _s12(i, j, nz - 3) +
                                ph4[0] * _s12(i, j - 2, nz - 3) +
                                ph4[1] * _s12(i, j - 1, nz - 3) +
                                ph4[3] * _s12(i, j + 1, nz - 3)) +
                           dhp4r[k][1] * _g_c(nz - 2) *
                               (ph4[2] * _s12(i, j, nz - 2) +
                                ph4[0] * _s12(i, j - 2, nz - 2) +
                                ph4[1] * _s12(i, j - 1, nz - 2) +
                                ph4[3] * _s12(i, j + 1, nz - 2)) +
                           dhp4r[k][0] * _g_c(nz - 1) *
                               (ph4[2] * _s12(i, j, nz - 1) +
                                ph4[0] * _s12(i, j - 2, nz - 1) +
                                ph4[1] * _s12(i, j - 1, nz - 1) +
                                ph4[3] * _s12(i, j + 1, nz - 1))))) *
      f_dcrj;
  _u2(i, j, nz - 1 - k) =
      (a * _u2(i, j, nz - 1 - k) +
       Ai2 *
           (d4[1] * _f(i, j) * _g3_c(nz - 1 - k) * _s12(i, j, nz - 1 - k) +
            d4[0] * _f(i - 1, j) * _g3_c(nz - 1 - k) *
                _s12(i - 1, j, nz - 1 - k) +
            d4[2] * _f(i + 1, j) * _g3_c(nz - 1 - k) *
                _s12(i + 1, j, nz - 1 - k) +
            d4[3] * _f(i + 2, j) * _g3_c(nz - 1 - k) *
                _s12(i + 2, j, nz - 1 - k) +
            d4[1] * _f_c(i, j) * _g3_c(nz - 1 - k) * _s22(i, j, nz - 1 - k) +
            d4[0] * _f_c(i, j - 1) * _g3_c(nz - 1 - k) *
                _s22(i, j - 1, nz - 1 - k) +
            d4[2] * _f_c(i, j + 1) * _g3_c(nz - 1 - k) *
                _s22(i, j + 1, nz - 1 - k) +
            d4[3] * _f_c(i, j + 2) * _g3_c(nz - 1 - k) *
                _s22(i, j + 2, nz - 1 - k) +
            dh4r[k][7] * _s23(i, j, nz - 8) + dh4r[k][6] * _s23(i, j, nz - 7) +
            dh4r[k][5] * _s23(i, j, nz - 6) + dh4r[k][4] * _s23(i, j, nz - 5) +
            dh4r[k][3] * _s23(i, j, nz - 4) + dh4r[k][2] * _s23(i, j, nz - 3) +
            dh4r[k][1] * _s23(i, j, nz - 2) + dh4r[k][0] * _s23(i, j, nz - 1) -
            _f1_2(i, j) * (dhp4r[k][8] * _g_c(nz - 9) *
                               (p4[1] * _s12(i, j, nz - 9) +
                                p4[0] * _s12(i - 1, j, nz - 9) +
                                p4[2] * _s12(i + 1, j, nz - 9) +
                                p4[3] * _s12(i + 2, j, nz - 9)) +
                           dhp4r[k][7] * _g_c(nz - 8) *
                               (p4[1] * _s12(i, j, nz - 8) +
                                p4[0] * _s12(i - 1, j, nz - 8) +
                                p4[2] * _s12(i + 1, j, nz - 8) +
                                p4[3] * _s12(i + 2, j, nz - 8)) +
                           dhp4r[k][6] * _g_c(nz - 7) *
                               (p4[1] * _s12(i, j, nz - 7) +
                                p4[0] * _s12(i - 1, j, nz - 7) +
                                p4[2] * _s12(i + 1, j, nz - 7) +
                                p4[3] * _s12(i + 2, j, nz - 7)) +
                           dhp4r[k][5] * _g_c(nz - 6) *
                               (p4[1] * _s12(i, j, nz - 6) +
                                p4[0] * _s12(i - 1, j, nz - 6) +
                                p4[2] * _s12(i + 1, j, nz - 6) +
                                p4[3] * _s12(i + 2, j, nz - 6)) +
                           dhp4r[k][4] * _g_c(nz - 5) *
                               (p4[1] * _s12(i, j, nz - 5) +
                                p4[0] * _s12(i - 1, j, nz - 5) +
                                p4[2] * _s12(i + 1, j, nz - 5) +
                                p4[3] * _s12(i + 2, j, nz - 5)) +
                           dhp4r[k][3] * _g_c(nz - 4) *
                               (p4[1] * _s12(i, j, nz - 4) +
                                p4[0] * _s12(i - 1, j, nz - 4) +
                                p4[2] * _s12(i + 1, j, nz - 4) +
                                p4[3] * _s12(i + 2, j, nz - 4)) +
                           dhp4r[k][2] * _g_c(nz - 3) *
                               (p4[1] * _s12(i, j, nz - 3) +
                                p4[0] * _s12(i - 1, j, nz - 3) +
                                p4[2] * _s12(i + 1, j, nz - 3) +
                                p4[3] * _s12(i + 2, j, nz - 3)) +
                           dhp4r[k][1] * _g_c(nz - 2) *
                               (p4[1] * _s12(i, j, nz - 2) +
                                p4[0] * _s12(i - 1, j, nz - 2) +
                                p4[2] * _s12(i + 1, j, nz - 2) +
                                p4[3] * _s12(i + 2, j, nz - 2)) +
                           dhp4r[k][0] * _g_c(nz - 1) *
                               (p4[1] * _s12(i, j, nz - 1) +
                                p4[0] * _s12(i - 1, j, nz - 1) +
                                p4[2] * _s12(i + 1, j, nz - 1) +
                                p4[3] * _s12(i + 2, j, nz - 1))) -
            _f2_2(i, j) * (dhp4r[k][8] * _g_c(nz - 9) *
                               (p4[1] * _s22(i, j, nz - 9) +
                                p4[0] * _s22(i, j - 1, nz - 9) +
                                p4[2] * _s22(i, j + 1, nz - 9) +
                                p4[3] * _s22(i, j + 2, nz - 9)) +
                           dhp4r[k][7] * _g_c(nz - 8) *
                               (p4[1] * _s22(i, j, nz - 8) +
                                p4[0] * _s22(i, j - 1, nz - 8) +
                                p4[2] * _s22(i, j + 1, nz - 8) +
                                p4[3] * _s22(i, j + 2, nz - 8)) +
                           dhp4r[k][6] * _g_c(nz - 7) *
                               (p4[1] * _s22(i, j, nz - 7) +
                                p4[0] * _s22(i, j - 1, nz - 7) +
                                p4[2] * _s22(i, j + 1, nz - 7) +
                                p4[3] * _s22(i, j + 2, nz - 7)) +
                           dhp4r[k][5] * _g_c(nz - 6) *
                               (p4[1] * _s22(i, j, nz - 6) +
                                p4[0] * _s22(i, j - 1, nz - 6) +
                                p4[2] * _s22(i, j + 1, nz - 6) +
                                p4[3] * _s22(i, j + 2, nz - 6)) +
                           dhp4r[k][4] * _g_c(nz - 5) *
                               (p4[1] * _s22(i, j, nz - 5) +
                                p4[0] * _s22(i, j - 1, nz - 5) +
                                p4[2] * _s22(i, j + 1, nz - 5) +
                                p4[3] * _s22(i, j + 2, nz - 5)) +
                           dhp4r[k][3] * _g_c(nz - 4) *
                               (p4[1] * _s22(i, j, nz - 4) +
                                p4[0] * _s22(i, j - 1, nz - 4) +
                                p4[2] * _s22(i, j + 1, nz - 4) +
                                p4[3] * _s22(i, j + 2, nz - 4)) +
                           dhp4r[k][2] * _g_c(nz - 3) *
                               (p4[1] * _s22(i, j, nz - 3) +
                                p4[0] * _s22(i, j - 1, nz - 3) +
                                p4[2] * _s22(i, j + 1, nz - 3) +
                                p4[3] * _s22(i, j + 2, nz - 3)) +
                           dhp4r[k][1] * _g_c(nz - 2) *
                               (p4[1] * _s22(i, j, nz - 2) +
                                p4[0] * _s22(i, j - 1, nz - 2) +
                                p4[2] * _s22(i, j + 1, nz - 2) +
                                p4[3] * _s22(i, j + 2, nz - 2)) +
                           dhp4r[k][0] * _g_c(nz - 1) *
                               (p4[1] * _s22(i, j, nz - 1) +
                                p4[0] * _s22(i, j - 1, nz - 1) +
                                p4[2] * _s22(i, j + 1, nz - 1) +
                                p4[3] * _s22(i, j + 2, nz - 1))))) *
      f_dcrj;
  _u3(i, j, nz - 1 - k) =
      (a * _u3(i, j, nz - 1 - k) +
       Ai3 * (d4[1] * _f_1(i, j) * _g3(nz - 1 - k) * _s13(i, j, nz - 1 - k) +
              d4[0] * _f_1(i - 1, j) * _g3(nz - 1 - k) *
                  _s13(i - 1, j, nz - 1 - k) +
              d4[2] * _f_1(i + 1, j) * _g3(nz - 1 - k) *
                  _s13(i + 1, j, nz - 1 - k) +
              d4[3] * _f_1(i + 2, j) * _g3(nz - 1 - k) *
                  _s13(i + 2, j, nz - 1 - k) +
              d4r[k][6] * _s33(i, j, nz - 7) + d4r[k][5] * _s33(i, j, nz - 6) +
              d4r[k][4] * _s33(i, j, nz - 5) + d4r[k][3] * _s33(i, j, nz - 4) +
              d4r[k][2] * _s33(i, j, nz - 3) + d4r[k][1] * _s33(i, j, nz - 2) +
              d4r[k][0] * _s33(i, j, nz - 1) +
              dh4[2] * _f_2(i, j) * _g3(nz - 1 - k) * _s23(i, j, nz - 1 - k) +
              dh4[0] * _f_2(i, j - 2) * _g3(nz - 1 - k) *
                  _s23(i, j - 2, nz - 1 - k) +
              dh4[1] * _f_2(i, j - 1) * _g3(nz - 1 - k) *
                  _s23(i, j - 1, nz - 1 - k) +
              dh4[3] * _f_2(i, j + 1) * _g3(nz - 1 - k) *
                  _s23(i, j + 1, nz - 1 - k) -
              _f1_c(i, j) * (dph4r[k][8] * _g(nz - 9) *
                                 (p4[1] * _s13(i, j, nz - 9) +
                                  p4[0] * _s13(i - 1, j, nz - 9) +
                                  p4[2] * _s13(i + 1, j, nz - 9) +
                                  p4[3] * _s13(i + 2, j, nz - 9)) +
                             dph4r[k][7] * _g(nz - 8) *
                                 (p4[1] * _s13(i, j, nz - 8) +
                                  p4[0] * _s13(i - 1, j, nz - 8) +
                                  p4[2] * _s13(i + 1, j, nz - 8) +
                                  p4[3] * _s13(i + 2, j, nz - 8)) +
                             dph4r[k][6] * _g(nz - 7) *
                                 (p4[1] * _s13(i, j, nz - 7) +
                                  p4[0] * _s13(i - 1, j, nz - 7) +
                                  p4[2] * _s13(i + 1, j, nz - 7) +
                                  p4[3] * _s13(i + 2, j, nz - 7)) +
                             dph4r[k][5] * _g(nz - 6) *
                                 (p4[1] * _s13(i, j, nz - 6) +
                                  p4[0] * _s13(i - 1, j, nz - 6) +
                                  p4[2] * _s13(i + 1, j, nz - 6) +
                                  p4[3] * _s13(i + 2, j, nz - 6)) +
                             dph4r[k][4] * _g(nz - 5) *
                                 (p4[1] * _s13(i, j, nz - 5) +
                                  p4[0] * _s13(i - 1, j, nz - 5) +
                                  p4[2] * _s13(i + 1, j, nz - 5) +
                                  p4[3] * _s13(i + 2, j, nz - 5)) +
                             dph4r[k][3] * _g(nz - 4) *
                                 (p4[1] * _s13(i, j, nz - 4) +
                                  p4[0] * _s13(i - 1, j, nz - 4) +
                                  p4[2] * _s13(i + 1, j, nz - 4) +
                                  p4[3] * _s13(i + 2, j, nz - 4)) +
                             dph4r[k][2] * _g(nz - 3) *
                                 (p4[1] * _s13(i, j, nz - 3) +
                                  p4[0] * _s13(i - 1, j, nz - 3) +
                                  p4[2] * _s13(i + 1, j, nz - 3) +
                                  p4[3] * _s13(i + 2, j, nz - 3)) +
                             dph4r[k][1] * _g(nz - 2) *
                                 (p4[1] * _s13(i, j, nz - 2) +
                                  p4[0] * _s13(i - 1, j, nz - 2) +
                                  p4[2] * _s13(i + 1, j, nz - 2) +
                                  p4[3] * _s13(i + 2, j, nz - 2)) +
                             dph4r[k][0] * _g(nz - 1) *
                                 (p4[1] * _s13(i, j, nz - 1) +
                                  p4[0] * _s13(i - 1, j, nz - 1) +
                                  p4[2] * _s13(i + 1, j, nz - 1) +
                                  p4[3] * _s13(i + 2, j, nz - 1))) -
              _f2_c(i, j) * (dph4r[k][8] * _g(nz - 9) *
                                 (ph4[2] * _s23(i, j, nz - 9) +
                                  ph4[0] * _s23(i, j - 2, nz - 9) +
                                  ph4[1] * _s23(i, j - 1, nz - 9) +
                                  ph4[3] * _s23(i, j + 1, nz - 9)) +
                             dph4r[k][7] * _g(nz - 8) *
                                 (ph4[2] * _s23(i, j, nz - 8) +
                                  ph4[0] * _s23(i, j - 2, nz - 8) +
                                  ph4[1] * _s23(i, j - 1, nz - 8) +
                                  ph4[3] * _s23(i, j + 1, nz - 8)) +
                             dph4r[k][6] * _g(nz - 7) *
                                 (ph4[2] * _s23(i, j, nz - 7) +
                                  ph4[0] * _s23(i, j - 2, nz - 7) +
                                  ph4[1] * _s23(i, j - 1, nz - 7) +
                                  ph4[3] * _s23(i, j + 1, nz - 7)) +
                             dph4r[k][5] * _g(nz - 6) *
                                 (ph4[2] * _s23(i, j, nz - 6) +
                                  ph4[0] * _s23(i, j - 2, nz - 6) +
                                  ph4[1] * _s23(i, j - 1, nz - 6) +
                                  ph4[3] * _s23(i, j + 1, nz - 6)) +
                             dph4r[k][4] * _g(nz - 5) *
                                 (ph4[2] * _s23(i, j, nz - 5) +
                                  ph4[0] * _s23(i, j - 2, nz - 5) +
                                  ph4[1] * _s23(i, j - 1, nz - 5) +
                                  ph4[3] * _s23(i, j + 1, nz - 5)) +
                             dph4r[k][3] * _g(nz - 4) *
                                 (ph4[2] * _s23(i, j, nz - 4) +
                                  ph4[0] * _s23(i, j - 2, nz - 4) +
                                  ph4[1] * _s23(i, j - 1, nz - 4) +
                                  ph4[3] * _s23(i, j + 1, nz - 4)) +
                             dph4r[k][2] * _g(nz - 3) *
                                 (ph4[2] * _s23(i, j, nz - 3) +
                                  ph4[0] * _s23(i, j - 2, nz - 3) +
                                  ph4[1] * _s23(i, j - 1, nz - 3) +
                                  ph4[3] * _s23(i, j + 1, nz - 3)) +
                             dph4r[k][1] * _g(nz - 2) *
                                 (ph4[2] * _s23(i, j, nz - 2) +
                                  ph4[0] * _s23(i, j - 2, nz - 2) +
                                  ph4[1] * _s23(i, j - 1, nz - 2) +
                                  ph4[3] * _s23(i, j + 1, nz - 2)) +
                             dph4r[k][0] * _g(nz - 1) *
                                 (ph4[2] * _s23(i, j, nz - 1) +
                                  ph4[0] * _s23(i, j - 2, nz - 1) +
                                  ph4[1] * _s23(i, j - 1, nz - 1) +
                                  ph4[3] * _s23(i, j + 1, nz - 1))))) *
      f_dcrj;
#undef _rho
#undef _f_1
#undef _g3_c
#undef _f_2
#undef _g3
#undef _f_c
#undef _dcrjz
#undef _dcrjx
#undef _dcrjy
#undef _g_c
#undef _s11
#undef _f2_1
#undef _u1
#undef _f1_1
#undef _s12
#undef _f
#undef _s13
#undef _f1_2
#undef _s22
#undef _s23
#undef _u2
#undef _f2_2
#undef _f2_c
#undef _g
#undef _f1_c
#undef _s33
#undef _u3
}

__global__ void dtopo_buf_vel_110(
    float *__restrict__ buf_u1, float *__restrict__ buf_u2,
    float *__restrict__ buf_u3, const float *__restrict__ dcrjx,
    const float *__restrict__ dcrjy, const float *__restrict__ dcrjz,
    const float *__restrict__ f, const float *__restrict__ f1_1,
    const float *__restrict__ f1_2, const float *__restrict__ f1_c,
    const float *__restrict__ f2_1, const float *__restrict__ f2_2,
    const float *__restrict__ f2_c, const float *__restrict__ f_1,
    const float *__restrict__ f_2, const float *__restrict__ f_c,
    const float *__restrict__ g, const float *__restrict__ g3,
    const float *__restrict__ g3_c, const float *__restrict__ g_c,
    const float *__restrict__ rho, const float *__restrict__ s11,
    const float *__restrict__ s12, const float *__restrict__ s13,
    const float *__restrict__ s22, const float *__restrict__ s23,
    const float *__restrict__ s33, const float *__restrict__ u1,
    const float *__restrict__ u2, const float *__restrict__ u3, const float a,
    const float nu, const int nx, const int ny, const int nz, const int bj,
    const int ej, const int rj0) {
  const float ph2l[6][7] = {
      {1.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.5000000000000000, 0.5000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.5000000000000000, 0.5000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.5000000000000000,
       0.5000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.5000000000000000, 0.5000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.5000000000000000, 0.5000000000000000,
       0.0000000000000000}};
  const float ph2[2] = {0.5000000000000000, 0.5000000000000000};
  const float dhp4l[6][9] = {
      {-1.4276800979942257, 0.2875185051606178, 2.0072491465276454,
       -0.8773816261307504, 0.0075022330101095, 0.0027918394266035,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {-0.8139439685257414, -0.1273679143938725, 1.1932750007455708,
       -0.1475120181828087, -0.1125814499297686, 0.0081303502866204,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {-0.1639182541610305, -0.3113839909089031, 0.0536007135209480,
       0.3910958927076031, 0.0401741813821989, -0.0095685425408165,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {-0.0171478318814576, 0.0916600077207278, -0.7187220404622644,
       0.1434031863528334, 0.5827389738506837, -0.0847863081664324,
       0.0028540125859095, 0.0000000000000000, 0.0000000000000000},
      {0.0579176640853654, -0.0022069616616207, -0.0108792602269819,
       -0.6803612607837533, 0.0530169938441241, 0.6736586580761996,
       -0.0937500000000000, 0.0026041666666667, 0.0000000000000000},
      {-0.0020323834153791, -0.0002106933140862, 0.0013351454085978,
       0.0938400881871787, -0.6816971139746001, 0.0002232904416222,
       0.6796875000000000, -0.0937500000000000, 0.0026041666666667}};
  const float ph4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float dh4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float dh4l[6][7] = {
      {-1.4511412472637157, 1.8534237417911470, -0.3534237417911469,
       -0.0488587527362844, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {-0.8577143189081458, 0.5731429567244373, 0.4268570432755628,
       -0.1422856810918542, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {-0.1674548505882877, -0.4976354482351368, 0.4976354482351368,
       0.1674548505882877, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.1027061113405124, -0.2624541326469860, -0.8288742701021167,
       1.0342864927831414, -0.0456642013745513, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0416666666666667,
       -1.1250000000000000, 1.1250000000000000, -0.0416666666666667,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0416666666666667, -1.1250000000000000, 1.1250000000000000,
       -0.0416666666666667}};
  const float p4[4] = {-0.0625000000000000, 0.5625000000000000,
                       0.5625000000000000, -0.0625000000000000};
  const float d4[4] = {0.0416666666666667, -1.1250000000000000,
                       1.1250000000000000, -0.0416666666666667};
  const float dph4l[6][9] = {
      {-1.3764648947859957, 1.8523239861274134, -0.5524268681758195,
       0.0537413571133823, 0.0228264197210198, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {-0.4428256655817484, 0.0574614517751293, 0.2022259589759502,
       0.1944663890497050, -0.0113281342190362, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.3360140866060757, -1.2113298407847195, 0.3111668377093505,
       0.6714462506479003, -0.1111440843153523, 0.0038467501367455,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {-0.0338560531369653, 0.0409943223643901, -0.5284757132923059,
       -0.0115571196122084, 0.6162252315536445, -0.0857115441015996,
       0.0023808762250444, 0.0000000000000000, 0.0000000000000000},
      {0.0040378273193044, -0.0064139372778371, 0.0890062133451850,
       -0.6749219241340761, -0.0002498459192428, 0.6796875000000000,
       -0.0937500000000000, 0.0026041666666667, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, -0.0026041666666667,
       0.0937500000000000, -0.6796875000000000, 0.0000000000000000,
       0.6796875000000000, -0.0937500000000000, 0.0026041666666667}};
  const float d4l[6][8] = {
      {-1.7779989465546748, 1.3337480247900155, 0.7775013168066564,
       -0.3332503950419969, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {-0.4410217341392059, -0.1730842484889890, 0.4487228323259926,
       0.1653831503022022, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.1798793213882701, -0.2757257254150788, -0.9597948548284453,
       1.1171892610431817, -0.0615480021879277, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0153911381507088, 0.0568851455503591, -0.1998976464597171,
       -0.8628231468598346, 1.0285385292191949, -0.0380940196007109,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0416666666666667, -1.1250000000000000, 1.1250000000000000,
       -0.0416666666666667, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0416666666666667, -1.1250000000000000,
       1.1250000000000000, -0.0416666666666667}};
  const int i = threadIdx.z + blockIdx.z * blockDim.z;
  if (i >= nx)
    return;
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ny)
    return;
  if (j >= ej)
    return;
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= 6)
    return;
#define _rho(i, j, k)                                                          \
  rho[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f_1(i, j)                                                             \
  f_1[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3_c(k) g3_c[(k) + align]
#define _f_2(i, j)                                                             \
  f_2[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3(k) g3[(k) + align]
#define _f_c(i, j)                                                             \
  f_c[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _dcrjz(k) dcrjz[(k) + align]
#define _dcrjx(i) dcrjx[(i) + ngsl + 2]
#define _dcrjy(j) dcrjy[(j) + ngsl + 2]
#define _g_c(k) g_c[(k) + align]
#define _s11(i, j, k)                                                          \
  s11[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_1(i, j)                                                            \
  f2_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _u1(i, j, k)                                                           \
  u1[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_1(i, j)                                                            \
  f1_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s12(i, j, k)                                                          \
  s12[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f(i, j)                                                               \
  f[(j) + align + ngsl + ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s13(i, j, k)                                                          \
  s13[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_2(i, j)                                                            \
  f1_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s22(i, j, k)                                                          \
  s22[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s23(i, j, k)                                                          \
  s23[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _u2(i, j, k)                                                           \
  u2[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_2(i, j)                                                            \
  f2_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_c(i, j)                                                            \
  f2_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g(k) g[(k) + align]
#define _f1_c(i, j)                                                            \
  f1_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s33(i, j, k)                                                          \
  s33[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _u3(i, j, k)                                                           \
  u3[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _buf_u1(i, j, k)                                                       \
  buf_u1[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
#define _buf_u2(i, j, k)                                                       \
  buf_u2[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
#define _buf_u3(i, j, k)                                                       \
  buf_u3[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
  float rho1 =
      ph2l[k][0] *
          (ph2[1] * _rho(i, j + rj0, 0) + ph2[0] * _rho(i, j + rj0 - 1, 0)) +
      ph2l[k][1] *
          (ph2[1] * _rho(i, j + rj0, 1) + ph2[0] * _rho(i, j + rj0 - 1, 1)) +
      ph2l[k][2] *
          (ph2[1] * _rho(i, j + rj0, 2) + ph2[0] * _rho(i, j + rj0 - 1, 2)) +
      ph2l[k][3] *
          (ph2[1] * _rho(i, j + rj0, 3) + ph2[0] * _rho(i, j + rj0 - 1, 3)) +
      ph2l[k][4] *
          (ph2[1] * _rho(i, j + rj0, 4) + ph2[0] * _rho(i, j + rj0 - 1, 4)) +
      ph2l[k][5] *
          (ph2[1] * _rho(i, j + rj0, 5) + ph2[0] * _rho(i, j + rj0 - 1, 5)) +
      ph2l[k][6] *
          (ph2[1] * _rho(i, j + rj0, 6) + ph2[0] * _rho(i, j + rj0 - 1, 6));
  float rho2 =
      ph2l[k][0] *
          (ph2[1] * _rho(i, j + rj0, 0) + ph2[0] * _rho(i - 1, j + rj0, 0)) +
      ph2l[k][1] *
          (ph2[1] * _rho(i, j + rj0, 1) + ph2[0] * _rho(i - 1, j + rj0, 1)) +
      ph2l[k][2] *
          (ph2[1] * _rho(i, j + rj0, 2) + ph2[0] * _rho(i - 1, j + rj0, 2)) +
      ph2l[k][3] *
          (ph2[1] * _rho(i, j + rj0, 3) + ph2[0] * _rho(i - 1, j + rj0, 3)) +
      ph2l[k][4] *
          (ph2[1] * _rho(i, j + rj0, 4) + ph2[0] * _rho(i - 1, j + rj0, 4)) +
      ph2l[k][5] *
          (ph2[1] * _rho(i, j + rj0, 5) + ph2[0] * _rho(i - 1, j + rj0, 5)) +
      ph2l[k][6] *
          (ph2[1] * _rho(i, j + rj0, 6) + ph2[0] * _rho(i - 1, j + rj0, 6));
  float rho3 = ph2[1] * (ph2[1] * _rho(i, j + rj0, k) +
                         ph2[0] * _rho(i - 1, j + rj0, k)) +
               ph2[0] * (ph2[1] * _rho(i, j + rj0 - 1, k) +
                         ph2[0] * _rho(i - 1, j + rj0 - 1, k));
  float Ai1 = _f_1(i, j + rj0) * _g3_c(k) * rho1;
  Ai1 = nu * 1.0 / Ai1;
  float Ai2 = _f_2(i, j + rj0) * _g3_c(k) * rho2;
  Ai2 = nu * 1.0 / Ai2;
  float Ai3 = _f_c(i, j + rj0) * _g3(k) * rho3;
  Ai3 = nu * 1.0 / Ai3;
  float f_dcrj = _dcrjx(i) * _dcrjy(j + rj0) * _dcrjz(k);
  _buf_u1(i, j, k) =
      (a * _u1(i, j + rj0, k) +
       Ai1 *
           (dh4[2] * _f(i, j + rj0) * _g3_c(k) * _s12(i, j + rj0, k) +
            dh4[0] * _f(i, j + rj0 - 2) * _g3_c(k) * _s12(i, j + rj0 - 2, k) +
            dh4[1] * _f(i, j + rj0 - 1) * _g3_c(k) * _s12(i, j + rj0 - 1, k) +
            dh4[3] * _f(i, j + rj0 + 1) * _g3_c(k) * _s12(i, j + rj0 + 1, k) +
            dh4[2] * _f_c(i, j + rj0) * _g3_c(k) * _s11(i, j + rj0, k) +
            dh4[0] * _f_c(i - 2, j + rj0) * _g3_c(k) * _s11(i - 2, j + rj0, k) +
            dh4[1] * _f_c(i - 1, j + rj0) * _g3_c(k) * _s11(i - 1, j + rj0, k) +
            dh4[3] * _f_c(i + 1, j + rj0) * _g3_c(k) * _s11(i + 1, j + rj0, k) +
            dh4l[k][0] * _s13(i, j + rj0, 0) +
            dh4l[k][1] * _s13(i, j + rj0, 1) +
            dh4l[k][2] * _s13(i, j + rj0, 2) +
            dh4l[k][3] * _s13(i, j + rj0, 3) +
            dh4l[k][4] * _s13(i, j + rj0, 4) +
            dh4l[k][5] * _s13(i, j + rj0, 5) +
            dh4l[k][6] * _s13(i, j + rj0, 6) -
            _f1_1(i, j + rj0) * (dhp4l[k][0] * _g_c(0) *
                                     (ph4[2] * _s11(i, j + rj0, 0) +
                                      ph4[0] * _s11(i - 2, j + rj0, 0) +
                                      ph4[1] * _s11(i - 1, j + rj0, 0) +
                                      ph4[3] * _s11(i + 1, j + rj0, 0)) +
                                 dhp4l[k][1] * _g_c(1) *
                                     (ph4[2] * _s11(i, j + rj0, 1) +
                                      ph4[0] * _s11(i - 2, j + rj0, 1) +
                                      ph4[1] * _s11(i - 1, j + rj0, 1) +
                                      ph4[3] * _s11(i + 1, j + rj0, 1)) +
                                 dhp4l[k][2] * _g_c(2) *
                                     (ph4[2] * _s11(i, j + rj0, 2) +
                                      ph4[0] * _s11(i - 2, j + rj0, 2) +
                                      ph4[1] * _s11(i - 1, j + rj0, 2) +
                                      ph4[3] * _s11(i + 1, j + rj0, 2)) +
                                 dhp4l[k][3] * _g_c(3) *
                                     (ph4[2] * _s11(i, j + rj0, 3) +
                                      ph4[0] * _s11(i - 2, j + rj0, 3) +
                                      ph4[1] * _s11(i - 1, j + rj0, 3) +
                                      ph4[3] * _s11(i + 1, j + rj0, 3)) +
                                 dhp4l[k][4] * _g_c(4) *
                                     (ph4[2] * _s11(i, j + rj0, 4) +
                                      ph4[0] * _s11(i - 2, j + rj0, 4) +
                                      ph4[1] * _s11(i - 1, j + rj0, 4) +
                                      ph4[3] * _s11(i + 1, j + rj0, 4)) +
                                 dhp4l[k][5] * _g_c(5) *
                                     (ph4[2] * _s11(i, j + rj0, 5) +
                                      ph4[0] * _s11(i - 2, j + rj0, 5) +
                                      ph4[1] * _s11(i - 1, j + rj0, 5) +
                                      ph4[3] * _s11(i + 1, j + rj0, 5)) +
                                 dhp4l[k][6] * _g_c(6) *
                                     (ph4[2] * _s11(i, j + rj0, 6) +
                                      ph4[0] * _s11(i - 2, j + rj0, 6) +
                                      ph4[1] * _s11(i - 1, j + rj0, 6) +
                                      ph4[3] * _s11(i + 1, j + rj0, 6)) +
                                 dhp4l[k][7] * _g_c(7) *
                                     (ph4[2] * _s11(i, j + rj0, 7) +
                                      ph4[0] * _s11(i - 2, j + rj0, 7) +
                                      ph4[1] * _s11(i - 1, j + rj0, 7) +
                                      ph4[3] * _s11(i + 1, j + rj0, 7)) +
                                 dhp4l[k][8] * _g_c(8) *
                                     (ph4[2] * _s11(i, j + rj0, 8) +
                                      ph4[0] * _s11(i - 2, j + rj0, 8) +
                                      ph4[1] * _s11(i - 1, j + rj0, 8) +
                                      ph4[3] * _s11(i + 1, j + rj0, 8))) -
            _f2_1(i, j + rj0) * (dhp4l[k][0] * _g_c(0) *
                                     (ph4[2] * _s12(i, j + rj0, 0) +
                                      ph4[0] * _s12(i, j + rj0 - 2, 0) +
                                      ph4[1] * _s12(i, j + rj0 - 1, 0) +
                                      ph4[3] * _s12(i, j + rj0 + 1, 0)) +
                                 dhp4l[k][1] * _g_c(1) *
                                     (ph4[2] * _s12(i, j + rj0, 1) +
                                      ph4[0] * _s12(i, j + rj0 - 2, 1) +
                                      ph4[1] * _s12(i, j + rj0 - 1, 1) +
                                      ph4[3] * _s12(i, j + rj0 + 1, 1)) +
                                 dhp4l[k][2] * _g_c(2) *
                                     (ph4[2] * _s12(i, j + rj0, 2) +
                                      ph4[0] * _s12(i, j + rj0 - 2, 2) +
                                      ph4[1] * _s12(i, j + rj0 - 1, 2) +
                                      ph4[3] * _s12(i, j + rj0 + 1, 2)) +
                                 dhp4l[k][3] * _g_c(3) *
                                     (ph4[2] * _s12(i, j + rj0, 3) +
                                      ph4[0] * _s12(i, j + rj0 - 2, 3) +
                                      ph4[1] * _s12(i, j + rj0 - 1, 3) +
                                      ph4[3] * _s12(i, j + rj0 + 1, 3)) +
                                 dhp4l[k][4] * _g_c(4) *
                                     (ph4[2] * _s12(i, j + rj0, 4) +
                                      ph4[0] * _s12(i, j + rj0 - 2, 4) +
                                      ph4[1] * _s12(i, j + rj0 - 1, 4) +
                                      ph4[3] * _s12(i, j + rj0 + 1, 4)) +
                                 dhp4l[k][5] * _g_c(5) *
                                     (ph4[2] * _s12(i, j + rj0, 5) +
                                      ph4[0] * _s12(i, j + rj0 - 2, 5) +
                                      ph4[1] * _s12(i, j + rj0 - 1, 5) +
                                      ph4[3] * _s12(i, j + rj0 + 1, 5)) +
                                 dhp4l[k][6] * _g_c(6) *
                                     (ph4[2] * _s12(i, j + rj0, 6) +
                                      ph4[0] * _s12(i, j + rj0 - 2, 6) +
                                      ph4[1] * _s12(i, j + rj0 - 1, 6) +
                                      ph4[3] * _s12(i, j + rj0 + 1, 6)) +
                                 dhp4l[k][7] * _g_c(7) *
                                     (ph4[2] * _s12(i, j + rj0, 7) +
                                      ph4[0] * _s12(i, j + rj0 - 2, 7) +
                                      ph4[1] * _s12(i, j + rj0 - 1, 7) +
                                      ph4[3] * _s12(i, j + rj0 + 1, 7)) +
                                 dhp4l[k][8] * _g_c(8) *
                                     (ph4[2] * _s12(i, j + rj0, 8) +
                                      ph4[0] * _s12(i, j + rj0 - 2, 8) +
                                      ph4[1] * _s12(i, j + rj0 - 1, 8) +
                                      ph4[3] * _s12(i, j + rj0 + 1, 8))))) *
      f_dcrj;
  _buf_u2(i, j, k) =
      (a * _u2(i, j + rj0, k) +
       Ai2 *
           (d4[1] * _f(i, j + rj0) * _g3_c(k) * _s12(i, j + rj0, k) +
            d4[0] * _f(i - 1, j + rj0) * _g3_c(k) * _s12(i - 1, j + rj0, k) +
            d4[2] * _f(i + 1, j + rj0) * _g3_c(k) * _s12(i + 1, j + rj0, k) +
            d4[3] * _f(i + 2, j + rj0) * _g3_c(k) * _s12(i + 2, j + rj0, k) +
            d4[1] * _f_c(i, j + rj0) * _g3_c(k) * _s22(i, j + rj0, k) +
            d4[0] * _f_c(i, j + rj0 - 1) * _g3_c(k) * _s22(i, j + rj0 - 1, k) +
            d4[2] * _f_c(i, j + rj0 + 1) * _g3_c(k) * _s22(i, j + rj0 + 1, k) +
            d4[3] * _f_c(i, j + rj0 + 2) * _g3_c(k) * _s22(i, j + rj0 + 2, k) +
            dh4l[k][0] * _s23(i, j + rj0, 0) +
            dh4l[k][1] * _s23(i, j + rj0, 1) +
            dh4l[k][2] * _s23(i, j + rj0, 2) +
            dh4l[k][3] * _s23(i, j + rj0, 3) +
            dh4l[k][4] * _s23(i, j + rj0, 4) +
            dh4l[k][5] * _s23(i, j + rj0, 5) +
            dh4l[k][6] * _s23(i, j + rj0, 6) -
            _f1_2(i, j + rj0) * (dhp4l[k][0] * _g_c(0) *
                                     (p4[1] * _s12(i, j + rj0, 0) +
                                      p4[0] * _s12(i - 1, j + rj0, 0) +
                                      p4[2] * _s12(i + 1, j + rj0, 0) +
                                      p4[3] * _s12(i + 2, j + rj0, 0)) +
                                 dhp4l[k][1] * _g_c(1) *
                                     (p4[1] * _s12(i, j + rj0, 1) +
                                      p4[0] * _s12(i - 1, j + rj0, 1) +
                                      p4[2] * _s12(i + 1, j + rj0, 1) +
                                      p4[3] * _s12(i + 2, j + rj0, 1)) +
                                 dhp4l[k][2] * _g_c(2) *
                                     (p4[1] * _s12(i, j + rj0, 2) +
                                      p4[0] * _s12(i - 1, j + rj0, 2) +
                                      p4[2] * _s12(i + 1, j + rj0, 2) +
                                      p4[3] * _s12(i + 2, j + rj0, 2)) +
                                 dhp4l[k][3] * _g_c(3) *
                                     (p4[1] * _s12(i, j + rj0, 3) +
                                      p4[0] * _s12(i - 1, j + rj0, 3) +
                                      p4[2] * _s12(i + 1, j + rj0, 3) +
                                      p4[3] * _s12(i + 2, j + rj0, 3)) +
                                 dhp4l[k][4] * _g_c(4) *
                                     (p4[1] * _s12(i, j + rj0, 4) +
                                      p4[0] * _s12(i - 1, j + rj0, 4) +
                                      p4[2] * _s12(i + 1, j + rj0, 4) +
                                      p4[3] * _s12(i + 2, j + rj0, 4)) +
                                 dhp4l[k][5] * _g_c(5) *
                                     (p4[1] * _s12(i, j + rj0, 5) +
                                      p4[0] * _s12(i - 1, j + rj0, 5) +
                                      p4[2] * _s12(i + 1, j + rj0, 5) +
                                      p4[3] * _s12(i + 2, j + rj0, 5)) +
                                 dhp4l[k][6] * _g_c(6) *
                                     (p4[1] * _s12(i, j + rj0, 6) +
                                      p4[0] * _s12(i - 1, j + rj0, 6) +
                                      p4[2] * _s12(i + 1, j + rj0, 6) +
                                      p4[3] * _s12(i + 2, j + rj0, 6)) +
                                 dhp4l[k][7] * _g_c(7) *
                                     (p4[1] * _s12(i, j + rj0, 7) +
                                      p4[0] * _s12(i - 1, j + rj0, 7) +
                                      p4[2] * _s12(i + 1, j + rj0, 7) +
                                      p4[3] * _s12(i + 2, j + rj0, 7)) +
                                 dhp4l[k][8] * _g_c(8) *
                                     (p4[1] * _s12(i, j + rj0, 8) +
                                      p4[0] * _s12(i - 1, j + rj0, 8) +
                                      p4[2] * _s12(i + 1, j + rj0, 8) +
                                      p4[3] * _s12(i + 2, j + rj0, 8))) -
            _f2_2(i, j + rj0) * (dhp4l[k][0] * _g_c(0) *
                                     (p4[1] * _s22(i, j + rj0, 0) +
                                      p4[0] * _s22(i, j + rj0 - 1, 0) +
                                      p4[2] * _s22(i, j + rj0 + 1, 0) +
                                      p4[3] * _s22(i, j + rj0 + 2, 0)) +
                                 dhp4l[k][1] * _g_c(1) *
                                     (p4[1] * _s22(i, j + rj0, 1) +
                                      p4[0] * _s22(i, j + rj0 - 1, 1) +
                                      p4[2] * _s22(i, j + rj0 + 1, 1) +
                                      p4[3] * _s22(i, j + rj0 + 2, 1)) +
                                 dhp4l[k][2] * _g_c(2) *
                                     (p4[1] * _s22(i, j + rj0, 2) +
                                      p4[0] * _s22(i, j + rj0 - 1, 2) +
                                      p4[2] * _s22(i, j + rj0 + 1, 2) +
                                      p4[3] * _s22(i, j + rj0 + 2, 2)) +
                                 dhp4l[k][3] * _g_c(3) *
                                     (p4[1] * _s22(i, j + rj0, 3) +
                                      p4[0] * _s22(i, j + rj0 - 1, 3) +
                                      p4[2] * _s22(i, j + rj0 + 1, 3) +
                                      p4[3] * _s22(i, j + rj0 + 2, 3)) +
                                 dhp4l[k][4] * _g_c(4) *
                                     (p4[1] * _s22(i, j + rj0, 4) +
                                      p4[0] * _s22(i, j + rj0 - 1, 4) +
                                      p4[2] * _s22(i, j + rj0 + 1, 4) +
                                      p4[3] * _s22(i, j + rj0 + 2, 4)) +
                                 dhp4l[k][5] * _g_c(5) *
                                     (p4[1] * _s22(i, j + rj0, 5) +
                                      p4[0] * _s22(i, j + rj0 - 1, 5) +
                                      p4[2] * _s22(i, j + rj0 + 1, 5) +
                                      p4[3] * _s22(i, j + rj0 + 2, 5)) +
                                 dhp4l[k][6] * _g_c(6) *
                                     (p4[1] * _s22(i, j + rj0, 6) +
                                      p4[0] * _s22(i, j + rj0 - 1, 6) +
                                      p4[2] * _s22(i, j + rj0 + 1, 6) +
                                      p4[3] * _s22(i, j + rj0 + 2, 6)) +
                                 dhp4l[k][7] * _g_c(7) *
                                     (p4[1] * _s22(i, j + rj0, 7) +
                                      p4[0] * _s22(i, j + rj0 - 1, 7) +
                                      p4[2] * _s22(i, j + rj0 + 1, 7) +
                                      p4[3] * _s22(i, j + rj0 + 2, 7)) +
                                 dhp4l[k][8] * _g_c(8) *
                                     (p4[1] * _s22(i, j + rj0, 8) +
                                      p4[0] * _s22(i, j + rj0 - 1, 8) +
                                      p4[2] * _s22(i, j + rj0 + 1, 8) +
                                      p4[3] * _s22(i, j + rj0 + 2, 8))))) *
      f_dcrj;
  _buf_u3(i, j, k) =
      (a * _u3(i, j + rj0, k) +
       Ai3 *
           (d4[1] * _f_1(i, j + rj0) * _g3(k) * _s13(i, j + rj0, k) +
            d4[0] * _f_1(i - 1, j + rj0) * _g3(k) * _s13(i - 1, j + rj0, k) +
            d4[2] * _f_1(i + 1, j + rj0) * _g3(k) * _s13(i + 1, j + rj0, k) +
            d4[3] * _f_1(i + 2, j + rj0) * _g3(k) * _s13(i + 2, j + rj0, k) +
            d4l[k][0] * _s33(i, j + rj0, 0) + d4l[k][1] * _s33(i, j + rj0, 1) +
            d4l[k][2] * _s33(i, j + rj0, 2) + d4l[k][3] * _s33(i, j + rj0, 3) +
            d4l[k][4] * _s33(i, j + rj0, 4) + d4l[k][5] * _s33(i, j + rj0, 5) +
            d4l[k][6] * _s33(i, j + rj0, 6) + d4l[k][7] * _s33(i, j + rj0, 7) +
            dh4[2] * _f_2(i, j + rj0) * _g3(k) * _s23(i, j + rj0, k) +
            dh4[0] * _f_2(i, j + rj0 - 2) * _g3(k) * _s23(i, j + rj0 - 2, k) +
            dh4[1] * _f_2(i, j + rj0 - 1) * _g3(k) * _s23(i, j + rj0 - 1, k) +
            dh4[3] * _f_2(i, j + rj0 + 1) * _g3(k) * _s23(i, j + rj0 + 1, k) -
            _f1_c(i, j + rj0) * (dph4l[k][0] * _g(0) *
                                     (p4[1] * _s13(i, j + rj0, 0) +
                                      p4[0] * _s13(i - 1, j + rj0, 0) +
                                      p4[2] * _s13(i + 1, j + rj0, 0) +
                                      p4[3] * _s13(i + 2, j + rj0, 0)) +
                                 dph4l[k][1] * _g(1) *
                                     (p4[1] * _s13(i, j + rj0, 1) +
                                      p4[0] * _s13(i - 1, j + rj0, 1) +
                                      p4[2] * _s13(i + 1, j + rj0, 1) +
                                      p4[3] * _s13(i + 2, j + rj0, 1)) +
                                 dph4l[k][2] * _g(2) *
                                     (p4[1] * _s13(i, j + rj0, 2) +
                                      p4[0] * _s13(i - 1, j + rj0, 2) +
                                      p4[2] * _s13(i + 1, j + rj0, 2) +
                                      p4[3] * _s13(i + 2, j + rj0, 2)) +
                                 dph4l[k][3] * _g(3) *
                                     (p4[1] * _s13(i, j + rj0, 3) +
                                      p4[0] * _s13(i - 1, j + rj0, 3) +
                                      p4[2] * _s13(i + 1, j + rj0, 3) +
                                      p4[3] * _s13(i + 2, j + rj0, 3)) +
                                 dph4l[k][4] * _g(4) *
                                     (p4[1] * _s13(i, j + rj0, 4) +
                                      p4[0] * _s13(i - 1, j + rj0, 4) +
                                      p4[2] * _s13(i + 1, j + rj0, 4) +
                                      p4[3] * _s13(i + 2, j + rj0, 4)) +
                                 dph4l[k][5] * _g(5) *
                                     (p4[1] * _s13(i, j + rj0, 5) +
                                      p4[0] * _s13(i - 1, j + rj0, 5) +
                                      p4[2] * _s13(i + 1, j + rj0, 5) +
                                      p4[3] * _s13(i + 2, j + rj0, 5)) +
                                 dph4l[k][6] * _g(6) *
                                     (p4[1] * _s13(i, j + rj0, 6) +
                                      p4[0] * _s13(i - 1, j + rj0, 6) +
                                      p4[2] * _s13(i + 1, j + rj0, 6) +
                                      p4[3] * _s13(i + 2, j + rj0, 6)) +
                                 dph4l[k][7] * _g(7) *
                                     (p4[1] * _s13(i, j + rj0, 7) +
                                      p4[0] * _s13(i - 1, j + rj0, 7) +
                                      p4[2] * _s13(i + 1, j + rj0, 7) +
                                      p4[3] * _s13(i + 2, j + rj0, 7)) +
                                 dph4l[k][8] * _g(8) *
                                     (p4[1] * _s13(i, j + rj0, 8) +
                                      p4[0] * _s13(i - 1, j + rj0, 8) +
                                      p4[2] * _s13(i + 1, j + rj0, 8) +
                                      p4[3] * _s13(i + 2, j + rj0, 8))) -
            _f2_c(i, j + rj0) * (dph4l[k][0] * _g(0) *
                                     (ph4[2] * _s23(i, j + rj0, 0) +
                                      ph4[0] * _s23(i, j + rj0 - 2, 0) +
                                      ph4[1] * _s23(i, j + rj0 - 1, 0) +
                                      ph4[3] * _s23(i, j + rj0 + 1, 0)) +
                                 dph4l[k][1] * _g(1) *
                                     (ph4[2] * _s23(i, j + rj0, 1) +
                                      ph4[0] * _s23(i, j + rj0 - 2, 1) +
                                      ph4[1] * _s23(i, j + rj0 - 1, 1) +
                                      ph4[3] * _s23(i, j + rj0 + 1, 1)) +
                                 dph4l[k][2] * _g(2) *
                                     (ph4[2] * _s23(i, j + rj0, 2) +
                                      ph4[0] * _s23(i, j + rj0 - 2, 2) +
                                      ph4[1] * _s23(i, j + rj0 - 1, 2) +
                                      ph4[3] * _s23(i, j + rj0 + 1, 2)) +
                                 dph4l[k][3] * _g(3) *
                                     (ph4[2] * _s23(i, j + rj0, 3) +
                                      ph4[0] * _s23(i, j + rj0 - 2, 3) +
                                      ph4[1] * _s23(i, j + rj0 - 1, 3) +
                                      ph4[3] * _s23(i, j + rj0 + 1, 3)) +
                                 dph4l[k][4] * _g(4) *
                                     (ph4[2] * _s23(i, j + rj0, 4) +
                                      ph4[0] * _s23(i, j + rj0 - 2, 4) +
                                      ph4[1] * _s23(i, j + rj0 - 1, 4) +
                                      ph4[3] * _s23(i, j + rj0 + 1, 4)) +
                                 dph4l[k][5] * _g(5) *
                                     (ph4[2] * _s23(i, j + rj0, 5) +
                                      ph4[0] * _s23(i, j + rj0 - 2, 5) +
                                      ph4[1] * _s23(i, j + rj0 - 1, 5) +
                                      ph4[3] * _s23(i, j + rj0 + 1, 5)) +
                                 dph4l[k][6] * _g(6) *
                                     (ph4[2] * _s23(i, j + rj0, 6) +
                                      ph4[0] * _s23(i, j + rj0 - 2, 6) +
                                      ph4[1] * _s23(i, j + rj0 - 1, 6) +
                                      ph4[3] * _s23(i, j + rj0 + 1, 6)) +
                                 dph4l[k][7] * _g(7) *
                                     (ph4[2] * _s23(i, j + rj0, 7) +
                                      ph4[0] * _s23(i, j + rj0 - 2, 7) +
                                      ph4[1] * _s23(i, j + rj0 - 1, 7) +
                                      ph4[3] * _s23(i, j + rj0 + 1, 7)) +
                                 dph4l[k][8] * _g(8) *
                                     (ph4[2] * _s23(i, j + rj0, 8) +
                                      ph4[0] * _s23(i, j + rj0 - 2, 8) +
                                      ph4[1] * _s23(i, j + rj0 - 1, 8) +
                                      ph4[3] * _s23(i, j + rj0 + 1, 8))))) *
      f_dcrj;
#undef _rho
#undef _f_1
#undef _g3_c
#undef _f_2
#undef _g3
#undef _f_c
#undef _dcrjz
#undef _dcrjx
#undef _dcrjy
#undef _g_c
#undef _s11
#undef _f2_1
#undef _u1
#undef _f1_1
#undef _s12
#undef _f
#undef _s13
#undef _f1_2
#undef _s22
#undef _s23
#undef _u2
#undef _f2_2
#undef _f2_c
#undef _g
#undef _f1_c
#undef _s33
#undef _u3
#undef _buf_u1
#undef _buf_u2
#undef _buf_u3
}

__global__ void dtopo_buf_vel_111(
    float *__restrict__ buf_u1, float *__restrict__ buf_u2,
    float *__restrict__ buf_u3, const float *__restrict__ dcrjx,
    const float *__restrict__ dcrjy, const float *__restrict__ dcrjz,
    const float *__restrict__ f, const float *__restrict__ f1_1,
    const float *__restrict__ f1_2, const float *__restrict__ f1_c,
    const float *__restrict__ f2_1, const float *__restrict__ f2_2,
    const float *__restrict__ f2_c, const float *__restrict__ f_1,
    const float *__restrict__ f_2, const float *__restrict__ f_c,
    const float *__restrict__ g, const float *__restrict__ g3,
    const float *__restrict__ g3_c, const float *__restrict__ g_c,
    const float *__restrict__ rho, const float *__restrict__ s11,
    const float *__restrict__ s12, const float *__restrict__ s13,
    const float *__restrict__ s22, const float *__restrict__ s23,
    const float *__restrict__ s33, const float *__restrict__ u1,
    const float *__restrict__ u2, const float *__restrict__ u3, const float a,
    const float nu, const int nx, const int ny, const int nz, const int bj,
    const int ej, const int rj0) {
  const float ph2[2] = {0.5000000000000000, 0.5000000000000000};
  const float dhp4[7] = {-0.0026041666666667, 0.0937500000000000,
                         -0.6796875000000000, 0.0000000000000000,
                         0.6796875000000000,  -0.0937500000000000,
                         0.0026041666666667};
  const float ph4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float dh4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float p4[4] = {-0.0625000000000000, 0.5625000000000000,
                       0.5625000000000000, -0.0625000000000000};
  const float d4[4] = {0.0416666666666667, -1.1250000000000000,
                       1.1250000000000000, -0.0416666666666667};
  const float dph4[7] = {-0.0026041666666667, 0.0937500000000000,
                         -0.6796875000000000, 0.0000000000000000,
                         0.6796875000000000,  -0.0937500000000000,
                         0.0026041666666667};
  const int i = threadIdx.z + blockIdx.z * blockDim.z;
  if (i >= nx)
    return;
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ny)
    return;
  if (j >= ej)
    return;
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= nz - 12)
    return;
#define _rho(i, j, k)                                                          \
  rho[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f_1(i, j)                                                             \
  f_1[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3_c(k) g3_c[(k) + align]
#define _f_2(i, j)                                                             \
  f_2[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3(k) g3[(k) + align]
#define _f_c(i, j)                                                             \
  f_c[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _dcrjz(k) dcrjz[(k) + align]
#define _dcrjx(i) dcrjx[(i) + ngsl + 2]
#define _dcrjy(j) dcrjy[(j) + ngsl + 2]
#define _g_c(k) g_c[(k) + align]
#define _s11(i, j, k)                                                          \
  s11[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_1(i, j)                                                            \
  f2_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _u1(i, j, k)                                                           \
  u1[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_1(i, j)                                                            \
  f1_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s12(i, j, k)                                                          \
  s12[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f(i, j)                                                               \
  f[(j) + align + ngsl + ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s13(i, j, k)                                                          \
  s13[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_2(i, j)                                                            \
  f1_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s22(i, j, k)                                                          \
  s22[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s23(i, j, k)                                                          \
  s23[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _u2(i, j, k)                                                           \
  u2[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_2(i, j)                                                            \
  f2_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_c(i, j)                                                            \
  f2_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g(k) g[(k) + align]
#define _f1_c(i, j)                                                            \
  f1_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s33(i, j, k)                                                          \
  s33[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _u3(i, j, k)                                                           \
  u3[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _buf_u1(i, j, k)                                                       \
  buf_u1[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
#define _buf_u2(i, j, k)                                                       \
  buf_u2[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
#define _buf_u3(i, j, k)                                                       \
  buf_u3[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
  float rho1 = ph2[0] * (ph2[1] * _rho(i, j + rj0, k + 5) +
                         ph2[0] * _rho(i, j + rj0 - 1, k + 5)) +
               ph2[1] * (ph2[1] * _rho(i, j + rj0, k + 6) +
                         ph2[0] * _rho(i, j + rj0 - 1, k + 6));
  float rho2 = ph2[0] * (ph2[1] * _rho(i, j + rj0, k + 5) +
                         ph2[0] * _rho(i - 1, j + rj0, k + 5)) +
               ph2[1] * (ph2[1] * _rho(i, j + rj0, k + 6) +
                         ph2[0] * _rho(i - 1, j + rj0, k + 6));
  float rho3 = ph2[1] * (ph2[1] * _rho(i, j + rj0, k + 6) +
                         ph2[0] * _rho(i - 1, j + rj0, k + 6)) +
               ph2[0] * (ph2[0] * _rho(i, j + rj0 - 1, k + 6) +
                         ph2[0] * _rho(i - 1, j + rj0 - 1, k + 6));
  float Ai1 = _f_1(i, j + rj0) * _g3_c(k + 6) * rho1;
  Ai1 = nu * 1.0 / Ai1;
  float Ai2 = _f_2(i, j + rj0) * _g3_c(k + 6) * rho2;
  Ai2 = nu * 1.0 / Ai2;
  float Ai3 = _f_c(i, j + rj0) * _g3(k + 6) * rho3;
  Ai3 = nu * 1.0 / Ai3;
  float f_dcrj = _dcrjx(i) * _dcrjy(j + rj0) * _dcrjz(k + 6);
  _buf_u1(i, j, k + 6) =
      (a * _u1(i, j + rj0, k + 6) +
       Ai1 *
           (dh4[2] * _f(i, j + rj0) * _g3_c(k + 6) * _s12(i, j + rj0, k + 6) +
            dh4[0] * _f(i, j + rj0 - 2) * _g3_c(k + 6) *
                _s12(i, j + rj0 - 2, k + 6) +
            dh4[1] * _f(i, j + rj0 - 1) * _g3_c(k + 6) *
                _s12(i, j + rj0 - 1, k + 6) +
            dh4[3] * _f(i, j + rj0 + 1) * _g3_c(k + 6) *
                _s12(i, j + rj0 + 1, k + 6) +
            dh4[2] * _f_c(i, j + rj0) * _g3_c(k + 6) * _s11(i, j + rj0, k + 6) +
            dh4[0] * _f_c(i - 2, j + rj0) * _g3_c(k + 6) *
                _s11(i - 2, j + rj0, k + 6) +
            dh4[1] * _f_c(i - 1, j + rj0) * _g3_c(k + 6) *
                _s11(i - 1, j + rj0, k + 6) +
            dh4[3] * _f_c(i + 1, j + rj0) * _g3_c(k + 6) *
                _s11(i + 1, j + rj0, k + 6) +
            dh4[0] * _s13(i, j + rj0, k + 4) +
            dh4[1] * _s13(i, j + rj0, k + 5) +
            dh4[2] * _s13(i, j + rj0, k + 6) +
            dh4[3] * _s13(i, j + rj0, k + 7) -
            _f1_1(i, j + rj0) * (dhp4[0] * _g_c(k + 3) *
                                     (ph4[2] * _s11(i, j + rj0, k + 3) +
                                      ph4[0] * _s11(i - 2, j + rj0, k + 3) +
                                      ph4[1] * _s11(i - 1, j + rj0, k + 3) +
                                      ph4[3] * _s11(i + 1, j + rj0, k + 3)) +
                                 dhp4[1] * _g_c(k + 4) *
                                     (ph4[2] * _s11(i, j + rj0, k + 4) +
                                      ph4[0] * _s11(i - 2, j + rj0, k + 4) +
                                      ph4[1] * _s11(i - 1, j + rj0, k + 4) +
                                      ph4[3] * _s11(i + 1, j + rj0, k + 4)) +
                                 dhp4[2] * _g_c(k + 5) *
                                     (ph4[2] * _s11(i, j + rj0, k + 5) +
                                      ph4[0] * _s11(i - 2, j + rj0, k + 5) +
                                      ph4[1] * _s11(i - 1, j + rj0, k + 5) +
                                      ph4[3] * _s11(i + 1, j + rj0, k + 5)) +
                                 dhp4[3] * _g_c(k + 6) *
                                     (ph4[2] * _s11(i, j + rj0, k + 6) +
                                      ph4[0] * _s11(i - 2, j + rj0, k + 6) +
                                      ph4[1] * _s11(i - 1, j + rj0, k + 6) +
                                      ph4[3] * _s11(i + 1, j + rj0, k + 6)) +
                                 dhp4[4] * _g_c(k + 7) *
                                     (ph4[2] * _s11(i, j + rj0, k + 7) +
                                      ph4[0] * _s11(i - 2, j + rj0, k + 7) +
                                      ph4[1] * _s11(i - 1, j + rj0, k + 7) +
                                      ph4[3] * _s11(i + 1, j + rj0, k + 7)) +
                                 dhp4[5] * _g_c(k + 8) *
                                     (ph4[2] * _s11(i, j + rj0, k + 8) +
                                      ph4[0] * _s11(i - 2, j + rj0, k + 8) +
                                      ph4[1] * _s11(i - 1, j + rj0, k + 8) +
                                      ph4[3] * _s11(i + 1, j + rj0, k + 8)) +
                                 dhp4[6] * _g_c(k + 9) *
                                     (ph4[2] * _s11(i, j + rj0, k + 9) +
                                      ph4[0] * _s11(i - 2, j + rj0, k + 9) +
                                      ph4[1] * _s11(i - 1, j + rj0, k + 9) +
                                      ph4[3] * _s11(i + 1, j + rj0, k + 9))) -
            _f2_1(i, j + rj0) * (dhp4[0] * _g_c(k + 3) *
                                     (ph4[2] * _s12(i, j + rj0, k + 3) +
                                      ph4[0] * _s12(i, j + rj0 - 2, k + 3) +
                                      ph4[1] * _s12(i, j + rj0 - 1, k + 3) +
                                      ph4[3] * _s12(i, j + rj0 + 1, k + 3)) +
                                 dhp4[1] * _g_c(k + 4) *
                                     (ph4[2] * _s12(i, j + rj0, k + 4) +
                                      ph4[0] * _s12(i, j + rj0 - 2, k + 4) +
                                      ph4[1] * _s12(i, j + rj0 - 1, k + 4) +
                                      ph4[3] * _s12(i, j + rj0 + 1, k + 4)) +
                                 dhp4[2] * _g_c(k + 5) *
                                     (ph4[2] * _s12(i, j + rj0, k + 5) +
                                      ph4[0] * _s12(i, j + rj0 - 2, k + 5) +
                                      ph4[1] * _s12(i, j + rj0 - 1, k + 5) +
                                      ph4[3] * _s12(i, j + rj0 + 1, k + 5)) +
                                 dhp4[3] * _g_c(k + 6) *
                                     (ph4[2] * _s12(i, j + rj0, k + 6) +
                                      ph4[0] * _s12(i, j + rj0 - 2, k + 6) +
                                      ph4[1] * _s12(i, j + rj0 - 1, k + 6) +
                                      ph4[3] * _s12(i, j + rj0 + 1, k + 6)) +
                                 dhp4[4] * _g_c(k + 7) *
                                     (ph4[2] * _s12(i, j + rj0, k + 7) +
                                      ph4[0] * _s12(i, j + rj0 - 2, k + 7) +
                                      ph4[1] * _s12(i, j + rj0 - 1, k + 7) +
                                      ph4[3] * _s12(i, j + rj0 + 1, k + 7)) +
                                 dhp4[5] * _g_c(k + 8) *
                                     (ph4[2] * _s12(i, j + rj0, k + 8) +
                                      ph4[0] * _s12(i, j + rj0 - 2, k + 8) +
                                      ph4[1] * _s12(i, j + rj0 - 1, k + 8) +
                                      ph4[3] * _s12(i, j + rj0 + 1, k + 8)) +
                                 dhp4[6] * _g_c(k + 9) *
                                     (ph4[2] * _s12(i, j + rj0, k + 9) +
                                      ph4[0] * _s12(i, j + rj0 - 2, k + 9) +
                                      ph4[1] * _s12(i, j + rj0 - 1, k + 9) +
                                      ph4[3] * _s12(i, j + rj0 + 1, k + 9))))) *
      f_dcrj;
  _buf_u2(i, j, k + 6) =
      (a * _u2(i, j + rj0, k + 6) +
       Ai2 *
           (d4[1] * _f(i, j + rj0) * _g3_c(k + 6) * _s12(i, j + rj0, k + 6) +
            d4[0] * _f(i - 1, j + rj0) * _g3_c(k + 6) *
                _s12(i - 1, j + rj0, k + 6) +
            d4[2] * _f(i + 1, j + rj0) * _g3_c(k + 6) *
                _s12(i + 1, j + rj0, k + 6) +
            d4[3] * _f(i + 2, j + rj0) * _g3_c(k + 6) *
                _s12(i + 2, j + rj0, k + 6) +
            d4[1] * _f_c(i, j + rj0) * _g3_c(k + 6) * _s22(i, j + rj0, k + 6) +
            d4[0] * _f_c(i, j + rj0 - 1) * _g3_c(k + 6) *
                _s22(i, j + rj0 - 1, k + 6) +
            d4[2] * _f_c(i, j + rj0 + 1) * _g3_c(k + 6) *
                _s22(i, j + rj0 + 1, k + 6) +
            d4[3] * _f_c(i, j + rj0 + 2) * _g3_c(k + 6) *
                _s22(i, j + rj0 + 2, k + 6) +
            dh4[0] * _s23(i, j + rj0, k + 4) +
            dh4[1] * _s23(i, j + rj0, k + 5) +
            dh4[2] * _s23(i, j + rj0, k + 6) +
            dh4[3] * _s23(i, j + rj0, k + 7) -
            _f1_2(i, j + rj0) * (dhp4[0] * _g_c(k + 3) *
                                     (p4[1] * _s12(i, j + rj0, k + 3) +
                                      p4[0] * _s12(i - 1, j + rj0, k + 3) +
                                      p4[2] * _s12(i + 1, j + rj0, k + 3) +
                                      p4[3] * _s12(i + 2, j + rj0, k + 3)) +
                                 dhp4[1] * _g_c(k + 4) *
                                     (p4[1] * _s12(i, j + rj0, k + 4) +
                                      p4[0] * _s12(i - 1, j + rj0, k + 4) +
                                      p4[2] * _s12(i + 1, j + rj0, k + 4) +
                                      p4[3] * _s12(i + 2, j + rj0, k + 4)) +
                                 dhp4[2] * _g_c(k + 5) *
                                     (p4[1] * _s12(i, j + rj0, k + 5) +
                                      p4[0] * _s12(i - 1, j + rj0, k + 5) +
                                      p4[2] * _s12(i + 1, j + rj0, k + 5) +
                                      p4[3] * _s12(i + 2, j + rj0, k + 5)) +
                                 dhp4[3] * _g_c(k + 6) *
                                     (p4[1] * _s12(i, j + rj0, k + 6) +
                                      p4[0] * _s12(i - 1, j + rj0, k + 6) +
                                      p4[2] * _s12(i + 1, j + rj0, k + 6) +
                                      p4[3] * _s12(i + 2, j + rj0, k + 6)) +
                                 dhp4[4] * _g_c(k + 7) *
                                     (p4[1] * _s12(i, j + rj0, k + 7) +
                                      p4[0] * _s12(i - 1, j + rj0, k + 7) +
                                      p4[2] * _s12(i + 1, j + rj0, k + 7) +
                                      p4[3] * _s12(i + 2, j + rj0, k + 7)) +
                                 dhp4[5] * _g_c(k + 8) *
                                     (p4[1] * _s12(i, j + rj0, k + 8) +
                                      p4[0] * _s12(i - 1, j + rj0, k + 8) +
                                      p4[2] * _s12(i + 1, j + rj0, k + 8) +
                                      p4[3] * _s12(i + 2, j + rj0, k + 8)) +
                                 dhp4[6] * _g_c(k + 9) *
                                     (p4[1] * _s12(i, j + rj0, k + 9) +
                                      p4[0] * _s12(i - 1, j + rj0, k + 9) +
                                      p4[2] * _s12(i + 1, j + rj0, k + 9) +
                                      p4[3] * _s12(i + 2, j + rj0, k + 9))) -
            _f2_2(i, j + rj0) * (dhp4[0] * _g_c(k + 3) *
                                     (p4[1] * _s22(i, j + rj0, k + 3) +
                                      p4[0] * _s22(i, j + rj0 - 1, k + 3) +
                                      p4[2] * _s22(i, j + rj0 + 1, k + 3) +
                                      p4[3] * _s22(i, j + rj0 + 2, k + 3)) +
                                 dhp4[1] * _g_c(k + 4) *
                                     (p4[1] * _s22(i, j + rj0, k + 4) +
                                      p4[0] * _s22(i, j + rj0 - 1, k + 4) +
                                      p4[2] * _s22(i, j + rj0 + 1, k + 4) +
                                      p4[3] * _s22(i, j + rj0 + 2, k + 4)) +
                                 dhp4[2] * _g_c(k + 5) *
                                     (p4[1] * _s22(i, j + rj0, k + 5) +
                                      p4[0] * _s22(i, j + rj0 - 1, k + 5) +
                                      p4[2] * _s22(i, j + rj0 + 1, k + 5) +
                                      p4[3] * _s22(i, j + rj0 + 2, k + 5)) +
                                 dhp4[3] * _g_c(k + 6) *
                                     (p4[1] * _s22(i, j + rj0, k + 6) +
                                      p4[0] * _s22(i, j + rj0 - 1, k + 6) +
                                      p4[2] * _s22(i, j + rj0 + 1, k + 6) +
                                      p4[3] * _s22(i, j + rj0 + 2, k + 6)) +
                                 dhp4[4] * _g_c(k + 7) *
                                     (p4[1] * _s22(i, j + rj0, k + 7) +
                                      p4[0] * _s22(i, j + rj0 - 1, k + 7) +
                                      p4[2] * _s22(i, j + rj0 + 1, k + 7) +
                                      p4[3] * _s22(i, j + rj0 + 2, k + 7)) +
                                 dhp4[5] * _g_c(k + 8) *
                                     (p4[1] * _s22(i, j + rj0, k + 8) +
                                      p4[0] * _s22(i, j + rj0 - 1, k + 8) +
                                      p4[2] * _s22(i, j + rj0 + 1, k + 8) +
                                      p4[3] * _s22(i, j + rj0 + 2, k + 8)) +
                                 dhp4[6] * _g_c(k + 9) *
                                     (p4[1] * _s22(i, j + rj0, k + 9) +
                                      p4[0] * _s22(i, j + rj0 - 1, k + 9) +
                                      p4[2] * _s22(i, j + rj0 + 1, k + 9) +
                                      p4[3] * _s22(i, j + rj0 + 2, k + 9))))) *
      f_dcrj;
  _buf_u3(i, j, k + 6) =
      (a * _u3(i, j + rj0, k + 6) +
       Ai3 *
           (d4[1] * _f_1(i, j + rj0) * _g3(k + 6) * _s13(i, j + rj0, k + 6) +
            d4[0] * _f_1(i - 1, j + rj0) * _g3(k + 6) *
                _s13(i - 1, j + rj0, k + 6) +
            d4[2] * _f_1(i + 1, j + rj0) * _g3(k + 6) *
                _s13(i + 1, j + rj0, k + 6) +
            d4[3] * _f_1(i + 2, j + rj0) * _g3(k + 6) *
                _s13(i + 2, j + rj0, k + 6) +
            d4[0] * _s33(i, j + rj0, k + 5) + d4[1] * _s33(i, j + rj0, k + 6) +
            d4[2] * _s33(i, j + rj0, k + 7) + d4[3] * _s33(i, j + rj0, k + 8) +
            dh4[2] * _f_2(i, j + rj0) * _g3(k + 6) * _s23(i, j + rj0, k + 6) +
            dh4[0] * _f_2(i, j + rj0 - 2) * _g3(k + 6) *
                _s23(i, j + rj0 - 2, k + 6) +
            dh4[1] * _f_2(i, j + rj0 - 1) * _g3(k + 6) *
                _s23(i, j + rj0 - 1, k + 6) +
            dh4[3] * _f_2(i, j + rj0 + 1) * _g3(k + 6) *
                _s23(i, j + rj0 + 1, k + 6) -
            _f1_c(i, j + rj0) * (dph4[0] * _g(k + 3) *
                                     (p4[1] * _s13(i, j + rj0, k + 3) +
                                      p4[0] * _s13(i - 1, j + rj0, k + 3) +
                                      p4[2] * _s13(i + 1, j + rj0, k + 3) +
                                      p4[3] * _s13(i + 2, j + rj0, k + 3)) +
                                 dph4[1] * _g(k + 4) *
                                     (p4[1] * _s13(i, j + rj0, k + 4) +
                                      p4[0] * _s13(i - 1, j + rj0, k + 4) +
                                      p4[2] * _s13(i + 1, j + rj0, k + 4) +
                                      p4[3] * _s13(i + 2, j + rj0, k + 4)) +
                                 dph4[2] * _g(k + 5) *
                                     (p4[1] * _s13(i, j + rj0, k + 5) +
                                      p4[0] * _s13(i - 1, j + rj0, k + 5) +
                                      p4[2] * _s13(i + 1, j + rj0, k + 5) +
                                      p4[3] * _s13(i + 2, j + rj0, k + 5)) +
                                 dph4[3] * _g(k + 6) *
                                     (p4[1] * _s13(i, j + rj0, k + 6) +
                                      p4[0] * _s13(i - 1, j + rj0, k + 6) +
                                      p4[2] * _s13(i + 1, j + rj0, k + 6) +
                                      p4[3] * _s13(i + 2, j + rj0, k + 6)) +
                                 dph4[4] * _g(k + 7) *
                                     (p4[1] * _s13(i, j + rj0, k + 7) +
                                      p4[0] * _s13(i - 1, j + rj0, k + 7) +
                                      p4[2] * _s13(i + 1, j + rj0, k + 7) +
                                      p4[3] * _s13(i + 2, j + rj0, k + 7)) +
                                 dph4[5] * _g(k + 8) *
                                     (p4[1] * _s13(i, j + rj0, k + 8) +
                                      p4[0] * _s13(i - 1, j + rj0, k + 8) +
                                      p4[2] * _s13(i + 1, j + rj0, k + 8) +
                                      p4[3] * _s13(i + 2, j + rj0, k + 8)) +
                                 dph4[6] * _g(k + 9) *
                                     (p4[1] * _s13(i, j + rj0, k + 9) +
                                      p4[0] * _s13(i - 1, j + rj0, k + 9) +
                                      p4[2] * _s13(i + 1, j + rj0, k + 9) +
                                      p4[3] * _s13(i + 2, j + rj0, k + 9))) -
            _f2_c(i, j + rj0) * (dph4[0] * _g(k + 3) *
                                     (ph4[2] * _s23(i, j + rj0, k + 3) +
                                      ph4[0] * _s23(i, j + rj0 - 2, k + 3) +
                                      ph4[1] * _s23(i, j + rj0 - 1, k + 3) +
                                      ph4[3] * _s23(i, j + rj0 + 1, k + 3)) +
                                 dph4[1] * _g(k + 4) *
                                     (ph4[2] * _s23(i, j + rj0, k + 4) +
                                      ph4[0] * _s23(i, j + rj0 - 2, k + 4) +
                                      ph4[1] * _s23(i, j + rj0 - 1, k + 4) +
                                      ph4[3] * _s23(i, j + rj0 + 1, k + 4)) +
                                 dph4[2] * _g(k + 5) *
                                     (ph4[2] * _s23(i, j + rj0, k + 5) +
                                      ph4[0] * _s23(i, j + rj0 - 2, k + 5) +
                                      ph4[1] * _s23(i, j + rj0 - 1, k + 5) +
                                      ph4[3] * _s23(i, j + rj0 + 1, k + 5)) +
                                 dph4[3] * _g(k + 6) *
                                     (ph4[2] * _s23(i, j + rj0, k + 6) +
                                      ph4[0] * _s23(i, j + rj0 - 2, k + 6) +
                                      ph4[1] * _s23(i, j + rj0 - 1, k + 6) +
                                      ph4[3] * _s23(i, j + rj0 + 1, k + 6)) +
                                 dph4[4] * _g(k + 7) *
                                     (ph4[2] * _s23(i, j + rj0, k + 7) +
                                      ph4[0] * _s23(i, j + rj0 - 2, k + 7) +
                                      ph4[1] * _s23(i, j + rj0 - 1, k + 7) +
                                      ph4[3] * _s23(i, j + rj0 + 1, k + 7)) +
                                 dph4[5] * _g(k + 8) *
                                     (ph4[2] * _s23(i, j + rj0, k + 8) +
                                      ph4[0] * _s23(i, j + rj0 - 2, k + 8) +
                                      ph4[1] * _s23(i, j + rj0 - 1, k + 8) +
                                      ph4[3] * _s23(i, j + rj0 + 1, k + 8)) +
                                 dph4[6] * _g(k + 9) *
                                     (ph4[2] * _s23(i, j + rj0, k + 9) +
                                      ph4[0] * _s23(i, j + rj0 - 2, k + 9) +
                                      ph4[1] * _s23(i, j + rj0 - 1, k + 9) +
                                      ph4[3] * _s23(i, j + rj0 + 1, k + 9))))) *
      f_dcrj;
#undef _rho
#undef _f_1
#undef _g3_c
#undef _f_2
#undef _g3
#undef _f_c
#undef _dcrjz
#undef _dcrjx
#undef _dcrjy
#undef _g_c
#undef _s11
#undef _f2_1
#undef _u1
#undef _f1_1
#undef _s12
#undef _f
#undef _s13
#undef _f1_2
#undef _s22
#undef _s23
#undef _u2
#undef _f2_2
#undef _f2_c
#undef _g
#undef _f1_c
#undef _s33
#undef _u3
#undef _buf_u1
#undef _buf_u2
#undef _buf_u3
}

__global__ void dtopo_buf_vel_112(
    float *__restrict__ buf_u1, float *__restrict__ buf_u2,
    float *__restrict__ buf_u3, const float *__restrict__ dcrjx,
    const float *__restrict__ dcrjy, const float *__restrict__ dcrjz,
    const float *__restrict__ f, const float *__restrict__ f1_1,
    const float *__restrict__ f1_2, const float *__restrict__ f1_c,
    const float *__restrict__ f2_1, const float *__restrict__ f2_2,
    const float *__restrict__ f2_c, const float *__restrict__ f_1,
    const float *__restrict__ f_2, const float *__restrict__ f_c,
    const float *__restrict__ g, const float *__restrict__ g3,
    const float *__restrict__ g3_c, const float *__restrict__ g_c,
    const float *__restrict__ rho, const float *__restrict__ s11,
    const float *__restrict__ s12, const float *__restrict__ s13,
    const float *__restrict__ s22, const float *__restrict__ s23,
    const float *__restrict__ s33, const float *__restrict__ u1,
    const float *__restrict__ u2, const float *__restrict__ u3, const float a,
    const float nu, const int nx, const int ny, const int nz, const int bj,
    const int ej, const int rj0) {
  const float ph2r[6][8] = {
      {0.0000000000000000, 1.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.5000000000000000, 0.5000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.5000000000000000,
       0.5000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.5000000000000000, 0.5000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.5000000000000000, 0.5000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.5000000000000000,
       0.5000000000000000, 0.0000000000000000}};
  const float ph2[2] = {0.5000000000000000, 0.5000000000000000};
  const float dhp4r[6][9] = {
      {-1.5373923010673118, -1.1059180740634813, -0.2134752473866528,
       -0.0352027995732726, -0.0075022330101095, -0.0027918394266035,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.8139439685257414, 0.1273679143938725, -1.1932750007455710,
       0.1475120181828087, 0.1125814499297686, -0.0081303502866204,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.1639182541610305, 0.3113839909089030, -0.0536007135209480,
       -0.3910958927076030, -0.0401741813821989, 0.0095685425408165,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.0171478318814576, -0.0916600077207278, 0.7187220404622645,
       -0.1434031863528334, -0.5827389738506837, 0.0847863081664324,
       -0.0028540125859095, 0.0000000000000000, 0.0000000000000000},
      {-0.0579176640853654, 0.0022069616616207, 0.0108792602269819,
       0.6803612607837533, -0.0530169938441240, -0.6736586580761996,
       0.0937500000000000, -0.0026041666666667, 0.0000000000000000},
      {0.0020323834153791, 0.0002106933140862, -0.0013351454085978,
       -0.0938400881871787, 0.6816971139746001, -0.0002232904416222,
       -0.6796875000000000, 0.0937500000000000, -0.0026041666666667}};
  const float ph4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float dh4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float dh4r[6][8] = {
      {0.0000000000000000, -1.4511412472637157, -1.8534237417911470,
       0.3534237417911469, 0.0488587527362844, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.8577143189081458, -0.5731429567244373,
       -0.4268570432755628, 0.1422856810918542, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.1674548505882877, 0.4976354482351368,
       -0.4976354482351368, -0.1674548505882877, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, -0.1027061113405124, 0.2624541326469860,
       0.8288742701021167, -1.0342864927831414, 0.0456642013745513,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       -0.0416666666666667, 1.1250000000000000, -1.1250000000000000,
       0.0416666666666667, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, -0.0416666666666667, 1.1250000000000000,
       -1.1250000000000000, 0.0416666666666667}};
  const float p4[4] = {-0.0625000000000000, 0.5625000000000000,
                       0.5625000000000000, -0.0625000000000000};
  const float d4[4] = {0.0416666666666667, -1.1250000000000000,
                       1.1250000000000000, -0.0416666666666667};
  const float dph4r[6][9] = {
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, -1.5886075042755421, -2.4835574634505861,
       0.0421173406787286, 0.4968761536590695, -0.0228264197210198,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.4428256655817484, -0.0574614517751294,
       -0.2022259589759502, -0.1944663890497050, 0.0113281342190362,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, -0.3360140866060758, 1.2113298407847195,
       -0.3111668377093505, -0.6714462506479002, 0.1111440843153523,
       -0.0038467501367455, 0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0338560531369653, -0.0409943223643902,
       0.5284757132923059, 0.0115571196122084, -0.6162252315536446,
       0.0857115441015996, -0.0023808762250444, 0.0000000000000000},
      {0.0000000000000000, -0.0040378273193044, 0.0064139372778371,
       -0.0890062133451850, 0.6749219241340761, 0.0002498459192428,
       -0.6796875000000000, 0.0937500000000000, -0.0026041666666667}};
  const float d4r[6][7] = {
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {-1.7779989465546753, -1.3337480247900155, -0.7775013168066564,
       0.3332503950419969, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.4410217341392059, 0.1730842484889890, -0.4487228323259926,
       -0.1653831503022022, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {-0.1798793213882701, 0.2757257254150788, 0.9597948548284453,
       -1.1171892610431817, 0.0615480021879277, 0.0000000000000000,
       0.0000000000000000},
      {-0.0153911381507088, -0.0568851455503591, 0.1998976464597171,
       0.8628231468598346, -1.0285385292191949, 0.0380940196007109,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       -0.0416666666666667, 1.1250000000000000, -1.1250000000000000,
       0.0416666666666667}};
  const int i = threadIdx.z + blockIdx.z * blockDim.z;
  if (i >= nx)
    return;
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ny)
    return;
  if (j >= ej)
    return;
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= 6)
    return;
#define _rho(i, j, k)                                                          \
  rho[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f_1(i, j)                                                             \
  f_1[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3_c(k) g3_c[(k) + align]
#define _f_2(i, j)                                                             \
  f_2[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3(k) g3[(k) + align]
#define _f_c(i, j)                                                             \
  f_c[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _dcrjz(k) dcrjz[(k) + align]
#define _dcrjx(i) dcrjx[(i) + ngsl + 2]
#define _dcrjy(j) dcrjy[(j) + ngsl + 2]
#define _g_c(k) g_c[(k) + align]
#define _s11(i, j, k)                                                          \
  s11[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_1(i, j)                                                            \
  f2_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _u1(i, j, k)                                                           \
  u1[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_1(i, j)                                                            \
  f1_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s12(i, j, k)                                                          \
  s12[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f(i, j)                                                               \
  f[(j) + align + ngsl + ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s13(i, j, k)                                                          \
  s13[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_2(i, j)                                                            \
  f1_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s22(i, j, k)                                                          \
  s22[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s23(i, j, k)                                                          \
  s23[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _u2(i, j, k)                                                           \
  u2[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_2(i, j)                                                            \
  f2_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_c(i, j)                                                            \
  f2_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g(k) g[(k) + align]
#define _f1_c(i, j)                                                            \
  f1_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s33(i, j, k)                                                          \
  s33[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _u3(i, j, k)                                                           \
  u3[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _buf_u1(i, j, k)                                                       \
  buf_u1[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
#define _buf_u2(i, j, k)                                                       \
  buf_u2[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
#define _buf_u3(i, j, k)                                                       \
  buf_u3[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
  float rho1 = ph2r[k][7] * (ph2[1] * _rho(i, j + rj0, nz - 8) +
                             ph2[0] * _rho(i, j + rj0 - 1, nz - 8)) +
               ph2r[k][6] * (ph2[1] * _rho(i, j + rj0, nz - 7) +
                             ph2[0] * _rho(i, j + rj0 - 1, nz - 7)) +
               ph2r[k][5] * (ph2[1] * _rho(i, j + rj0, nz - 6) +
                             ph2[0] * _rho(i, j + rj0 - 1, nz - 6)) +
               ph2r[k][4] * (ph2[1] * _rho(i, j + rj0, nz - 5) +
                             ph2[0] * _rho(i, j + rj0 - 1, nz - 5)) +
               ph2r[k][3] * (ph2[1] * _rho(i, j + rj0, nz - 4) +
                             ph2[0] * _rho(i, j + rj0 - 1, nz - 4)) +
               ph2r[k][2] * (ph2[1] * _rho(i, j + rj0, nz - 3) +
                             ph2[0] * _rho(i, j + rj0 - 1, nz - 3)) +
               ph2r[k][1] * (ph2[1] * _rho(i, j + rj0, nz - 2) +
                             ph2[0] * _rho(i, j + rj0 - 1, nz - 2)) +
               ph2r[k][0] * (ph2[1] * _rho(i, j + rj0, nz - 1) +
                             ph2[0] * _rho(i, j + rj0 - 1, nz - 1));
  float rho2 = ph2r[k][7] * (ph2[1] * _rho(i, j + rj0, nz - 8) +
                             ph2[0] * _rho(i - 1, j + rj0, nz - 8)) +
               ph2r[k][6] * (ph2[1] * _rho(i, j + rj0, nz - 7) +
                             ph2[0] * _rho(i - 1, j + rj0, nz - 7)) +
               ph2r[k][5] * (ph2[1] * _rho(i, j + rj0, nz - 6) +
                             ph2[0] * _rho(i - 1, j + rj0, nz - 6)) +
               ph2r[k][4] * (ph2[1] * _rho(i, j + rj0, nz - 5) +
                             ph2[0] * _rho(i - 1, j + rj0, nz - 5)) +
               ph2r[k][3] * (ph2[1] * _rho(i, j + rj0, nz - 4) +
                             ph2[0] * _rho(i - 1, j + rj0, nz - 4)) +
               ph2r[k][2] * (ph2[1] * _rho(i, j + rj0, nz - 3) +
                             ph2[0] * _rho(i - 1, j + rj0, nz - 3)) +
               ph2r[k][1] * (ph2[1] * _rho(i, j + rj0, nz - 2) +
                             ph2[0] * _rho(i - 1, j + rj0, nz - 2)) +
               ph2r[k][0] * (ph2[1] * _rho(i, j + rj0, nz - 1) +
                             ph2[0] * _rho(i - 1, j + rj0, nz - 1));
  float rho3 = ph2[1] * (ph2[1] * _rho(i, j + rj0, nz - 1 - k) +
                         ph2[0] * _rho(i - 1, j + rj0, nz - 1 - k)) +
               ph2[0] * (ph2[1] * _rho(i, j + rj0 - 1, nz - 1 - k) +
                         ph2[0] * _rho(i - 1, j + rj0 - 1, nz - 1 - k));
  float Ai1 = _f_1(i, j + rj0) * _g3_c(nz - 1 - k) * rho1;
  Ai1 = nu * 1.0 / Ai1;
  float Ai2 = _f_2(i, j + rj0) * _g3_c(nz - 1 - k) * rho2;
  Ai2 = nu * 1.0 / Ai2;
  float Ai3 = _f_c(i, j + rj0) * _g3(nz - 1 - k) * rho3;
  Ai3 = nu * 1.0 / Ai3;
  float f_dcrj = _dcrjx(i) * _dcrjy(j + rj0) * _dcrjz(nz - 1 - k);
  _buf_u1(i, j, nz - 1 - k) =
      (a * _u1(i, j + rj0, nz - 1 - k) +
       Ai1 *
           (dh4[2] * _f(i, j + rj0) * _g3_c(nz - 1 - k) *
                _s12(i, j + rj0, nz - 1 - k) +
            dh4[0] * _f(i, j + rj0 - 2) * _g3_c(nz - 1 - k) *
                _s12(i, j + rj0 - 2, nz - 1 - k) +
            dh4[1] * _f(i, j + rj0 - 1) * _g3_c(nz - 1 - k) *
                _s12(i, j + rj0 - 1, nz - 1 - k) +
            dh4[3] * _f(i, j + rj0 + 1) * _g3_c(nz - 1 - k) *
                _s12(i, j + rj0 + 1, nz - 1 - k) +
            dh4[2] * _f_c(i, j + rj0) * _g3_c(nz - 1 - k) *
                _s11(i, j + rj0, nz - 1 - k) +
            dh4[0] * _f_c(i - 2, j + rj0) * _g3_c(nz - 1 - k) *
                _s11(i - 2, j + rj0, nz - 1 - k) +
            dh4[1] * _f_c(i - 1, j + rj0) * _g3_c(nz - 1 - k) *
                _s11(i - 1, j + rj0, nz - 1 - k) +
            dh4[3] * _f_c(i + 1, j + rj0) * _g3_c(nz - 1 - k) *
                _s11(i + 1, j + rj0, nz - 1 - k) +
            dh4r[k][7] * _s13(i, j + rj0, nz - 8) +
            dh4r[k][6] * _s13(i, j + rj0, nz - 7) +
            dh4r[k][5] * _s13(i, j + rj0, nz - 6) +
            dh4r[k][4] * _s13(i, j + rj0, nz - 5) +
            dh4r[k][3] * _s13(i, j + rj0, nz - 4) +
            dh4r[k][2] * _s13(i, j + rj0, nz - 3) +
            dh4r[k][1] * _s13(i, j + rj0, nz - 2) +
            dh4r[k][0] * _s13(i, j + rj0, nz - 1) -
            _f1_1(i, j + rj0) * (dhp4r[k][8] * _g_c(nz - 9) *
                                     (ph4[2] * _s11(i, j + rj0, nz - 9) +
                                      ph4[0] * _s11(i - 2, j + rj0, nz - 9) +
                                      ph4[1] * _s11(i - 1, j + rj0, nz - 9) +
                                      ph4[3] * _s11(i + 1, j + rj0, nz - 9)) +
                                 dhp4r[k][7] * _g_c(nz - 8) *
                                     (ph4[2] * _s11(i, j + rj0, nz - 8) +
                                      ph4[0] * _s11(i - 2, j + rj0, nz - 8) +
                                      ph4[1] * _s11(i - 1, j + rj0, nz - 8) +
                                      ph4[3] * _s11(i + 1, j + rj0, nz - 8)) +
                                 dhp4r[k][6] * _g_c(nz - 7) *
                                     (ph4[2] * _s11(i, j + rj0, nz - 7) +
                                      ph4[0] * _s11(i - 2, j + rj0, nz - 7) +
                                      ph4[1] * _s11(i - 1, j + rj0, nz - 7) +
                                      ph4[3] * _s11(i + 1, j + rj0, nz - 7)) +
                                 dhp4r[k][5] * _g_c(nz - 6) *
                                     (ph4[2] * _s11(i, j + rj0, nz - 6) +
                                      ph4[0] * _s11(i - 2, j + rj0, nz - 6) +
                                      ph4[1] * _s11(i - 1, j + rj0, nz - 6) +
                                      ph4[3] * _s11(i + 1, j + rj0, nz - 6)) +
                                 dhp4r[k][4] * _g_c(nz - 5) *
                                     (ph4[2] * _s11(i, j + rj0, nz - 5) +
                                      ph4[0] * _s11(i - 2, j + rj0, nz - 5) +
                                      ph4[1] * _s11(i - 1, j + rj0, nz - 5) +
                                      ph4[3] * _s11(i + 1, j + rj0, nz - 5)) +
                                 dhp4r[k][3] * _g_c(nz - 4) *
                                     (ph4[2] * _s11(i, j + rj0, nz - 4) +
                                      ph4[0] * _s11(i - 2, j + rj0, nz - 4) +
                                      ph4[1] * _s11(i - 1, j + rj0, nz - 4) +
                                      ph4[3] * _s11(i + 1, j + rj0, nz - 4)) +
                                 dhp4r[k][2] * _g_c(nz - 3) *
                                     (ph4[2] * _s11(i, j + rj0, nz - 3) +
                                      ph4[0] * _s11(i - 2, j + rj0, nz - 3) +
                                      ph4[1] * _s11(i - 1, j + rj0, nz - 3) +
                                      ph4[3] * _s11(i + 1, j + rj0, nz - 3)) +
                                 dhp4r[k][1] * _g_c(nz - 2) *
                                     (ph4[2] * _s11(i, j + rj0, nz - 2) +
                                      ph4[0] * _s11(i - 2, j + rj0, nz - 2) +
                                      ph4[1] * _s11(i - 1, j + rj0, nz - 2) +
                                      ph4[3] * _s11(i + 1, j + rj0, nz - 2)) +
                                 dhp4r[k][0] * _g_c(nz - 1) *
                                     (ph4[2] * _s11(i, j + rj0, nz - 1) +
                                      ph4[0] * _s11(i - 2, j + rj0, nz - 1) +
                                      ph4[1] * _s11(i - 1, j + rj0, nz - 1) +
                                      ph4[3] * _s11(i + 1, j + rj0, nz - 1))) -
            _f2_1(i, j + rj0) *
                (dhp4r[k][8] * _g_c(nz - 9) *
                     (ph4[2] * _s12(i, j + rj0, nz - 9) +
                      ph4[0] * _s12(i, j + rj0 - 2, nz - 9) +
                      ph4[1] * _s12(i, j + rj0 - 1, nz - 9) +
                      ph4[3] * _s12(i, j + rj0 + 1, nz - 9)) +
                 dhp4r[k][7] * _g_c(nz - 8) *
                     (ph4[2] * _s12(i, j + rj0, nz - 8) +
                      ph4[0] * _s12(i, j + rj0 - 2, nz - 8) +
                      ph4[1] * _s12(i, j + rj0 - 1, nz - 8) +
                      ph4[3] * _s12(i, j + rj0 + 1, nz - 8)) +
                 dhp4r[k][6] * _g_c(nz - 7) *
                     (ph4[2] * _s12(i, j + rj0, nz - 7) +
                      ph4[0] * _s12(i, j + rj0 - 2, nz - 7) +
                      ph4[1] * _s12(i, j + rj0 - 1, nz - 7) +
                      ph4[3] * _s12(i, j + rj0 + 1, nz - 7)) +
                 dhp4r[k][5] * _g_c(nz - 6) *
                     (ph4[2] * _s12(i, j + rj0, nz - 6) +
                      ph4[0] * _s12(i, j + rj0 - 2, nz - 6) +
                      ph4[1] * _s12(i, j + rj0 - 1, nz - 6) +
                      ph4[3] * _s12(i, j + rj0 + 1, nz - 6)) +
                 dhp4r[k][4] * _g_c(nz - 5) *
                     (ph4[2] * _s12(i, j + rj0, nz - 5) +
                      ph4[0] * _s12(i, j + rj0 - 2, nz - 5) +
                      ph4[1] * _s12(i, j + rj0 - 1, nz - 5) +
                      ph4[3] * _s12(i, j + rj0 + 1, nz - 5)) +
                 dhp4r[k][3] * _g_c(nz - 4) *
                     (ph4[2] * _s12(i, j + rj0, nz - 4) +
                      ph4[0] * _s12(i, j + rj0 - 2, nz - 4) +
                      ph4[1] * _s12(i, j + rj0 - 1, nz - 4) +
                      ph4[3] * _s12(i, j + rj0 + 1, nz - 4)) +
                 dhp4r[k][2] * _g_c(nz - 3) *
                     (ph4[2] * _s12(i, j + rj0, nz - 3) +
                      ph4[0] * _s12(i, j + rj0 - 2, nz - 3) +
                      ph4[1] * _s12(i, j + rj0 - 1, nz - 3) +
                      ph4[3] * _s12(i, j + rj0 + 1, nz - 3)) +
                 dhp4r[k][1] * _g_c(nz - 2) *
                     (ph4[2] * _s12(i, j + rj0, nz - 2) +
                      ph4[0] * _s12(i, j + rj0 - 2, nz - 2) +
                      ph4[1] * _s12(i, j + rj0 - 1, nz - 2) +
                      ph4[3] * _s12(i, j + rj0 + 1, nz - 2)) +
                 dhp4r[k][0] * _g_c(nz - 1) *
                     (ph4[2] * _s12(i, j + rj0, nz - 1) +
                      ph4[0] * _s12(i, j + rj0 - 2, nz - 1) +
                      ph4[1] * _s12(i, j + rj0 - 1, nz - 1) +
                      ph4[3] * _s12(i, j + rj0 + 1, nz - 1))))) *
      f_dcrj;
  _buf_u2(i, j, nz - 1 - k) =
      (a * _u2(i, j + rj0, nz - 1 - k) +
       Ai2 *
           (d4[1] * _f(i, j + rj0) * _g3_c(nz - 1 - k) *
                _s12(i, j + rj0, nz - 1 - k) +
            d4[0] * _f(i - 1, j + rj0) * _g3_c(nz - 1 - k) *
                _s12(i - 1, j + rj0, nz - 1 - k) +
            d4[2] * _f(i + 1, j + rj0) * _g3_c(nz - 1 - k) *
                _s12(i + 1, j + rj0, nz - 1 - k) +
            d4[3] * _f(i + 2, j + rj0) * _g3_c(nz - 1 - k) *
                _s12(i + 2, j + rj0, nz - 1 - k) +
            d4[1] * _f_c(i, j + rj0) * _g3_c(nz - 1 - k) *
                _s22(i, j + rj0, nz - 1 - k) +
            d4[0] * _f_c(i, j + rj0 - 1) * _g3_c(nz - 1 - k) *
                _s22(i, j + rj0 - 1, nz - 1 - k) +
            d4[2] * _f_c(i, j + rj0 + 1) * _g3_c(nz - 1 - k) *
                _s22(i, j + rj0 + 1, nz - 1 - k) +
            d4[3] * _f_c(i, j + rj0 + 2) * _g3_c(nz - 1 - k) *
                _s22(i, j + rj0 + 2, nz - 1 - k) +
            dh4r[k][7] * _s23(i, j + rj0, nz - 8) +
            dh4r[k][6] * _s23(i, j + rj0, nz - 7) +
            dh4r[k][5] * _s23(i, j + rj0, nz - 6) +
            dh4r[k][4] * _s23(i, j + rj0, nz - 5) +
            dh4r[k][3] * _s23(i, j + rj0, nz - 4) +
            dh4r[k][2] * _s23(i, j + rj0, nz - 3) +
            dh4r[k][1] * _s23(i, j + rj0, nz - 2) +
            dh4r[k][0] * _s23(i, j + rj0, nz - 1) -
            _f1_2(i, j + rj0) * (dhp4r[k][8] * _g_c(nz - 9) *
                                     (p4[1] * _s12(i, j + rj0, nz - 9) +
                                      p4[0] * _s12(i - 1, j + rj0, nz - 9) +
                                      p4[2] * _s12(i + 1, j + rj0, nz - 9) +
                                      p4[3] * _s12(i + 2, j + rj0, nz - 9)) +
                                 dhp4r[k][7] * _g_c(nz - 8) *
                                     (p4[1] * _s12(i, j + rj0, nz - 8) +
                                      p4[0] * _s12(i - 1, j + rj0, nz - 8) +
                                      p4[2] * _s12(i + 1, j + rj0, nz - 8) +
                                      p4[3] * _s12(i + 2, j + rj0, nz - 8)) +
                                 dhp4r[k][6] * _g_c(nz - 7) *
                                     (p4[1] * _s12(i, j + rj0, nz - 7) +
                                      p4[0] * _s12(i - 1, j + rj0, nz - 7) +
                                      p4[2] * _s12(i + 1, j + rj0, nz - 7) +
                                      p4[3] * _s12(i + 2, j + rj0, nz - 7)) +
                                 dhp4r[k][5] * _g_c(nz - 6) *
                                     (p4[1] * _s12(i, j + rj0, nz - 6) +
                                      p4[0] * _s12(i - 1, j + rj0, nz - 6) +
                                      p4[2] * _s12(i + 1, j + rj0, nz - 6) +
                                      p4[3] * _s12(i + 2, j + rj0, nz - 6)) +
                                 dhp4r[k][4] * _g_c(nz - 5) *
                                     (p4[1] * _s12(i, j + rj0, nz - 5) +
                                      p4[0] * _s12(i - 1, j + rj0, nz - 5) +
                                      p4[2] * _s12(i + 1, j + rj0, nz - 5) +
                                      p4[3] * _s12(i + 2, j + rj0, nz - 5)) +
                                 dhp4r[k][3] * _g_c(nz - 4) *
                                     (p4[1] * _s12(i, j + rj0, nz - 4) +
                                      p4[0] * _s12(i - 1, j + rj0, nz - 4) +
                                      p4[2] * _s12(i + 1, j + rj0, nz - 4) +
                                      p4[3] * _s12(i + 2, j + rj0, nz - 4)) +
                                 dhp4r[k][2] * _g_c(nz - 3) *
                                     (p4[1] * _s12(i, j + rj0, nz - 3) +
                                      p4[0] * _s12(i - 1, j + rj0, nz - 3) +
                                      p4[2] * _s12(i + 1, j + rj0, nz - 3) +
                                      p4[3] * _s12(i + 2, j + rj0, nz - 3)) +
                                 dhp4r[k][1] * _g_c(nz - 2) *
                                     (p4[1] * _s12(i, j + rj0, nz - 2) +
                                      p4[0] * _s12(i - 1, j + rj0, nz - 2) +
                                      p4[2] * _s12(i + 1, j + rj0, nz - 2) +
                                      p4[3] * _s12(i + 2, j + rj0, nz - 2)) +
                                 dhp4r[k][0] * _g_c(nz - 1) *
                                     (p4[1] * _s12(i, j + rj0, nz - 1) +
                                      p4[0] * _s12(i - 1, j + rj0, nz - 1) +
                                      p4[2] * _s12(i + 1, j + rj0, nz - 1) +
                                      p4[3] * _s12(i + 2, j + rj0, nz - 1))) -
            _f2_2(i, j + rj0) * (dhp4r[k][8] * _g_c(nz - 9) *
                                     (p4[1] * _s22(i, j + rj0, nz - 9) +
                                      p4[0] * _s22(i, j + rj0 - 1, nz - 9) +
                                      p4[2] * _s22(i, j + rj0 + 1, nz - 9) +
                                      p4[3] * _s22(i, j + rj0 + 2, nz - 9)) +
                                 dhp4r[k][7] * _g_c(nz - 8) *
                                     (p4[1] * _s22(i, j + rj0, nz - 8) +
                                      p4[0] * _s22(i, j + rj0 - 1, nz - 8) +
                                      p4[2] * _s22(i, j + rj0 + 1, nz - 8) +
                                      p4[3] * _s22(i, j + rj0 + 2, nz - 8)) +
                                 dhp4r[k][6] * _g_c(nz - 7) *
                                     (p4[1] * _s22(i, j + rj0, nz - 7) +
                                      p4[0] * _s22(i, j + rj0 - 1, nz - 7) +
                                      p4[2] * _s22(i, j + rj0 + 1, nz - 7) +
                                      p4[3] * _s22(i, j + rj0 + 2, nz - 7)) +
                                 dhp4r[k][5] * _g_c(nz - 6) *
                                     (p4[1] * _s22(i, j + rj0, nz - 6) +
                                      p4[0] * _s22(i, j + rj0 - 1, nz - 6) +
                                      p4[2] * _s22(i, j + rj0 + 1, nz - 6) +
                                      p4[3] * _s22(i, j + rj0 + 2, nz - 6)) +
                                 dhp4r[k][4] * _g_c(nz - 5) *
                                     (p4[1] * _s22(i, j + rj0, nz - 5) +
                                      p4[0] * _s22(i, j + rj0 - 1, nz - 5) +
                                      p4[2] * _s22(i, j + rj0 + 1, nz - 5) +
                                      p4[3] * _s22(i, j + rj0 + 2, nz - 5)) +
                                 dhp4r[k][3] * _g_c(nz - 4) *
                                     (p4[1] * _s22(i, j + rj0, nz - 4) +
                                      p4[0] * _s22(i, j + rj0 - 1, nz - 4) +
                                      p4[2] * _s22(i, j + rj0 + 1, nz - 4) +
                                      p4[3] * _s22(i, j + rj0 + 2, nz - 4)) +
                                 dhp4r[k][2] * _g_c(nz - 3) *
                                     (p4[1] * _s22(i, j + rj0, nz - 3) +
                                      p4[0] * _s22(i, j + rj0 - 1, nz - 3) +
                                      p4[2] * _s22(i, j + rj0 + 1, nz - 3) +
                                      p4[3] * _s22(i, j + rj0 + 2, nz - 3)) +
                                 dhp4r[k][1] * _g_c(nz - 2) *
                                     (p4[1] * _s22(i, j + rj0, nz - 2) +
                                      p4[0] * _s22(i, j + rj0 - 1, nz - 2) +
                                      p4[2] * _s22(i, j + rj0 + 1, nz - 2) +
                                      p4[3] * _s22(i, j + rj0 + 2, nz - 2)) +
                                 dhp4r[k][0] * _g_c(nz - 1) *
                                     (p4[1] * _s22(i, j + rj0, nz - 1) +
                                      p4[0] * _s22(i, j + rj0 - 1, nz - 1) +
                                      p4[2] * _s22(i, j + rj0 + 1, nz - 1) +
                                      p4[3] * _s22(i, j + rj0 + 2, nz - 1))))) *
      f_dcrj;
  _buf_u3(i, j, nz - 1 - k) =
      (a * _u3(i, j + rj0, nz - 1 - k) +
       Ai3 * (d4[1] * _f_1(i, j + rj0) * _g3(nz - 1 - k) *
                  _s13(i, j + rj0, nz - 1 - k) +
              d4[0] * _f_1(i - 1, j + rj0) * _g3(nz - 1 - k) *
                  _s13(i - 1, j + rj0, nz - 1 - k) +
              d4[2] * _f_1(i + 1, j + rj0) * _g3(nz - 1 - k) *
                  _s13(i + 1, j + rj0, nz - 1 - k) +
              d4[3] * _f_1(i + 2, j + rj0) * _g3(nz - 1 - k) *
                  _s13(i + 2, j + rj0, nz - 1 - k) +
              d4r[k][6] * _s33(i, j + rj0, nz - 7) +
              d4r[k][5] * _s33(i, j + rj0, nz - 6) +
              d4r[k][4] * _s33(i, j + rj0, nz - 5) +
              d4r[k][3] * _s33(i, j + rj0, nz - 4) +
              d4r[k][2] * _s33(i, j + rj0, nz - 3) +
              d4r[k][1] * _s33(i, j + rj0, nz - 2) +
              d4r[k][0] * _s33(i, j + rj0, nz - 1) +
              dh4[2] * _f_2(i, j + rj0) * _g3(nz - 1 - k) *
                  _s23(i, j + rj0, nz - 1 - k) +
              dh4[0] * _f_2(i, j + rj0 - 2) * _g3(nz - 1 - k) *
                  _s23(i, j + rj0 - 2, nz - 1 - k) +
              dh4[1] * _f_2(i, j + rj0 - 1) * _g3(nz - 1 - k) *
                  _s23(i, j + rj0 - 1, nz - 1 - k) +
              dh4[3] * _f_2(i, j + rj0 + 1) * _g3(nz - 1 - k) *
                  _s23(i, j + rj0 + 1, nz - 1 - k) -
              _f1_c(i, j + rj0) * (dph4r[k][8] * _g(nz - 9) *
                                       (p4[1] * _s13(i, j + rj0, nz - 9) +
                                        p4[0] * _s13(i - 1, j + rj0, nz - 9) +
                                        p4[2] * _s13(i + 1, j + rj0, nz - 9) +
                                        p4[3] * _s13(i + 2, j + rj0, nz - 9)) +
                                   dph4r[k][7] * _g(nz - 8) *
                                       (p4[1] * _s13(i, j + rj0, nz - 8) +
                                        p4[0] * _s13(i - 1, j + rj0, nz - 8) +
                                        p4[2] * _s13(i + 1, j + rj0, nz - 8) +
                                        p4[3] * _s13(i + 2, j + rj0, nz - 8)) +
                                   dph4r[k][6] * _g(nz - 7) *
                                       (p4[1] * _s13(i, j + rj0, nz - 7) +
                                        p4[0] * _s13(i - 1, j + rj0, nz - 7) +
                                        p4[2] * _s13(i + 1, j + rj0, nz - 7) +
                                        p4[3] * _s13(i + 2, j + rj0, nz - 7)) +
                                   dph4r[k][5] * _g(nz - 6) *
                                       (p4[1] * _s13(i, j + rj0, nz - 6) +
                                        p4[0] * _s13(i - 1, j + rj0, nz - 6) +
                                        p4[2] * _s13(i + 1, j + rj0, nz - 6) +
                                        p4[3] * _s13(i + 2, j + rj0, nz - 6)) +
                                   dph4r[k][4] * _g(nz - 5) *
                                       (p4[1] * _s13(i, j + rj0, nz - 5) +
                                        p4[0] * _s13(i - 1, j + rj0, nz - 5) +
                                        p4[2] * _s13(i + 1, j + rj0, nz - 5) +
                                        p4[3] * _s13(i + 2, j + rj0, nz - 5)) +
                                   dph4r[k][3] * _g(nz - 4) *
                                       (p4[1] * _s13(i, j + rj0, nz - 4) +
                                        p4[0] * _s13(i - 1, j + rj0, nz - 4) +
                                        p4[2] * _s13(i + 1, j + rj0, nz - 4) +
                                        p4[3] * _s13(i + 2, j + rj0, nz - 4)) +
                                   dph4r[k][2] * _g(nz - 3) *
                                       (p4[1] * _s13(i, j + rj0, nz - 3) +
                                        p4[0] * _s13(i - 1, j + rj0, nz - 3) +
                                        p4[2] * _s13(i + 1, j + rj0, nz - 3) +
                                        p4[3] * _s13(i + 2, j + rj0, nz - 3)) +
                                   dph4r[k][1] * _g(nz - 2) *
                                       (p4[1] * _s13(i, j + rj0, nz - 2) +
                                        p4[0] * _s13(i - 1, j + rj0, nz - 2) +
                                        p4[2] * _s13(i + 1, j + rj0, nz - 2) +
                                        p4[3] * _s13(i + 2, j + rj0, nz - 2)) +
                                   dph4r[k][0] * _g(nz - 1) *
                                       (p4[1] * _s13(i, j + rj0, nz - 1) +
                                        p4[0] * _s13(i - 1, j + rj0, nz - 1) +
                                        p4[2] * _s13(i + 1, j + rj0, nz - 1) +
                                        p4[3] * _s13(i + 2, j + rj0, nz - 1))) -
              _f2_c(i, j + rj0) *
                  (dph4r[k][8] * _g(nz - 9) *
                       (ph4[2] * _s23(i, j + rj0, nz - 9) +
                        ph4[0] * _s23(i, j + rj0 - 2, nz - 9) +
                        ph4[1] * _s23(i, j + rj0 - 1, nz - 9) +
                        ph4[3] * _s23(i, j + rj0 + 1, nz - 9)) +
                   dph4r[k][7] * _g(nz - 8) *
                       (ph4[2] * _s23(i, j + rj0, nz - 8) +
                        ph4[0] * _s23(i, j + rj0 - 2, nz - 8) +
                        ph4[1] * _s23(i, j + rj0 - 1, nz - 8) +
                        ph4[3] * _s23(i, j + rj0 + 1, nz - 8)) +
                   dph4r[k][6] * _g(nz - 7) *
                       (ph4[2] * _s23(i, j + rj0, nz - 7) +
                        ph4[0] * _s23(i, j + rj0 - 2, nz - 7) +
                        ph4[1] * _s23(i, j + rj0 - 1, nz - 7) +
                        ph4[3] * _s23(i, j + rj0 + 1, nz - 7)) +
                   dph4r[k][5] * _g(nz - 6) *
                       (ph4[2] * _s23(i, j + rj0, nz - 6) +
                        ph4[0] * _s23(i, j + rj0 - 2, nz - 6) +
                        ph4[1] * _s23(i, j + rj0 - 1, nz - 6) +
                        ph4[3] * _s23(i, j + rj0 + 1, nz - 6)) +
                   dph4r[k][4] * _g(nz - 5) *
                       (ph4[2] * _s23(i, j + rj0, nz - 5) +
                        ph4[0] * _s23(i, j + rj0 - 2, nz - 5) +
                        ph4[1] * _s23(i, j + rj0 - 1, nz - 5) +
                        ph4[3] * _s23(i, j + rj0 + 1, nz - 5)) +
                   dph4r[k][3] * _g(nz - 4) *
                       (ph4[2] * _s23(i, j + rj0, nz - 4) +
                        ph4[0] * _s23(i, j + rj0 - 2, nz - 4) +
                        ph4[1] * _s23(i, j + rj0 - 1, nz - 4) +
                        ph4[3] * _s23(i, j + rj0 + 1, nz - 4)) +
                   dph4r[k][2] * _g(nz - 3) *
                       (ph4[2] * _s23(i, j + rj0, nz - 3) +
                        ph4[0] * _s23(i, j + rj0 - 2, nz - 3) +
                        ph4[1] * _s23(i, j + rj0 - 1, nz - 3) +
                        ph4[3] * _s23(i, j + rj0 + 1, nz - 3)) +
                   dph4r[k][1] * _g(nz - 2) *
                       (ph4[2] * _s23(i, j + rj0, nz - 2) +
                        ph4[0] * _s23(i, j + rj0 - 2, nz - 2) +
                        ph4[1] * _s23(i, j + rj0 - 1, nz - 2) +
                        ph4[3] * _s23(i, j + rj0 + 1, nz - 2)) +
                   dph4r[k][0] * _g(nz - 1) *
                       (ph4[2] * _s23(i, j + rj0, nz - 1) +
                        ph4[0] * _s23(i, j + rj0 - 2, nz - 1) +
                        ph4[1] * _s23(i, j + rj0 - 1, nz - 1) +
                        ph4[3] * _s23(i, j + rj0 + 1, nz - 1))))) *
      f_dcrj;
#undef _rho
#undef _f_1
#undef _g3_c
#undef _f_2
#undef _g3
#undef _f_c
#undef _dcrjz
#undef _dcrjx
#undef _dcrjy
#undef _g_c
#undef _s11
#undef _f2_1
#undef _u1
#undef _f1_1
#undef _s12
#undef _f
#undef _s13
#undef _f1_2
#undef _s22
#undef _s23
#undef _u2
#undef _f2_2
#undef _f2_c
#undef _g
#undef _f1_c
#undef _s33
#undef _u3
#undef _buf_u1
#undef _buf_u2
#undef _buf_u3
}

__global__ void dtopo_str_110(
    float *__restrict__ s11, float *__restrict__ s12, float *__restrict__ s13,
    float *__restrict__ s22, float *__restrict__ s23, float *__restrict__ s33,
    float *__restrict__ u1, float *__restrict__ u2, float *__restrict__ u3,
    const float *__restrict__ dcrjx, const float *__restrict__ dcrjy,
    const float *__restrict__ dcrjz, const float *__restrict__ f,
    const float *__restrict__ f1_1, const float *__restrict__ f1_2,
    const float *__restrict__ f1_c, const float *__restrict__ f2_1,
    const float *__restrict__ f2_2, const float *__restrict__ f2_c,
    const float *__restrict__ f_1, const float *__restrict__ f_2,
    const float *__restrict__ f_c, const float *__restrict__ g,
    const float *__restrict__ g3, const float *__restrict__ g3_c,
    const float *__restrict__ g_c, const float *__restrict__ lami,
    const float *__restrict__ mui, const float a, const float nu, const int nx,
    const int ny, const int nz, const int bi, const int bj, const int ei,
    const int ej) {
  const float ph2l[6][7] = {
      {1.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.5000000000000000, 0.5000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.5000000000000000, 0.5000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.5000000000000000,
       0.5000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.5000000000000000, 0.5000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.5000000000000000, 0.5000000000000000,
       0.0000000000000000}};
  const float ph2[2] = {0.5000000000000000, 0.5000000000000000};
  const float p2[2] = {0.5000000000000000, 0.5000000000000000};
  const float dh4l[6][7] = {
      {-1.4511412472637157, 1.8534237417911470, -0.3534237417911469,
       -0.0488587527362844, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {-0.8577143189081458, 0.5731429567244373, 0.4268570432755628,
       -0.1422856810918542, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {-0.1674548505882877, -0.4976354482351368, 0.4976354482351368,
       0.1674548505882877, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.1027061113405124, -0.2624541326469860, -0.8288742701021167,
       1.0342864927831414, -0.0456642013745513, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0416666666666667,
       -1.1250000000000000, 1.1250000000000000, -0.0416666666666667,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0416666666666667, -1.1250000000000000, 1.1250000000000000,
       -0.0416666666666667}};
  const float p4[4] = {-0.0625000000000000, 0.5625000000000000,
                       0.5625000000000000, -0.0625000000000000};
  const float phd4l[6][9] = {
      {-1.5373923010673116, 1.0330083346742178, 0.6211677623382129,
       0.0454110758451345, -0.1680934225988761, 0.0058985508086226,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {-0.8713921425924011, 0.1273679143938725, 0.9297550647681330,
       -0.1912595577524762, 0.0050469052908678, 0.0004818158920039,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {-0.0563333965151294, -0.3996393739211770, -0.0536007135209481,
       0.5022638816465500, 0.0083321572725344, -0.0010225549618299,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {-0.0132930497153990, 0.0706942590708847, -0.5596445380498725,
       -0.1434031863528334, 0.7456356868769503, -0.1028431844156395,
       0.0028540125859095, 0.0000000000000000, 0.0000000000000000},
      {-0.0025849423769932, 0.0492307522105194, -0.0524552477068130,
       -0.5317248489238559, -0.0530169938441241, 0.6816971139746001,
       -0.0937500000000000, 0.0026041666666667, 0.0000000000000000},
      {-0.0009619461344193, -0.0035553215968974, 0.0124936029037323,
       0.0773639466787397, -0.6736586580761996, -0.0002232904416222,
       0.6796875000000000, -0.0937500000000000, 0.0026041666666667}};
  const float ph4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float d4[4] = {0.0416666666666667, -1.1250000000000000,
                       1.1250000000000000, -0.0416666666666667};
  const float dh4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float d4l[6][8] = {
      {-1.7779989465546748, 1.3337480247900155, 0.7775013168066564,
       -0.3332503950419969, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {-0.4410217341392059, -0.1730842484889890, 0.4487228323259926,
       0.1653831503022022, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.1798793213882701, -0.2757257254150788, -0.9597948548284453,
       1.1171892610431817, -0.0615480021879277, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0153911381507088, 0.0568851455503591, -0.1998976464597171,
       -0.8628231468598346, 1.0285385292191949, -0.0380940196007109,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0416666666666667, -1.1250000000000000, 1.1250000000000000,
       -0.0416666666666667, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0416666666666667, -1.1250000000000000,
       1.1250000000000000, -0.0416666666666667}};
  const float pdh4l[6][9] = {
      {-1.5886075042755416, 2.2801810182668110, -0.8088980291471827,
       0.1316830205960989, -0.0143585054401857, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {-0.4823226655921296, -0.0574614517751294, 0.5663203488781653,
       -0.0309656800624243, 0.0044294485515179, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.0174954311279016, -0.4325508330649350, -0.3111668377093504,
       0.8538512002386446, -0.1314757107290064, 0.0038467501367455,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.1277481742492071, -0.2574468839590017, -0.4155794781917712,
       0.0115571196122084, 0.6170517361659126, -0.0857115441015996,
       0.0023808762250444, 0.0000000000000000, 0.0000000000000000},
      {-0.0064191319587820, 0.0164033832904366, 0.0752421418813823,
       -0.6740179057989464, 0.0002498459192428, 0.6796875000000000,
       -0.0937500000000000, 0.0026041666666667, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, -0.0026041666666667,
       0.0937500000000000, -0.6796875000000000, -0.0000000000000000,
       0.6796875000000000, -0.0937500000000000, 0.0026041666666667}};
  const int i = threadIdx.z + blockIdx.z * blockDim.z + bi;
  if (i >= ngsl + nx)
    return;
  if (i >= ei)
    return;
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ngsl + ny)
    return;
  if (j >= ej)
    return;
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= 6)
    return;
#define _g3_c(k) g3_c[(k) + align]
#define _f_c(i, j)                                                             \
  f_c[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f(i, j)                                                               \
  f[(j) + align + ngsl + ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f_1(i, j)                                                             \
  f_1[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3(k) g3[(k) + align]
#define _f_2(i, j)                                                             \
  f_2[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _lami(i, j, k)                                                         \
  lami[(k) + align +                                                           \
       (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +             \
       (2 * align + nz) * ((j) + ngsl + 2)]
#define _mui(i, j, k)                                                          \
  mui[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_1(i, j)                                                            \
  f1_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _u1(i, j, k)                                                           \
  u1[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _g_c(k) g_c[(k) + align]
#define _u2(i, j, k)                                                           \
  u2[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_2(i, j)                                                            \
  f2_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _u3(i, j, k)                                                           \
  u3[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _dcrjz(k) dcrjz[(k) + align]
#define _dcrjx(i) dcrjx[(i) + ngsl + 2]
#define _dcrjy(j) dcrjy[(j) + ngsl + 2]
#define _s11(i, j, k)                                                          \
  s11[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s22(i, j, k)                                                          \
  s22[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s33(i, j, k)                                                          \
  s33[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_2(i, j)                                                            \
  f1_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_1(i, j)                                                            \
  f2_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s12(i, j, k)                                                          \
  s12[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _g(k) g[(k) + align]
#define _f1_c(i, j)                                                            \
  f1_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s13(i, j, k)                                                          \
  s13[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_c(i, j)                                                            \
  f2_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s23(i, j, k)                                                          \
  s23[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
  float Jii = _f_c(i, j) * _g3_c(k);
  Jii = 1.0 * 1.0 / Jii;
  float J12i = _f(i, j) * _g3_c(k);
  J12i = 1.0 * 1.0 / J12i;
  float J13i = _f_1(i, j) * _g3(k);
  J13i = 1.0 * 1.0 / J13i;
  float J23i = _f_2(i, j) * _g3(k);
  J23i = 1.0 * 1.0 / J23i;
  float lam =
      nu * 1.0 /
      (ph2l[k][0] *
           (ph2[1] * (p2[0] * _lami(i, j, 0) + p2[1] * _lami(i + 1, j, 0)) +
            ph2[0] *
                (p2[0] * _lami(i, j - 1, 0) + p2[1] * _lami(i + 1, j - 1, 0))) +
       ph2l[k][1] *
           (ph2[1] * (p2[0] * _lami(i, j, 1) + p2[1] * _lami(i + 1, j, 1)) +
            ph2[0] *
                (p2[0] * _lami(i, j - 1, 1) + p2[1] * _lami(i + 1, j - 1, 1))) +
       ph2l[k][2] *
           (ph2[1] * (p2[0] * _lami(i, j, 2) + p2[1] * _lami(i + 1, j, 2)) +
            ph2[0] *
                (p2[0] * _lami(i, j - 1, 2) + p2[1] * _lami(i + 1, j - 1, 2))) +
       ph2l[k][3] *
           (ph2[1] * (p2[0] * _lami(i, j, 3) + p2[1] * _lami(i + 1, j, 3)) +
            ph2[0] *
                (p2[0] * _lami(i, j - 1, 3) + p2[1] * _lami(i + 1, j - 1, 3))) +
       ph2l[k][4] *
           (ph2[1] * (p2[0] * _lami(i, j, 4) + p2[1] * _lami(i + 1, j, 4)) +
            ph2[0] *
                (p2[0] * _lami(i, j - 1, 4) + p2[1] * _lami(i + 1, j - 1, 4))) +
       ph2l[k][5] *
           (ph2[1] * (p2[0] * _lami(i, j, 5) + p2[1] * _lami(i + 1, j, 5)) +
            ph2[0] *
                (p2[0] * _lami(i, j - 1, 5) + p2[1] * _lami(i + 1, j - 1, 5))) +
       ph2l[k][6] *
           (ph2[1] * (p2[0] * _lami(i, j, 6) + p2[1] * _lami(i + 1, j, 6)) +
            ph2[0] *
                (p2[0] * _lami(i, j - 1, 6) + p2[1] * _lami(i + 1, j - 1, 6))));
  float twomu =
      2 * nu * 1.0 /
      (ph2l[k][0] *
           (ph2[1] * (p2[0] * _mui(i, j, 0) + p2[1] * _mui(i + 1, j, 0)) +
            ph2[0] *
                (p2[0] * _mui(i, j - 1, 0) + p2[1] * _mui(i + 1, j - 1, 0))) +
       ph2l[k][1] *
           (ph2[1] * (p2[0] * _mui(i, j, 1) + p2[1] * _mui(i + 1, j, 1)) +
            ph2[0] *
                (p2[0] * _mui(i, j - 1, 1) + p2[1] * _mui(i + 1, j - 1, 1))) +
       ph2l[k][2] *
           (ph2[1] * (p2[0] * _mui(i, j, 2) + p2[1] * _mui(i + 1, j, 2)) +
            ph2[0] *
                (p2[0] * _mui(i, j - 1, 2) + p2[1] * _mui(i + 1, j - 1, 2))) +
       ph2l[k][3] *
           (ph2[1] * (p2[0] * _mui(i, j, 3) + p2[1] * _mui(i + 1, j, 3)) +
            ph2[0] *
                (p2[0] * _mui(i, j - 1, 3) + p2[1] * _mui(i + 1, j - 1, 3))) +
       ph2l[k][4] *
           (ph2[1] * (p2[0] * _mui(i, j, 4) + p2[1] * _mui(i + 1, j, 4)) +
            ph2[0] *
                (p2[0] * _mui(i, j - 1, 4) + p2[1] * _mui(i + 1, j - 1, 4))) +
       ph2l[k][5] *
           (ph2[1] * (p2[0] * _mui(i, j, 5) + p2[1] * _mui(i + 1, j, 5)) +
            ph2[0] *
                (p2[0] * _mui(i, j - 1, 5) + p2[1] * _mui(i + 1, j - 1, 5))) +
       ph2l[k][6] *
           (ph2[1] * (p2[0] * _mui(i, j, 6) + p2[1] * _mui(i + 1, j, 6)) +
            ph2[0] *
                (p2[0] * _mui(i, j - 1, 6) + p2[1] * _mui(i + 1, j - 1, 6))));
  float mu12 = nu * 1.0 /
               (ph2l[k][0] * _mui(i, j, 0) + ph2l[k][1] * _mui(i, j, 1) +
                ph2l[k][2] * _mui(i, j, 2) + ph2l[k][3] * _mui(i, j, 3) +
                ph2l[k][4] * _mui(i, j, 4) + ph2l[k][5] * _mui(i, j, 5) +
                ph2l[k][6] * _mui(i, j, 6));
  float mu13 = nu * 1.0 / (ph2[1] * _mui(i, j, k) + ph2[0] * _mui(i, j - 1, k));
  float mu23 = nu * 1.0 / (p2[0] * _mui(i, j, k) + p2[1] * _mui(i + 1, j, k));
  float div =
      d4[1] * _u1(i, j, k) + d4[0] * _u1(i - 1, j, k) +
      d4[2] * _u1(i + 1, j, k) + d4[3] * _u1(i + 2, j, k) +
      dh4[2] * _u2(i, j, k) + dh4[0] * _u2(i, j - 2, k) +
      dh4[1] * _u2(i, j - 1, k) + dh4[3] * _u2(i, j + 1, k) +
      Jii * (dh4l[k][0] * _u3(i, j, 0) + dh4l[k][1] * _u3(i, j, 1) +
             dh4l[k][2] * _u3(i, j, 2) + dh4l[k][3] * _u3(i, j, 3) +
             dh4l[k][4] * _u3(i, j, 4) + dh4l[k][5] * _u3(i, j, 5) +
             dh4l[k][6] * _u3(i, j, 6)) -
      Jii * _g_c(k) *
          (p4[1] * _f1_1(i, j) *
               (phd4l[k][0] * _u1(i, j, 0) + phd4l[k][1] * _u1(i, j, 1) +
                phd4l[k][2] * _u1(i, j, 2) + phd4l[k][3] * _u1(i, j, 3) +
                phd4l[k][4] * _u1(i, j, 4) + phd4l[k][5] * _u1(i, j, 5) +
                phd4l[k][6] * _u1(i, j, 6) + phd4l[k][7] * _u1(i, j, 7) +
                phd4l[k][8] * _u1(i, j, 8)) +
           p4[0] * _f1_1(i - 1, j) *
               (phd4l[k][0] * _u1(i - 1, j, 0) +
                phd4l[k][1] * _u1(i - 1, j, 1) +
                phd4l[k][2] * _u1(i - 1, j, 2) +
                phd4l[k][3] * _u1(i - 1, j, 3) +
                phd4l[k][4] * _u1(i - 1, j, 4) +
                phd4l[k][5] * _u1(i - 1, j, 5) +
                phd4l[k][6] * _u1(i - 1, j, 6) +
                phd4l[k][7] * _u1(i - 1, j, 7) +
                phd4l[k][8] * _u1(i - 1, j, 8)) +
           p4[2] * _f1_1(i + 1, j) *
               (phd4l[k][0] * _u1(i + 1, j, 0) +
                phd4l[k][1] * _u1(i + 1, j, 1) +
                phd4l[k][2] * _u1(i + 1, j, 2) +
                phd4l[k][3] * _u1(i + 1, j, 3) +
                phd4l[k][4] * _u1(i + 1, j, 4) +
                phd4l[k][5] * _u1(i + 1, j, 5) +
                phd4l[k][6] * _u1(i + 1, j, 6) +
                phd4l[k][7] * _u1(i + 1, j, 7) +
                phd4l[k][8] * _u1(i + 1, j, 8)) +
           p4[3] * _f1_1(i + 2, j) *
               (phd4l[k][0] * _u1(i + 2, j, 0) +
                phd4l[k][1] * _u1(i + 2, j, 1) +
                phd4l[k][2] * _u1(i + 2, j, 2) +
                phd4l[k][3] * _u1(i + 2, j, 3) +
                phd4l[k][4] * _u1(i + 2, j, 4) +
                phd4l[k][5] * _u1(i + 2, j, 5) +
                phd4l[k][6] * _u1(i + 2, j, 6) +
                phd4l[k][7] * _u1(i + 2, j, 7) +
                phd4l[k][8] * _u1(i + 2, j, 8))) -
      Jii * _g_c(k) *
          (ph4[2] * _f2_2(i, j) *
               (phd4l[k][0] * _u2(i, j, 0) + phd4l[k][1] * _u2(i, j, 1) +
                phd4l[k][2] * _u2(i, j, 2) + phd4l[k][3] * _u2(i, j, 3) +
                phd4l[k][4] * _u2(i, j, 4) + phd4l[k][5] * _u2(i, j, 5) +
                phd4l[k][6] * _u2(i, j, 6) + phd4l[k][7] * _u2(i, j, 7) +
                phd4l[k][8] * _u2(i, j, 8)) +
           ph4[0] * _f2_2(i, j - 2) *
               (phd4l[k][0] * _u2(i, j - 2, 0) +
                phd4l[k][1] * _u2(i, j - 2, 1) +
                phd4l[k][2] * _u2(i, j - 2, 2) +
                phd4l[k][3] * _u2(i, j - 2, 3) +
                phd4l[k][4] * _u2(i, j - 2, 4) +
                phd4l[k][5] * _u2(i, j - 2, 5) +
                phd4l[k][6] * _u2(i, j - 2, 6) +
                phd4l[k][7] * _u2(i, j - 2, 7) +
                phd4l[k][8] * _u2(i, j - 2, 8)) +
           ph4[1] * _f2_2(i, j - 1) *
               (phd4l[k][0] * _u2(i, j - 1, 0) +
                phd4l[k][1] * _u2(i, j - 1, 1) +
                phd4l[k][2] * _u2(i, j - 1, 2) +
                phd4l[k][3] * _u2(i, j - 1, 3) +
                phd4l[k][4] * _u2(i, j - 1, 4) +
                phd4l[k][5] * _u2(i, j - 1, 5) +
                phd4l[k][6] * _u2(i, j - 1, 6) +
                phd4l[k][7] * _u2(i, j - 1, 7) +
                phd4l[k][8] * _u2(i, j - 1, 8)) +
           ph4[3] * _f2_2(i, j + 1) *
               (phd4l[k][0] * _u2(i, j + 1, 0) +
                phd4l[k][1] * _u2(i, j + 1, 1) +
                phd4l[k][2] * _u2(i, j + 1, 2) +
                phd4l[k][3] * _u2(i, j + 1, 3) +
                phd4l[k][4] * _u2(i, j + 1, 4) +
                phd4l[k][5] * _u2(i, j + 1, 5) +
                phd4l[k][6] * _u2(i, j + 1, 6) +
                phd4l[k][7] * _u2(i, j + 1, 7) +
                phd4l[k][8] * _u2(i, j + 1, 8)));
  float f_dcrj = _dcrjx(i) * _dcrjy(j) * _dcrjz(k);
  _s11(i, j, k) =
      (a * _s11(i, j, k) + lam * div +
       twomu * (d4[1] * _u1(i, j, k) + d4[0] * _u1(i - 1, j, k) +
                d4[2] * _u1(i + 1, j, k) + d4[3] * _u1(i + 2, j, k)) -
       twomu * Jii * _g_c(k) *
           (p4[1] * _f1_1(i, j) *
                (phd4l[k][0] * _u1(i, j, 0) + phd4l[k][1] * _u1(i, j, 1) +
                 phd4l[k][2] * _u1(i, j, 2) + phd4l[k][3] * _u1(i, j, 3) +
                 phd4l[k][4] * _u1(i, j, 4) + phd4l[k][5] * _u1(i, j, 5) +
                 phd4l[k][6] * _u1(i, j, 6) + phd4l[k][7] * _u1(i, j, 7) +
                 phd4l[k][8] * _u1(i, j, 8)) +
            p4[0] * _f1_1(i - 1, j) *
                (phd4l[k][0] * _u1(i - 1, j, 0) +
                 phd4l[k][1] * _u1(i - 1, j, 1) +
                 phd4l[k][2] * _u1(i - 1, j, 2) +
                 phd4l[k][3] * _u1(i - 1, j, 3) +
                 phd4l[k][4] * _u1(i - 1, j, 4) +
                 phd4l[k][5] * _u1(i - 1, j, 5) +
                 phd4l[k][6] * _u1(i - 1, j, 6) +
                 phd4l[k][7] * _u1(i - 1, j, 7) +
                 phd4l[k][8] * _u1(i - 1, j, 8)) +
            p4[2] * _f1_1(i + 1, j) *
                (phd4l[k][0] * _u1(i + 1, j, 0) +
                 phd4l[k][1] * _u1(i + 1, j, 1) +
                 phd4l[k][2] * _u1(i + 1, j, 2) +
                 phd4l[k][3] * _u1(i + 1, j, 3) +
                 phd4l[k][4] * _u1(i + 1, j, 4) +
                 phd4l[k][5] * _u1(i + 1, j, 5) +
                 phd4l[k][6] * _u1(i + 1, j, 6) +
                 phd4l[k][7] * _u1(i + 1, j, 7) +
                 phd4l[k][8] * _u1(i + 1, j, 8)) +
            p4[3] * _f1_1(i + 2, j) *
                (phd4l[k][0] * _u1(i + 2, j, 0) +
                 phd4l[k][1] * _u1(i + 2, j, 1) +
                 phd4l[k][2] * _u1(i + 2, j, 2) +
                 phd4l[k][3] * _u1(i + 2, j, 3) +
                 phd4l[k][4] * _u1(i + 2, j, 4) +
                 phd4l[k][5] * _u1(i + 2, j, 5) +
                 phd4l[k][6] * _u1(i + 2, j, 6) +
                 phd4l[k][7] * _u1(i + 2, j, 7) +
                 phd4l[k][8] * _u1(i + 2, j, 8)))) *
      f_dcrj;
  _s22(i, j, k) =
      (a * _s22(i, j, k) + lam * div +
       twomu * (dh4[2] * _u2(i, j, k) + dh4[0] * _u2(i, j - 2, k) +
                dh4[1] * _u2(i, j - 1, k) + dh4[3] * _u2(i, j + 1, k)) -
       twomu * Jii * _g_c(k) *
           (ph4[2] * _f2_2(i, j) *
                (phd4l[k][0] * _u2(i, j, 0) + phd4l[k][1] * _u2(i, j, 1) +
                 phd4l[k][2] * _u2(i, j, 2) + phd4l[k][3] * _u2(i, j, 3) +
                 phd4l[k][4] * _u2(i, j, 4) + phd4l[k][5] * _u2(i, j, 5) +
                 phd4l[k][6] * _u2(i, j, 6) + phd4l[k][7] * _u2(i, j, 7) +
                 phd4l[k][8] * _u2(i, j, 8)) +
            ph4[0] * _f2_2(i, j - 2) *
                (phd4l[k][0] * _u2(i, j - 2, 0) +
                 phd4l[k][1] * _u2(i, j - 2, 1) +
                 phd4l[k][2] * _u2(i, j - 2, 2) +
                 phd4l[k][3] * _u2(i, j - 2, 3) +
                 phd4l[k][4] * _u2(i, j - 2, 4) +
                 phd4l[k][5] * _u2(i, j - 2, 5) +
                 phd4l[k][6] * _u2(i, j - 2, 6) +
                 phd4l[k][7] * _u2(i, j - 2, 7) +
                 phd4l[k][8] * _u2(i, j - 2, 8)) +
            ph4[1] * _f2_2(i, j - 1) *
                (phd4l[k][0] * _u2(i, j - 1, 0) +
                 phd4l[k][1] * _u2(i, j - 1, 1) +
                 phd4l[k][2] * _u2(i, j - 1, 2) +
                 phd4l[k][3] * _u2(i, j - 1, 3) +
                 phd4l[k][4] * _u2(i, j - 1, 4) +
                 phd4l[k][5] * _u2(i, j - 1, 5) +
                 phd4l[k][6] * _u2(i, j - 1, 6) +
                 phd4l[k][7] * _u2(i, j - 1, 7) +
                 phd4l[k][8] * _u2(i, j - 1, 8)) +
            ph4[3] * _f2_2(i, j + 1) *
                (phd4l[k][0] * _u2(i, j + 1, 0) +
                 phd4l[k][1] * _u2(i, j + 1, 1) +
                 phd4l[k][2] * _u2(i, j + 1, 2) +
                 phd4l[k][3] * _u2(i, j + 1, 3) +
                 phd4l[k][4] * _u2(i, j + 1, 4) +
                 phd4l[k][5] * _u2(i, j + 1, 5) +
                 phd4l[k][6] * _u2(i, j + 1, 6) +
                 phd4l[k][7] * _u2(i, j + 1, 7) +
                 phd4l[k][8] * _u2(i, j + 1, 8)))) *
      f_dcrj;
  _s33(i, j, k) = (a * _s33(i, j, k) + lam * div +
                   twomu * Jii *
                       (dh4l[k][0] * _u3(i, j, 0) + dh4l[k][1] * _u3(i, j, 1) +
                        dh4l[k][2] * _u3(i, j, 2) + dh4l[k][3] * _u3(i, j, 3) +
                        dh4l[k][4] * _u3(i, j, 4) + dh4l[k][5] * _u3(i, j, 5) +
                        dh4l[k][6] * _u3(i, j, 6))) *
                  f_dcrj;
  _s12(i, j, k) =
      (a * _s12(i, j, k) +
       mu12 *
           (d4[1] * _u1(i, j, k) + d4[0] * _u1(i, j - 1, k) +
            d4[2] * _u1(i, j + 1, k) + d4[3] * _u1(i, j + 2, k) +
            dh4[2] * _u2(i, j, k) + dh4[0] * _u2(i - 2, j, k) +
            dh4[1] * _u2(i - 1, j, k) + dh4[3] * _u2(i + 1, j, k) -
            J12i * _g_c(k) *
                (p4[1] * _f2_1(i, j) *
                     (phd4l[k][0] * _u1(i, j, 0) + phd4l[k][1] * _u1(i, j, 1) +
                      phd4l[k][2] * _u1(i, j, 2) + phd4l[k][3] * _u1(i, j, 3) +
                      phd4l[k][4] * _u1(i, j, 4) + phd4l[k][5] * _u1(i, j, 5) +
                      phd4l[k][6] * _u1(i, j, 6) + phd4l[k][7] * _u1(i, j, 7) +
                      phd4l[k][8] * _u1(i, j, 8)) +
                 p4[0] * _f2_1(i, j - 1) *
                     (phd4l[k][0] * _u1(i, j - 1, 0) +
                      phd4l[k][1] * _u1(i, j - 1, 1) +
                      phd4l[k][2] * _u1(i, j - 1, 2) +
                      phd4l[k][3] * _u1(i, j - 1, 3) +
                      phd4l[k][4] * _u1(i, j - 1, 4) +
                      phd4l[k][5] * _u1(i, j - 1, 5) +
                      phd4l[k][6] * _u1(i, j - 1, 6) +
                      phd4l[k][7] * _u1(i, j - 1, 7) +
                      phd4l[k][8] * _u1(i, j - 1, 8)) +
                 p4[2] * _f2_1(i, j + 1) *
                     (phd4l[k][0] * _u1(i, j + 1, 0) +
                      phd4l[k][1] * _u1(i, j + 1, 1) +
                      phd4l[k][2] * _u1(i, j + 1, 2) +
                      phd4l[k][3] * _u1(i, j + 1, 3) +
                      phd4l[k][4] * _u1(i, j + 1, 4) +
                      phd4l[k][5] * _u1(i, j + 1, 5) +
                      phd4l[k][6] * _u1(i, j + 1, 6) +
                      phd4l[k][7] * _u1(i, j + 1, 7) +
                      phd4l[k][8] * _u1(i, j + 1, 8)) +
                 p4[3] * _f2_1(i, j + 2) *
                     (phd4l[k][0] * _u1(i, j + 2, 0) +
                      phd4l[k][1] * _u1(i, j + 2, 1) +
                      phd4l[k][2] * _u1(i, j + 2, 2) +
                      phd4l[k][3] * _u1(i, j + 2, 3) +
                      phd4l[k][4] * _u1(i, j + 2, 4) +
                      phd4l[k][5] * _u1(i, j + 2, 5) +
                      phd4l[k][6] * _u1(i, j + 2, 6) +
                      phd4l[k][7] * _u1(i, j + 2, 7) +
                      phd4l[k][8] * _u1(i, j + 2, 8))) -
            J12i * _g_c(k) *
                (ph4[2] * _f1_2(i, j) *
                     (phd4l[k][0] * _u2(i, j, 0) + phd4l[k][1] * _u2(i, j, 1) +
                      phd4l[k][2] * _u2(i, j, 2) + phd4l[k][3] * _u2(i, j, 3) +
                      phd4l[k][4] * _u2(i, j, 4) + phd4l[k][5] * _u2(i, j, 5) +
                      phd4l[k][6] * _u2(i, j, 6) + phd4l[k][7] * _u2(i, j, 7) +
                      phd4l[k][8] * _u2(i, j, 8)) +
                 ph4[0] * _f1_2(i - 2, j) *
                     (phd4l[k][0] * _u2(i - 2, j, 0) +
                      phd4l[k][1] * _u2(i - 2, j, 1) +
                      phd4l[k][2] * _u2(i - 2, j, 2) +
                      phd4l[k][3] * _u2(i - 2, j, 3) +
                      phd4l[k][4] * _u2(i - 2, j, 4) +
                      phd4l[k][5] * _u2(i - 2, j, 5) +
                      phd4l[k][6] * _u2(i - 2, j, 6) +
                      phd4l[k][7] * _u2(i - 2, j, 7) +
                      phd4l[k][8] * _u2(i - 2, j, 8)) +
                 ph4[1] * _f1_2(i - 1, j) *
                     (phd4l[k][0] * _u2(i - 1, j, 0) +
                      phd4l[k][1] * _u2(i - 1, j, 1) +
                      phd4l[k][2] * _u2(i - 1, j, 2) +
                      phd4l[k][3] * _u2(i - 1, j, 3) +
                      phd4l[k][4] * _u2(i - 1, j, 4) +
                      phd4l[k][5] * _u2(i - 1, j, 5) +
                      phd4l[k][6] * _u2(i - 1, j, 6) +
                      phd4l[k][7] * _u2(i - 1, j, 7) +
                      phd4l[k][8] * _u2(i - 1, j, 8)) +
                 ph4[3] * _f1_2(i + 1, j) *
                     (phd4l[k][0] * _u2(i + 1, j, 0) +
                      phd4l[k][1] * _u2(i + 1, j, 1) +
                      phd4l[k][2] * _u2(i + 1, j, 2) +
                      phd4l[k][3] * _u2(i + 1, j, 3) +
                      phd4l[k][4] * _u2(i + 1, j, 4) +
                      phd4l[k][5] * _u2(i + 1, j, 5) +
                      phd4l[k][6] * _u2(i + 1, j, 6) +
                      phd4l[k][7] * _u2(i + 1, j, 7) +
                      phd4l[k][8] * _u2(i + 1, j, 8))))) *
      f_dcrj;
  _s13(i, j, k) =
      (a * _s13(i, j, k) +
       mu13 *
           (dh4[2] * _u3(i, j, k) + dh4[0] * _u3(i - 2, j, k) +
            dh4[1] * _u3(i - 1, j, k) + dh4[3] * _u3(i + 1, j, k) +
            J13i * (d4l[k][0] * _u1(i, j, 0) + d4l[k][1] * _u1(i, j, 1) +
                    d4l[k][2] * _u1(i, j, 2) + d4l[k][3] * _u1(i, j, 3) +
                    d4l[k][4] * _u1(i, j, 4) + d4l[k][5] * _u1(i, j, 5) +
                    d4l[k][6] * _u1(i, j, 6) + d4l[k][7] * _u1(i, j, 7)) -
            J13i * _g(k) *
                (ph4[2] * _f1_c(i, j) *
                     (pdh4l[k][0] * _u3(i, j, 0) + pdh4l[k][1] * _u3(i, j, 1) +
                      pdh4l[k][2] * _u3(i, j, 2) + pdh4l[k][3] * _u3(i, j, 3) +
                      pdh4l[k][4] * _u3(i, j, 4) + pdh4l[k][5] * _u3(i, j, 5) +
                      pdh4l[k][6] * _u3(i, j, 6) + pdh4l[k][7] * _u3(i, j, 7) +
                      pdh4l[k][8] * _u3(i, j, 8)) +
                 ph4[0] * _f1_c(i - 2, j) *
                     (pdh4l[k][0] * _u3(i - 2, j, 0) +
                      pdh4l[k][1] * _u3(i - 2, j, 1) +
                      pdh4l[k][2] * _u3(i - 2, j, 2) +
                      pdh4l[k][3] * _u3(i - 2, j, 3) +
                      pdh4l[k][4] * _u3(i - 2, j, 4) +
                      pdh4l[k][5] * _u3(i - 2, j, 5) +
                      pdh4l[k][6] * _u3(i - 2, j, 6) +
                      pdh4l[k][7] * _u3(i - 2, j, 7) +
                      pdh4l[k][8] * _u3(i - 2, j, 8)) +
                 ph4[1] * _f1_c(i - 1, j) *
                     (pdh4l[k][0] * _u3(i - 1, j, 0) +
                      pdh4l[k][1] * _u3(i - 1, j, 1) +
                      pdh4l[k][2] * _u3(i - 1, j, 2) +
                      pdh4l[k][3] * _u3(i - 1, j, 3) +
                      pdh4l[k][4] * _u3(i - 1, j, 4) +
                      pdh4l[k][5] * _u3(i - 1, j, 5) +
                      pdh4l[k][6] * _u3(i - 1, j, 6) +
                      pdh4l[k][7] * _u3(i - 1, j, 7) +
                      pdh4l[k][8] * _u3(i - 1, j, 8)) +
                 ph4[3] * _f1_c(i + 1, j) *
                     (pdh4l[k][0] * _u3(i + 1, j, 0) +
                      pdh4l[k][1] * _u3(i + 1, j, 1) +
                      pdh4l[k][2] * _u3(i + 1, j, 2) +
                      pdh4l[k][3] * _u3(i + 1, j, 3) +
                      pdh4l[k][4] * _u3(i + 1, j, 4) +
                      pdh4l[k][5] * _u3(i + 1, j, 5) +
                      pdh4l[k][6] * _u3(i + 1, j, 6) +
                      pdh4l[k][7] * _u3(i + 1, j, 7) +
                      pdh4l[k][8] * _u3(i + 1, j, 8))))) *
      f_dcrj;
  _s23(i, j, k) =
      (a * _s23(i, j, k) +
       mu23 *
           (d4[1] * _u3(i, j, k) + d4[0] * _u3(i, j - 1, k) +
            d4[2] * _u3(i, j + 1, k) + d4[3] * _u3(i, j + 2, k) +
            J23i * (d4l[k][0] * _u2(i, j, 0) + d4l[k][1] * _u2(i, j, 1) +
                    d4l[k][2] * _u2(i, j, 2) + d4l[k][3] * _u2(i, j, 3) +
                    d4l[k][4] * _u2(i, j, 4) + d4l[k][5] * _u2(i, j, 5) +
                    d4l[k][6] * _u2(i, j, 6) + d4l[k][7] * _u2(i, j, 7)) -
            J23i * _g(k) *
                (p4[1] * _f2_c(i, j) *
                     (pdh4l[k][0] * _u3(i, j, 0) + pdh4l[k][1] * _u3(i, j, 1) +
                      pdh4l[k][2] * _u3(i, j, 2) + pdh4l[k][3] * _u3(i, j, 3) +
                      pdh4l[k][4] * _u3(i, j, 4) + pdh4l[k][5] * _u3(i, j, 5) +
                      pdh4l[k][6] * _u3(i, j, 6) + pdh4l[k][7] * _u3(i, j, 7) +
                      pdh4l[k][8] * _u3(i, j, 8)) +
                 p4[0] * _f2_c(i, j - 1) *
                     (pdh4l[k][0] * _u3(i, j - 1, 0) +
                      pdh4l[k][1] * _u3(i, j - 1, 1) +
                      pdh4l[k][2] * _u3(i, j - 1, 2) +
                      pdh4l[k][3] * _u3(i, j - 1, 3) +
                      pdh4l[k][4] * _u3(i, j - 1, 4) +
                      pdh4l[k][5] * _u3(i, j - 1, 5) +
                      pdh4l[k][6] * _u3(i, j - 1, 6) +
                      pdh4l[k][7] * _u3(i, j - 1, 7) +
                      pdh4l[k][8] * _u3(i, j - 1, 8)) +
                 p4[2] * _f2_c(i, j + 1) *
                     (pdh4l[k][0] * _u3(i, j + 1, 0) +
                      pdh4l[k][1] * _u3(i, j + 1, 1) +
                      pdh4l[k][2] * _u3(i, j + 1, 2) +
                      pdh4l[k][3] * _u3(i, j + 1, 3) +
                      pdh4l[k][4] * _u3(i, j + 1, 4) +
                      pdh4l[k][5] * _u3(i, j + 1, 5) +
                      pdh4l[k][6] * _u3(i, j + 1, 6) +
                      pdh4l[k][7] * _u3(i, j + 1, 7) +
                      pdh4l[k][8] * _u3(i, j + 1, 8)) +
                 p4[3] * _f2_c(i, j + 2) *
                     (pdh4l[k][0] * _u3(i, j + 2, 0) +
                      pdh4l[k][1] * _u3(i, j + 2, 1) +
                      pdh4l[k][2] * _u3(i, j + 2, 2) +
                      pdh4l[k][3] * _u3(i, j + 2, 3) +
                      pdh4l[k][4] * _u3(i, j + 2, 4) +
                      pdh4l[k][5] * _u3(i, j + 2, 5) +
                      pdh4l[k][6] * _u3(i, j + 2, 6) +
                      pdh4l[k][7] * _u3(i, j + 2, 7) +
                      pdh4l[k][8] * _u3(i, j + 2, 8))))) *
      f_dcrj;
#undef _g3_c
#undef _f_c
#undef _f
#undef _f_1
#undef _g3
#undef _f_2
#undef _lami
#undef _mui
#undef _f1_1
#undef _u1
#undef _g_c
#undef _u2
#undef _f2_2
#undef _u3
#undef _dcrjz
#undef _dcrjx
#undef _dcrjy
#undef _s11
#undef _s22
#undef _s33
#undef _f1_2
#undef _f2_1
#undef _s12
#undef _g
#undef _f1_c
#undef _s13
#undef _f2_c
#undef _s23
}

__global__ void dtopo_str_111(
    float *__restrict__ s11, float *__restrict__ s12, float *__restrict__ s13,
    float *__restrict__ s22, float *__restrict__ s23, float *__restrict__ s33,
    float *__restrict__ u1, float *__restrict__ u2, float *__restrict__ u3,
    const float *__restrict__ dcrjx, const float *__restrict__ dcrjy,
    const float *__restrict__ dcrjz, const float *__restrict__ f,
    const float *__restrict__ f1_1, const float *__restrict__ f1_2,
    const float *__restrict__ f1_c, const float *__restrict__ f2_1,
    const float *__restrict__ f2_2, const float *__restrict__ f2_c,
    const float *__restrict__ f_1, const float *__restrict__ f_2,
    const float *__restrict__ f_c, const float *__restrict__ g,
    const float *__restrict__ g3, const float *__restrict__ g3_c,
    const float *__restrict__ g_c, const float *__restrict__ lami,
    const float *__restrict__ mui, const float a, const float nu, const int nx,
    const int ny, const int nz, const int bi, const int bj, const int ei,
    const int ej) {
  const float ph2[2] = {0.5000000000000000, 0.5000000000000000};
  const float p2[2] = {0.5000000000000000, 0.5000000000000000};
  const float dh4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float p4[4] = {-0.0625000000000000, 0.5625000000000000,
                       0.5625000000000000, -0.0625000000000000};
  const float phd4[7] = {-0.0026041666666667, 0.0937500000000000,
                         -0.6796875000000000, -0.0000000000000000,
                         0.6796875000000000,  -0.0937500000000000,
                         0.0026041666666667};
  const float ph4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float d4[4] = {0.0416666666666667, -1.1250000000000000,
                       1.1250000000000000, -0.0416666666666667};
  const float pdh4[7] = {-0.0026041666666667, 0.0937500000000000,
                         -0.6796875000000000, -0.0000000000000000,
                         0.6796875000000000,  -0.0937500000000000,
                         0.0026041666666667};
  const int i = threadIdx.z + blockIdx.z * blockDim.z + bi;
  if (i >= ngsl + nx)
    return;
  if (i >= ei)
    return;
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ngsl + ny)
    return;
  if (j >= ej)
    return;
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= nz - 12)
    return;
#define _g3_c(k) g3_c[(k) + align]
#define _f_c(i, j)                                                             \
  f_c[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f(i, j)                                                               \
  f[(j) + align + ngsl + ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f_1(i, j)                                                             \
  f_1[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3(k) g3[(k) + align]
#define _f_2(i, j)                                                             \
  f_2[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _lami(i, j, k)                                                         \
  lami[(k) + align +                                                           \
       (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +             \
       (2 * align + nz) * ((j) + ngsl + 2)]
#define _mui(i, j, k)                                                          \
  mui[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_1(i, j)                                                            \
  f1_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _u1(i, j, k)                                                           \
  u1[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _g_c(k) g_c[(k) + align]
#define _u2(i, j, k)                                                           \
  u2[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_2(i, j)                                                            \
  f2_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _u3(i, j, k)                                                           \
  u3[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _dcrjz(k) dcrjz[(k) + align]
#define _dcrjx(i) dcrjx[(i) + ngsl + 2]
#define _dcrjy(j) dcrjy[(j) + ngsl + 2]
#define _s11(i, j, k)                                                          \
  s11[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s22(i, j, k)                                                          \
  s22[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s33(i, j, k)                                                          \
  s33[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_2(i, j)                                                            \
  f1_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_1(i, j)                                                            \
  f2_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s12(i, j, k)                                                          \
  s12[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _g(k) g[(k) + align]
#define _f1_c(i, j)                                                            \
  f1_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s13(i, j, k)                                                          \
  s13[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_c(i, j)                                                            \
  f2_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s23(i, j, k)                                                          \
  s23[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
  float Jii = _f_c(i, j) * _g3_c(k + 6);
  Jii = 1.0 * 1.0 / Jii;
  float J12i = _f(i, j) * _g3_c(k + 6);
  J12i = 1.0 * 1.0 / J12i;
  float J13i = _f_1(i, j) * _g3(k + 6);
  J13i = 1.0 * 1.0 / J13i;
  float J23i = _f_2(i, j) * _g3(k + 6);
  J23i = 1.0 * 1.0 / J23i;
  float lam = nu * 1.0 /
              (ph2[0] * (ph2[1] * (p2[0] * _lami(i, j, k + 5) +
                                   p2[1] * _lami(i + 1, j, k + 5)) +
                         ph2[0] * (p2[0] * _lami(i, j - 1, k + 5) +
                                   p2[1] * _lami(i + 1, j - 1, k + 5))) +
               ph2[1] * (ph2[1] * (p2[0] * _lami(i, j, k + 6) +
                                   p2[1] * _lami(i + 1, j, k + 6)) +
                         ph2[0] * (p2[0] * _lami(i, j - 1, k + 6) +
                                   p2[1] * _lami(i + 1, j - 1, k + 6))));
  float twomu = 2 * nu * 1.0 /
                (ph2[0] * (ph2[1] * (p2[0] * _mui(i, j, k + 5) +
                                     p2[1] * _mui(i + 1, j, k + 5)) +
                           ph2[0] * (p2[0] * _mui(i, j - 1, k + 5) +
                                     p2[1] * _mui(i + 1, j - 1, k + 5))) +
                 ph2[1] * (ph2[1] * (p2[0] * _mui(i, j, k + 6) +
                                     p2[1] * _mui(i + 1, j, k + 6)) +
                           ph2[0] * (p2[0] * _mui(i, j - 1, k + 6) +
                                     p2[1] * _mui(i + 1, j - 1, k + 6))));
  float mu12 =
      nu * 1.0 / (ph2[0] * _mui(i, j, k + 5) + ph2[1] * _mui(i, j, k + 6));
  float mu13 =
      nu * 1.0 / (ph2[1] * _mui(i, j, k + 6) + ph2[0] * _mui(i, j - 1, k + 6));
  float mu23 =
      nu * 1.0 / (p2[0] * _mui(i, j, k + 6) + p2[1] * _mui(i + 1, j, k + 6));
  float div =
      d4[1] * _u1(i, j, k + 6) + d4[0] * _u1(i - 1, j, k + 6) +
      d4[2] * _u1(i + 1, j, k + 6) + d4[3] * _u1(i + 2, j, k + 6) +
      dh4[2] * _u2(i, j, k + 6) + dh4[0] * _u2(i, j - 2, k + 6) +
      dh4[1] * _u2(i, j - 1, k + 6) + dh4[3] * _u2(i, j + 1, k + 6) +
      Jii * (dh4[0] * _u3(i, j, k + 4) + dh4[1] * _u3(i, j, k + 5) +
             dh4[2] * _u3(i, j, k + 6) + dh4[3] * _u3(i, j, k + 7)) -
      Jii * _g_c(k + 6) *
          (p4[1] * _f1_1(i, j) *
               (phd4[0] * _u1(i, j, k + 3) + phd4[1] * _u1(i, j, k + 4) +
                phd4[2] * _u1(i, j, k + 5) + phd4[3] * _u1(i, j, k + 6) +
                phd4[4] * _u1(i, j, k + 7) + phd4[5] * _u1(i, j, k + 8) +
                phd4[6] * _u1(i, j, k + 9)) +
           p4[0] * _f1_1(i - 1, j) *
               (phd4[0] * _u1(i - 1, j, k + 3) +
                phd4[1] * _u1(i - 1, j, k + 4) +
                phd4[2] * _u1(i - 1, j, k + 5) +
                phd4[3] * _u1(i - 1, j, k + 6) +
                phd4[4] * _u1(i - 1, j, k + 7) +
                phd4[5] * _u1(i - 1, j, k + 8) +
                phd4[6] * _u1(i - 1, j, k + 9)) +
           p4[2] * _f1_1(i + 1, j) *
               (phd4[0] * _u1(i + 1, j, k + 3) +
                phd4[1] * _u1(i + 1, j, k + 4) +
                phd4[2] * _u1(i + 1, j, k + 5) +
                phd4[3] * _u1(i + 1, j, k + 6) +
                phd4[4] * _u1(i + 1, j, k + 7) +
                phd4[5] * _u1(i + 1, j, k + 8) +
                phd4[6] * _u1(i + 1, j, k + 9)) +
           p4[3] * _f1_1(i + 2, j) *
               (phd4[0] * _u1(i + 2, j, k + 3) +
                phd4[1] * _u1(i + 2, j, k + 4) +
                phd4[2] * _u1(i + 2, j, k + 5) +
                phd4[3] * _u1(i + 2, j, k + 6) +
                phd4[4] * _u1(i + 2, j, k + 7) +
                phd4[5] * _u1(i + 2, j, k + 8) +
                phd4[6] * _u1(i + 2, j, k + 9))) -
      Jii * _g_c(k + 6) *
          (ph4[2] * _f2_2(i, j) *
               (phd4[0] * _u2(i, j, k + 3) + phd4[1] * _u2(i, j, k + 4) +
                phd4[2] * _u2(i, j, k + 5) + phd4[3] * _u2(i, j, k + 6) +
                phd4[4] * _u2(i, j, k + 7) + phd4[5] * _u2(i, j, k + 8) +
                phd4[6] * _u2(i, j, k + 9)) +
           ph4[0] * _f2_2(i, j - 2) *
               (phd4[0] * _u2(i, j - 2, k + 3) +
                phd4[1] * _u2(i, j - 2, k + 4) +
                phd4[2] * _u2(i, j - 2, k + 5) +
                phd4[3] * _u2(i, j - 2, k + 6) +
                phd4[4] * _u2(i, j - 2, k + 7) +
                phd4[5] * _u2(i, j - 2, k + 8) +
                phd4[6] * _u2(i, j - 2, k + 9)) +
           ph4[1] * _f2_2(i, j - 1) *
               (phd4[0] * _u2(i, j - 1, k + 3) +
                phd4[1] * _u2(i, j - 1, k + 4) +
                phd4[2] * _u2(i, j - 1, k + 5) +
                phd4[3] * _u2(i, j - 1, k + 6) +
                phd4[4] * _u2(i, j - 1, k + 7) +
                phd4[5] * _u2(i, j - 1, k + 8) +
                phd4[6] * _u2(i, j - 1, k + 9)) +
           ph4[3] * _f2_2(i, j + 1) *
               (phd4[0] * _u2(i, j + 1, k + 3) +
                phd4[1] * _u2(i, j + 1, k + 4) +
                phd4[2] * _u2(i, j + 1, k + 5) +
                phd4[3] * _u2(i, j + 1, k + 6) +
                phd4[4] * _u2(i, j + 1, k + 7) +
                phd4[5] * _u2(i, j + 1, k + 8) +
                phd4[6] * _u2(i, j + 1, k + 9)));
  float f_dcrj = _dcrjx(i) * _dcrjy(j) * _dcrjz(k + 6);
  _s11(i, j, k + 6) =
      (a * _s11(i, j, k + 6) + lam * div +
       twomu * (d4[1] * _u1(i, j, k + 6) + d4[0] * _u1(i - 1, j, k + 6) +
                d4[2] * _u1(i + 1, j, k + 6) + d4[3] * _u1(i + 2, j, k + 6)) -
       twomu * Jii * _g_c(k + 6) *
           (p4[1] * _f1_1(i, j) *
                (phd4[0] * _u1(i, j, k + 3) + phd4[1] * _u1(i, j, k + 4) +
                 phd4[2] * _u1(i, j, k + 5) + phd4[3] * _u1(i, j, k + 6) +
                 phd4[4] * _u1(i, j, k + 7) + phd4[5] * _u1(i, j, k + 8) +
                 phd4[6] * _u1(i, j, k + 9)) +
            p4[0] * _f1_1(i - 1, j) *
                (phd4[0] * _u1(i - 1, j, k + 3) +
                 phd4[1] * _u1(i - 1, j, k + 4) +
                 phd4[2] * _u1(i - 1, j, k + 5) +
                 phd4[3] * _u1(i - 1, j, k + 6) +
                 phd4[4] * _u1(i - 1, j, k + 7) +
                 phd4[5] * _u1(i - 1, j, k + 8) +
                 phd4[6] * _u1(i - 1, j, k + 9)) +
            p4[2] * _f1_1(i + 1, j) *
                (phd4[0] * _u1(i + 1, j, k + 3) +
                 phd4[1] * _u1(i + 1, j, k + 4) +
                 phd4[2] * _u1(i + 1, j, k + 5) +
                 phd4[3] * _u1(i + 1, j, k + 6) +
                 phd4[4] * _u1(i + 1, j, k + 7) +
                 phd4[5] * _u1(i + 1, j, k + 8) +
                 phd4[6] * _u1(i + 1, j, k + 9)) +
            p4[3] * _f1_1(i + 2, j) *
                (phd4[0] * _u1(i + 2, j, k + 3) +
                 phd4[1] * _u1(i + 2, j, k + 4) +
                 phd4[2] * _u1(i + 2, j, k + 5) +
                 phd4[3] * _u1(i + 2, j, k + 6) +
                 phd4[4] * _u1(i + 2, j, k + 7) +
                 phd4[5] * _u1(i + 2, j, k + 8) +
                 phd4[6] * _u1(i + 2, j, k + 9)))) *
      f_dcrj;
  _s22(i, j, k + 6) =
      (a * _s22(i, j, k + 6) + lam * div +
       twomu * (dh4[2] * _u2(i, j, k + 6) + dh4[0] * _u2(i, j - 2, k + 6) +
                dh4[1] * _u2(i, j - 1, k + 6) + dh4[3] * _u2(i, j + 1, k + 6)) -
       twomu * Jii * _g_c(k + 6) *
           (ph4[2] * _f2_2(i, j) *
                (phd4[0] * _u2(i, j, k + 3) + phd4[1] * _u2(i, j, k + 4) +
                 phd4[2] * _u2(i, j, k + 5) + phd4[3] * _u2(i, j, k + 6) +
                 phd4[4] * _u2(i, j, k + 7) + phd4[5] * _u2(i, j, k + 8) +
                 phd4[6] * _u2(i, j, k + 9)) +
            ph4[0] * _f2_2(i, j - 2) *
                (phd4[0] * _u2(i, j - 2, k + 3) +
                 phd4[1] * _u2(i, j - 2, k + 4) +
                 phd4[2] * _u2(i, j - 2, k + 5) +
                 phd4[3] * _u2(i, j - 2, k + 6) +
                 phd4[4] * _u2(i, j - 2, k + 7) +
                 phd4[5] * _u2(i, j - 2, k + 8) +
                 phd4[6] * _u2(i, j - 2, k + 9)) +
            ph4[1] * _f2_2(i, j - 1) *
                (phd4[0] * _u2(i, j - 1, k + 3) +
                 phd4[1] * _u2(i, j - 1, k + 4) +
                 phd4[2] * _u2(i, j - 1, k + 5) +
                 phd4[3] * _u2(i, j - 1, k + 6) +
                 phd4[4] * _u2(i, j - 1, k + 7) +
                 phd4[5] * _u2(i, j - 1, k + 8) +
                 phd4[6] * _u2(i, j - 1, k + 9)) +
            ph4[3] * _f2_2(i, j + 1) *
                (phd4[0] * _u2(i, j + 1, k + 3) +
                 phd4[1] * _u2(i, j + 1, k + 4) +
                 phd4[2] * _u2(i, j + 1, k + 5) +
                 phd4[3] * _u2(i, j + 1, k + 6) +
                 phd4[4] * _u2(i, j + 1, k + 7) +
                 phd4[5] * _u2(i, j + 1, k + 8) +
                 phd4[6] * _u2(i, j + 1, k + 9)))) *
      f_dcrj;
  _s33(i, j, k + 6) =
      (a * _s33(i, j, k + 6) + lam * div +
       twomu * Jii *
           (dh4[0] * _u3(i, j, k + 4) + dh4[1] * _u3(i, j, k + 5) +
            dh4[2] * _u3(i, j, k + 6) + dh4[3] * _u3(i, j, k + 7))) *
      f_dcrj;
  _s12(i, j, k + 6) =
      (a * _s12(i, j, k + 6) +
       mu12 *
           (d4[1] * _u1(i, j, k + 6) + d4[0] * _u1(i, j - 1, k + 6) +
            d4[2] * _u1(i, j + 1, k + 6) + d4[3] * _u1(i, j + 2, k + 6) +
            dh4[2] * _u2(i, j, k + 6) + dh4[0] * _u2(i - 2, j, k + 6) +
            dh4[1] * _u2(i - 1, j, k + 6) + dh4[3] * _u2(i + 1, j, k + 6) -
            J12i * _g_c(k + 6) *
                (p4[1] * _f2_1(i, j) *
                     (phd4[0] * _u1(i, j, k + 3) + phd4[1] * _u1(i, j, k + 4) +
                      phd4[2] * _u1(i, j, k + 5) + phd4[3] * _u1(i, j, k + 6) +
                      phd4[4] * _u1(i, j, k + 7) + phd4[5] * _u1(i, j, k + 8) +
                      phd4[6] * _u1(i, j, k + 9)) +
                 p4[0] * _f2_1(i, j - 1) *
                     (phd4[0] * _u1(i, j - 1, k + 3) +
                      phd4[1] * _u1(i, j - 1, k + 4) +
                      phd4[2] * _u1(i, j - 1, k + 5) +
                      phd4[3] * _u1(i, j - 1, k + 6) +
                      phd4[4] * _u1(i, j - 1, k + 7) +
                      phd4[5] * _u1(i, j - 1, k + 8) +
                      phd4[6] * _u1(i, j - 1, k + 9)) +
                 p4[2] * _f2_1(i, j + 1) *
                     (phd4[0] * _u1(i, j + 1, k + 3) +
                      phd4[1] * _u1(i, j + 1, k + 4) +
                      phd4[2] * _u1(i, j + 1, k + 5) +
                      phd4[3] * _u1(i, j + 1, k + 6) +
                      phd4[4] * _u1(i, j + 1, k + 7) +
                      phd4[5] * _u1(i, j + 1, k + 8) +
                      phd4[6] * _u1(i, j + 1, k + 9)) +
                 p4[3] * _f2_1(i, j + 2) *
                     (phd4[0] * _u1(i, j + 2, k + 3) +
                      phd4[1] * _u1(i, j + 2, k + 4) +
                      phd4[2] * _u1(i, j + 2, k + 5) +
                      phd4[3] * _u1(i, j + 2, k + 6) +
                      phd4[4] * _u1(i, j + 2, k + 7) +
                      phd4[5] * _u1(i, j + 2, k + 8) +
                      phd4[6] * _u1(i, j + 2, k + 9))) -
            J12i * _g_c(k + 6) *
                (ph4[2] * _f1_2(i, j) *
                     (phd4[0] * _u2(i, j, k + 3) + phd4[1] * _u2(i, j, k + 4) +
                      phd4[2] * _u2(i, j, k + 5) + phd4[3] * _u2(i, j, k + 6) +
                      phd4[4] * _u2(i, j, k + 7) + phd4[5] * _u2(i, j, k + 8) +
                      phd4[6] * _u2(i, j, k + 9)) +
                 ph4[0] * _f1_2(i - 2, j) *
                     (phd4[0] * _u2(i - 2, j, k + 3) +
                      phd4[1] * _u2(i - 2, j, k + 4) +
                      phd4[2] * _u2(i - 2, j, k + 5) +
                      phd4[3] * _u2(i - 2, j, k + 6) +
                      phd4[4] * _u2(i - 2, j, k + 7) +
                      phd4[5] * _u2(i - 2, j, k + 8) +
                      phd4[6] * _u2(i - 2, j, k + 9)) +
                 ph4[1] * _f1_2(i - 1, j) *
                     (phd4[0] * _u2(i - 1, j, k + 3) +
                      phd4[1] * _u2(i - 1, j, k + 4) +
                      phd4[2] * _u2(i - 1, j, k + 5) +
                      phd4[3] * _u2(i - 1, j, k + 6) +
                      phd4[4] * _u2(i - 1, j, k + 7) +
                      phd4[5] * _u2(i - 1, j, k + 8) +
                      phd4[6] * _u2(i - 1, j, k + 9)) +
                 ph4[3] * _f1_2(i + 1, j) *
                     (phd4[0] * _u2(i + 1, j, k + 3) +
                      phd4[1] * _u2(i + 1, j, k + 4) +
                      phd4[2] * _u2(i + 1, j, k + 5) +
                      phd4[3] * _u2(i + 1, j, k + 6) +
                      phd4[4] * _u2(i + 1, j, k + 7) +
                      phd4[5] * _u2(i + 1, j, k + 8) +
                      phd4[6] * _u2(i + 1, j, k + 9))))) *
      f_dcrj;
  _s13(i, j, k + 6) =
      (a * _s13(i, j, k + 6) +
       mu13 *
           (dh4[2] * _u3(i, j, k + 6) + dh4[0] * _u3(i - 2, j, k + 6) +
            dh4[1] * _u3(i - 1, j, k + 6) + dh4[3] * _u3(i + 1, j, k + 6) +
            J13i * (d4[0] * _u1(i, j, k + 5) + d4[1] * _u1(i, j, k + 6) +
                    d4[2] * _u1(i, j, k + 7) + d4[3] * _u1(i, j, k + 8)) -
            J13i * _g(k + 6) *
                (ph4[2] * _f1_c(i, j) *
                     (pdh4[0] * _u3(i, j, k + 3) + pdh4[1] * _u3(i, j, k + 4) +
                      pdh4[2] * _u3(i, j, k + 5) + pdh4[3] * _u3(i, j, k + 6) +
                      pdh4[4] * _u3(i, j, k + 7) + pdh4[5] * _u3(i, j, k + 8) +
                      pdh4[6] * _u3(i, j, k + 9)) +
                 ph4[0] * _f1_c(i - 2, j) *
                     (pdh4[0] * _u3(i - 2, j, k + 3) +
                      pdh4[1] * _u3(i - 2, j, k + 4) +
                      pdh4[2] * _u3(i - 2, j, k + 5) +
                      pdh4[3] * _u3(i - 2, j, k + 6) +
                      pdh4[4] * _u3(i - 2, j, k + 7) +
                      pdh4[5] * _u3(i - 2, j, k + 8) +
                      pdh4[6] * _u3(i - 2, j, k + 9)) +
                 ph4[1] * _f1_c(i - 1, j) *
                     (pdh4[0] * _u3(i - 1, j, k + 3) +
                      pdh4[1] * _u3(i - 1, j, k + 4) +
                      pdh4[2] * _u3(i - 1, j, k + 5) +
                      pdh4[3] * _u3(i - 1, j, k + 6) +
                      pdh4[4] * _u3(i - 1, j, k + 7) +
                      pdh4[5] * _u3(i - 1, j, k + 8) +
                      pdh4[6] * _u3(i - 1, j, k + 9)) +
                 ph4[3] * _f1_c(i + 1, j) *
                     (pdh4[0] * _u3(i + 1, j, k + 3) +
                      pdh4[1] * _u3(i + 1, j, k + 4) +
                      pdh4[2] * _u3(i + 1, j, k + 5) +
                      pdh4[3] * _u3(i + 1, j, k + 6) +
                      pdh4[4] * _u3(i + 1, j, k + 7) +
                      pdh4[5] * _u3(i + 1, j, k + 8) +
                      pdh4[6] * _u3(i + 1, j, k + 9))))) *
      f_dcrj;
  _s23(i, j, k + 6) =
      (a * _s23(i, j, k + 6) +
       mu23 *
           (d4[1] * _u3(i, j, k + 6) + d4[0] * _u3(i, j - 1, k + 6) +
            d4[2] * _u3(i, j + 1, k + 6) + d4[3] * _u3(i, j + 2, k + 6) +
            J23i * (d4[0] * _u2(i, j, k + 5) + d4[1] * _u2(i, j, k + 6) +
                    d4[2] * _u2(i, j, k + 7) + d4[3] * _u2(i, j, k + 8)) -
            J23i * _g(k + 6) *
                (p4[1] * _f2_c(i, j) *
                     (pdh4[0] * _u3(i, j, k + 3) + pdh4[1] * _u3(i, j, k + 4) +
                      pdh4[2] * _u3(i, j, k + 5) + pdh4[3] * _u3(i, j, k + 6) +
                      pdh4[4] * _u3(i, j, k + 7) + pdh4[5] * _u3(i, j, k + 8) +
                      pdh4[6] * _u3(i, j, k + 9)) +
                 p4[0] * _f2_c(i, j - 1) *
                     (pdh4[0] * _u3(i, j - 1, k + 3) +
                      pdh4[1] * _u3(i, j - 1, k + 4) +
                      pdh4[2] * _u3(i, j - 1, k + 5) +
                      pdh4[3] * _u3(i, j - 1, k + 6) +
                      pdh4[4] * _u3(i, j - 1, k + 7) +
                      pdh4[5] * _u3(i, j - 1, k + 8) +
                      pdh4[6] * _u3(i, j - 1, k + 9)) +
                 p4[2] * _f2_c(i, j + 1) *
                     (pdh4[0] * _u3(i, j + 1, k + 3) +
                      pdh4[1] * _u3(i, j + 1, k + 4) +
                      pdh4[2] * _u3(i, j + 1, k + 5) +
                      pdh4[3] * _u3(i, j + 1, k + 6) +
                      pdh4[4] * _u3(i, j + 1, k + 7) +
                      pdh4[5] * _u3(i, j + 1, k + 8) +
                      pdh4[6] * _u3(i, j + 1, k + 9)) +
                 p4[3] * _f2_c(i, j + 2) *
                     (pdh4[0] * _u3(i, j + 2, k + 3) +
                      pdh4[1] * _u3(i, j + 2, k + 4) +
                      pdh4[2] * _u3(i, j + 2, k + 5) +
                      pdh4[3] * _u3(i, j + 2, k + 6) +
                      pdh4[4] * _u3(i, j + 2, k + 7) +
                      pdh4[5] * _u3(i, j + 2, k + 8) +
                      pdh4[6] * _u3(i, j + 2, k + 9))))) *
      f_dcrj;
#undef _g3_c
#undef _f_c
#undef _f
#undef _f_1
#undef _g3
#undef _f_2
#undef _lami
#undef _mui
#undef _f1_1
#undef _u1
#undef _g_c
#undef _u2
#undef _f2_2
#undef _u3
#undef _dcrjz
#undef _dcrjx
#undef _dcrjy
#undef _s11
#undef _s22
#undef _s33
#undef _f1_2
#undef _f2_1
#undef _s12
#undef _g
#undef _f1_c
#undef _s13
#undef _f2_c
#undef _s23
}

__global__ void dtopo_str_112(
    float *__restrict__ s11, float *__restrict__ s12, float *__restrict__ s13,
    float *__restrict__ s22, float *__restrict__ s23, float *__restrict__ s33,
    float *__restrict__ u1, float *__restrict__ u2, float *__restrict__ u3,
    const float *__restrict__ dcrjx, const float *__restrict__ dcrjy,
    const float *__restrict__ dcrjz, const float *__restrict__ f,
    const float *__restrict__ f1_1, const float *__restrict__ f1_2,
    const float *__restrict__ f1_c, const float *__restrict__ f2_1,
    const float *__restrict__ f2_2, const float *__restrict__ f2_c,
    const float *__restrict__ f_1, const float *__restrict__ f_2,
    const float *__restrict__ f_c, const float *__restrict__ g,
    const float *__restrict__ g3, const float *__restrict__ g3_c,
    const float *__restrict__ g_c, const float *__restrict__ lami,
    const float *__restrict__ mui, const float a, const float nu, const int nx,
    const int ny, const int nz, const int bi, const int bj, const int ei,
    const int ej) {
  const float ph2r[6][8] = {
      {0.0000000000000000, 1.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.5000000000000000, 0.5000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.5000000000000000,
       0.5000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.5000000000000000, 0.5000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.5000000000000000, 0.5000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.5000000000000000,
       0.5000000000000000, 0.0000000000000000}};
  const float ph2[2] = {0.5000000000000000, 0.5000000000000000};
  const float p2[2] = {0.5000000000000000, 0.5000000000000000};
  const float dh4r[6][8] = {
      {0.0000000000000000, 1.4511412472637157, -1.8534237417911470,
       0.3534237417911469, 0.0488587527362844, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.8577143189081458, -0.5731429567244373,
       -0.4268570432755628, 0.1422856810918542, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.1674548505882877, 0.4976354482351368,
       -0.4976354482351368, -0.1674548505882877, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, -0.1027061113405124, 0.2624541326469860,
       0.8288742701021167, -1.0342864927831414, 0.0456642013745513,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       -0.0416666666666667, 1.1250000000000000, -1.1250000000000000,
       0.0416666666666667, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, -0.0416666666666667, 1.1250000000000000,
       -1.1250000000000000, 0.0416666666666667}};
  const float p4[4] = {-0.0625000000000000, 0.5625000000000000,
                       0.5625000000000000, -0.0625000000000000};
  const float phd4r[6][9] = {
      {1.5373923010673116, -1.0330083346742178, -0.6211677623382129,
       -0.0454110758451345, 0.1680934225988761, -0.0058985508086226,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.8713921425924012, -0.1273679143938725, -0.9297550647681331,
       0.1912595577524762, -0.0050469052908678, -0.0004818158920039,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.0563333965151294, 0.3996393739211770, 0.0536007135209481,
       -0.5022638816465500, -0.0083321572725344, 0.0010225549618299,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.0132930497153990, -0.0706942590708847, 0.5596445380498726,
       0.1434031863528334, -0.7456356868769503, 0.1028431844156395,
       -0.0028540125859095, 0.0000000000000000, 0.0000000000000000},
      {0.0025849423769932, -0.0492307522105194, 0.0524552477068130,
       0.5317248489238559, 0.0530169938441240, -0.6816971139746001,
       0.0937500000000000, -0.0026041666666667, 0.0000000000000000},
      {0.0009619461344193, 0.0035553215968974, -0.0124936029037323,
       -0.0773639466787397, 0.6736586580761996, 0.0002232904416222,
       -0.6796875000000000, 0.0937500000000000, -0.0026041666666667}};
  const float ph4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float d4[4] = {0.0416666666666667, -1.1250000000000000,
                       1.1250000000000000, -0.0416666666666667};
  const float dh4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float d4r[6][7] = {
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {1.7779989465546748, -1.3337480247900155, -0.7775013168066564,
       0.3332503950419969, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.4410217341392059, 0.1730842484889890, -0.4487228323259926,
       -0.1653831503022022, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {-0.1798793213882701, 0.2757257254150788, 0.9597948548284453,
       -1.1171892610431817, 0.0615480021879277, 0.0000000000000000,
       0.0000000000000000},
      {-0.0153911381507088, -0.0568851455503591, 0.1998976464597171,
       0.8628231468598346, -1.0285385292191949, 0.0380940196007109,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       -0.0416666666666667, 1.1250000000000000, -1.1250000000000000,
       0.0416666666666667}};
  const float pdh4r[6][9] = {
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 1.5886075042755419, -2.2801810182668114,
       0.8088980291471826, -0.1316830205960989, 0.0143585054401857,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.4823226655921295, 0.0574614517751295,
       -0.5663203488781653, 0.0309656800624243, -0.0044294485515179,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, -0.0174954311279016, 0.4325508330649349,
       0.3111668377093504, -0.8538512002386446, 0.1314757107290064,
       -0.0038467501367455, 0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, -0.1277481742492071, 0.2574468839590017,
       0.4155794781917712, -0.0115571196122084, -0.6170517361659126,
       0.0857115441015996, -0.0023808762250444, 0.0000000000000000},
      {0.0000000000000000, 0.0064191319587820, -0.0164033832904366,
       -0.0752421418813823, 0.6740179057989464, -0.0002498459192428,
       -0.6796875000000000, 0.0937500000000000, -0.0026041666666667}};
  const int i = threadIdx.z + blockIdx.z * blockDim.z + bi;
  if (i >= ngsl + nx)
    return;
  if (i >= ei)
    return;
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ngsl + ny)
    return;
  if (j >= ej)
    return;
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= 6)
    return;
#define _g3_c(k) g3_c[(k) + align]
#define _f_c(i, j)                                                             \
  f_c[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f(i, j)                                                               \
  f[(j) + align + ngsl + ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f_1(i, j)                                                             \
  f_1[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _g3(k) g3[(k) + align]
#define _f_2(i, j)                                                             \
  f_2[(j) + align + ngsl +                                                     \
      ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _lami(i, j, k)                                                         \
  lami[(k) + align +                                                           \
       (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +             \
       (2 * align + nz) * ((j) + ngsl + 2)]
#define _mui(i, j, k)                                                          \
  mui[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_1(i, j)                                                            \
  f1_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _u1(i, j, k)                                                           \
  u1[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _g_c(k) g_c[(k) + align]
#define _u2(i, j, k)                                                           \
  u2[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_2(i, j)                                                            \
  f2_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _u3(i, j, k)                                                           \
  u3[(k) + align + (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * ((j) + ngsl + 2)]
#define _dcrjz(k) dcrjz[(k) + align]
#define _dcrjx(i) dcrjx[(i) + ngsl + 2]
#define _dcrjy(j) dcrjy[(j) + ngsl + 2]
#define _s11(i, j, k)                                                          \
  s11[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s22(i, j, k)                                                          \
  s22[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _s33(i, j, k)                                                          \
  s33[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f1_2(i, j)                                                            \
  f1_2[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _f2_1(i, j)                                                            \
  f2_1[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s12(i, j, k)                                                          \
  s12[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _g(k) g[(k) + align]
#define _f1_c(i, j)                                                            \
  f1_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s13(i, j, k)                                                          \
  s13[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
#define _f2_c(i, j)                                                            \
  f2_c[(j) + align + ngsl +                                                    \
       ((i) + ngsl + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _s23(i, j, k)                                                          \
  s23[(k) + align +                                                            \
      (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +              \
      (2 * align + nz) * ((j) + ngsl + 2)]
  float Jii = _f_c(i, j) * _g3_c(nz - 1 - k);
  Jii = 1.0 * 1.0 / Jii;
  float J12i = _f(i, j) * _g3_c(nz - 1 - k);
  J12i = 1.0 * 1.0 / J12i;
  float J13i = _f_1(i, j) * _g3(nz - 1 - k);
  J13i = 1.0 * 1.0 / J13i;
  float J23i = _f_2(i, j) * _g3(nz - 1 - k);
  J23i = 1.0 * 1.0 / J23i;
  float lam = nu * 1.0 /
              (ph2r[k][7] * (ph2[1] * (p2[0] * _lami(i, j, nz - 8) +
                                       p2[1] * _lami(i + 1, j, nz - 8)) +
                             ph2[0] * (p2[0] * _lami(i, j - 1, nz - 8) +
                                       p2[1] * _lami(i + 1, j - 1, nz - 8))) +
               ph2r[k][6] * (ph2[1] * (p2[0] * _lami(i, j, nz - 7) +
                                       p2[1] * _lami(i + 1, j, nz - 7)) +
                             ph2[0] * (p2[0] * _lami(i, j - 1, nz - 7) +
                                       p2[1] * _lami(i + 1, j - 1, nz - 7))) +
               ph2r[k][5] * (ph2[1] * (p2[0] * _lami(i, j, nz - 6) +
                                       p2[1] * _lami(i + 1, j, nz - 6)) +
                             ph2[0] * (p2[0] * _lami(i, j - 1, nz - 6) +
                                       p2[1] * _lami(i + 1, j - 1, nz - 6))) +
               ph2r[k][4] * (ph2[1] * (p2[0] * _lami(i, j, nz - 5) +
                                       p2[1] * _lami(i + 1, j, nz - 5)) +
                             ph2[0] * (p2[0] * _lami(i, j - 1, nz - 5) +
                                       p2[1] * _lami(i + 1, j - 1, nz - 5))) +
               ph2r[k][3] * (ph2[1] * (p2[0] * _lami(i, j, nz - 4) +
                                       p2[1] * _lami(i + 1, j, nz - 4)) +
                             ph2[0] * (p2[0] * _lami(i, j - 1, nz - 4) +
                                       p2[1] * _lami(i + 1, j - 1, nz - 4))) +
               ph2r[k][2] * (ph2[1] * (p2[0] * _lami(i, j, nz - 3) +
                                       p2[1] * _lami(i + 1, j, nz - 3)) +
                             ph2[0] * (p2[0] * _lami(i, j - 1, nz - 3) +
                                       p2[1] * _lami(i + 1, j - 1, nz - 3))) +
               ph2r[k][1] * (ph2[1] * (p2[0] * _lami(i, j, nz - 2) +
                                       p2[1] * _lami(i + 1, j, nz - 2)) +
                             ph2[0] * (p2[0] * _lami(i, j - 1, nz - 2) +
                                       p2[1] * _lami(i + 1, j - 1, nz - 2))) +
               ph2r[k][0] * (ph2[1] * (p2[0] * _lami(i, j, nz - 1) +
                                       p2[1] * _lami(i + 1, j, nz - 1)) +
                             ph2[0] * (p2[0] * _lami(i, j - 1, nz - 1) +
                                       p2[1] * _lami(i + 1, j - 1, nz - 1))));
  float twomu = 2 * nu * 1.0 /
                (ph2r[k][7] * (ph2[1] * (p2[0] * _mui(i, j, nz - 8) +
                                         p2[1] * _mui(i + 1, j, nz - 8)) +
                               ph2[0] * (p2[0] * _mui(i, j - 1, nz - 8) +
                                         p2[1] * _mui(i + 1, j - 1, nz - 8))) +
                 ph2r[k][6] * (ph2[1] * (p2[0] * _mui(i, j, nz - 7) +
                                         p2[1] * _mui(i + 1, j, nz - 7)) +
                               ph2[0] * (p2[0] * _mui(i, j - 1, nz - 7) +
                                         p2[1] * _mui(i + 1, j - 1, nz - 7))) +
                 ph2r[k][5] * (ph2[1] * (p2[0] * _mui(i, j, nz - 6) +
                                         p2[1] * _mui(i + 1, j, nz - 6)) +
                               ph2[0] * (p2[0] * _mui(i, j - 1, nz - 6) +
                                         p2[1] * _mui(i + 1, j - 1, nz - 6))) +
                 ph2r[k][4] * (ph2[1] * (p2[0] * _mui(i, j, nz - 5) +
                                         p2[1] * _mui(i + 1, j, nz - 5)) +
                               ph2[0] * (p2[0] * _mui(i, j - 1, nz - 5) +
                                         p2[1] * _mui(i + 1, j - 1, nz - 5))) +
                 ph2r[k][3] * (ph2[1] * (p2[0] * _mui(i, j, nz - 4) +
                                         p2[1] * _mui(i + 1, j, nz - 4)) +
                               ph2[0] * (p2[0] * _mui(i, j - 1, nz - 4) +
                                         p2[1] * _mui(i + 1, j - 1, nz - 4))) +
                 ph2r[k][2] * (ph2[1] * (p2[0] * _mui(i, j, nz - 3) +
                                         p2[1] * _mui(i + 1, j, nz - 3)) +
                               ph2[0] * (p2[0] * _mui(i, j - 1, nz - 3) +
                                         p2[1] * _mui(i + 1, j - 1, nz - 3))) +
                 ph2r[k][1] * (ph2[1] * (p2[0] * _mui(i, j, nz - 2) +
                                         p2[1] * _mui(i + 1, j, nz - 2)) +
                               ph2[0] * (p2[0] * _mui(i, j - 1, nz - 2) +
                                         p2[1] * _mui(i + 1, j - 1, nz - 2))) +
                 ph2r[k][0] * (ph2[1] * (p2[0] * _mui(i, j, nz - 1) +
                                         p2[1] * _mui(i + 1, j, nz - 1)) +
                               ph2[0] * (p2[0] * _mui(i, j - 1, nz - 1) +
                                         p2[1] * _mui(i + 1, j - 1, nz - 1))));
  float mu12 =
      nu * 1.0 /
      (ph2r[k][7] * _mui(i, j, nz - 8) + ph2r[k][6] * _mui(i, j, nz - 7) +
       ph2r[k][5] * _mui(i, j, nz - 6) + ph2r[k][4] * _mui(i, j, nz - 5) +
       ph2r[k][3] * _mui(i, j, nz - 4) + ph2r[k][2] * _mui(i, j, nz - 3) +
       ph2r[k][1] * _mui(i, j, nz - 2) + ph2r[k][0] * _mui(i, j, nz - 1));
  float mu13 =
      nu * 1.0 /
      (ph2[1] * _mui(i, j, nz - 1 - k) + ph2[0] * _mui(i, j - 1, nz - 1 - k));
  float mu23 =
      nu * 1.0 /
      (p2[0] * _mui(i, j, nz - 1 - k) + p2[1] * _mui(i + 1, j, nz - 1 - k));
  float div =
      d4[1] * _u1(i, j, nz - 1 - k) + d4[0] * _u1(i - 1, j, nz - 1 - k) +
      d4[2] * _u1(i + 1, j, nz - 1 - k) + d4[3] * _u1(i + 2, j, nz - 1 - k) +
      dh4[2] * _u2(i, j, nz - 1 - k) + dh4[0] * _u2(i, j - 2, nz - 1 - k) +
      dh4[1] * _u2(i, j - 1, nz - 1 - k) + dh4[3] * _u2(i, j + 1, nz - 1 - k) +
      Jii * (dh4r[k][7] * _u3(i, j, nz - 8) + dh4r[k][6] * _u3(i, j, nz - 7) +
             dh4r[k][5] * _u3(i, j, nz - 6) + dh4r[k][4] * _u3(i, j, nz - 5) +
             dh4r[k][3] * _u3(i, j, nz - 4) + dh4r[k][2] * _u3(i, j, nz - 3) +
             dh4r[k][1] * _u3(i, j, nz - 2) + dh4r[k][0] * _u3(i, j, nz - 1)) -
      Jii * _g_c(nz - 1 - k) *
          (p4[1] * _f1_1(i, j) *
               (phd4r[k][8] * _u1(i, j, nz - 9) +
                phd4r[k][7] * _u1(i, j, nz - 8) +
                phd4r[k][6] * _u1(i, j, nz - 7) +
                phd4r[k][5] * _u1(i, j, nz - 6) +
                phd4r[k][4] * _u1(i, j, nz - 5) +
                phd4r[k][3] * _u1(i, j, nz - 4) +
                phd4r[k][2] * _u1(i, j, nz - 3) +
                phd4r[k][1] * _u1(i, j, nz - 2) +
                phd4r[k][0] * _u1(i, j, nz - 1)) +
           p4[0] * _f1_1(i - 1, j) *
               (phd4r[k][8] * _u1(i - 1, j, nz - 9) +
                phd4r[k][7] * _u1(i - 1, j, nz - 8) +
                phd4r[k][6] * _u1(i - 1, j, nz - 7) +
                phd4r[k][5] * _u1(i - 1, j, nz - 6) +
                phd4r[k][4] * _u1(i - 1, j, nz - 5) +
                phd4r[k][3] * _u1(i - 1, j, nz - 4) +
                phd4r[k][2] * _u1(i - 1, j, nz - 3) +
                phd4r[k][1] * _u1(i - 1, j, nz - 2) +
                phd4r[k][0] * _u1(i - 1, j, nz - 1)) +
           p4[2] * _f1_1(i + 1, j) *
               (phd4r[k][8] * _u1(i + 1, j, nz - 9) +
                phd4r[k][7] * _u1(i + 1, j, nz - 8) +
                phd4r[k][6] * _u1(i + 1, j, nz - 7) +
                phd4r[k][5] * _u1(i + 1, j, nz - 6) +
                phd4r[k][4] * _u1(i + 1, j, nz - 5) +
                phd4r[k][3] * _u1(i + 1, j, nz - 4) +
                phd4r[k][2] * _u1(i + 1, j, nz - 3) +
                phd4r[k][1] * _u1(i + 1, j, nz - 2) +
                phd4r[k][0] * _u1(i + 1, j, nz - 1)) +
           p4[3] * _f1_1(i + 2, j) *
               (phd4r[k][8] * _u1(i + 2, j, nz - 9) +
                phd4r[k][7] * _u1(i + 2, j, nz - 8) +
                phd4r[k][6] * _u1(i + 2, j, nz - 7) +
                phd4r[k][5] * _u1(i + 2, j, nz - 6) +
                phd4r[k][4] * _u1(i + 2, j, nz - 5) +
                phd4r[k][3] * _u1(i + 2, j, nz - 4) +
                phd4r[k][2] * _u1(i + 2, j, nz - 3) +
                phd4r[k][1] * _u1(i + 2, j, nz - 2) +
                phd4r[k][0] * _u1(i + 2, j, nz - 1))) -
      Jii * _g_c(nz - 1 - k) *
          (ph4[2] * _f2_2(i, j) *
               (phd4r[k][8] * _u2(i, j, nz - 9) +
                phd4r[k][7] * _u2(i, j, nz - 8) +
                phd4r[k][6] * _u2(i, j, nz - 7) +
                phd4r[k][5] * _u2(i, j, nz - 6) +
                phd4r[k][4] * _u2(i, j, nz - 5) +
                phd4r[k][3] * _u2(i, j, nz - 4) +
                phd4r[k][2] * _u2(i, j, nz - 3) +
                phd4r[k][1] * _u2(i, j, nz - 2) +
                phd4r[k][0] * _u2(i, j, nz - 1)) +
           ph4[0] * _f2_2(i, j - 2) *
               (phd4r[k][8] * _u2(i, j - 2, nz - 9) +
                phd4r[k][7] * _u2(i, j - 2, nz - 8) +
                phd4r[k][6] * _u2(i, j - 2, nz - 7) +
                phd4r[k][5] * _u2(i, j - 2, nz - 6) +
                phd4r[k][4] * _u2(i, j - 2, nz - 5) +
                phd4r[k][3] * _u2(i, j - 2, nz - 4) +
                phd4r[k][2] * _u2(i, j - 2, nz - 3) +
                phd4r[k][1] * _u2(i, j - 2, nz - 2) +
                phd4r[k][0] * _u2(i, j - 2, nz - 1)) +
           ph4[1] * _f2_2(i, j - 1) *
               (phd4r[k][8] * _u2(i, j - 1, nz - 9) +
                phd4r[k][7] * _u2(i, j - 1, nz - 8) +
                phd4r[k][6] * _u2(i, j - 1, nz - 7) +
                phd4r[k][5] * _u2(i, j - 1, nz - 6) +
                phd4r[k][4] * _u2(i, j - 1, nz - 5) +
                phd4r[k][3] * _u2(i, j - 1, nz - 4) +
                phd4r[k][2] * _u2(i, j - 1, nz - 3) +
                phd4r[k][1] * _u2(i, j - 1, nz - 2) +
                phd4r[k][0] * _u2(i, j - 1, nz - 1)) +
           ph4[3] * _f2_2(i, j + 1) *
               (phd4r[k][8] * _u2(i, j + 1, nz - 9) +
                phd4r[k][7] * _u2(i, j + 1, nz - 8) +
                phd4r[k][6] * _u2(i, j + 1, nz - 7) +
                phd4r[k][5] * _u2(i, j + 1, nz - 6) +
                phd4r[k][4] * _u2(i, j + 1, nz - 5) +
                phd4r[k][3] * _u2(i, j + 1, nz - 4) +
                phd4r[k][2] * _u2(i, j + 1, nz - 3) +
                phd4r[k][1] * _u2(i, j + 1, nz - 2) +
                phd4r[k][0] * _u2(i, j + 1, nz - 1)));
  float f_dcrj = _dcrjx(i) * _dcrjy(j) * _dcrjz(nz - 1 - k);
  _s11(i, j, nz - 1 - k) = (a * _s11(i, j, nz - 1 - k) + lam * div +
                            twomu * (d4[1] * _u1(i, j, nz - 1 - k) +
                                     d4[0] * _u1(i - 1, j, nz - 1 - k) +
                                     d4[2] * _u1(i + 1, j, nz - 1 - k) +
                                     d4[3] * _u1(i + 2, j, nz - 1 - k)) -
                            twomu * Jii * _g_c(nz - 1 - k) *
                                (p4[1] * _f1_1(i, j) *
                                     (phd4r[k][8] * _u1(i, j, nz - 9) +
                                      phd4r[k][7] * _u1(i, j, nz - 8) +
                                      phd4r[k][6] * _u1(i, j, nz - 7) +
                                      phd4r[k][5] * _u1(i, j, nz - 6) +
                                      phd4r[k][4] * _u1(i, j, nz - 5) +
                                      phd4r[k][3] * _u1(i, j, nz - 4) +
                                      phd4r[k][2] * _u1(i, j, nz - 3) +
                                      phd4r[k][1] * _u1(i, j, nz - 2) +
                                      phd4r[k][0] * _u1(i, j, nz - 1)) +
                                 p4[0] * _f1_1(i - 1, j) *
                                     (phd4r[k][8] * _u1(i - 1, j, nz - 9) +
                                      phd4r[k][7] * _u1(i - 1, j, nz - 8) +
                                      phd4r[k][6] * _u1(i - 1, j, nz - 7) +
                                      phd4r[k][5] * _u1(i - 1, j, nz - 6) +
                                      phd4r[k][4] * _u1(i - 1, j, nz - 5) +
                                      phd4r[k][3] * _u1(i - 1, j, nz - 4) +
                                      phd4r[k][2] * _u1(i - 1, j, nz - 3) +
                                      phd4r[k][1] * _u1(i - 1, j, nz - 2) +
                                      phd4r[k][0] * _u1(i - 1, j, nz - 1)) +
                                 p4[2] * _f1_1(i + 1, j) *
                                     (phd4r[k][8] * _u1(i + 1, j, nz - 9) +
                                      phd4r[k][7] * _u1(i + 1, j, nz - 8) +
                                      phd4r[k][6] * _u1(i + 1, j, nz - 7) +
                                      phd4r[k][5] * _u1(i + 1, j, nz - 6) +
                                      phd4r[k][4] * _u1(i + 1, j, nz - 5) +
                                      phd4r[k][3] * _u1(i + 1, j, nz - 4) +
                                      phd4r[k][2] * _u1(i + 1, j, nz - 3) +
                                      phd4r[k][1] * _u1(i + 1, j, nz - 2) +
                                      phd4r[k][0] * _u1(i + 1, j, nz - 1)) +
                                 p4[3] * _f1_1(i + 2, j) *
                                     (phd4r[k][8] * _u1(i + 2, j, nz - 9) +
                                      phd4r[k][7] * _u1(i + 2, j, nz - 8) +
                                      phd4r[k][6] * _u1(i + 2, j, nz - 7) +
                                      phd4r[k][5] * _u1(i + 2, j, nz - 6) +
                                      phd4r[k][4] * _u1(i + 2, j, nz - 5) +
                                      phd4r[k][3] * _u1(i + 2, j, nz - 4) +
                                      phd4r[k][2] * _u1(i + 2, j, nz - 3) +
                                      phd4r[k][1] * _u1(i + 2, j, nz - 2) +
                                      phd4r[k][0] * _u1(i + 2, j, nz - 1)))) *
                           f_dcrj;
  _s22(i, j, nz - 1 - k) = (a * _s22(i, j, nz - 1 - k) + lam * div +
                            twomu * (dh4[2] * _u2(i, j, nz - 1 - k) +
                                     dh4[0] * _u2(i, j - 2, nz - 1 - k) +
                                     dh4[1] * _u2(i, j - 1, nz - 1 - k) +
                                     dh4[3] * _u2(i, j + 1, nz - 1 - k)) -
                            twomu * Jii * _g_c(nz - 1 - k) *
                                (ph4[2] * _f2_2(i, j) *
                                     (phd4r[k][8] * _u2(i, j, nz - 9) +
                                      phd4r[k][7] * _u2(i, j, nz - 8) +
                                      phd4r[k][6] * _u2(i, j, nz - 7) +
                                      phd4r[k][5] * _u2(i, j, nz - 6) +
                                      phd4r[k][4] * _u2(i, j, nz - 5) +
                                      phd4r[k][3] * _u2(i, j, nz - 4) +
                                      phd4r[k][2] * _u2(i, j, nz - 3) +
                                      phd4r[k][1] * _u2(i, j, nz - 2) +
                                      phd4r[k][0] * _u2(i, j, nz - 1)) +
                                 ph4[0] * _f2_2(i, j - 2) *
                                     (phd4r[k][8] * _u2(i, j - 2, nz - 9) +
                                      phd4r[k][7] * _u2(i, j - 2, nz - 8) +
                                      phd4r[k][6] * _u2(i, j - 2, nz - 7) +
                                      phd4r[k][5] * _u2(i, j - 2, nz - 6) +
                                      phd4r[k][4] * _u2(i, j - 2, nz - 5) +
                                      phd4r[k][3] * _u2(i, j - 2, nz - 4) +
                                      phd4r[k][2] * _u2(i, j - 2, nz - 3) +
                                      phd4r[k][1] * _u2(i, j - 2, nz - 2) +
                                      phd4r[k][0] * _u2(i, j - 2, nz - 1)) +
                                 ph4[1] * _f2_2(i, j - 1) *
                                     (phd4r[k][8] * _u2(i, j - 1, nz - 9) +
                                      phd4r[k][7] * _u2(i, j - 1, nz - 8) +
                                      phd4r[k][6] * _u2(i, j - 1, nz - 7) +
                                      phd4r[k][5] * _u2(i, j - 1, nz - 6) +
                                      phd4r[k][4] * _u2(i, j - 1, nz - 5) +
                                      phd4r[k][3] * _u2(i, j - 1, nz - 4) +
                                      phd4r[k][2] * _u2(i, j - 1, nz - 3) +
                                      phd4r[k][1] * _u2(i, j - 1, nz - 2) +
                                      phd4r[k][0] * _u2(i, j - 1, nz - 1)) +
                                 ph4[3] * _f2_2(i, j + 1) *
                                     (phd4r[k][8] * _u2(i, j + 1, nz - 9) +
                                      phd4r[k][7] * _u2(i, j + 1, nz - 8) +
                                      phd4r[k][6] * _u2(i, j + 1, nz - 7) +
                                      phd4r[k][5] * _u2(i, j + 1, nz - 6) +
                                      phd4r[k][4] * _u2(i, j + 1, nz - 5) +
                                      phd4r[k][3] * _u2(i, j + 1, nz - 4) +
                                      phd4r[k][2] * _u2(i, j + 1, nz - 3) +
                                      phd4r[k][1] * _u2(i, j + 1, nz - 2) +
                                      phd4r[k][0] * _u2(i, j + 1, nz - 1)))) *
                           f_dcrj;
  _s33(i, j, nz - 1 - k) =
      (a * _s33(i, j, nz - 1 - k) + lam * div +
       twomu * Jii *
           (dh4r[k][7] * _u3(i, j, nz - 8) + dh4r[k][6] * _u3(i, j, nz - 7) +
            dh4r[k][5] * _u3(i, j, nz - 6) + dh4r[k][4] * _u3(i, j, nz - 5) +
            dh4r[k][3] * _u3(i, j, nz - 4) + dh4r[k][2] * _u3(i, j, nz - 3) +
            dh4r[k][1] * _u3(i, j, nz - 2) + dh4r[k][0] * _u3(i, j, nz - 1))) *
      f_dcrj;
  _s12(i, j, nz - 1 - k) =
      (a * _s12(i, j, nz - 1 - k) +
       mu12 *
           (d4[1] * _u1(i, j, nz - 1 - k) + d4[0] * _u1(i, j - 1, nz - 1 - k) +
            d4[2] * _u1(i, j + 1, nz - 1 - k) +
            d4[3] * _u1(i, j + 2, nz - 1 - k) + dh4[2] * _u2(i, j, nz - 1 - k) +
            dh4[0] * _u2(i - 2, j, nz - 1 - k) +
            dh4[1] * _u2(i - 1, j, nz - 1 - k) +
            dh4[3] * _u2(i + 1, j, nz - 1 - k) -
            J12i * _g_c(nz - 1 - k) *
                (p4[1] * _f2_1(i, j) *
                     (phd4r[k][8] * _u1(i, j, nz - 9) +
                      phd4r[k][7] * _u1(i, j, nz - 8) +
                      phd4r[k][6] * _u1(i, j, nz - 7) +
                      phd4r[k][5] * _u1(i, j, nz - 6) +
                      phd4r[k][4] * _u1(i, j, nz - 5) +
                      phd4r[k][3] * _u1(i, j, nz - 4) +
                      phd4r[k][2] * _u1(i, j, nz - 3) +
                      phd4r[k][1] * _u1(i, j, nz - 2) +
                      phd4r[k][0] * _u1(i, j, nz - 1)) +
                 p4[0] * _f2_1(i, j - 1) *
                     (phd4r[k][8] * _u1(i, j - 1, nz - 9) +
                      phd4r[k][7] * _u1(i, j - 1, nz - 8) +
                      phd4r[k][6] * _u1(i, j - 1, nz - 7) +
                      phd4r[k][5] * _u1(i, j - 1, nz - 6) +
                      phd4r[k][4] * _u1(i, j - 1, nz - 5) +
                      phd4r[k][3] * _u1(i, j - 1, nz - 4) +
                      phd4r[k][2] * _u1(i, j - 1, nz - 3) +
                      phd4r[k][1] * _u1(i, j - 1, nz - 2) +
                      phd4r[k][0] * _u1(i, j - 1, nz - 1)) +
                 p4[2] * _f2_1(i, j + 1) *
                     (phd4r[k][8] * _u1(i, j + 1, nz - 9) +
                      phd4r[k][7] * _u1(i, j + 1, nz - 8) +
                      phd4r[k][6] * _u1(i, j + 1, nz - 7) +
                      phd4r[k][5] * _u1(i, j + 1, nz - 6) +
                      phd4r[k][4] * _u1(i, j + 1, nz - 5) +
                      phd4r[k][3] * _u1(i, j + 1, nz - 4) +
                      phd4r[k][2] * _u1(i, j + 1, nz - 3) +
                      phd4r[k][1] * _u1(i, j + 1, nz - 2) +
                      phd4r[k][0] * _u1(i, j + 1, nz - 1)) +
                 p4[3] * _f2_1(i, j + 2) *
                     (phd4r[k][8] * _u1(i, j + 2, nz - 9) +
                      phd4r[k][7] * _u1(i, j + 2, nz - 8) +
                      phd4r[k][6] * _u1(i, j + 2, nz - 7) +
                      phd4r[k][5] * _u1(i, j + 2, nz - 6) +
                      phd4r[k][4] * _u1(i, j + 2, nz - 5) +
                      phd4r[k][3] * _u1(i, j + 2, nz - 4) +
                      phd4r[k][2] * _u1(i, j + 2, nz - 3) +
                      phd4r[k][1] * _u1(i, j + 2, nz - 2) +
                      phd4r[k][0] * _u1(i, j + 2, nz - 1))) -
            J12i * _g_c(nz - 1 - k) *
                (ph4[2] * _f1_2(i, j) *
                     (phd4r[k][8] * _u2(i, j, nz - 9) +
                      phd4r[k][7] * _u2(i, j, nz - 8) +
                      phd4r[k][6] * _u2(i, j, nz - 7) +
                      phd4r[k][5] * _u2(i, j, nz - 6) +
                      phd4r[k][4] * _u2(i, j, nz - 5) +
                      phd4r[k][3] * _u2(i, j, nz - 4) +
                      phd4r[k][2] * _u2(i, j, nz - 3) +
                      phd4r[k][1] * _u2(i, j, nz - 2) +
                      phd4r[k][0] * _u2(i, j, nz - 1)) +
                 ph4[0] * _f1_2(i - 2, j) *
                     (phd4r[k][8] * _u2(i - 2, j, nz - 9) +
                      phd4r[k][7] * _u2(i - 2, j, nz - 8) +
                      phd4r[k][6] * _u2(i - 2, j, nz - 7) +
                      phd4r[k][5] * _u2(i - 2, j, nz - 6) +
                      phd4r[k][4] * _u2(i - 2, j, nz - 5) +
                      phd4r[k][3] * _u2(i - 2, j, nz - 4) +
                      phd4r[k][2] * _u2(i - 2, j, nz - 3) +
                      phd4r[k][1] * _u2(i - 2, j, nz - 2) +
                      phd4r[k][0] * _u2(i - 2, j, nz - 1)) +
                 ph4[1] * _f1_2(i - 1, j) *
                     (phd4r[k][8] * _u2(i - 1, j, nz - 9) +
                      phd4r[k][7] * _u2(i - 1, j, nz - 8) +
                      phd4r[k][6] * _u2(i - 1, j, nz - 7) +
                      phd4r[k][5] * _u2(i - 1, j, nz - 6) +
                      phd4r[k][4] * _u2(i - 1, j, nz - 5) +
                      phd4r[k][3] * _u2(i - 1, j, nz - 4) +
                      phd4r[k][2] * _u2(i - 1, j, nz - 3) +
                      phd4r[k][1] * _u2(i - 1, j, nz - 2) +
                      phd4r[k][0] * _u2(i - 1, j, nz - 1)) +
                 ph4[3] * _f1_2(i + 1, j) *
                     (phd4r[k][8] * _u2(i + 1, j, nz - 9) +
                      phd4r[k][7] * _u2(i + 1, j, nz - 8) +
                      phd4r[k][6] * _u2(i + 1, j, nz - 7) +
                      phd4r[k][5] * _u2(i + 1, j, nz - 6) +
                      phd4r[k][4] * _u2(i + 1, j, nz - 5) +
                      phd4r[k][3] * _u2(i + 1, j, nz - 4) +
                      phd4r[k][2] * _u2(i + 1, j, nz - 3) +
                      phd4r[k][1] * _u2(i + 1, j, nz - 2) +
                      phd4r[k][0] * _u2(i + 1, j, nz - 1))))) *
      f_dcrj;
  _s13(i, j, nz - 1 - k) =
      (a * _s13(i, j, nz - 1 - k) +
       mu13 *
           (dh4[2] * _u3(i, j, nz - 1 - k) +
            dh4[0] * _u3(i - 2, j, nz - 1 - k) +
            dh4[1] * _u3(i - 1, j, nz - 1 - k) +
            dh4[3] * _u3(i + 1, j, nz - 1 - k) +
            J13i *
                (d4r[k][6] * _u1(i, j, nz - 7) + d4r[k][5] * _u1(i, j, nz - 6) +
                 d4r[k][4] * _u1(i, j, nz - 5) + d4r[k][3] * _u1(i, j, nz - 4) +
                 d4r[k][2] * _u1(i, j, nz - 3) + d4r[k][1] * _u1(i, j, nz - 2) +
                 d4r[k][0] * _u1(i, j, nz - 1)) -
            J13i * _g(nz - 1 - k) *
                (ph4[2] * _f1_c(i, j) *
                     (pdh4r[k][8] * _u3(i, j, nz - 9) +
                      pdh4r[k][7] * _u3(i, j, nz - 8) +
                      pdh4r[k][6] * _u3(i, j, nz - 7) +
                      pdh4r[k][5] * _u3(i, j, nz - 6) +
                      pdh4r[k][4] * _u3(i, j, nz - 5) +
                      pdh4r[k][3] * _u3(i, j, nz - 4) +
                      pdh4r[k][2] * _u3(i, j, nz - 3) +
                      pdh4r[k][1] * _u3(i, j, nz - 2) +
                      pdh4r[k][0] * _u3(i, j, nz - 1)) +
                 ph4[0] * _f1_c(i - 2, j) *
                     (pdh4r[k][8] * _u3(i - 2, j, nz - 9) +
                      pdh4r[k][7] * _u3(i - 2, j, nz - 8) +
                      pdh4r[k][6] * _u3(i - 2, j, nz - 7) +
                      pdh4r[k][5] * _u3(i - 2, j, nz - 6) +
                      pdh4r[k][4] * _u3(i - 2, j, nz - 5) +
                      pdh4r[k][3] * _u3(i - 2, j, nz - 4) +
                      pdh4r[k][2] * _u3(i - 2, j, nz - 3) +
                      pdh4r[k][1] * _u3(i - 2, j, nz - 2) +
                      pdh4r[k][0] * _u3(i - 2, j, nz - 1)) +
                 ph4[1] * _f1_c(i - 1, j) *
                     (pdh4r[k][8] * _u3(i - 1, j, nz - 9) +
                      pdh4r[k][7] * _u3(i - 1, j, nz - 8) +
                      pdh4r[k][6] * _u3(i - 1, j, nz - 7) +
                      pdh4r[k][5] * _u3(i - 1, j, nz - 6) +
                      pdh4r[k][4] * _u3(i - 1, j, nz - 5) +
                      pdh4r[k][3] * _u3(i - 1, j, nz - 4) +
                      pdh4r[k][2] * _u3(i - 1, j, nz - 3) +
                      pdh4r[k][1] * _u3(i - 1, j, nz - 2) +
                      pdh4r[k][0] * _u3(i - 1, j, nz - 1)) +
                 ph4[3] * _f1_c(i + 1, j) *
                     (pdh4r[k][8] * _u3(i + 1, j, nz - 9) +
                      pdh4r[k][7] * _u3(i + 1, j, nz - 8) +
                      pdh4r[k][6] * _u3(i + 1, j, nz - 7) +
                      pdh4r[k][5] * _u3(i + 1, j, nz - 6) +
                      pdh4r[k][4] * _u3(i + 1, j, nz - 5) +
                      pdh4r[k][3] * _u3(i + 1, j, nz - 4) +
                      pdh4r[k][2] * _u3(i + 1, j, nz - 3) +
                      pdh4r[k][1] * _u3(i + 1, j, nz - 2) +
                      pdh4r[k][0] * _u3(i + 1, j, nz - 1))))) *
      f_dcrj;
  _s23(i, j, nz - 1 - k) =
      (a * _s23(i, j, nz - 1 - k) +
       mu23 *
           (d4[1] * _u3(i, j, nz - 1 - k) + d4[0] * _u3(i, j - 1, nz - 1 - k) +
            d4[2] * _u3(i, j + 1, nz - 1 - k) +
            d4[3] * _u3(i, j + 2, nz - 1 - k) +
            J23i *
                (d4r[k][6] * _u2(i, j, nz - 7) + d4r[k][5] * _u2(i, j, nz - 6) +
                 d4r[k][4] * _u2(i, j, nz - 5) + d4r[k][3] * _u2(i, j, nz - 4) +
                 d4r[k][2] * _u2(i, j, nz - 3) + d4r[k][1] * _u2(i, j, nz - 2) +
                 d4r[k][0] * _u2(i, j, nz - 1)) -
            J23i * _g(nz - 1 - k) *
                (p4[1] * _f2_c(i, j) *
                     (pdh4r[k][8] * _u3(i, j, nz - 9) +
                      pdh4r[k][7] * _u3(i, j, nz - 8) +
                      pdh4r[k][6] * _u3(i, j, nz - 7) +
                      pdh4r[k][5] * _u3(i, j, nz - 6) +
                      pdh4r[k][4] * _u3(i, j, nz - 5) +
                      pdh4r[k][3] * _u3(i, j, nz - 4) +
                      pdh4r[k][2] * _u3(i, j, nz - 3) +
                      pdh4r[k][1] * _u3(i, j, nz - 2) +
                      pdh4r[k][0] * _u3(i, j, nz - 1)) +
                 p4[0] * _f2_c(i, j - 1) *
                     (pdh4r[k][8] * _u3(i, j - 1, nz - 9) +
                      pdh4r[k][7] * _u3(i, j - 1, nz - 8) +
                      pdh4r[k][6] * _u3(i, j - 1, nz - 7) +
                      pdh4r[k][5] * _u3(i, j - 1, nz - 6) +
                      pdh4r[k][4] * _u3(i, j - 1, nz - 5) +
                      pdh4r[k][3] * _u3(i, j - 1, nz - 4) +
                      pdh4r[k][2] * _u3(i, j - 1, nz - 3) +
                      pdh4r[k][1] * _u3(i, j - 1, nz - 2) +
                      pdh4r[k][0] * _u3(i, j - 1, nz - 1)) +
                 p4[2] * _f2_c(i, j + 1) *
                     (pdh4r[k][8] * _u3(i, j + 1, nz - 9) +
                      pdh4r[k][7] * _u3(i, j + 1, nz - 8) +
                      pdh4r[k][6] * _u3(i, j + 1, nz - 7) +
                      pdh4r[k][5] * _u3(i, j + 1, nz - 6) +
                      pdh4r[k][4] * _u3(i, j + 1, nz - 5) +
                      pdh4r[k][3] * _u3(i, j + 1, nz - 4) +
                      pdh4r[k][2] * _u3(i, j + 1, nz - 3) +
                      pdh4r[k][1] * _u3(i, j + 1, nz - 2) +
                      pdh4r[k][0] * _u3(i, j + 1, nz - 1)) +
                 p4[3] * _f2_c(i, j + 2) *
                     (pdh4r[k][8] * _u3(i, j + 2, nz - 9) +
                      pdh4r[k][7] * _u3(i, j + 2, nz - 8) +
                      pdh4r[k][6] * _u3(i, j + 2, nz - 7) +
                      pdh4r[k][5] * _u3(i, j + 2, nz - 6) +
                      pdh4r[k][4] * _u3(i, j + 2, nz - 5) +
                      pdh4r[k][3] * _u3(i, j + 2, nz - 4) +
                      pdh4r[k][2] * _u3(i, j + 2, nz - 3) +
                      pdh4r[k][1] * _u3(i, j + 2, nz - 2) +
                      pdh4r[k][0] * _u3(i, j + 2, nz - 1))))) *
      f_dcrj;
#undef _g3_c
#undef _f_c
#undef _f
#undef _f_1
#undef _g3
#undef _f_2
#undef _lami
#undef _mui
#undef _f1_1
#undef _u1
#undef _g_c
#undef _u2
#undef _f2_2
#undef _u3
#undef _dcrjz
#undef _dcrjx
#undef _dcrjy
#undef _s11
#undef _s22
#undef _s33
#undef _f1_2
#undef _f2_1
#undef _s12
#undef _g
#undef _f1_c
#undef _s13
#undef _f2_c
#undef _s23
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
#define _rho(i, j, k) rho[(i)*ny * nz + (j)*nz + (k)]
#define _lami(i, j, k) lami[(i)*ny * nz + (j)*nz + (k)]
#define _mui(i, j, k) mui[(i)*ny * nz + (j)*nz + (k)]
  _rho(i, j, k) = 1.0;
  _lami(i, j, k) = 1.0;
  _mui(i, j, k) = 1.0;
#undef _rho
#undef _lami
#undef _mui
}
