#include <topography/kernels/optimized_launch_config.cuh>
#include <topography/kernels/optimized_velocity.cuh>

__launch_bounds__(DTOPO_VEL_110_MAX_THREADS_PER_BLOCK,
                  DTOPO_VEL_110_MIN_BLOCKS_PER_SM)

    __global__ void dtopo_vel_110(
        float *__restrict__ u1, float *__restrict__ u2, float *__restrict__ u3,
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
  const float phz4l[6][7] = {
      {0.8338228784688313, 0.1775123316429260, 0.1435067013076542,
       -0.1548419114194114, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.1813404047323969, 1.1246711188154426, -0.2933634518280757,
       -0.0126480717197637, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {-0.1331142706282399, 0.7930714675884345, 0.3131998767078508,
       0.0268429263319546, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.0969078556633046, -0.1539344946680898, 0.4486491202844389,
       0.6768738207821733, -0.0684963020618270, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, -0.0625000000000000,
       0.5625000000000000, 0.5625000000000000, -0.0625000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       -0.0625000000000000, 0.5625000000000000, 0.5625000000000000,
       -0.0625000000000000}};
  const float phy4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float phx4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float dhpz4l[6][9] = {
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
  const float dhy4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const float dhx4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const float dhz4l[6][7] = {
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
  const float px4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float py4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float dx4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float dy4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float dphz4l[6][9] = {
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
  const float dz4l[6][8] = {
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
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ny)
    return;
  if (j >= ej)
    return;
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= 6)
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
  for (int i = bi; i < ei; ++i) {
    float rho1 =
        phz4l[k][0] *
            (phy4[2] * _rho(i, j, 0) + phy4[0] * _rho(i, j - 2, 0) +
             phy4[1] * _rho(i, j - 1, 0) + phy4[3] * _rho(i, j + 1, 0)) +
        phz4l[k][1] *
            (phy4[2] * _rho(i, j, 1) + phy4[0] * _rho(i, j - 2, 1) +
             phy4[1] * _rho(i, j - 1, 1) + phy4[3] * _rho(i, j + 1, 1)) +
        phz4l[k][2] *
            (phy4[2] * _rho(i, j, 2) + phy4[0] * _rho(i, j - 2, 2) +
             phy4[1] * _rho(i, j - 1, 2) + phy4[3] * _rho(i, j + 1, 2)) +
        phz4l[k][3] *
            (phy4[2] * _rho(i, j, 3) + phy4[0] * _rho(i, j - 2, 3) +
             phy4[1] * _rho(i, j - 1, 3) + phy4[3] * _rho(i, j + 1, 3)) +
        phz4l[k][4] *
            (phy4[2] * _rho(i, j, 4) + phy4[0] * _rho(i, j - 2, 4) +
             phy4[1] * _rho(i, j - 1, 4) + phy4[3] * _rho(i, j + 1, 4)) +
        phz4l[k][5] *
            (phy4[2] * _rho(i, j, 5) + phy4[0] * _rho(i, j - 2, 5) +
             phy4[1] * _rho(i, j - 1, 5) + phy4[3] * _rho(i, j + 1, 5)) +
        phz4l[k][6] *
            (phy4[2] * _rho(i, j, 6) + phy4[0] * _rho(i, j - 2, 6) +
             phy4[1] * _rho(i, j - 1, 6) + phy4[3] * _rho(i, j + 1, 6));
    float rho2 =
        phz4l[k][0] *
            (phx4[2] * _rho(i, j, 0) + phx4[0] * _rho(i - 2, j, 0) +
             phx4[1] * _rho(i - 1, j, 0) + phx4[3] * _rho(i + 1, j, 0)) +
        phz4l[k][1] *
            (phx4[2] * _rho(i, j, 1) + phx4[0] * _rho(i - 2, j, 1) +
             phx4[1] * _rho(i - 1, j, 1) + phx4[3] * _rho(i + 1, j, 1)) +
        phz4l[k][2] *
            (phx4[2] * _rho(i, j, 2) + phx4[0] * _rho(i - 2, j, 2) +
             phx4[1] * _rho(i - 1, j, 2) + phx4[3] * _rho(i + 1, j, 2)) +
        phz4l[k][3] *
            (phx4[2] * _rho(i, j, 3) + phx4[0] * _rho(i - 2, j, 3) +
             phx4[1] * _rho(i - 1, j, 3) + phx4[3] * _rho(i + 1, j, 3)) +
        phz4l[k][4] *
            (phx4[2] * _rho(i, j, 4) + phx4[0] * _rho(i - 2, j, 4) +
             phx4[1] * _rho(i - 1, j, 4) + phx4[3] * _rho(i + 1, j, 4)) +
        phz4l[k][5] *
            (phx4[2] * _rho(i, j, 5) + phx4[0] * _rho(i - 2, j, 5) +
             phx4[1] * _rho(i - 1, j, 5) + phx4[3] * _rho(i + 1, j, 5)) +
        phz4l[k][6] *
            (phx4[2] * _rho(i, j, 6) + phx4[0] * _rho(i - 2, j, 6) +
             phx4[1] * _rho(i - 1, j, 6) + phx4[3] * _rho(i + 1, j, 6));
    float rho3 =
        phy4[2] * (phx4[2] * _rho(i, j, k) + phx4[0] * _rho(i - 2, j, k) +
                   phx4[1] * _rho(i - 1, j, k) + phx4[3] * _rho(i + 1, j, k)) +
        phy4[0] *
            (phx4[2] * _rho(i, j - 2, k) + phx4[0] * _rho(i - 2, j - 2, k) +
             phx4[1] * _rho(i - 1, j - 2, k) +
             phx4[3] * _rho(i + 1, j - 2, k)) +
        phy4[1] *
            (phx4[2] * _rho(i, j - 1, k) + phx4[0] * _rho(i - 2, j - 1, k) +
             phx4[1] * _rho(i - 1, j - 1, k) +
             phx4[3] * _rho(i + 1, j - 1, k)) +
        phy4[3] *
            (phx4[2] * _rho(i, j + 1, k) + phx4[0] * _rho(i - 2, j + 1, k) +
             phx4[1] * _rho(i - 1, j + 1, k) + phx4[3] * _rho(i + 1, j + 1, k));
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
             (dhx4[2] * _f_c(i, j) * _g3_c(k) * _s11(i, j, k) +
              dhx4[0] * _f_c(i - 2, j) * _g3_c(k) * _s11(i - 2, j, k) +
              dhx4[1] * _f_c(i - 1, j) * _g3_c(k) * _s11(i - 1, j, k) +
              dhx4[3] * _f_c(i + 1, j) * _g3_c(k) * _s11(i + 1, j, k) +
              dhy4[2] * _f(i, j) * _g3_c(k) * _s12(i, j, k) +
              dhy4[0] * _f(i, j - 2) * _g3_c(k) * _s12(i, j - 2, k) +
              dhy4[1] * _f(i, j - 1) * _g3_c(k) * _s12(i, j - 1, k) +
              dhy4[3] * _f(i, j + 1) * _g3_c(k) * _s12(i, j + 1, k) +
              dhz4l[k][0] * _s13(i, j, 0) + dhz4l[k][1] * _s13(i, j, 1) +
              dhz4l[k][2] * _s13(i, j, 2) + dhz4l[k][3] * _s13(i, j, 3) +
              dhz4l[k][4] * _s13(i, j, 4) + dhz4l[k][5] * _s13(i, j, 5) +
              dhz4l[k][6] * _s13(i, j, 6) -
              _f1_1(i, j) *
                  (dhpz4l[k][0] * _g_c(0) *
                       (phx4[2] * _s11(i, j, 0) + phx4[0] * _s11(i - 2, j, 0) +
                        phx4[1] * _s11(i - 1, j, 0) +
                        phx4[3] * _s11(i + 1, j, 0)) +
                   dhpz4l[k][1] * _g_c(1) *
                       (phx4[2] * _s11(i, j, 1) + phx4[0] * _s11(i - 2, j, 1) +
                        phx4[1] * _s11(i - 1, j, 1) +
                        phx4[3] * _s11(i + 1, j, 1)) +
                   dhpz4l[k][2] * _g_c(2) *
                       (phx4[2] * _s11(i, j, 2) + phx4[0] * _s11(i - 2, j, 2) +
                        phx4[1] * _s11(i - 1, j, 2) +
                        phx4[3] * _s11(i + 1, j, 2)) +
                   dhpz4l[k][3] * _g_c(3) *
                       (phx4[2] * _s11(i, j, 3) + phx4[0] * _s11(i - 2, j, 3) +
                        phx4[1] * _s11(i - 1, j, 3) +
                        phx4[3] * _s11(i + 1, j, 3)) +
                   dhpz4l[k][4] * _g_c(4) *
                       (phx4[2] * _s11(i, j, 4) + phx4[0] * _s11(i - 2, j, 4) +
                        phx4[1] * _s11(i - 1, j, 4) +
                        phx4[3] * _s11(i + 1, j, 4)) +
                   dhpz4l[k][5] * _g_c(5) *
                       (phx4[2] * _s11(i, j, 5) + phx4[0] * _s11(i - 2, j, 5) +
                        phx4[1] * _s11(i - 1, j, 5) +
                        phx4[3] * _s11(i + 1, j, 5)) +
                   dhpz4l[k][6] * _g_c(6) *
                       (phx4[2] * _s11(i, j, 6) + phx4[0] * _s11(i - 2, j, 6) +
                        phx4[1] * _s11(i - 1, j, 6) +
                        phx4[3] * _s11(i + 1, j, 6)) +
                   dhpz4l[k][7] * _g_c(7) *
                       (phx4[2] * _s11(i, j, 7) + phx4[0] * _s11(i - 2, j, 7) +
                        phx4[1] * _s11(i - 1, j, 7) +
                        phx4[3] * _s11(i + 1, j, 7)) +
                   dhpz4l[k][8] * _g_c(8) *
                       (phx4[2] * _s11(i, j, 8) + phx4[0] * _s11(i - 2, j, 8) +
                        phx4[1] * _s11(i - 1, j, 8) +
                        phx4[3] * _s11(i + 1, j, 8))) -
              _f2_1(i, j) *
                  (dhpz4l[k][0] * _g_c(0) *
                       (phy4[2] * _s12(i, j, 0) + phy4[0] * _s12(i, j - 2, 0) +
                        phy4[1] * _s12(i, j - 1, 0) +
                        phy4[3] * _s12(i, j + 1, 0)) +
                   dhpz4l[k][1] * _g_c(1) *
                       (phy4[2] * _s12(i, j, 1) + phy4[0] * _s12(i, j - 2, 1) +
                        phy4[1] * _s12(i, j - 1, 1) +
                        phy4[3] * _s12(i, j + 1, 1)) +
                   dhpz4l[k][2] * _g_c(2) *
                       (phy4[2] * _s12(i, j, 2) + phy4[0] * _s12(i, j - 2, 2) +
                        phy4[1] * _s12(i, j - 1, 2) +
                        phy4[3] * _s12(i, j + 1, 2)) +
                   dhpz4l[k][3] * _g_c(3) *
                       (phy4[2] * _s12(i, j, 3) + phy4[0] * _s12(i, j - 2, 3) +
                        phy4[1] * _s12(i, j - 1, 3) +
                        phy4[3] * _s12(i, j + 1, 3)) +
                   dhpz4l[k][4] * _g_c(4) *
                       (phy4[2] * _s12(i, j, 4) + phy4[0] * _s12(i, j - 2, 4) +
                        phy4[1] * _s12(i, j - 1, 4) +
                        phy4[3] * _s12(i, j + 1, 4)) +
                   dhpz4l[k][5] * _g_c(5) *
                       (phy4[2] * _s12(i, j, 5) + phy4[0] * _s12(i, j - 2, 5) +
                        phy4[1] * _s12(i, j - 1, 5) +
                        phy4[3] * _s12(i, j + 1, 5)) +
                   dhpz4l[k][6] * _g_c(6) *
                       (phy4[2] * _s12(i, j, 6) + phy4[0] * _s12(i, j - 2, 6) +
                        phy4[1] * _s12(i, j - 1, 6) +
                        phy4[3] * _s12(i, j + 1, 6)) +
                   dhpz4l[k][7] * _g_c(7) *
                       (phy4[2] * _s12(i, j, 7) + phy4[0] * _s12(i, j - 2, 7) +
                        phy4[1] * _s12(i, j - 1, 7) +
                        phy4[3] * _s12(i, j + 1, 7)) +
                   dhpz4l[k][8] * _g_c(8) *
                       (phy4[2] * _s12(i, j, 8) + phy4[0] * _s12(i, j - 2, 8) +
                        phy4[1] * _s12(i, j - 1, 8) +
                        phy4[3] * _s12(i, j + 1, 8))))) *
        f_dcrj;
    _u2(i, j, k) =
        (a * _u2(i, j, k) +
         Ai2 * (dhz4l[k][0] * _s23(i, j, 0) + dhz4l[k][1] * _s23(i, j, 1) +
                dhz4l[k][2] * _s23(i, j, 2) + dhz4l[k][3] * _s23(i, j, 3) +
                dhz4l[k][4] * _s23(i, j, 4) + dhz4l[k][5] * _s23(i, j, 5) +
                dhz4l[k][6] * _s23(i, j, 6) +
                dx4[1] * _f(i, j) * _g3_c(k) * _s12(i, j, k) +
                dx4[0] * _f(i - 1, j) * _g3_c(k) * _s12(i - 1, j, k) +
                dx4[2] * _f(i + 1, j) * _g3_c(k) * _s12(i + 1, j, k) +
                dx4[3] * _f(i + 2, j) * _g3_c(k) * _s12(i + 2, j, k) +
                dy4[1] * _f_c(i, j) * _g3_c(k) * _s22(i, j, k) +
                dy4[0] * _f_c(i, j - 1) * _g3_c(k) * _s22(i, j - 1, k) +
                dy4[2] * _f_c(i, j + 1) * _g3_c(k) * _s22(i, j + 1, k) +
                dy4[3] * _f_c(i, j + 2) * _g3_c(k) * _s22(i, j + 2, k) -
                _f1_2(i, j) *
                    (dhpz4l[k][0] * _g_c(0) *
                         (px4[1] * _s12(i, j, 0) + px4[0] * _s12(i - 1, j, 0) +
                          px4[2] * _s12(i + 1, j, 0) +
                          px4[3] * _s12(i + 2, j, 0)) +
                     dhpz4l[k][1] * _g_c(1) *
                         (px4[1] * _s12(i, j, 1) + px4[0] * _s12(i - 1, j, 1) +
                          px4[2] * _s12(i + 1, j, 1) +
                          px4[3] * _s12(i + 2, j, 1)) +
                     dhpz4l[k][2] * _g_c(2) *
                         (px4[1] * _s12(i, j, 2) + px4[0] * _s12(i - 1, j, 2) +
                          px4[2] * _s12(i + 1, j, 2) +
                          px4[3] * _s12(i + 2, j, 2)) +
                     dhpz4l[k][3] * _g_c(3) *
                         (px4[1] * _s12(i, j, 3) + px4[0] * _s12(i - 1, j, 3) +
                          px4[2] * _s12(i + 1, j, 3) +
                          px4[3] * _s12(i + 2, j, 3)) +
                     dhpz4l[k][4] * _g_c(4) *
                         (px4[1] * _s12(i, j, 4) + px4[0] * _s12(i - 1, j, 4) +
                          px4[2] * _s12(i + 1, j, 4) +
                          px4[3] * _s12(i + 2, j, 4)) +
                     dhpz4l[k][5] * _g_c(5) *
                         (px4[1] * _s12(i, j, 5) + px4[0] * _s12(i - 1, j, 5) +
                          px4[2] * _s12(i + 1, j, 5) +
                          px4[3] * _s12(i + 2, j, 5)) +
                     dhpz4l[k][6] * _g_c(6) *
                         (px4[1] * _s12(i, j, 6) + px4[0] * _s12(i - 1, j, 6) +
                          px4[2] * _s12(i + 1, j, 6) +
                          px4[3] * _s12(i + 2, j, 6)) +
                     dhpz4l[k][7] * _g_c(7) *
                         (px4[1] * _s12(i, j, 7) + px4[0] * _s12(i - 1, j, 7) +
                          px4[2] * _s12(i + 1, j, 7) +
                          px4[3] * _s12(i + 2, j, 7)) +
                     dhpz4l[k][8] * _g_c(8) *
                         (px4[1] * _s12(i, j, 8) + px4[0] * _s12(i - 1, j, 8) +
                          px4[2] * _s12(i + 1, j, 8) +
                          px4[3] * _s12(i + 2, j, 8))) -
                _f2_2(i, j) *
                    (dhpz4l[k][0] * _g_c(0) *
                         (py4[1] * _s22(i, j, 0) + py4[0] * _s22(i, j - 1, 0) +
                          py4[2] * _s22(i, j + 1, 0) +
                          py4[3] * _s22(i, j + 2, 0)) +
                     dhpz4l[k][1] * _g_c(1) *
                         (py4[1] * _s22(i, j, 1) + py4[0] * _s22(i, j - 1, 1) +
                          py4[2] * _s22(i, j + 1, 1) +
                          py4[3] * _s22(i, j + 2, 1)) +
                     dhpz4l[k][2] * _g_c(2) *
                         (py4[1] * _s22(i, j, 2) + py4[0] * _s22(i, j - 1, 2) +
                          py4[2] * _s22(i, j + 1, 2) +
                          py4[3] * _s22(i, j + 2, 2)) +
                     dhpz4l[k][3] * _g_c(3) *
                         (py4[1] * _s22(i, j, 3) + py4[0] * _s22(i, j - 1, 3) +
                          py4[2] * _s22(i, j + 1, 3) +
                          py4[3] * _s22(i, j + 2, 3)) +
                     dhpz4l[k][4] * _g_c(4) *
                         (py4[1] * _s22(i, j, 4) + py4[0] * _s22(i, j - 1, 4) +
                          py4[2] * _s22(i, j + 1, 4) +
                          py4[3] * _s22(i, j + 2, 4)) +
                     dhpz4l[k][5] * _g_c(5) *
                         (py4[1] * _s22(i, j, 5) + py4[0] * _s22(i, j - 1, 5) +
                          py4[2] * _s22(i, j + 1, 5) +
                          py4[3] * _s22(i, j + 2, 5)) +
                     dhpz4l[k][6] * _g_c(6) *
                         (py4[1] * _s22(i, j, 6) + py4[0] * _s22(i, j - 1, 6) +
                          py4[2] * _s22(i, j + 1, 6) +
                          py4[3] * _s22(i, j + 2, 6)) +
                     dhpz4l[k][7] * _g_c(7) *
                         (py4[1] * _s22(i, j, 7) + py4[0] * _s22(i, j - 1, 7) +
                          py4[2] * _s22(i, j + 1, 7) +
                          py4[3] * _s22(i, j + 2, 7)) +
                     dhpz4l[k][8] * _g_c(8) *
                         (py4[1] * _s22(i, j, 8) + py4[0] * _s22(i, j - 1, 8) +
                          py4[2] * _s22(i, j + 1, 8) +
                          py4[3] * _s22(i, j + 2, 8))))) *
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
              dz4l[k][0] * _s33(i, j, 0) + dz4l[k][1] * _s33(i, j, 1) +
              dz4l[k][2] * _s33(i, j, 2) + dz4l[k][3] * _s33(i, j, 3) +
              dz4l[k][4] * _s33(i, j, 4) + dz4l[k][5] * _s33(i, j, 5) +
              dz4l[k][6] * _s33(i, j, 6) + dz4l[k][7] * _s33(i, j, 7) -
              _f1_c(i, j) *
                  (dphz4l[k][0] * _g(0) *
                       (px4[1] * _s13(i, j, 0) + px4[0] * _s13(i - 1, j, 0) +
                        px4[2] * _s13(i + 1, j, 0) +
                        px4[3] * _s13(i + 2, j, 0)) +
                   dphz4l[k][1] * _g(1) *
                       (px4[1] * _s13(i, j, 1) + px4[0] * _s13(i - 1, j, 1) +
                        px4[2] * _s13(i + 1, j, 1) +
                        px4[3] * _s13(i + 2, j, 1)) +
                   dphz4l[k][2] * _g(2) *
                       (px4[1] * _s13(i, j, 2) + px4[0] * _s13(i - 1, j, 2) +
                        px4[2] * _s13(i + 1, j, 2) +
                        px4[3] * _s13(i + 2, j, 2)) +
                   dphz4l[k][3] * _g(3) *
                       (px4[1] * _s13(i, j, 3) + px4[0] * _s13(i - 1, j, 3) +
                        px4[2] * _s13(i + 1, j, 3) +
                        px4[3] * _s13(i + 2, j, 3)) +
                   dphz4l[k][4] * _g(4) *
                       (px4[1] * _s13(i, j, 4) + px4[0] * _s13(i - 1, j, 4) +
                        px4[2] * _s13(i + 1, j, 4) +
                        px4[3] * _s13(i + 2, j, 4)) +
                   dphz4l[k][5] * _g(5) *
                       (px4[1] * _s13(i, j, 5) + px4[0] * _s13(i - 1, j, 5) +
                        px4[2] * _s13(i + 1, j, 5) +
                        px4[3] * _s13(i + 2, j, 5)) +
                   dphz4l[k][6] * _g(6) *
                       (px4[1] * _s13(i, j, 6) + px4[0] * _s13(i - 1, j, 6) +
                        px4[2] * _s13(i + 1, j, 6) +
                        px4[3] * _s13(i + 2, j, 6)) +
                   dphz4l[k][7] * _g(7) *
                       (px4[1] * _s13(i, j, 7) + px4[0] * _s13(i - 1, j, 7) +
                        px4[2] * _s13(i + 1, j, 7) +
                        px4[3] * _s13(i + 2, j, 7)) +
                   dphz4l[k][8] * _g(8) *
                       (px4[1] * _s13(i, j, 8) + px4[0] * _s13(i - 1, j, 8) +
                        px4[2] * _s13(i + 1, j, 8) +
                        px4[3] * _s13(i + 2, j, 8))) -
              _f2_c(i, j) *
                  (dphz4l[k][0] * _g(0) *
                       (phy4[2] * _s23(i, j, 0) + phy4[0] * _s23(i, j - 2, 0) +
                        phy4[1] * _s23(i, j - 1, 0) +
                        phy4[3] * _s23(i, j + 1, 0)) +
                   dphz4l[k][1] * _g(1) *
                       (phy4[2] * _s23(i, j, 1) + phy4[0] * _s23(i, j - 2, 1) +
                        phy4[1] * _s23(i, j - 1, 1) +
                        phy4[3] * _s23(i, j + 1, 1)) +
                   dphz4l[k][2] * _g(2) *
                       (phy4[2] * _s23(i, j, 2) + phy4[0] * _s23(i, j - 2, 2) +
                        phy4[1] * _s23(i, j - 1, 2) +
                        phy4[3] * _s23(i, j + 1, 2)) +
                   dphz4l[k][3] * _g(3) *
                       (phy4[2] * _s23(i, j, 3) + phy4[0] * _s23(i, j - 2, 3) +
                        phy4[1] * _s23(i, j - 1, 3) +
                        phy4[3] * _s23(i, j + 1, 3)) +
                   dphz4l[k][4] * _g(4) *
                       (phy4[2] * _s23(i, j, 4) + phy4[0] * _s23(i, j - 2, 4) +
                        phy4[1] * _s23(i, j - 1, 4) +
                        phy4[3] * _s23(i, j + 1, 4)) +
                   dphz4l[k][5] * _g(5) *
                       (phy4[2] * _s23(i, j, 5) + phy4[0] * _s23(i, j - 2, 5) +
                        phy4[1] * _s23(i, j - 1, 5) +
                        phy4[3] * _s23(i, j + 1, 5)) +
                   dphz4l[k][6] * _g(6) *
                       (phy4[2] * _s23(i, j, 6) + phy4[0] * _s23(i, j - 2, 6) +
                        phy4[1] * _s23(i, j - 1, 6) +
                        phy4[3] * _s23(i, j + 1, 6)) +
                   dphz4l[k][7] * _g(7) *
                       (phy4[2] * _s23(i, j, 7) + phy4[0] * _s23(i, j - 2, 7) +
                        phy4[1] * _s23(i, j - 1, 7) +
                        phy4[3] * _s23(i, j + 1, 7)) +
                   dphz4l[k][8] * _g(8) *
                       (phy4[2] * _s23(i, j, 8) + phy4[0] * _s23(i, j - 2, 8) +
                        phy4[1] * _s23(i, j - 1, 8) +
                        phy4[3] * _s23(i, j + 1, 8))))) *
        f_dcrj;
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

__launch_bounds__(DTOPO_VEL_111_MAX_THREADS_PER_BLOCK,
                  DTOPO_VEL_111_MIN_BLOCKS_PER_SM)

    __global__ void dtopo_vel_111(
        float *__restrict__ u1, float *__restrict__ u2, float *__restrict__ u3,
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
  const float phz4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float phy4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float phx4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float dhpz4[7] = {-0.0026041666666667, 0.0937500000000000,
                          -0.6796875000000000, 0.0000000000000000,
                          0.6796875000000000,  -0.0937500000000000,
                          0.0026041666666667};
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
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= nz - 12)
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
    float rho1 =
        phz4[0] *
            (phy4[2] * _rho(i, j, k + 4) + phy4[0] * _rho(i, j - 2, k + 4) +
             phy4[1] * _rho(i, j - 1, k + 4) +
             phy4[3] * _rho(i, j + 1, k + 4)) +
        phz4[1] *
            (phy4[2] * _rho(i, j, k + 5) + phy4[0] * _rho(i, j - 2, k + 5) +
             phy4[1] * _rho(i, j - 1, k + 5) +
             phy4[3] * _rho(i, j + 1, k + 5)) +
        phz4[2] *
            (phy4[2] * _rho(i, j, k + 6) + phy4[0] * _rho(i, j - 2, k + 6) +
             phy4[1] * _rho(i, j - 1, k + 6) +
             phy4[3] * _rho(i, j + 1, k + 6)) +
        phz4[3] *
            (phy4[2] * _rho(i, j, k + 7) + phy4[0] * _rho(i, j - 2, k + 7) +
             phy4[1] * _rho(i, j - 1, k + 7) + phy4[3] * _rho(i, j + 1, k + 7));
    float rho2 =
        phz4[0] *
            (phx4[2] * _rho(i, j, k + 4) + phx4[0] * _rho(i - 2, j, k + 4) +
             phx4[1] * _rho(i - 1, j, k + 4) +
             phx4[3] * _rho(i + 1, j, k + 4)) +
        phz4[1] *
            (phx4[2] * _rho(i, j, k + 5) + phx4[0] * _rho(i - 2, j, k + 5) +
             phx4[1] * _rho(i - 1, j, k + 5) +
             phx4[3] * _rho(i + 1, j, k + 5)) +
        phz4[2] *
            (phx4[2] * _rho(i, j, k + 6) + phx4[0] * _rho(i - 2, j, k + 6) +
             phx4[1] * _rho(i - 1, j, k + 6) +
             phx4[3] * _rho(i + 1, j, k + 6)) +
        phz4[3] *
            (phx4[2] * _rho(i, j, k + 7) + phx4[0] * _rho(i - 2, j, k + 7) +
             phx4[1] * _rho(i - 1, j, k + 7) + phx4[3] * _rho(i + 1, j, k + 7));
    float rho3 = phy4[2] * (phx4[2] * _rho(i, j, k + 6) +
                            phx4[0] * _rho(i - 2, j, k + 6) +
                            phx4[1] * _rho(i - 1, j, k + 6) +
                            phx4[3] * _rho(i + 1, j, k + 6)) +
                 phy4[0] * (phx4[2] * _rho(i, j - 2, k + 6) +
                            phx4[0] * _rho(i - 2, j - 2, k + 6) +
                            phx4[1] * _rho(i - 1, j - 2, k + 6) +
                            phx4[3] * _rho(i + 1, j - 2, k + 6)) +
                 phy4[1] * (phx4[2] * _rho(i, j - 1, k + 6) +
                            phx4[0] * _rho(i - 2, j - 1, k + 6) +
                            phx4[1] * _rho(i - 1, j - 1, k + 6) +
                            phx4[3] * _rho(i + 1, j - 1, k + 6)) +
                 phy4[3] * (phx4[2] * _rho(i, j + 1, k + 6) +
                            phx4[0] * _rho(i - 2, j + 1, k + 6) +
                            phx4[1] * _rho(i - 1, j + 1, k + 6) +
                            phx4[3] * _rho(i + 1, j + 1, k + 6));
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
             (dhx4[2] * _f_c(i, j) * _g3_c(k + 6) * _s11(i, j, k + 6) +
              dhx4[0] * _f_c(i - 2, j) * _g3_c(k + 6) * _s11(i - 2, j, k + 6) +
              dhx4[1] * _f_c(i - 1, j) * _g3_c(k + 6) * _s11(i - 1, j, k + 6) +
              dhx4[3] * _f_c(i + 1, j) * _g3_c(k + 6) * _s11(i + 1, j, k + 6) +
              dhy4[2] * _f(i, j) * _g3_c(k + 6) * _s12(i, j, k + 6) +
              dhy4[0] * _f(i, j - 2) * _g3_c(k + 6) * _s12(i, j - 2, k + 6) +
              dhy4[1] * _f(i, j - 1) * _g3_c(k + 6) * _s12(i, j - 1, k + 6) +
              dhy4[3] * _f(i, j + 1) * _g3_c(k + 6) * _s12(i, j + 1, k + 6) +
              dhz4[0] * _s13(i, j, k + 4) + dhz4[1] * _s13(i, j, k + 5) +
              dhz4[2] * _s13(i, j, k + 6) + dhz4[3] * _s13(i, j, k + 7) -
              _f1_1(i, j) * (dhpz4[0] * _g_c(k + 3) *
                                 (phx4[2] * _s11(i, j, k + 3) +
                                  phx4[0] * _s11(i - 2, j, k + 3) +
                                  phx4[1] * _s11(i - 1, j, k + 3) +
                                  phx4[3] * _s11(i + 1, j, k + 3)) +
                             dhpz4[1] * _g_c(k + 4) *
                                 (phx4[2] * _s11(i, j, k + 4) +
                                  phx4[0] * _s11(i - 2, j, k + 4) +
                                  phx4[1] * _s11(i - 1, j, k + 4) +
                                  phx4[3] * _s11(i + 1, j, k + 4)) +
                             dhpz4[2] * _g_c(k + 5) *
                                 (phx4[2] * _s11(i, j, k + 5) +
                                  phx4[0] * _s11(i - 2, j, k + 5) +
                                  phx4[1] * _s11(i - 1, j, k + 5) +
                                  phx4[3] * _s11(i + 1, j, k + 5)) +
                             dhpz4[3] * _g_c(k + 6) *
                                 (phx4[2] * _s11(i, j, k + 6) +
                                  phx4[0] * _s11(i - 2, j, k + 6) +
                                  phx4[1] * _s11(i - 1, j, k + 6) +
                                  phx4[3] * _s11(i + 1, j, k + 6)) +
                             dhpz4[4] * _g_c(k + 7) *
                                 (phx4[2] * _s11(i, j, k + 7) +
                                  phx4[0] * _s11(i - 2, j, k + 7) +
                                  phx4[1] * _s11(i - 1, j, k + 7) +
                                  phx4[3] * _s11(i + 1, j, k + 7)) +
                             dhpz4[5] * _g_c(k + 8) *
                                 (phx4[2] * _s11(i, j, k + 8) +
                                  phx4[0] * _s11(i - 2, j, k + 8) +
                                  phx4[1] * _s11(i - 1, j, k + 8) +
                                  phx4[3] * _s11(i + 1, j, k + 8)) +
                             dhpz4[6] * _g_c(k + 9) *
                                 (phx4[2] * _s11(i, j, k + 9) +
                                  phx4[0] * _s11(i - 2, j, k + 9) +
                                  phx4[1] * _s11(i - 1, j, k + 9) +
                                  phx4[3] * _s11(i + 1, j, k + 9))) -
              _f2_1(i, j) * (dhpz4[0] * _g_c(k + 3) *
                                 (phy4[2] * _s12(i, j, k + 3) +
                                  phy4[0] * _s12(i, j - 2, k + 3) +
                                  phy4[1] * _s12(i, j - 1, k + 3) +
                                  phy4[3] * _s12(i, j + 1, k + 3)) +
                             dhpz4[1] * _g_c(k + 4) *
                                 (phy4[2] * _s12(i, j, k + 4) +
                                  phy4[0] * _s12(i, j - 2, k + 4) +
                                  phy4[1] * _s12(i, j - 1, k + 4) +
                                  phy4[3] * _s12(i, j + 1, k + 4)) +
                             dhpz4[2] * _g_c(k + 5) *
                                 (phy4[2] * _s12(i, j, k + 5) +
                                  phy4[0] * _s12(i, j - 2, k + 5) +
                                  phy4[1] * _s12(i, j - 1, k + 5) +
                                  phy4[3] * _s12(i, j + 1, k + 5)) +
                             dhpz4[3] * _g_c(k + 6) *
                                 (phy4[2] * _s12(i, j, k + 6) +
                                  phy4[0] * _s12(i, j - 2, k + 6) +
                                  phy4[1] * _s12(i, j - 1, k + 6) +
                                  phy4[3] * _s12(i, j + 1, k + 6)) +
                             dhpz4[4] * _g_c(k + 7) *
                                 (phy4[2] * _s12(i, j, k + 7) +
                                  phy4[0] * _s12(i, j - 2, k + 7) +
                                  phy4[1] * _s12(i, j - 1, k + 7) +
                                  phy4[3] * _s12(i, j + 1, k + 7)) +
                             dhpz4[5] * _g_c(k + 8) *
                                 (phy4[2] * _s12(i, j, k + 8) +
                                  phy4[0] * _s12(i, j - 2, k + 8) +
                                  phy4[1] * _s12(i, j - 1, k + 8) +
                                  phy4[3] * _s12(i, j + 1, k + 8)) +
                             dhpz4[6] * _g_c(k + 9) *
                                 (phy4[2] * _s12(i, j, k + 9) +
                                  phy4[0] * _s12(i, j - 2, k + 9) +
                                  phy4[1] * _s12(i, j - 1, k + 9) +
                                  phy4[3] * _s12(i, j + 1, k + 9))))) *
        f_dcrj;
    _u2(i, j, k + 6) =
        (a * _u2(i, j, k + 6) +
         Ai2 * (dhz4[0] * _s23(i, j, k + 4) + dhz4[1] * _s23(i, j, k + 5) +
                dhz4[2] * _s23(i, j, k + 6) + dhz4[3] * _s23(i, j, k + 7) +
                dx4[1] * _f(i, j) * _g3_c(k + 6) * _s12(i, j, k + 6) +
                dx4[0] * _f(i - 1, j) * _g3_c(k + 6) * _s12(i - 1, j, k + 6) +
                dx4[2] * _f(i + 1, j) * _g3_c(k + 6) * _s12(i + 1, j, k + 6) +
                dx4[3] * _f(i + 2, j) * _g3_c(k + 6) * _s12(i + 2, j, k + 6) +
                dy4[1] * _f_c(i, j) * _g3_c(k + 6) * _s22(i, j, k + 6) +
                dy4[0] * _f_c(i, j - 1) * _g3_c(k + 6) * _s22(i, j - 1, k + 6) +
                dy4[2] * _f_c(i, j + 1) * _g3_c(k + 6) * _s22(i, j + 1, k + 6) +
                dy4[3] * _f_c(i, j + 2) * _g3_c(k + 6) * _s22(i, j + 2, k + 6) -
                _f1_2(i, j) * (dhpz4[0] * _g_c(k + 3) *
                                   (px4[1] * _s12(i, j, k + 3) +
                                    px4[0] * _s12(i - 1, j, k + 3) +
                                    px4[2] * _s12(i + 1, j, k + 3) +
                                    px4[3] * _s12(i + 2, j, k + 3)) +
                               dhpz4[1] * _g_c(k + 4) *
                                   (px4[1] * _s12(i, j, k + 4) +
                                    px4[0] * _s12(i - 1, j, k + 4) +
                                    px4[2] * _s12(i + 1, j, k + 4) +
                                    px4[3] * _s12(i + 2, j, k + 4)) +
                               dhpz4[2] * _g_c(k + 5) *
                                   (px4[1] * _s12(i, j, k + 5) +
                                    px4[0] * _s12(i - 1, j, k + 5) +
                                    px4[2] * _s12(i + 1, j, k + 5) +
                                    px4[3] * _s12(i + 2, j, k + 5)) +
                               dhpz4[3] * _g_c(k + 6) *
                                   (px4[1] * _s12(i, j, k + 6) +
                                    px4[0] * _s12(i - 1, j, k + 6) +
                                    px4[2] * _s12(i + 1, j, k + 6) +
                                    px4[3] * _s12(i + 2, j, k + 6)) +
                               dhpz4[4] * _g_c(k + 7) *
                                   (px4[1] * _s12(i, j, k + 7) +
                                    px4[0] * _s12(i - 1, j, k + 7) +
                                    px4[2] * _s12(i + 1, j, k + 7) +
                                    px4[3] * _s12(i + 2, j, k + 7)) +
                               dhpz4[5] * _g_c(k + 8) *
                                   (px4[1] * _s12(i, j, k + 8) +
                                    px4[0] * _s12(i - 1, j, k + 8) +
                                    px4[2] * _s12(i + 1, j, k + 8) +
                                    px4[3] * _s12(i + 2, j, k + 8)) +
                               dhpz4[6] * _g_c(k + 9) *
                                   (px4[1] * _s12(i, j, k + 9) +
                                    px4[0] * _s12(i - 1, j, k + 9) +
                                    px4[2] * _s12(i + 1, j, k + 9) +
                                    px4[3] * _s12(i + 2, j, k + 9))) -
                _f2_2(i, j) * (dhpz4[0] * _g_c(k + 3) *
                                   (py4[1] * _s22(i, j, k + 3) +
                                    py4[0] * _s22(i, j - 1, k + 3) +
                                    py4[2] * _s22(i, j + 1, k + 3) +
                                    py4[3] * _s22(i, j + 2, k + 3)) +
                               dhpz4[1] * _g_c(k + 4) *
                                   (py4[1] * _s22(i, j, k + 4) +
                                    py4[0] * _s22(i, j - 1, k + 4) +
                                    py4[2] * _s22(i, j + 1, k + 4) +
                                    py4[3] * _s22(i, j + 2, k + 4)) +
                               dhpz4[2] * _g_c(k + 5) *
                                   (py4[1] * _s22(i, j, k + 5) +
                                    py4[0] * _s22(i, j - 1, k + 5) +
                                    py4[2] * _s22(i, j + 1, k + 5) +
                                    py4[3] * _s22(i, j + 2, k + 5)) +
                               dhpz4[3] * _g_c(k + 6) *
                                   (py4[1] * _s22(i, j, k + 6) +
                                    py4[0] * _s22(i, j - 1, k + 6) +
                                    py4[2] * _s22(i, j + 1, k + 6) +
                                    py4[3] * _s22(i, j + 2, k + 6)) +
                               dhpz4[4] * _g_c(k + 7) *
                                   (py4[1] * _s22(i, j, k + 7) +
                                    py4[0] * _s22(i, j - 1, k + 7) +
                                    py4[2] * _s22(i, j + 1, k + 7) +
                                    py4[3] * _s22(i, j + 2, k + 7)) +
                               dhpz4[5] * _g_c(k + 8) *
                                   (py4[1] * _s22(i, j, k + 8) +
                                    py4[0] * _s22(i, j - 1, k + 8) +
                                    py4[2] * _s22(i, j + 1, k + 8) +
                                    py4[3] * _s22(i, j + 2, k + 8)) +
                               dhpz4[6] * _g_c(k + 9) *
                                   (py4[1] * _s22(i, j, k + 9) +
                                    py4[0] * _s22(i, j - 1, k + 9) +
                                    py4[2] * _s22(i, j + 1, k + 9) +
                                    py4[3] * _s22(i, j + 2, k + 9))))) *
        f_dcrj;
    _u3(i, j, k + 6) =
        (a * _u3(i, j, k + 6) +
         Ai3 * (dhy4[2] * _f_2(i, j) * _g3(k + 6) * _s23(i, j, k + 6) +
                dhy4[0] * _f_2(i, j - 2) * _g3(k + 6) * _s23(i, j - 2, k + 6) +
                dhy4[1] * _f_2(i, j - 1) * _g3(k + 6) * _s23(i, j - 1, k + 6) +
                dhy4[3] * _f_2(i, j + 1) * _g3(k + 6) * _s23(i, j + 1, k + 6) +
                dx4[1] * _f_1(i, j) * _g3(k + 6) * _s13(i, j, k + 6) +
                dx4[0] * _f_1(i - 1, j) * _g3(k + 6) * _s13(i - 1, j, k + 6) +
                dx4[2] * _f_1(i + 1, j) * _g3(k + 6) * _s13(i + 1, j, k + 6) +
                dx4[3] * _f_1(i + 2, j) * _g3(k + 6) * _s13(i + 2, j, k + 6) +
                dz4[0] * _s33(i, j, k + 5) + dz4[1] * _s33(i, j, k + 6) +
                dz4[2] * _s33(i, j, k + 7) + dz4[3] * _s33(i, j, k + 8) -
                _f1_c(i, j) * (dphz4[0] * _g(k + 3) *
                                   (px4[1] * _s13(i, j, k + 3) +
                                    px4[0] * _s13(i - 1, j, k + 3) +
                                    px4[2] * _s13(i + 1, j, k + 3) +
                                    px4[3] * _s13(i + 2, j, k + 3)) +
                               dphz4[1] * _g(k + 4) *
                                   (px4[1] * _s13(i, j, k + 4) +
                                    px4[0] * _s13(i - 1, j, k + 4) +
                                    px4[2] * _s13(i + 1, j, k + 4) +
                                    px4[3] * _s13(i + 2, j, k + 4)) +
                               dphz4[2] * _g(k + 5) *
                                   (px4[1] * _s13(i, j, k + 5) +
                                    px4[0] * _s13(i - 1, j, k + 5) +
                                    px4[2] * _s13(i + 1, j, k + 5) +
                                    px4[3] * _s13(i + 2, j, k + 5)) +
                               dphz4[3] * _g(k + 6) *
                                   (px4[1] * _s13(i, j, k + 6) +
                                    px4[0] * _s13(i - 1, j, k + 6) +
                                    px4[2] * _s13(i + 1, j, k + 6) +
                                    px4[3] * _s13(i + 2, j, k + 6)) +
                               dphz4[4] * _g(k + 7) *
                                   (px4[1] * _s13(i, j, k + 7) +
                                    px4[0] * _s13(i - 1, j, k + 7) +
                                    px4[2] * _s13(i + 1, j, k + 7) +
                                    px4[3] * _s13(i + 2, j, k + 7)) +
                               dphz4[5] * _g(k + 8) *
                                   (px4[1] * _s13(i, j, k + 8) +
                                    px4[0] * _s13(i - 1, j, k + 8) +
                                    px4[2] * _s13(i + 1, j, k + 8) +
                                    px4[3] * _s13(i + 2, j, k + 8)) +
                               dphz4[6] * _g(k + 9) *
                                   (px4[1] * _s13(i, j, k + 9) +
                                    px4[0] * _s13(i - 1, j, k + 9) +
                                    px4[2] * _s13(i + 1, j, k + 9) +
                                    px4[3] * _s13(i + 2, j, k + 9))) -
                _f2_c(i, j) * (dphz4[0] * _g(k + 3) *
                                   (phy4[2] * _s23(i, j, k + 3) +
                                    phy4[0] * _s23(i, j - 2, k + 3) +
                                    phy4[1] * _s23(i, j - 1, k + 3) +
                                    phy4[3] * _s23(i, j + 1, k + 3)) +
                               dphz4[1] * _g(k + 4) *
                                   (phy4[2] * _s23(i, j, k + 4) +
                                    phy4[0] * _s23(i, j - 2, k + 4) +
                                    phy4[1] * _s23(i, j - 1, k + 4) +
                                    phy4[3] * _s23(i, j + 1, k + 4)) +
                               dphz4[2] * _g(k + 5) *
                                   (phy4[2] * _s23(i, j, k + 5) +
                                    phy4[0] * _s23(i, j - 2, k + 5) +
                                    phy4[1] * _s23(i, j - 1, k + 5) +
                                    phy4[3] * _s23(i, j + 1, k + 5)) +
                               dphz4[3] * _g(k + 6) *
                                   (phy4[2] * _s23(i, j, k + 6) +
                                    phy4[0] * _s23(i, j - 2, k + 6) +
                                    phy4[1] * _s23(i, j - 1, k + 6) +
                                    phy4[3] * _s23(i, j + 1, k + 6)) +
                               dphz4[4] * _g(k + 7) *
                                   (phy4[2] * _s23(i, j, k + 7) +
                                    phy4[0] * _s23(i, j - 2, k + 7) +
                                    phy4[1] * _s23(i, j - 1, k + 7) +
                                    phy4[3] * _s23(i, j + 1, k + 7)) +
                               dphz4[5] * _g(k + 8) *
                                   (phy4[2] * _s23(i, j, k + 8) +
                                    phy4[0] * _s23(i, j - 2, k + 8) +
                                    phy4[1] * _s23(i, j - 1, k + 8) +
                                    phy4[3] * _s23(i, j + 1, k + 8)) +
                               dphz4[6] * _g(k + 9) *
                                   (phy4[2] * _s23(i, j, k + 9) +
                                    phy4[0] * _s23(i, j - 2, k + 9) +
                                    phy4[1] * _s23(i, j - 1, k + 9) +
                                    phy4[3] * _s23(i, j + 1, k + 9))))) *
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

__launch_bounds__(DTOPO_VEL_112_MAX_THREADS_PER_BLOCK,
                  DTOPO_VEL_112_MIN_BLOCKS_PER_SM)

    __global__ void dtopo_vel_112(
        float *__restrict__ u1, float *__restrict__ u2, float *__restrict__ u3,
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
  const float phz4r[6][8] = {
      {0.0000000000000000, 0.8338228784688313, 0.1775123316429260,
       0.1435067013076542, -0.1548419114194114, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.1813404047323969, 1.1246711188154426,
       -0.2933634518280757, -0.0126480717197637, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, -0.1331142706282399, 0.7930714675884345,
       0.3131998767078508, 0.0268429263319546, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0969078556633046, -0.1539344946680898,
       0.4486491202844389, 0.6768738207821733, -0.0684963020618270,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       -0.0625000000000000, 0.5625000000000000, 0.5625000000000000,
       -0.0625000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, -0.0625000000000000, 0.5625000000000000,
       0.5625000000000000, -0.0625000000000000}};
  const float phy4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float phx4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float dhpz4r[6][9] = {
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
  const float dhy4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const float dhx4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const float dhz4r[6][8] = {
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
  const float px4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float py4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float dx4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float dy4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float dphz4r[6][9] = {
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
  const float dz4r[6][7] = {
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
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ny)
    return;
  if (j >= ej)
    return;
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= 6)
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
  for (int i = bi; i < ei; ++i) {
    float rho1 = phz4r[k][7] * (phy4[2] * _rho(i, j, nz - 8) +
                                phy4[0] * _rho(i, j - 2, nz - 8) +
                                phy4[1] * _rho(i, j - 1, nz - 8) +
                                phy4[3] * _rho(i, j + 1, nz - 8)) +
                 phz4r[k][6] * (phy4[2] * _rho(i, j, nz - 7) +
                                phy4[0] * _rho(i, j - 2, nz - 7) +
                                phy4[1] * _rho(i, j - 1, nz - 7) +
                                phy4[3] * _rho(i, j + 1, nz - 7)) +
                 phz4r[k][5] * (phy4[2] * _rho(i, j, nz - 6) +
                                phy4[0] * _rho(i, j - 2, nz - 6) +
                                phy4[1] * _rho(i, j - 1, nz - 6) +
                                phy4[3] * _rho(i, j + 1, nz - 6)) +
                 phz4r[k][4] * (phy4[2] * _rho(i, j, nz - 5) +
                                phy4[0] * _rho(i, j - 2, nz - 5) +
                                phy4[1] * _rho(i, j - 1, nz - 5) +
                                phy4[3] * _rho(i, j + 1, nz - 5)) +
                 phz4r[k][3] * (phy4[2] * _rho(i, j, nz - 4) +
                                phy4[0] * _rho(i, j - 2, nz - 4) +
                                phy4[1] * _rho(i, j - 1, nz - 4) +
                                phy4[3] * _rho(i, j + 1, nz - 4)) +
                 phz4r[k][2] * (phy4[2] * _rho(i, j, nz - 3) +
                                phy4[0] * _rho(i, j - 2, nz - 3) +
                                phy4[1] * _rho(i, j - 1, nz - 3) +
                                phy4[3] * _rho(i, j + 1, nz - 3)) +
                 phz4r[k][1] * (phy4[2] * _rho(i, j, nz - 2) +
                                phy4[0] * _rho(i, j - 2, nz - 2) +
                                phy4[1] * _rho(i, j - 1, nz - 2) +
                                phy4[3] * _rho(i, j + 1, nz - 2)) +
                 phz4r[k][0] * (phy4[2] * _rho(i, j, nz - 1) +
                                phy4[0] * _rho(i, j - 2, nz - 1) +
                                phy4[1] * _rho(i, j - 1, nz - 1) +
                                phy4[3] * _rho(i, j + 1, nz - 1));
    float rho2 = phz4r[k][7] * (phx4[2] * _rho(i, j, nz - 8) +
                                phx4[0] * _rho(i - 2, j, nz - 8) +
                                phx4[1] * _rho(i - 1, j, nz - 8) +
                                phx4[3] * _rho(i + 1, j, nz - 8)) +
                 phz4r[k][6] * (phx4[2] * _rho(i, j, nz - 7) +
                                phx4[0] * _rho(i - 2, j, nz - 7) +
                                phx4[1] * _rho(i - 1, j, nz - 7) +
                                phx4[3] * _rho(i + 1, j, nz - 7)) +
                 phz4r[k][5] * (phx4[2] * _rho(i, j, nz - 6) +
                                phx4[0] * _rho(i - 2, j, nz - 6) +
                                phx4[1] * _rho(i - 1, j, nz - 6) +
                                phx4[3] * _rho(i + 1, j, nz - 6)) +
                 phz4r[k][4] * (phx4[2] * _rho(i, j, nz - 5) +
                                phx4[0] * _rho(i - 2, j, nz - 5) +
                                phx4[1] * _rho(i - 1, j, nz - 5) +
                                phx4[3] * _rho(i + 1, j, nz - 5)) +
                 phz4r[k][3] * (phx4[2] * _rho(i, j, nz - 4) +
                                phx4[0] * _rho(i - 2, j, nz - 4) +
                                phx4[1] * _rho(i - 1, j, nz - 4) +
                                phx4[3] * _rho(i + 1, j, nz - 4)) +
                 phz4r[k][2] * (phx4[2] * _rho(i, j, nz - 3) +
                                phx4[0] * _rho(i - 2, j, nz - 3) +
                                phx4[1] * _rho(i - 1, j, nz - 3) +
                                phx4[3] * _rho(i + 1, j, nz - 3)) +
                 phz4r[k][1] * (phx4[2] * _rho(i, j, nz - 2) +
                                phx4[0] * _rho(i - 2, j, nz - 2) +
                                phx4[1] * _rho(i - 1, j, nz - 2) +
                                phx4[3] * _rho(i + 1, j, nz - 2)) +
                 phz4r[k][0] * (phx4[2] * _rho(i, j, nz - 1) +
                                phx4[0] * _rho(i - 2, j, nz - 1) +
                                phx4[1] * _rho(i - 1, j, nz - 1) +
                                phx4[3] * _rho(i + 1, j, nz - 1));
    float rho3 = phy4[2] * (phx4[2] * _rho(i, j, nz - 1 - k) +
                            phx4[0] * _rho(i - 2, j, nz - 1 - k) +
                            phx4[1] * _rho(i - 1, j, nz - 1 - k) +
                            phx4[3] * _rho(i + 1, j, nz - 1 - k)) +
                 phy4[0] * (phx4[2] * _rho(i, j - 2, nz - 1 - k) +
                            phx4[0] * _rho(i - 2, j - 2, nz - 1 - k) +
                            phx4[1] * _rho(i - 1, j - 2, nz - 1 - k) +
                            phx4[3] * _rho(i + 1, j - 2, nz - 1 - k)) +
                 phy4[1] * (phx4[2] * _rho(i, j - 1, nz - 1 - k) +
                            phx4[0] * _rho(i - 2, j - 1, nz - 1 - k) +
                            phx4[1] * _rho(i - 1, j - 1, nz - 1 - k) +
                            phx4[3] * _rho(i + 1, j - 1, nz - 1 - k)) +
                 phy4[3] * (phx4[2] * _rho(i, j + 1, nz - 1 - k) +
                            phx4[0] * _rho(i - 2, j + 1, nz - 1 - k) +
                            phx4[1] * _rho(i - 1, j + 1, nz - 1 - k) +
                            phx4[3] * _rho(i + 1, j + 1, nz - 1 - k));
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
             (dhx4[2] * _f_c(i, j) * _g3_c(nz - 1 - k) *
                  _s11(i, j, nz - 1 - k) +
              dhx4[0] * _f_c(i - 2, j) * _g3_c(nz - 1 - k) *
                  _s11(i - 2, j, nz - 1 - k) +
              dhx4[1] * _f_c(i - 1, j) * _g3_c(nz - 1 - k) *
                  _s11(i - 1, j, nz - 1 - k) +
              dhx4[3] * _f_c(i + 1, j) * _g3_c(nz - 1 - k) *
                  _s11(i + 1, j, nz - 1 - k) +
              dhy4[2] * _f(i, j) * _g3_c(nz - 1 - k) * _s12(i, j, nz - 1 - k) +
              dhy4[0] * _f(i, j - 2) * _g3_c(nz - 1 - k) *
                  _s12(i, j - 2, nz - 1 - k) +
              dhy4[1] * _f(i, j - 1) * _g3_c(nz - 1 - k) *
                  _s12(i, j - 1, nz - 1 - k) +
              dhy4[3] * _f(i, j + 1) * _g3_c(nz - 1 - k) *
                  _s12(i, j + 1, nz - 1 - k) +
              dhz4r[k][7] * _s13(i, j, nz - 8) +
              dhz4r[k][6] * _s13(i, j, nz - 7) +
              dhz4r[k][5] * _s13(i, j, nz - 6) +
              dhz4r[k][4] * _s13(i, j, nz - 5) +
              dhz4r[k][3] * _s13(i, j, nz - 4) +
              dhz4r[k][2] * _s13(i, j, nz - 3) +
              dhz4r[k][1] * _s13(i, j, nz - 2) +
              dhz4r[k][0] * _s13(i, j, nz - 1) -
              _f1_1(i, j) * (dhpz4r[k][8] * _g_c(nz - 9) *
                                 (phx4[2] * _s11(i, j, nz - 9) +
                                  phx4[0] * _s11(i - 2, j, nz - 9) +
                                  phx4[1] * _s11(i - 1, j, nz - 9) +
                                  phx4[3] * _s11(i + 1, j, nz - 9)) +
                             dhpz4r[k][7] * _g_c(nz - 8) *
                                 (phx4[2] * _s11(i, j, nz - 8) +
                                  phx4[0] * _s11(i - 2, j, nz - 8) +
                                  phx4[1] * _s11(i - 1, j, nz - 8) +
                                  phx4[3] * _s11(i + 1, j, nz - 8)) +
                             dhpz4r[k][6] * _g_c(nz - 7) *
                                 (phx4[2] * _s11(i, j, nz - 7) +
                                  phx4[0] * _s11(i - 2, j, nz - 7) +
                                  phx4[1] * _s11(i - 1, j, nz - 7) +
                                  phx4[3] * _s11(i + 1, j, nz - 7)) +
                             dhpz4r[k][5] * _g_c(nz - 6) *
                                 (phx4[2] * _s11(i, j, nz - 6) +
                                  phx4[0] * _s11(i - 2, j, nz - 6) +
                                  phx4[1] * _s11(i - 1, j, nz - 6) +
                                  phx4[3] * _s11(i + 1, j, nz - 6)) +
                             dhpz4r[k][4] * _g_c(nz - 5) *
                                 (phx4[2] * _s11(i, j, nz - 5) +
                                  phx4[0] * _s11(i - 2, j, nz - 5) +
                                  phx4[1] * _s11(i - 1, j, nz - 5) +
                                  phx4[3] * _s11(i + 1, j, nz - 5)) +
                             dhpz4r[k][3] * _g_c(nz - 4) *
                                 (phx4[2] * _s11(i, j, nz - 4) +
                                  phx4[0] * _s11(i - 2, j, nz - 4) +
                                  phx4[1] * _s11(i - 1, j, nz - 4) +
                                  phx4[3] * _s11(i + 1, j, nz - 4)) +
                             dhpz4r[k][2] * _g_c(nz - 3) *
                                 (phx4[2] * _s11(i, j, nz - 3) +
                                  phx4[0] * _s11(i - 2, j, nz - 3) +
                                  phx4[1] * _s11(i - 1, j, nz - 3) +
                                  phx4[3] * _s11(i + 1, j, nz - 3)) +
                             dhpz4r[k][1] * _g_c(nz - 2) *
                                 (phx4[2] * _s11(i, j, nz - 2) +
                                  phx4[0] * _s11(i - 2, j, nz - 2) +
                                  phx4[1] * _s11(i - 1, j, nz - 2) +
                                  phx4[3] * _s11(i + 1, j, nz - 2)) +
                             dhpz4r[k][0] * _g_c(nz - 1) *
                                 (phx4[2] * _s11(i, j, nz - 1) +
                                  phx4[0] * _s11(i - 2, j, nz - 1) +
                                  phx4[1] * _s11(i - 1, j, nz - 1) +
                                  phx4[3] * _s11(i + 1, j, nz - 1))) -
              _f2_1(i, j) * (dhpz4r[k][8] * _g_c(nz - 9) *
                                 (phy4[2] * _s12(i, j, nz - 9) +
                                  phy4[0] * _s12(i, j - 2, nz - 9) +
                                  phy4[1] * _s12(i, j - 1, nz - 9) +
                                  phy4[3] * _s12(i, j + 1, nz - 9)) +
                             dhpz4r[k][7] * _g_c(nz - 8) *
                                 (phy4[2] * _s12(i, j, nz - 8) +
                                  phy4[0] * _s12(i, j - 2, nz - 8) +
                                  phy4[1] * _s12(i, j - 1, nz - 8) +
                                  phy4[3] * _s12(i, j + 1, nz - 8)) +
                             dhpz4r[k][6] * _g_c(nz - 7) *
                                 (phy4[2] * _s12(i, j, nz - 7) +
                                  phy4[0] * _s12(i, j - 2, nz - 7) +
                                  phy4[1] * _s12(i, j - 1, nz - 7) +
                                  phy4[3] * _s12(i, j + 1, nz - 7)) +
                             dhpz4r[k][5] * _g_c(nz - 6) *
                                 (phy4[2] * _s12(i, j, nz - 6) +
                                  phy4[0] * _s12(i, j - 2, nz - 6) +
                                  phy4[1] * _s12(i, j - 1, nz - 6) +
                                  phy4[3] * _s12(i, j + 1, nz - 6)) +
                             dhpz4r[k][4] * _g_c(nz - 5) *
                                 (phy4[2] * _s12(i, j, nz - 5) +
                                  phy4[0] * _s12(i, j - 2, nz - 5) +
                                  phy4[1] * _s12(i, j - 1, nz - 5) +
                                  phy4[3] * _s12(i, j + 1, nz - 5)) +
                             dhpz4r[k][3] * _g_c(nz - 4) *
                                 (phy4[2] * _s12(i, j, nz - 4) +
                                  phy4[0] * _s12(i, j - 2, nz - 4) +
                                  phy4[1] * _s12(i, j - 1, nz - 4) +
                                  phy4[3] * _s12(i, j + 1, nz - 4)) +
                             dhpz4r[k][2] * _g_c(nz - 3) *
                                 (phy4[2] * _s12(i, j, nz - 3) +
                                  phy4[0] * _s12(i, j - 2, nz - 3) +
                                  phy4[1] * _s12(i, j - 1, nz - 3) +
                                  phy4[3] * _s12(i, j + 1, nz - 3)) +
                             dhpz4r[k][1] * _g_c(nz - 2) *
                                 (phy4[2] * _s12(i, j, nz - 2) +
                                  phy4[0] * _s12(i, j - 2, nz - 2) +
                                  phy4[1] * _s12(i, j - 1, nz - 2) +
                                  phy4[3] * _s12(i, j + 1, nz - 2)) +
                             dhpz4r[k][0] * _g_c(nz - 1) *
                                 (phy4[2] * _s12(i, j, nz - 1) +
                                  phy4[0] * _s12(i, j - 2, nz - 1) +
                                  phy4[1] * _s12(i, j - 1, nz - 1) +
                                  phy4[3] * _s12(i, j + 1, nz - 1))))) *
        f_dcrj;
    _u2(i, j, nz - 1 - k) =
        (a * _u2(i, j, nz - 1 - k) +
         Ai2 *
             (dhz4r[k][7] * _s23(i, j, nz - 8) +
              dhz4r[k][6] * _s23(i, j, nz - 7) +
              dhz4r[k][5] * _s23(i, j, nz - 6) +
              dhz4r[k][4] * _s23(i, j, nz - 5) +
              dhz4r[k][3] * _s23(i, j, nz - 4) +
              dhz4r[k][2] * _s23(i, j, nz - 3) +
              dhz4r[k][1] * _s23(i, j, nz - 2) +
              dhz4r[k][0] * _s23(i, j, nz - 1) +
              dx4[1] * _f(i, j) * _g3_c(nz - 1 - k) * _s12(i, j, nz - 1 - k) +
              dx4[0] * _f(i - 1, j) * _g3_c(nz - 1 - k) *
                  _s12(i - 1, j, nz - 1 - k) +
              dx4[2] * _f(i + 1, j) * _g3_c(nz - 1 - k) *
                  _s12(i + 1, j, nz - 1 - k) +
              dx4[3] * _f(i + 2, j) * _g3_c(nz - 1 - k) *
                  _s12(i + 2, j, nz - 1 - k) +
              dy4[1] * _f_c(i, j) * _g3_c(nz - 1 - k) * _s22(i, j, nz - 1 - k) +
              dy4[0] * _f_c(i, j - 1) * _g3_c(nz - 1 - k) *
                  _s22(i, j - 1, nz - 1 - k) +
              dy4[2] * _f_c(i, j + 1) * _g3_c(nz - 1 - k) *
                  _s22(i, j + 1, nz - 1 - k) +
              dy4[3] * _f_c(i, j + 2) * _g3_c(nz - 1 - k) *
                  _s22(i, j + 2, nz - 1 - k) -
              _f1_2(i, j) * (dhpz4r[k][8] * _g_c(nz - 9) *
                                 (px4[1] * _s12(i, j, nz - 9) +
                                  px4[0] * _s12(i - 1, j, nz - 9) +
                                  px4[2] * _s12(i + 1, j, nz - 9) +
                                  px4[3] * _s12(i + 2, j, nz - 9)) +
                             dhpz4r[k][7] * _g_c(nz - 8) *
                                 (px4[1] * _s12(i, j, nz - 8) +
                                  px4[0] * _s12(i - 1, j, nz - 8) +
                                  px4[2] * _s12(i + 1, j, nz - 8) +
                                  px4[3] * _s12(i + 2, j, nz - 8)) +
                             dhpz4r[k][6] * _g_c(nz - 7) *
                                 (px4[1] * _s12(i, j, nz - 7) +
                                  px4[0] * _s12(i - 1, j, nz - 7) +
                                  px4[2] * _s12(i + 1, j, nz - 7) +
                                  px4[3] * _s12(i + 2, j, nz - 7)) +
                             dhpz4r[k][5] * _g_c(nz - 6) *
                                 (px4[1] * _s12(i, j, nz - 6) +
                                  px4[0] * _s12(i - 1, j, nz - 6) +
                                  px4[2] * _s12(i + 1, j, nz - 6) +
                                  px4[3] * _s12(i + 2, j, nz - 6)) +
                             dhpz4r[k][4] * _g_c(nz - 5) *
                                 (px4[1] * _s12(i, j, nz - 5) +
                                  px4[0] * _s12(i - 1, j, nz - 5) +
                                  px4[2] * _s12(i + 1, j, nz - 5) +
                                  px4[3] * _s12(i + 2, j, nz - 5)) +
                             dhpz4r[k][3] * _g_c(nz - 4) *
                                 (px4[1] * _s12(i, j, nz - 4) +
                                  px4[0] * _s12(i - 1, j, nz - 4) +
                                  px4[2] * _s12(i + 1, j, nz - 4) +
                                  px4[3] * _s12(i + 2, j, nz - 4)) +
                             dhpz4r[k][2] * _g_c(nz - 3) *
                                 (px4[1] * _s12(i, j, nz - 3) +
                                  px4[0] * _s12(i - 1, j, nz - 3) +
                                  px4[2] * _s12(i + 1, j, nz - 3) +
                                  px4[3] * _s12(i + 2, j, nz - 3)) +
                             dhpz4r[k][1] * _g_c(nz - 2) *
                                 (px4[1] * _s12(i, j, nz - 2) +
                                  px4[0] * _s12(i - 1, j, nz - 2) +
                                  px4[2] * _s12(i + 1, j, nz - 2) +
                                  px4[3] * _s12(i + 2, j, nz - 2)) +
                             dhpz4r[k][0] * _g_c(nz - 1) *
                                 (px4[1] * _s12(i, j, nz - 1) +
                                  px4[0] * _s12(i - 1, j, nz - 1) +
                                  px4[2] * _s12(i + 1, j, nz - 1) +
                                  px4[3] * _s12(i + 2, j, nz - 1))) -
              _f2_2(i, j) * (dhpz4r[k][8] * _g_c(nz - 9) *
                                 (py4[1] * _s22(i, j, nz - 9) +
                                  py4[0] * _s22(i, j - 1, nz - 9) +
                                  py4[2] * _s22(i, j + 1, nz - 9) +
                                  py4[3] * _s22(i, j + 2, nz - 9)) +
                             dhpz4r[k][7] * _g_c(nz - 8) *
                                 (py4[1] * _s22(i, j, nz - 8) +
                                  py4[0] * _s22(i, j - 1, nz - 8) +
                                  py4[2] * _s22(i, j + 1, nz - 8) +
                                  py4[3] * _s22(i, j + 2, nz - 8)) +
                             dhpz4r[k][6] * _g_c(nz - 7) *
                                 (py4[1] * _s22(i, j, nz - 7) +
                                  py4[0] * _s22(i, j - 1, nz - 7) +
                                  py4[2] * _s22(i, j + 1, nz - 7) +
                                  py4[3] * _s22(i, j + 2, nz - 7)) +
                             dhpz4r[k][5] * _g_c(nz - 6) *
                                 (py4[1] * _s22(i, j, nz - 6) +
                                  py4[0] * _s22(i, j - 1, nz - 6) +
                                  py4[2] * _s22(i, j + 1, nz - 6) +
                                  py4[3] * _s22(i, j + 2, nz - 6)) +
                             dhpz4r[k][4] * _g_c(nz - 5) *
                                 (py4[1] * _s22(i, j, nz - 5) +
                                  py4[0] * _s22(i, j - 1, nz - 5) +
                                  py4[2] * _s22(i, j + 1, nz - 5) +
                                  py4[3] * _s22(i, j + 2, nz - 5)) +
                             dhpz4r[k][3] * _g_c(nz - 4) *
                                 (py4[1] * _s22(i, j, nz - 4) +
                                  py4[0] * _s22(i, j - 1, nz - 4) +
                                  py4[2] * _s22(i, j + 1, nz - 4) +
                                  py4[3] * _s22(i, j + 2, nz - 4)) +
                             dhpz4r[k][2] * _g_c(nz - 3) *
                                 (py4[1] * _s22(i, j, nz - 3) +
                                  py4[0] * _s22(i, j - 1, nz - 3) +
                                  py4[2] * _s22(i, j + 1, nz - 3) +
                                  py4[3] * _s22(i, j + 2, nz - 3)) +
                             dhpz4r[k][1] * _g_c(nz - 2) *
                                 (py4[1] * _s22(i, j, nz - 2) +
                                  py4[0] * _s22(i, j - 1, nz - 2) +
                                  py4[2] * _s22(i, j + 1, nz - 2) +
                                  py4[3] * _s22(i, j + 2, nz - 2)) +
                             dhpz4r[k][0] * _g_c(nz - 1) *
                                 (py4[1] * _s22(i, j, nz - 1) +
                                  py4[0] * _s22(i, j - 1, nz - 1) +
                                  py4[2] * _s22(i, j + 1, nz - 1) +
                                  py4[3] * _s22(i, j + 2, nz - 1))))) *
        f_dcrj;
    _u3(i, j, nz - 1 - k) =
        (a * _u3(i, j, nz - 1 - k) +
         Ai3 *
             (dhy4[2] * _f_2(i, j) * _g3(nz - 1 - k) * _s23(i, j, nz - 1 - k) +
              dhy4[0] * _f_2(i, j - 2) * _g3(nz - 1 - k) *
                  _s23(i, j - 2, nz - 1 - k) +
              dhy4[1] * _f_2(i, j - 1) * _g3(nz - 1 - k) *
                  _s23(i, j - 1, nz - 1 - k) +
              dhy4[3] * _f_2(i, j + 1) * _g3(nz - 1 - k) *
                  _s23(i, j + 1, nz - 1 - k) +
              dx4[1] * _f_1(i, j) * _g3(nz - 1 - k) * _s13(i, j, nz - 1 - k) +
              dx4[0] * _f_1(i - 1, j) * _g3(nz - 1 - k) *
                  _s13(i - 1, j, nz - 1 - k) +
              dx4[2] * _f_1(i + 1, j) * _g3(nz - 1 - k) *
                  _s13(i + 1, j, nz - 1 - k) +
              dx4[3] * _f_1(i + 2, j) * _g3(nz - 1 - k) *
                  _s13(i + 2, j, nz - 1 - k) +
              dz4r[k][6] * _s33(i, j, nz - 7) +
              dz4r[k][5] * _s33(i, j, nz - 6) +
              dz4r[k][4] * _s33(i, j, nz - 5) +
              dz4r[k][3] * _s33(i, j, nz - 4) +
              dz4r[k][2] * _s33(i, j, nz - 3) +
              dz4r[k][1] * _s33(i, j, nz - 2) +
              dz4r[k][0] * _s33(i, j, nz - 1) -
              _f1_c(i, j) * (dphz4r[k][8] * _g(nz - 9) *
                                 (px4[1] * _s13(i, j, nz - 9) +
                                  px4[0] * _s13(i - 1, j, nz - 9) +
                                  px4[2] * _s13(i + 1, j, nz - 9) +
                                  px4[3] * _s13(i + 2, j, nz - 9)) +
                             dphz4r[k][7] * _g(nz - 8) *
                                 (px4[1] * _s13(i, j, nz - 8) +
                                  px4[0] * _s13(i - 1, j, nz - 8) +
                                  px4[2] * _s13(i + 1, j, nz - 8) +
                                  px4[3] * _s13(i + 2, j, nz - 8)) +
                             dphz4r[k][6] * _g(nz - 7) *
                                 (px4[1] * _s13(i, j, nz - 7) +
                                  px4[0] * _s13(i - 1, j, nz - 7) +
                                  px4[2] * _s13(i + 1, j, nz - 7) +
                                  px4[3] * _s13(i + 2, j, nz - 7)) +
                             dphz4r[k][5] * _g(nz - 6) *
                                 (px4[1] * _s13(i, j, nz - 6) +
                                  px4[0] * _s13(i - 1, j, nz - 6) +
                                  px4[2] * _s13(i + 1, j, nz - 6) +
                                  px4[3] * _s13(i + 2, j, nz - 6)) +
                             dphz4r[k][4] * _g(nz - 5) *
                                 (px4[1] * _s13(i, j, nz - 5) +
                                  px4[0] * _s13(i - 1, j, nz - 5) +
                                  px4[2] * _s13(i + 1, j, nz - 5) +
                                  px4[3] * _s13(i + 2, j, nz - 5)) +
                             dphz4r[k][3] * _g(nz - 4) *
                                 (px4[1] * _s13(i, j, nz - 4) +
                                  px4[0] * _s13(i - 1, j, nz - 4) +
                                  px4[2] * _s13(i + 1, j, nz - 4) +
                                  px4[3] * _s13(i + 2, j, nz - 4)) +
                             dphz4r[k][2] * _g(nz - 3) *
                                 (px4[1] * _s13(i, j, nz - 3) +
                                  px4[0] * _s13(i - 1, j, nz - 3) +
                                  px4[2] * _s13(i + 1, j, nz - 3) +
                                  px4[3] * _s13(i + 2, j, nz - 3)) +
                             dphz4r[k][1] * _g(nz - 2) *
                                 (px4[1] * _s13(i, j, nz - 2) +
                                  px4[0] * _s13(i - 1, j, nz - 2) +
                                  px4[2] * _s13(i + 1, j, nz - 2) +
                                  px4[3] * _s13(i + 2, j, nz - 2)) +
                             dphz4r[k][0] * _g(nz - 1) *
                                 (px4[1] * _s13(i, j, nz - 1) +
                                  px4[0] * _s13(i - 1, j, nz - 1) +
                                  px4[2] * _s13(i + 1, j, nz - 1) +
                                  px4[3] * _s13(i + 2, j, nz - 1))) -
              _f2_c(i, j) * (dphz4r[k][8] * _g(nz - 9) *
                                 (phy4[2] * _s23(i, j, nz - 9) +
                                  phy4[0] * _s23(i, j - 2, nz - 9) +
                                  phy4[1] * _s23(i, j - 1, nz - 9) +
                                  phy4[3] * _s23(i, j + 1, nz - 9)) +
                             dphz4r[k][7] * _g(nz - 8) *
                                 (phy4[2] * _s23(i, j, nz - 8) +
                                  phy4[0] * _s23(i, j - 2, nz - 8) +
                                  phy4[1] * _s23(i, j - 1, nz - 8) +
                                  phy4[3] * _s23(i, j + 1, nz - 8)) +
                             dphz4r[k][6] * _g(nz - 7) *
                                 (phy4[2] * _s23(i, j, nz - 7) +
                                  phy4[0] * _s23(i, j - 2, nz - 7) +
                                  phy4[1] * _s23(i, j - 1, nz - 7) +
                                  phy4[3] * _s23(i, j + 1, nz - 7)) +
                             dphz4r[k][5] * _g(nz - 6) *
                                 (phy4[2] * _s23(i, j, nz - 6) +
                                  phy4[0] * _s23(i, j - 2, nz - 6) +
                                  phy4[1] * _s23(i, j - 1, nz - 6) +
                                  phy4[3] * _s23(i, j + 1, nz - 6)) +
                             dphz4r[k][4] * _g(nz - 5) *
                                 (phy4[2] * _s23(i, j, nz - 5) +
                                  phy4[0] * _s23(i, j - 2, nz - 5) +
                                  phy4[1] * _s23(i, j - 1, nz - 5) +
                                  phy4[3] * _s23(i, j + 1, nz - 5)) +
                             dphz4r[k][3] * _g(nz - 4) *
                                 (phy4[2] * _s23(i, j, nz - 4) +
                                  phy4[0] * _s23(i, j - 2, nz - 4) +
                                  phy4[1] * _s23(i, j - 1, nz - 4) +
                                  phy4[3] * _s23(i, j + 1, nz - 4)) +
                             dphz4r[k][2] * _g(nz - 3) *
                                 (phy4[2] * _s23(i, j, nz - 3) +
                                  phy4[0] * _s23(i, j - 2, nz - 3) +
                                  phy4[1] * _s23(i, j - 1, nz - 3) +
                                  phy4[3] * _s23(i, j + 1, nz - 3)) +
                             dphz4r[k][1] * _g(nz - 2) *
                                 (phy4[2] * _s23(i, j, nz - 2) +
                                  phy4[0] * _s23(i, j - 2, nz - 2) +
                                  phy4[1] * _s23(i, j - 1, nz - 2) +
                                  phy4[3] * _s23(i, j + 1, nz - 2)) +
                             dphz4r[k][0] * _g(nz - 1) *
                                 (phy4[2] * _s23(i, j, nz - 1) +
                                  phy4[0] * _s23(i, j - 2, nz - 1) +
                                  phy4[1] * _s23(i, j - 1, nz - 1) +
                                  phy4[3] * _s23(i, j + 1, nz - 1))))) *
        f_dcrj;
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

__launch_bounds__(DTOPO_BUF_VEL_110_MAX_THREADS_PER_BLOCK,
                  DTOPO_BUF_VEL_110_MIN_BLOCKS_PER_SM)

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
        const float *__restrict__ u2, const float *__restrict__ u3,
        const float a, const float nu, const int nx, const int ny, const int nz,
        const int bj, const int ej, const int rj0) {
  const float phz4l[6][7] = {
      {0.8338228784688313, 0.1775123316429260, 0.1435067013076542,
       -0.1548419114194114, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.1813404047323969, 1.1246711188154426, -0.2933634518280757,
       -0.0126480717197637, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {-0.1331142706282399, 0.7930714675884345, 0.3131998767078508,
       0.0268429263319546, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000},
      {0.0969078556633046, -0.1539344946680898, 0.4486491202844389,
       0.6768738207821733, -0.0684963020618270, 0.0000000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, -0.0625000000000000,
       0.5625000000000000, 0.5625000000000000, -0.0625000000000000,
       0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       -0.0625000000000000, 0.5625000000000000, 0.5625000000000000,
       -0.0625000000000000}};
  const float phy4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float phx4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float dhpz4l[6][9] = {
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
  const float dhy4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const float dhx4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const float dhz4l[6][7] = {
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
  const float px4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float py4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float dx4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float dy4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float dphz4l[6][9] = {
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
  const float dz4l[6][8] = {
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
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ny)
    return;
  if (j >= ej)
    return;
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= 6)
    return;
#define _buf_u1(i, j, k)                                                       \
  buf_u1[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
#define _buf_u2(i, j, k)                                                       \
  buf_u2[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
#define _buf_u3(i, j, k)                                                       \
  buf_u3[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
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
  for (int i = 0; i < nx; ++i) {
    float rho1 = phz4l[k][0] * (phy4[2] * _rho(i, j + rj0, 0) +
                                phy4[0] * _rho(i, j + rj0 - 2, 0) +
                                phy4[1] * _rho(i, j + rj0 - 1, 0) +
                                phy4[3] * _rho(i, j + rj0 + 1, 0)) +
                 phz4l[k][1] * (phy4[2] * _rho(i, j + rj0, 1) +
                                phy4[0] * _rho(i, j + rj0 - 2, 1) +
                                phy4[1] * _rho(i, j + rj0 - 1, 1) +
                                phy4[3] * _rho(i, j + rj0 + 1, 1)) +
                 phz4l[k][2] * (phy4[2] * _rho(i, j + rj0, 2) +
                                phy4[0] * _rho(i, j + rj0 - 2, 2) +
                                phy4[1] * _rho(i, j + rj0 - 1, 2) +
                                phy4[3] * _rho(i, j + rj0 + 1, 2)) +
                 phz4l[k][3] * (phy4[2] * _rho(i, j + rj0, 3) +
                                phy4[0] * _rho(i, j + rj0 - 2, 3) +
                                phy4[1] * _rho(i, j + rj0 - 1, 3) +
                                phy4[3] * _rho(i, j + rj0 + 1, 3)) +
                 phz4l[k][4] * (phy4[2] * _rho(i, j + rj0, 4) +
                                phy4[0] * _rho(i, j + rj0 - 2, 4) +
                                phy4[1] * _rho(i, j + rj0 - 1, 4) +
                                phy4[3] * _rho(i, j + rj0 + 1, 4)) +
                 phz4l[k][5] * (phy4[2] * _rho(i, j + rj0, 5) +
                                phy4[0] * _rho(i, j + rj0 - 2, 5) +
                                phy4[1] * _rho(i, j + rj0 - 1, 5) +
                                phy4[3] * _rho(i, j + rj0 + 1, 5)) +
                 phz4l[k][6] * (phy4[2] * _rho(i, j + rj0, 6) +
                                phy4[0] * _rho(i, j + rj0 - 2, 6) +
                                phy4[1] * _rho(i, j + rj0 - 1, 6) +
                                phy4[3] * _rho(i, j + rj0 + 1, 6));
    float rho2 = phz4l[k][0] * (phx4[2] * _rho(i, j + rj0, 0) +
                                phx4[0] * _rho(i - 2, j + rj0, 0) +
                                phx4[1] * _rho(i - 1, j + rj0, 0) +
                                phx4[3] * _rho(i + 1, j + rj0, 0)) +
                 phz4l[k][1] * (phx4[2] * _rho(i, j + rj0, 1) +
                                phx4[0] * _rho(i - 2, j + rj0, 1) +
                                phx4[1] * _rho(i - 1, j + rj0, 1) +
                                phx4[3] * _rho(i + 1, j + rj0, 1)) +
                 phz4l[k][2] * (phx4[2] * _rho(i, j + rj0, 2) +
                                phx4[0] * _rho(i - 2, j + rj0, 2) +
                                phx4[1] * _rho(i - 1, j + rj0, 2) +
                                phx4[3] * _rho(i + 1, j + rj0, 2)) +
                 phz4l[k][3] * (phx4[2] * _rho(i, j + rj0, 3) +
                                phx4[0] * _rho(i - 2, j + rj0, 3) +
                                phx4[1] * _rho(i - 1, j + rj0, 3) +
                                phx4[3] * _rho(i + 1, j + rj0, 3)) +
                 phz4l[k][4] * (phx4[2] * _rho(i, j + rj0, 4) +
                                phx4[0] * _rho(i - 2, j + rj0, 4) +
                                phx4[1] * _rho(i - 1, j + rj0, 4) +
                                phx4[3] * _rho(i + 1, j + rj0, 4)) +
                 phz4l[k][5] * (phx4[2] * _rho(i, j + rj0, 5) +
                                phx4[0] * _rho(i - 2, j + rj0, 5) +
                                phx4[1] * _rho(i - 1, j + rj0, 5) +
                                phx4[3] * _rho(i + 1, j + rj0, 5)) +
                 phz4l[k][6] * (phx4[2] * _rho(i, j + rj0, 6) +
                                phx4[0] * _rho(i - 2, j + rj0, 6) +
                                phx4[1] * _rho(i - 1, j + rj0, 6) +
                                phx4[3] * _rho(i + 1, j + rj0, 6));
    float rho3 = phy4[2] * (phx4[2] * _rho(i, j + rj0, k) +
                            phx4[0] * _rho(i - 2, j + rj0, k) +
                            phx4[1] * _rho(i - 1, j + rj0, k) +
                            phx4[3] * _rho(i + 1, j + rj0, k)) +
                 phy4[0] * (phx4[2] * _rho(i, j + rj0 - 2, k) +
                            phx4[0] * _rho(i - 2, j + rj0 - 2, k) +
                            phx4[1] * _rho(i - 1, j + rj0 - 2, k) +
                            phx4[3] * _rho(i + 1, j + rj0 - 2, k)) +
                 phy4[1] * (phx4[2] * _rho(i, j + rj0 - 1, k) +
                            phx4[0] * _rho(i - 2, j + rj0 - 1, k) +
                            phx4[1] * _rho(i - 1, j + rj0 - 1, k) +
                            phx4[3] * _rho(i + 1, j + rj0 - 1, k)) +
                 phy4[3] * (phx4[2] * _rho(i, j + rj0 + 1, k) +
                            phx4[0] * _rho(i - 2, j + rj0 + 1, k) +
                            phx4[1] * _rho(i - 1, j + rj0 + 1, k) +
                            phx4[3] * _rho(i + 1, j + rj0 + 1, k));
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
             (dhx4[2] * _f_c(i, j + rj0) * _g3_c(k) * _s11(i, j + rj0, k) +
              dhx4[0] * _f_c(i - 2, j + rj0) * _g3_c(k) *
                  _s11(i - 2, j + rj0, k) +
              dhx4[1] * _f_c(i - 1, j + rj0) * _g3_c(k) *
                  _s11(i - 1, j + rj0, k) +
              dhx4[3] * _f_c(i + 1, j + rj0) * _g3_c(k) *
                  _s11(i + 1, j + rj0, k) +
              dhy4[2] * _f(i, j + rj0) * _g3_c(k) * _s12(i, j + rj0, k) +
              dhy4[0] * _f(i, j + rj0 - 2) * _g3_c(k) *
                  _s12(i, j + rj0 - 2, k) +
              dhy4[1] * _f(i, j + rj0 - 1) * _g3_c(k) *
                  _s12(i, j + rj0 - 1, k) +
              dhy4[3] * _f(i, j + rj0 + 1) * _g3_c(k) *
                  _s12(i, j + rj0 + 1, k) +
              dhz4l[k][0] * _s13(i, j + rj0, 0) +
              dhz4l[k][1] * _s13(i, j + rj0, 1) +
              dhz4l[k][2] * _s13(i, j + rj0, 2) +
              dhz4l[k][3] * _s13(i, j + rj0, 3) +
              dhz4l[k][4] * _s13(i, j + rj0, 4) +
              dhz4l[k][5] * _s13(i, j + rj0, 5) +
              dhz4l[k][6] * _s13(i, j + rj0, 6) -
              _f1_1(i, j + rj0) * (dhpz4l[k][0] * _g_c(0) *
                                       (phx4[2] * _s11(i, j + rj0, 0) +
                                        phx4[0] * _s11(i - 2, j + rj0, 0) +
                                        phx4[1] * _s11(i - 1, j + rj0, 0) +
                                        phx4[3] * _s11(i + 1, j + rj0, 0)) +
                                   dhpz4l[k][1] * _g_c(1) *
                                       (phx4[2] * _s11(i, j + rj0, 1) +
                                        phx4[0] * _s11(i - 2, j + rj0, 1) +
                                        phx4[1] * _s11(i - 1, j + rj0, 1) +
                                        phx4[3] * _s11(i + 1, j + rj0, 1)) +
                                   dhpz4l[k][2] * _g_c(2) *
                                       (phx4[2] * _s11(i, j + rj0, 2) +
                                        phx4[0] * _s11(i - 2, j + rj0, 2) +
                                        phx4[1] * _s11(i - 1, j + rj0, 2) +
                                        phx4[3] * _s11(i + 1, j + rj0, 2)) +
                                   dhpz4l[k][3] * _g_c(3) *
                                       (phx4[2] * _s11(i, j + rj0, 3) +
                                        phx4[0] * _s11(i - 2, j + rj0, 3) +
                                        phx4[1] * _s11(i - 1, j + rj0, 3) +
                                        phx4[3] * _s11(i + 1, j + rj0, 3)) +
                                   dhpz4l[k][4] * _g_c(4) *
                                       (phx4[2] * _s11(i, j + rj0, 4) +
                                        phx4[0] * _s11(i - 2, j + rj0, 4) +
                                        phx4[1] * _s11(i - 1, j + rj0, 4) +
                                        phx4[3] * _s11(i + 1, j + rj0, 4)) +
                                   dhpz4l[k][5] * _g_c(5) *
                                       (phx4[2] * _s11(i, j + rj0, 5) +
                                        phx4[0] * _s11(i - 2, j + rj0, 5) +
                                        phx4[1] * _s11(i - 1, j + rj0, 5) +
                                        phx4[3] * _s11(i + 1, j + rj0, 5)) +
                                   dhpz4l[k][6] * _g_c(6) *
                                       (phx4[2] * _s11(i, j + rj0, 6) +
                                        phx4[0] * _s11(i - 2, j + rj0, 6) +
                                        phx4[1] * _s11(i - 1, j + rj0, 6) +
                                        phx4[3] * _s11(i + 1, j + rj0, 6)) +
                                   dhpz4l[k][7] * _g_c(7) *
                                       (phx4[2] * _s11(i, j + rj0, 7) +
                                        phx4[0] * _s11(i - 2, j + rj0, 7) +
                                        phx4[1] * _s11(i - 1, j + rj0, 7) +
                                        phx4[3] * _s11(i + 1, j + rj0, 7)) +
                                   dhpz4l[k][8] * _g_c(8) *
                                       (phx4[2] * _s11(i, j + rj0, 8) +
                                        phx4[0] * _s11(i - 2, j + rj0, 8) +
                                        phx4[1] * _s11(i - 1, j + rj0, 8) +
                                        phx4[3] * _s11(i + 1, j + rj0, 8))) -
              _f2_1(i, j + rj0) * (dhpz4l[k][0] * _g_c(0) *
                                       (phy4[2] * _s12(i, j + rj0, 0) +
                                        phy4[0] * _s12(i, j + rj0 - 2, 0) +
                                        phy4[1] * _s12(i, j + rj0 - 1, 0) +
                                        phy4[3] * _s12(i, j + rj0 + 1, 0)) +
                                   dhpz4l[k][1] * _g_c(1) *
                                       (phy4[2] * _s12(i, j + rj0, 1) +
                                        phy4[0] * _s12(i, j + rj0 - 2, 1) +
                                        phy4[1] * _s12(i, j + rj0 - 1, 1) +
                                        phy4[3] * _s12(i, j + rj0 + 1, 1)) +
                                   dhpz4l[k][2] * _g_c(2) *
                                       (phy4[2] * _s12(i, j + rj0, 2) +
                                        phy4[0] * _s12(i, j + rj0 - 2, 2) +
                                        phy4[1] * _s12(i, j + rj0 - 1, 2) +
                                        phy4[3] * _s12(i, j + rj0 + 1, 2)) +
                                   dhpz4l[k][3] * _g_c(3) *
                                       (phy4[2] * _s12(i, j + rj0, 3) +
                                        phy4[0] * _s12(i, j + rj0 - 2, 3) +
                                        phy4[1] * _s12(i, j + rj0 - 1, 3) +
                                        phy4[3] * _s12(i, j + rj0 + 1, 3)) +
                                   dhpz4l[k][4] * _g_c(4) *
                                       (phy4[2] * _s12(i, j + rj0, 4) +
                                        phy4[0] * _s12(i, j + rj0 - 2, 4) +
                                        phy4[1] * _s12(i, j + rj0 - 1, 4) +
                                        phy4[3] * _s12(i, j + rj0 + 1, 4)) +
                                   dhpz4l[k][5] * _g_c(5) *
                                       (phy4[2] * _s12(i, j + rj0, 5) +
                                        phy4[0] * _s12(i, j + rj0 - 2, 5) +
                                        phy4[1] * _s12(i, j + rj0 - 1, 5) +
                                        phy4[3] * _s12(i, j + rj0 + 1, 5)) +
                                   dhpz4l[k][6] * _g_c(6) *
                                       (phy4[2] * _s12(i, j + rj0, 6) +
                                        phy4[0] * _s12(i, j + rj0 - 2, 6) +
                                        phy4[1] * _s12(i, j + rj0 - 1, 6) +
                                        phy4[3] * _s12(i, j + rj0 + 1, 6)) +
                                   dhpz4l[k][7] * _g_c(7) *
                                       (phy4[2] * _s12(i, j + rj0, 7) +
                                        phy4[0] * _s12(i, j + rj0 - 2, 7) +
                                        phy4[1] * _s12(i, j + rj0 - 1, 7) +
                                        phy4[3] * _s12(i, j + rj0 + 1, 7)) +
                                   dhpz4l[k][8] * _g_c(8) *
                                       (phy4[2] * _s12(i, j + rj0, 8) +
                                        phy4[0] * _s12(i, j + rj0 - 2, 8) +
                                        phy4[1] * _s12(i, j + rj0 - 1, 8) +
                                        phy4[3] * _s12(i, j + rj0 + 1, 8))))) *
        f_dcrj;
    _buf_u2(i, j, k) =
        (a * _u2(i, j + rj0, k) +
         Ai2 *
             (dhz4l[k][0] * _s23(i, j + rj0, 0) +
              dhz4l[k][1] * _s23(i, j + rj0, 1) +
              dhz4l[k][2] * _s23(i, j + rj0, 2) +
              dhz4l[k][3] * _s23(i, j + rj0, 3) +
              dhz4l[k][4] * _s23(i, j + rj0, 4) +
              dhz4l[k][5] * _s23(i, j + rj0, 5) +
              dhz4l[k][6] * _s23(i, j + rj0, 6) +
              dx4[1] * _f(i, j + rj0) * _g3_c(k) * _s12(i, j + rj0, k) +
              dx4[0] * _f(i - 1, j + rj0) * _g3_c(k) * _s12(i - 1, j + rj0, k) +
              dx4[2] * _f(i + 1, j + rj0) * _g3_c(k) * _s12(i + 1, j + rj0, k) +
              dx4[3] * _f(i + 2, j + rj0) * _g3_c(k) * _s12(i + 2, j + rj0, k) +
              dy4[1] * _f_c(i, j + rj0) * _g3_c(k) * _s22(i, j + rj0, k) +
              dy4[0] * _f_c(i, j + rj0 - 1) * _g3_c(k) *
                  _s22(i, j + rj0 - 1, k) +
              dy4[2] * _f_c(i, j + rj0 + 1) * _g3_c(k) *
                  _s22(i, j + rj0 + 1, k) +
              dy4[3] * _f_c(i, j + rj0 + 2) * _g3_c(k) *
                  _s22(i, j + rj0 + 2, k) -
              _f1_2(i, j + rj0) * (dhpz4l[k][0] * _g_c(0) *
                                       (px4[1] * _s12(i, j + rj0, 0) +
                                        px4[0] * _s12(i - 1, j + rj0, 0) +
                                        px4[2] * _s12(i + 1, j + rj0, 0) +
                                        px4[3] * _s12(i + 2, j + rj0, 0)) +
                                   dhpz4l[k][1] * _g_c(1) *
                                       (px4[1] * _s12(i, j + rj0, 1) +
                                        px4[0] * _s12(i - 1, j + rj0, 1) +
                                        px4[2] * _s12(i + 1, j + rj0, 1) +
                                        px4[3] * _s12(i + 2, j + rj0, 1)) +
                                   dhpz4l[k][2] * _g_c(2) *
                                       (px4[1] * _s12(i, j + rj0, 2) +
                                        px4[0] * _s12(i - 1, j + rj0, 2) +
                                        px4[2] * _s12(i + 1, j + rj0, 2) +
                                        px4[3] * _s12(i + 2, j + rj0, 2)) +
                                   dhpz4l[k][3] * _g_c(3) *
                                       (px4[1] * _s12(i, j + rj0, 3) +
                                        px4[0] * _s12(i - 1, j + rj0, 3) +
                                        px4[2] * _s12(i + 1, j + rj0, 3) +
                                        px4[3] * _s12(i + 2, j + rj0, 3)) +
                                   dhpz4l[k][4] * _g_c(4) *
                                       (px4[1] * _s12(i, j + rj0, 4) +
                                        px4[0] * _s12(i - 1, j + rj0, 4) +
                                        px4[2] * _s12(i + 1, j + rj0, 4) +
                                        px4[3] * _s12(i + 2, j + rj0, 4)) +
                                   dhpz4l[k][5] * _g_c(5) *
                                       (px4[1] * _s12(i, j + rj0, 5) +
                                        px4[0] * _s12(i - 1, j + rj0, 5) +
                                        px4[2] * _s12(i + 1, j + rj0, 5) +
                                        px4[3] * _s12(i + 2, j + rj0, 5)) +
                                   dhpz4l[k][6] * _g_c(6) *
                                       (px4[1] * _s12(i, j + rj0, 6) +
                                        px4[0] * _s12(i - 1, j + rj0, 6) +
                                        px4[2] * _s12(i + 1, j + rj0, 6) +
                                        px4[3] * _s12(i + 2, j + rj0, 6)) +
                                   dhpz4l[k][7] * _g_c(7) *
                                       (px4[1] * _s12(i, j + rj0, 7) +
                                        px4[0] * _s12(i - 1, j + rj0, 7) +
                                        px4[2] * _s12(i + 1, j + rj0, 7) +
                                        px4[3] * _s12(i + 2, j + rj0, 7)) +
                                   dhpz4l[k][8] * _g_c(8) *
                                       (px4[1] * _s12(i, j + rj0, 8) +
                                        px4[0] * _s12(i - 1, j + rj0, 8) +
                                        px4[2] * _s12(i + 1, j + rj0, 8) +
                                        px4[3] * _s12(i + 2, j + rj0, 8))) -
              _f2_2(i, j + rj0) * (dhpz4l[k][0] * _g_c(0) *
                                       (py4[1] * _s22(i, j + rj0, 0) +
                                        py4[0] * _s22(i, j + rj0 - 1, 0) +
                                        py4[2] * _s22(i, j + rj0 + 1, 0) +
                                        py4[3] * _s22(i, j + rj0 + 2, 0)) +
                                   dhpz4l[k][1] * _g_c(1) *
                                       (py4[1] * _s22(i, j + rj0, 1) +
                                        py4[0] * _s22(i, j + rj0 - 1, 1) +
                                        py4[2] * _s22(i, j + rj0 + 1, 1) +
                                        py4[3] * _s22(i, j + rj0 + 2, 1)) +
                                   dhpz4l[k][2] * _g_c(2) *
                                       (py4[1] * _s22(i, j + rj0, 2) +
                                        py4[0] * _s22(i, j + rj0 - 1, 2) +
                                        py4[2] * _s22(i, j + rj0 + 1, 2) +
                                        py4[3] * _s22(i, j + rj0 + 2, 2)) +
                                   dhpz4l[k][3] * _g_c(3) *
                                       (py4[1] * _s22(i, j + rj0, 3) +
                                        py4[0] * _s22(i, j + rj0 - 1, 3) +
                                        py4[2] * _s22(i, j + rj0 + 1, 3) +
                                        py4[3] * _s22(i, j + rj0 + 2, 3)) +
                                   dhpz4l[k][4] * _g_c(4) *
                                       (py4[1] * _s22(i, j + rj0, 4) +
                                        py4[0] * _s22(i, j + rj0 - 1, 4) +
                                        py4[2] * _s22(i, j + rj0 + 1, 4) +
                                        py4[3] * _s22(i, j + rj0 + 2, 4)) +
                                   dhpz4l[k][5] * _g_c(5) *
                                       (py4[1] * _s22(i, j + rj0, 5) +
                                        py4[0] * _s22(i, j + rj0 - 1, 5) +
                                        py4[2] * _s22(i, j + rj0 + 1, 5) +
                                        py4[3] * _s22(i, j + rj0 + 2, 5)) +
                                   dhpz4l[k][6] * _g_c(6) *
                                       (py4[1] * _s22(i, j + rj0, 6) +
                                        py4[0] * _s22(i, j + rj0 - 1, 6) +
                                        py4[2] * _s22(i, j + rj0 + 1, 6) +
                                        py4[3] * _s22(i, j + rj0 + 2, 6)) +
                                   dhpz4l[k][7] * _g_c(7) *
                                       (py4[1] * _s22(i, j + rj0, 7) +
                                        py4[0] * _s22(i, j + rj0 - 1, 7) +
                                        py4[2] * _s22(i, j + rj0 + 1, 7) +
                                        py4[3] * _s22(i, j + rj0 + 2, 7)) +
                                   dhpz4l[k][8] * _g_c(8) *
                                       (py4[1] * _s22(i, j + rj0, 8) +
                                        py4[0] * _s22(i, j + rj0 - 1, 8) +
                                        py4[2] * _s22(i, j + rj0 + 1, 8) +
                                        py4[3] * _s22(i, j + rj0 + 2, 8))))) *
        f_dcrj;
    _buf_u3(i, j, k) =
        (a * _u3(i, j + rj0, k) +
         Ai3 *
             (dhy4[2] * _f_2(i, j + rj0) * _g3(k) * _s23(i, j + rj0, k) +
              dhy4[0] * _f_2(i, j + rj0 - 2) * _g3(k) *
                  _s23(i, j + rj0 - 2, k) +
              dhy4[1] * _f_2(i, j + rj0 - 1) * _g3(k) *
                  _s23(i, j + rj0 - 1, k) +
              dhy4[3] * _f_2(i, j + rj0 + 1) * _g3(k) *
                  _s23(i, j + rj0 + 1, k) +
              dx4[1] * _f_1(i, j + rj0) * _g3(k) * _s13(i, j + rj0, k) +
              dx4[0] * _f_1(i - 1, j + rj0) * _g3(k) * _s13(i - 1, j + rj0, k) +
              dx4[2] * _f_1(i + 1, j + rj0) * _g3(k) * _s13(i + 1, j + rj0, k) +
              dx4[3] * _f_1(i + 2, j + rj0) * _g3(k) * _s13(i + 2, j + rj0, k) +
              dz4l[k][0] * _s33(i, j + rj0, 0) +
              dz4l[k][1] * _s33(i, j + rj0, 1) +
              dz4l[k][2] * _s33(i, j + rj0, 2) +
              dz4l[k][3] * _s33(i, j + rj0, 3) +
              dz4l[k][4] * _s33(i, j + rj0, 4) +
              dz4l[k][5] * _s33(i, j + rj0, 5) +
              dz4l[k][6] * _s33(i, j + rj0, 6) +
              dz4l[k][7] * _s33(i, j + rj0, 7) -
              _f1_c(i, j + rj0) * (dphz4l[k][0] * _g(0) *
                                       (px4[1] * _s13(i, j + rj0, 0) +
                                        px4[0] * _s13(i - 1, j + rj0, 0) +
                                        px4[2] * _s13(i + 1, j + rj0, 0) +
                                        px4[3] * _s13(i + 2, j + rj0, 0)) +
                                   dphz4l[k][1] * _g(1) *
                                       (px4[1] * _s13(i, j + rj0, 1) +
                                        px4[0] * _s13(i - 1, j + rj0, 1) +
                                        px4[2] * _s13(i + 1, j + rj0, 1) +
                                        px4[3] * _s13(i + 2, j + rj0, 1)) +
                                   dphz4l[k][2] * _g(2) *
                                       (px4[1] * _s13(i, j + rj0, 2) +
                                        px4[0] * _s13(i - 1, j + rj0, 2) +
                                        px4[2] * _s13(i + 1, j + rj0, 2) +
                                        px4[3] * _s13(i + 2, j + rj0, 2)) +
                                   dphz4l[k][3] * _g(3) *
                                       (px4[1] * _s13(i, j + rj0, 3) +
                                        px4[0] * _s13(i - 1, j + rj0, 3) +
                                        px4[2] * _s13(i + 1, j + rj0, 3) +
                                        px4[3] * _s13(i + 2, j + rj0, 3)) +
                                   dphz4l[k][4] * _g(4) *
                                       (px4[1] * _s13(i, j + rj0, 4) +
                                        px4[0] * _s13(i - 1, j + rj0, 4) +
                                        px4[2] * _s13(i + 1, j + rj0, 4) +
                                        px4[3] * _s13(i + 2, j + rj0, 4)) +
                                   dphz4l[k][5] * _g(5) *
                                       (px4[1] * _s13(i, j + rj0, 5) +
                                        px4[0] * _s13(i - 1, j + rj0, 5) +
                                        px4[2] * _s13(i + 1, j + rj0, 5) +
                                        px4[3] * _s13(i + 2, j + rj0, 5)) +
                                   dphz4l[k][6] * _g(6) *
                                       (px4[1] * _s13(i, j + rj0, 6) +
                                        px4[0] * _s13(i - 1, j + rj0, 6) +
                                        px4[2] * _s13(i + 1, j + rj0, 6) +
                                        px4[3] * _s13(i + 2, j + rj0, 6)) +
                                   dphz4l[k][7] * _g(7) *
                                       (px4[1] * _s13(i, j + rj0, 7) +
                                        px4[0] * _s13(i - 1, j + rj0, 7) +
                                        px4[2] * _s13(i + 1, j + rj0, 7) +
                                        px4[3] * _s13(i + 2, j + rj0, 7)) +
                                   dphz4l[k][8] * _g(8) *
                                       (px4[1] * _s13(i, j + rj0, 8) +
                                        px4[0] * _s13(i - 1, j + rj0, 8) +
                                        px4[2] * _s13(i + 1, j + rj0, 8) +
                                        px4[3] * _s13(i + 2, j + rj0, 8))) -
              _f2_c(i, j + rj0) * (dphz4l[k][0] * _g(0) *
                                       (phy4[2] * _s23(i, j + rj0, 0) +
                                        phy4[0] * _s23(i, j + rj0 - 2, 0) +
                                        phy4[1] * _s23(i, j + rj0 - 1, 0) +
                                        phy4[3] * _s23(i, j + rj0 + 1, 0)) +
                                   dphz4l[k][1] * _g(1) *
                                       (phy4[2] * _s23(i, j + rj0, 1) +
                                        phy4[0] * _s23(i, j + rj0 - 2, 1) +
                                        phy4[1] * _s23(i, j + rj0 - 1, 1) +
                                        phy4[3] * _s23(i, j + rj0 + 1, 1)) +
                                   dphz4l[k][2] * _g(2) *
                                       (phy4[2] * _s23(i, j + rj0, 2) +
                                        phy4[0] * _s23(i, j + rj0 - 2, 2) +
                                        phy4[1] * _s23(i, j + rj0 - 1, 2) +
                                        phy4[3] * _s23(i, j + rj0 + 1, 2)) +
                                   dphz4l[k][3] * _g(3) *
                                       (phy4[2] * _s23(i, j + rj0, 3) +
                                        phy4[0] * _s23(i, j + rj0 - 2, 3) +
                                        phy4[1] * _s23(i, j + rj0 - 1, 3) +
                                        phy4[3] * _s23(i, j + rj0 + 1, 3)) +
                                   dphz4l[k][4] * _g(4) *
                                       (phy4[2] * _s23(i, j + rj0, 4) +
                                        phy4[0] * _s23(i, j + rj0 - 2, 4) +
                                        phy4[1] * _s23(i, j + rj0 - 1, 4) +
                                        phy4[3] * _s23(i, j + rj0 + 1, 4)) +
                                   dphz4l[k][5] * _g(5) *
                                       (phy4[2] * _s23(i, j + rj0, 5) +
                                        phy4[0] * _s23(i, j + rj0 - 2, 5) +
                                        phy4[1] * _s23(i, j + rj0 - 1, 5) +
                                        phy4[3] * _s23(i, j + rj0 + 1, 5)) +
                                   dphz4l[k][6] * _g(6) *
                                       (phy4[2] * _s23(i, j + rj0, 6) +
                                        phy4[0] * _s23(i, j + rj0 - 2, 6) +
                                        phy4[1] * _s23(i, j + rj0 - 1, 6) +
                                        phy4[3] * _s23(i, j + rj0 + 1, 6)) +
                                   dphz4l[k][7] * _g(7) *
                                       (phy4[2] * _s23(i, j + rj0, 7) +
                                        phy4[0] * _s23(i, j + rj0 - 2, 7) +
                                        phy4[1] * _s23(i, j + rj0 - 1, 7) +
                                        phy4[3] * _s23(i, j + rj0 + 1, 7)) +
                                   dphz4l[k][8] * _g(8) *
                                       (phy4[2] * _s23(i, j + rj0, 8) +
                                        phy4[0] * _s23(i, j + rj0 - 2, 8) +
                                        phy4[1] * _s23(i, j + rj0 - 1, 8) +
                                        phy4[3] * _s23(i, j + rj0 + 1, 8))))) *
        f_dcrj;
  }
#undef _buf_u1
#undef _buf_u2
#undef _buf_u3
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

__launch_bounds__(DTOPO_BUF_VEL_111_MAX_THREADS_PER_BLOCK,
                  DTOPO_BUF_VEL_111_MIN_BLOCKS_PER_SM)

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
        const float *__restrict__ u2, const float *__restrict__ u3,
        const float a, const float nu, const int nx, const int ny, const int nz,
        const int bj, const int ej, const int rj0) {
  const float phz4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float phy4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float phx4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float dhpz4[7] = {-0.0026041666666667, 0.0937500000000000,
                          -0.6796875000000000, 0.0000000000000000,
                          0.6796875000000000,  -0.0937500000000000,
                          0.0026041666666667};
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
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ny)
    return;
  if (j >= ej)
    return;
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= nz - 12)
    return;
#define _buf_u1(i, j, k)                                                       \
  buf_u1[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
#define _buf_u2(i, j, k)                                                       \
  buf_u2[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
#define _buf_u3(i, j, k)                                                       \
  buf_u3[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
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
  for (int i = 0; i < nx; ++i) {
    float rho1 = phz4[0] * (phy4[2] * _rho(i, j + rj0, k + 4) +
                            phy4[0] * _rho(i, j + rj0 - 2, k + 4) +
                            phy4[1] * _rho(i, j + rj0 - 1, k + 4) +
                            phy4[3] * _rho(i, j + rj0 + 1, k + 4)) +
                 phz4[1] * (phy4[2] * _rho(i, j + rj0, k + 5) +
                            phy4[0] * _rho(i, j + rj0 - 2, k + 5) +
                            phy4[1] * _rho(i, j + rj0 - 1, k + 5) +
                            phy4[3] * _rho(i, j + rj0 + 1, k + 5)) +
                 phz4[2] * (phy4[2] * _rho(i, j + rj0, k + 6) +
                            phy4[0] * _rho(i, j + rj0 - 2, k + 6) +
                            phy4[1] * _rho(i, j + rj0 - 1, k + 6) +
                            phy4[3] * _rho(i, j + rj0 + 1, k + 6)) +
                 phz4[3] * (phy4[2] * _rho(i, j + rj0, k + 7) +
                            phy4[0] * _rho(i, j + rj0 - 2, k + 7) +
                            phy4[1] * _rho(i, j + rj0 - 1, k + 7) +
                            phy4[3] * _rho(i, j + rj0 + 1, k + 7));
    float rho2 = phz4[0] * (phx4[2] * _rho(i, j + rj0, k + 4) +
                            phx4[0] * _rho(i - 2, j + rj0, k + 4) +
                            phx4[1] * _rho(i - 1, j + rj0, k + 4) +
                            phx4[3] * _rho(i + 1, j + rj0, k + 4)) +
                 phz4[1] * (phx4[2] * _rho(i, j + rj0, k + 5) +
                            phx4[0] * _rho(i - 2, j + rj0, k + 5) +
                            phx4[1] * _rho(i - 1, j + rj0, k + 5) +
                            phx4[3] * _rho(i + 1, j + rj0, k + 5)) +
                 phz4[2] * (phx4[2] * _rho(i, j + rj0, k + 6) +
                            phx4[0] * _rho(i - 2, j + rj0, k + 6) +
                            phx4[1] * _rho(i - 1, j + rj0, k + 6) +
                            phx4[3] * _rho(i + 1, j + rj0, k + 6)) +
                 phz4[3] * (phx4[2] * _rho(i, j + rj0, k + 7) +
                            phx4[0] * _rho(i - 2, j + rj0, k + 7) +
                            phx4[1] * _rho(i - 1, j + rj0, k + 7) +
                            phx4[3] * _rho(i + 1, j + rj0, k + 7));
    float rho3 = phy4[2] * (phx4[2] * _rho(i, j + rj0, k + 6) +
                            phx4[0] * _rho(i - 2, j + rj0, k + 6) +
                            phx4[1] * _rho(i - 1, j + rj0, k + 6) +
                            phx4[3] * _rho(i + 1, j + rj0, k + 6)) +
                 phy4[0] * (phx4[2] * _rho(i, j + rj0 - 2, k + 6) +
                            phx4[0] * _rho(i - 2, j + rj0 - 2, k + 6) +
                            phx4[1] * _rho(i - 1, j + rj0 - 2, k + 6) +
                            phx4[3] * _rho(i + 1, j + rj0 - 2, k + 6)) +
                 phy4[1] * (phx4[2] * _rho(i, j + rj0 - 1, k + 6) +
                            phx4[0] * _rho(i - 2, j + rj0 - 1, k + 6) +
                            phx4[1] * _rho(i - 1, j + rj0 - 1, k + 6) +
                            phx4[3] * _rho(i + 1, j + rj0 - 1, k + 6)) +
                 phy4[3] * (phx4[2] * _rho(i, j + rj0 + 1, k + 6) +
                            phx4[0] * _rho(i - 2, j + rj0 + 1, k + 6) +
                            phx4[1] * _rho(i - 1, j + rj0 + 1, k + 6) +
                            phx4[3] * _rho(i + 1, j + rj0 + 1, k + 6));
    float Ai1 = _f_1(i, j + rj0) * _g3_c(k + 6) * rho1;
    Ai1 = nu * 1.0 / Ai1;
    float Ai2 = _f_2(i, j + rj0) * _g3_c(k + 6) * rho2;
    Ai2 = nu * 1.0 / Ai2;
    float Ai3 = _f_c(i, j + rj0) * _g3(k + 6) * rho3;
    Ai3 = nu * 1.0 / Ai3;
    float f_dcrj = _dcrjx(i) * _dcrjy(j + rj0) * _dcrjz(k + 6);
    _buf_u1(i, j, k + 6) =
        (a * _u1(i, j + rj0, k + 6) +
         Ai1 * (dhx4[2] * _f_c(i, j + rj0) * _g3_c(k + 6) *
                    _s11(i, j + rj0, k + 6) +
                dhx4[0] * _f_c(i - 2, j + rj0) * _g3_c(k + 6) *
                    _s11(i - 2, j + rj0, k + 6) +
                dhx4[1] * _f_c(i - 1, j + rj0) * _g3_c(k + 6) *
                    _s11(i - 1, j + rj0, k + 6) +
                dhx4[3] * _f_c(i + 1, j + rj0) * _g3_c(k + 6) *
                    _s11(i + 1, j + rj0, k + 6) +
                dhy4[2] * _f(i, j + rj0) * _g3_c(k + 6) *
                    _s12(i, j + rj0, k + 6) +
                dhy4[0] * _f(i, j + rj0 - 2) * _g3_c(k + 6) *
                    _s12(i, j + rj0 - 2, k + 6) +
                dhy4[1] * _f(i, j + rj0 - 1) * _g3_c(k + 6) *
                    _s12(i, j + rj0 - 1, k + 6) +
                dhy4[3] * _f(i, j + rj0 + 1) * _g3_c(k + 6) *
                    _s12(i, j + rj0 + 1, k + 6) +
                dhz4[0] * _s13(i, j + rj0, k + 4) +
                dhz4[1] * _s13(i, j + rj0, k + 5) +
                dhz4[2] * _s13(i, j + rj0, k + 6) +
                dhz4[3] * _s13(i, j + rj0, k + 7) -
                _f1_1(i, j + rj0) *
                    (dhpz4[0] * _g_c(k + 3) *
                         (phx4[2] * _s11(i, j + rj0, k + 3) +
                          phx4[0] * _s11(i - 2, j + rj0, k + 3) +
                          phx4[1] * _s11(i - 1, j + rj0, k + 3) +
                          phx4[3] * _s11(i + 1, j + rj0, k + 3)) +
                     dhpz4[1] * _g_c(k + 4) *
                         (phx4[2] * _s11(i, j + rj0, k + 4) +
                          phx4[0] * _s11(i - 2, j + rj0, k + 4) +
                          phx4[1] * _s11(i - 1, j + rj0, k + 4) +
                          phx4[3] * _s11(i + 1, j + rj0, k + 4)) +
                     dhpz4[2] * _g_c(k + 5) *
                         (phx4[2] * _s11(i, j + rj0, k + 5) +
                          phx4[0] * _s11(i - 2, j + rj0, k + 5) +
                          phx4[1] * _s11(i - 1, j + rj0, k + 5) +
                          phx4[3] * _s11(i + 1, j + rj0, k + 5)) +
                     dhpz4[3] * _g_c(k + 6) *
                         (phx4[2] * _s11(i, j + rj0, k + 6) +
                          phx4[0] * _s11(i - 2, j + rj0, k + 6) +
                          phx4[1] * _s11(i - 1, j + rj0, k + 6) +
                          phx4[3] * _s11(i + 1, j + rj0, k + 6)) +
                     dhpz4[4] * _g_c(k + 7) *
                         (phx4[2] * _s11(i, j + rj0, k + 7) +
                          phx4[0] * _s11(i - 2, j + rj0, k + 7) +
                          phx4[1] * _s11(i - 1, j + rj0, k + 7) +
                          phx4[3] * _s11(i + 1, j + rj0, k + 7)) +
                     dhpz4[5] * _g_c(k + 8) *
                         (phx4[2] * _s11(i, j + rj0, k + 8) +
                          phx4[0] * _s11(i - 2, j + rj0, k + 8) +
                          phx4[1] * _s11(i - 1, j + rj0, k + 8) +
                          phx4[3] * _s11(i + 1, j + rj0, k + 8)) +
                     dhpz4[6] * _g_c(k + 9) *
                         (phx4[2] * _s11(i, j + rj0, k + 9) +
                          phx4[0] * _s11(i - 2, j + rj0, k + 9) +
                          phx4[1] * _s11(i - 1, j + rj0, k + 9) +
                          phx4[3] * _s11(i + 1, j + rj0, k + 9))) -
                _f2_1(i, j + rj0) *
                    (dhpz4[0] * _g_c(k + 3) *
                         (phy4[2] * _s12(i, j + rj0, k + 3) +
                          phy4[0] * _s12(i, j + rj0 - 2, k + 3) +
                          phy4[1] * _s12(i, j + rj0 - 1, k + 3) +
                          phy4[3] * _s12(i, j + rj0 + 1, k + 3)) +
                     dhpz4[1] * _g_c(k + 4) *
                         (phy4[2] * _s12(i, j + rj0, k + 4) +
                          phy4[0] * _s12(i, j + rj0 - 2, k + 4) +
                          phy4[1] * _s12(i, j + rj0 - 1, k + 4) +
                          phy4[3] * _s12(i, j + rj0 + 1, k + 4)) +
                     dhpz4[2] * _g_c(k + 5) *
                         (phy4[2] * _s12(i, j + rj0, k + 5) +
                          phy4[0] * _s12(i, j + rj0 - 2, k + 5) +
                          phy4[1] * _s12(i, j + rj0 - 1, k + 5) +
                          phy4[3] * _s12(i, j + rj0 + 1, k + 5)) +
                     dhpz4[3] * _g_c(k + 6) *
                         (phy4[2] * _s12(i, j + rj0, k + 6) +
                          phy4[0] * _s12(i, j + rj0 - 2, k + 6) +
                          phy4[1] * _s12(i, j + rj0 - 1, k + 6) +
                          phy4[3] * _s12(i, j + rj0 + 1, k + 6)) +
                     dhpz4[4] * _g_c(k + 7) *
                         (phy4[2] * _s12(i, j + rj0, k + 7) +
                          phy4[0] * _s12(i, j + rj0 - 2, k + 7) +
                          phy4[1] * _s12(i, j + rj0 - 1, k + 7) +
                          phy4[3] * _s12(i, j + rj0 + 1, k + 7)) +
                     dhpz4[5] * _g_c(k + 8) *
                         (phy4[2] * _s12(i, j + rj0, k + 8) +
                          phy4[0] * _s12(i, j + rj0 - 2, k + 8) +
                          phy4[1] * _s12(i, j + rj0 - 1, k + 8) +
                          phy4[3] * _s12(i, j + rj0 + 1, k + 8)) +
                     dhpz4[6] * _g_c(k + 9) *
                         (phy4[2] * _s12(i, j + rj0, k + 9) +
                          phy4[0] * _s12(i, j + rj0 - 2, k + 9) +
                          phy4[1] * _s12(i, j + rj0 - 1, k + 9) +
                          phy4[3] * _s12(i, j + rj0 + 1, k + 9))))) *
        f_dcrj;
    _buf_u2(i, j, k + 6) =
        (a * _u2(i, j + rj0, k + 6) +
         Ai2 *
             (dhz4[0] * _s23(i, j + rj0, k + 4) +
              dhz4[1] * _s23(i, j + rj0, k + 5) +
              dhz4[2] * _s23(i, j + rj0, k + 6) +
              dhz4[3] * _s23(i, j + rj0, k + 7) +
              dx4[1] * _f(i, j + rj0) * _g3_c(k + 6) * _s12(i, j + rj0, k + 6) +
              dx4[0] * _f(i - 1, j + rj0) * _g3_c(k + 6) *
                  _s12(i - 1, j + rj0, k + 6) +
              dx4[2] * _f(i + 1, j + rj0) * _g3_c(k + 6) *
                  _s12(i + 1, j + rj0, k + 6) +
              dx4[3] * _f(i + 2, j + rj0) * _g3_c(k + 6) *
                  _s12(i + 2, j + rj0, k + 6) +
              dy4[1] * _f_c(i, j + rj0) * _g3_c(k + 6) *
                  _s22(i, j + rj0, k + 6) +
              dy4[0] * _f_c(i, j + rj0 - 1) * _g3_c(k + 6) *
                  _s22(i, j + rj0 - 1, k + 6) +
              dy4[2] * _f_c(i, j + rj0 + 1) * _g3_c(k + 6) *
                  _s22(i, j + rj0 + 1, k + 6) +
              dy4[3] * _f_c(i, j + rj0 + 2) * _g3_c(k + 6) *
                  _s22(i, j + rj0 + 2, k + 6) -
              _f1_2(i, j + rj0) * (dhpz4[0] * _g_c(k + 3) *
                                       (px4[1] * _s12(i, j + rj0, k + 3) +
                                        px4[0] * _s12(i - 1, j + rj0, k + 3) +
                                        px4[2] * _s12(i + 1, j + rj0, k + 3) +
                                        px4[3] * _s12(i + 2, j + rj0, k + 3)) +
                                   dhpz4[1] * _g_c(k + 4) *
                                       (px4[1] * _s12(i, j + rj0, k + 4) +
                                        px4[0] * _s12(i - 1, j + rj0, k + 4) +
                                        px4[2] * _s12(i + 1, j + rj0, k + 4) +
                                        px4[3] * _s12(i + 2, j + rj0, k + 4)) +
                                   dhpz4[2] * _g_c(k + 5) *
                                       (px4[1] * _s12(i, j + rj0, k + 5) +
                                        px4[0] * _s12(i - 1, j + rj0, k + 5) +
                                        px4[2] * _s12(i + 1, j + rj0, k + 5) +
                                        px4[3] * _s12(i + 2, j + rj0, k + 5)) +
                                   dhpz4[3] * _g_c(k + 6) *
                                       (px4[1] * _s12(i, j + rj0, k + 6) +
                                        px4[0] * _s12(i - 1, j + rj0, k + 6) +
                                        px4[2] * _s12(i + 1, j + rj0, k + 6) +
                                        px4[3] * _s12(i + 2, j + rj0, k + 6)) +
                                   dhpz4[4] * _g_c(k + 7) *
                                       (px4[1] * _s12(i, j + rj0, k + 7) +
                                        px4[0] * _s12(i - 1, j + rj0, k + 7) +
                                        px4[2] * _s12(i + 1, j + rj0, k + 7) +
                                        px4[3] * _s12(i + 2, j + rj0, k + 7)) +
                                   dhpz4[5] * _g_c(k + 8) *
                                       (px4[1] * _s12(i, j + rj0, k + 8) +
                                        px4[0] * _s12(i - 1, j + rj0, k + 8) +
                                        px4[2] * _s12(i + 1, j + rj0, k + 8) +
                                        px4[3] * _s12(i + 2, j + rj0, k + 8)) +
                                   dhpz4[6] * _g_c(k + 9) *
                                       (px4[1] * _s12(i, j + rj0, k + 9) +
                                        px4[0] * _s12(i - 1, j + rj0, k + 9) +
                                        px4[2] * _s12(i + 1, j + rj0, k + 9) +
                                        px4[3] * _s12(i + 2, j + rj0, k + 9))) -
              _f2_2(i, j + rj0) *
                  (dhpz4[0] * _g_c(k + 3) *
                       (py4[1] * _s22(i, j + rj0, k + 3) +
                        py4[0] * _s22(i, j + rj0 - 1, k + 3) +
                        py4[2] * _s22(i, j + rj0 + 1, k + 3) +
                        py4[3] * _s22(i, j + rj0 + 2, k + 3)) +
                   dhpz4[1] * _g_c(k + 4) *
                       (py4[1] * _s22(i, j + rj0, k + 4) +
                        py4[0] * _s22(i, j + rj0 - 1, k + 4) +
                        py4[2] * _s22(i, j + rj0 + 1, k + 4) +
                        py4[3] * _s22(i, j + rj0 + 2, k + 4)) +
                   dhpz4[2] * _g_c(k + 5) *
                       (py4[1] * _s22(i, j + rj0, k + 5) +
                        py4[0] * _s22(i, j + rj0 - 1, k + 5) +
                        py4[2] * _s22(i, j + rj0 + 1, k + 5) +
                        py4[3] * _s22(i, j + rj0 + 2, k + 5)) +
                   dhpz4[3] * _g_c(k + 6) *
                       (py4[1] * _s22(i, j + rj0, k + 6) +
                        py4[0] * _s22(i, j + rj0 - 1, k + 6) +
                        py4[2] * _s22(i, j + rj0 + 1, k + 6) +
                        py4[3] * _s22(i, j + rj0 + 2, k + 6)) +
                   dhpz4[4] * _g_c(k + 7) *
                       (py4[1] * _s22(i, j + rj0, k + 7) +
                        py4[0] * _s22(i, j + rj0 - 1, k + 7) +
                        py4[2] * _s22(i, j + rj0 + 1, k + 7) +
                        py4[3] * _s22(i, j + rj0 + 2, k + 7)) +
                   dhpz4[5] * _g_c(k + 8) *
                       (py4[1] * _s22(i, j + rj0, k + 8) +
                        py4[0] * _s22(i, j + rj0 - 1, k + 8) +
                        py4[2] * _s22(i, j + rj0 + 1, k + 8) +
                        py4[3] * _s22(i, j + rj0 + 2, k + 8)) +
                   dhpz4[6] * _g_c(k + 9) *
                       (py4[1] * _s22(i, j + rj0, k + 9) +
                        py4[0] * _s22(i, j + rj0 - 1, k + 9) +
                        py4[2] * _s22(i, j + rj0 + 1, k + 9) +
                        py4[3] * _s22(i, j + rj0 + 2, k + 9))))) *
        f_dcrj;
    _buf_u3(i, j, k + 6) =
        (a * _u3(i, j + rj0, k + 6) +
         Ai3 *
             (dhy4[2] * _f_2(i, j + rj0) * _g3(k + 6) *
                  _s23(i, j + rj0, k + 6) +
              dhy4[0] * _f_2(i, j + rj0 - 2) * _g3(k + 6) *
                  _s23(i, j + rj0 - 2, k + 6) +
              dhy4[1] * _f_2(i, j + rj0 - 1) * _g3(k + 6) *
                  _s23(i, j + rj0 - 1, k + 6) +
              dhy4[3] * _f_2(i, j + rj0 + 1) * _g3(k + 6) *
                  _s23(i, j + rj0 + 1, k + 6) +
              dx4[1] * _f_1(i, j + rj0) * _g3(k + 6) * _s13(i, j + rj0, k + 6) +
              dx4[0] * _f_1(i - 1, j + rj0) * _g3(k + 6) *
                  _s13(i - 1, j + rj0, k + 6) +
              dx4[2] * _f_1(i + 1, j + rj0) * _g3(k + 6) *
                  _s13(i + 1, j + rj0, k + 6) +
              dx4[3] * _f_1(i + 2, j + rj0) * _g3(k + 6) *
                  _s13(i + 2, j + rj0, k + 6) +
              dz4[0] * _s33(i, j + rj0, k + 5) +
              dz4[1] * _s33(i, j + rj0, k + 6) +
              dz4[2] * _s33(i, j + rj0, k + 7) +
              dz4[3] * _s33(i, j + rj0, k + 8) -
              _f1_c(i, j + rj0) * (dphz4[0] * _g(k + 3) *
                                       (px4[1] * _s13(i, j + rj0, k + 3) +
                                        px4[0] * _s13(i - 1, j + rj0, k + 3) +
                                        px4[2] * _s13(i + 1, j + rj0, k + 3) +
                                        px4[3] * _s13(i + 2, j + rj0, k + 3)) +
                                   dphz4[1] * _g(k + 4) *
                                       (px4[1] * _s13(i, j + rj0, k + 4) +
                                        px4[0] * _s13(i - 1, j + rj0, k + 4) +
                                        px4[2] * _s13(i + 1, j + rj0, k + 4) +
                                        px4[3] * _s13(i + 2, j + rj0, k + 4)) +
                                   dphz4[2] * _g(k + 5) *
                                       (px4[1] * _s13(i, j + rj0, k + 5) +
                                        px4[0] * _s13(i - 1, j + rj0, k + 5) +
                                        px4[2] * _s13(i + 1, j + rj0, k + 5) +
                                        px4[3] * _s13(i + 2, j + rj0, k + 5)) +
                                   dphz4[3] * _g(k + 6) *
                                       (px4[1] * _s13(i, j + rj0, k + 6) +
                                        px4[0] * _s13(i - 1, j + rj0, k + 6) +
                                        px4[2] * _s13(i + 1, j + rj0, k + 6) +
                                        px4[3] * _s13(i + 2, j + rj0, k + 6)) +
                                   dphz4[4] * _g(k + 7) *
                                       (px4[1] * _s13(i, j + rj0, k + 7) +
                                        px4[0] * _s13(i - 1, j + rj0, k + 7) +
                                        px4[2] * _s13(i + 1, j + rj0, k + 7) +
                                        px4[3] * _s13(i + 2, j + rj0, k + 7)) +
                                   dphz4[5] * _g(k + 8) *
                                       (px4[1] * _s13(i, j + rj0, k + 8) +
                                        px4[0] * _s13(i - 1, j + rj0, k + 8) +
                                        px4[2] * _s13(i + 1, j + rj0, k + 8) +
                                        px4[3] * _s13(i + 2, j + rj0, k + 8)) +
                                   dphz4[6] * _g(k + 9) *
                                       (px4[1] * _s13(i, j + rj0, k + 9) +
                                        px4[0] * _s13(i - 1, j + rj0, k + 9) +
                                        px4[2] * _s13(i + 1, j + rj0, k + 9) +
                                        px4[3] * _s13(i + 2, j + rj0, k + 9))) -
              _f2_c(i, j + rj0) *
                  (dphz4[0] * _g(k + 3) *
                       (phy4[2] * _s23(i, j + rj0, k + 3) +
                        phy4[0] * _s23(i, j + rj0 - 2, k + 3) +
                        phy4[1] * _s23(i, j + rj0 - 1, k + 3) +
                        phy4[3] * _s23(i, j + rj0 + 1, k + 3)) +
                   dphz4[1] * _g(k + 4) *
                       (phy4[2] * _s23(i, j + rj0, k + 4) +
                        phy4[0] * _s23(i, j + rj0 - 2, k + 4) +
                        phy4[1] * _s23(i, j + rj0 - 1, k + 4) +
                        phy4[3] * _s23(i, j + rj0 + 1, k + 4)) +
                   dphz4[2] * _g(k + 5) *
                       (phy4[2] * _s23(i, j + rj0, k + 5) +
                        phy4[0] * _s23(i, j + rj0 - 2, k + 5) +
                        phy4[1] * _s23(i, j + rj0 - 1, k + 5) +
                        phy4[3] * _s23(i, j + rj0 + 1, k + 5)) +
                   dphz4[3] * _g(k + 6) *
                       (phy4[2] * _s23(i, j + rj0, k + 6) +
                        phy4[0] * _s23(i, j + rj0 - 2, k + 6) +
                        phy4[1] * _s23(i, j + rj0 - 1, k + 6) +
                        phy4[3] * _s23(i, j + rj0 + 1, k + 6)) +
                   dphz4[4] * _g(k + 7) *
                       (phy4[2] * _s23(i, j + rj0, k + 7) +
                        phy4[0] * _s23(i, j + rj0 - 2, k + 7) +
                        phy4[1] * _s23(i, j + rj0 - 1, k + 7) +
                        phy4[3] * _s23(i, j + rj0 + 1, k + 7)) +
                   dphz4[5] * _g(k + 8) *
                       (phy4[2] * _s23(i, j + rj0, k + 8) +
                        phy4[0] * _s23(i, j + rj0 - 2, k + 8) +
                        phy4[1] * _s23(i, j + rj0 - 1, k + 8) +
                        phy4[3] * _s23(i, j + rj0 + 1, k + 8)) +
                   dphz4[6] * _g(k + 9) *
                       (phy4[2] * _s23(i, j + rj0, k + 9) +
                        phy4[0] * _s23(i, j + rj0 - 2, k + 9) +
                        phy4[1] * _s23(i, j + rj0 - 1, k + 9) +
                        phy4[3] * _s23(i, j + rj0 + 1, k + 9))))) *
        f_dcrj;
  }
#undef _buf_u1
#undef _buf_u2
#undef _buf_u3
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

__launch_bounds__(DTOPO_BUF_VEL_112_MAX_THREADS_PER_BLOCK,
                  DTOPO_BUF_VEL_112_MIN_BLOCKS_PER_SM)

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
        const float *__restrict__ u2, const float *__restrict__ u3,
        const float a, const float nu, const int nx, const int ny, const int nz,
        const int bj, const int ej, const int rj0) {
  const float phz4r[6][8] = {
      {0.0000000000000000, 0.8338228784688313, 0.1775123316429260,
       0.1435067013076542, -0.1548419114194114, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.1813404047323969, 1.1246711188154426,
       -0.2933634518280757, -0.0126480717197637, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, -0.1331142706282399, 0.7930714675884345,
       0.3131998767078508, 0.0268429263319546, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0969078556633046, -0.1539344946680898,
       0.4486491202844389, 0.6768738207821733, -0.0684963020618270,
       0.0000000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       -0.0625000000000000, 0.5625000000000000, 0.5625000000000000,
       -0.0625000000000000, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
       0.0000000000000000, -0.0625000000000000, 0.5625000000000000,
       0.5625000000000000, -0.0625000000000000}};
  const float phy4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float phx4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float dhpz4r[6][9] = {
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
  const float dhy4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const float dhx4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const float dhz4r[6][8] = {
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
  const float px4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float py4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float dx4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float dy4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float dphz4r[6][9] = {
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
  const float dz4r[6][7] = {
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
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ny)
    return;
  if (j >= ej)
    return;
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= 6)
    return;
#define _buf_u1(i, j, k)                                                       \
  buf_u1[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
#define _buf_u2(i, j, k)                                                       \
  buf_u2[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
#define _buf_u3(i, j, k)                                                       \
  buf_u3[(j) * (2 * align + nz) + (k) + align +                                \
         ngsl * (2 * align + nz) * ((i) + ngsl + 2)]
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
  for (int i = 0; i < nx; ++i) {
    float rho1 = phz4r[k][7] * (phy4[2] * _rho(i, j + rj0, nz - 8) +
                                phy4[0] * _rho(i, j + rj0 - 2, nz - 8) +
                                phy4[1] * _rho(i, j + rj0 - 1, nz - 8) +
                                phy4[3] * _rho(i, j + rj0 + 1, nz - 8)) +
                 phz4r[k][6] * (phy4[2] * _rho(i, j + rj0, nz - 7) +
                                phy4[0] * _rho(i, j + rj0 - 2, nz - 7) +
                                phy4[1] * _rho(i, j + rj0 - 1, nz - 7) +
                                phy4[3] * _rho(i, j + rj0 + 1, nz - 7)) +
                 phz4r[k][5] * (phy4[2] * _rho(i, j + rj0, nz - 6) +
                                phy4[0] * _rho(i, j + rj0 - 2, nz - 6) +
                                phy4[1] * _rho(i, j + rj0 - 1, nz - 6) +
                                phy4[3] * _rho(i, j + rj0 + 1, nz - 6)) +
                 phz4r[k][4] * (phy4[2] * _rho(i, j + rj0, nz - 5) +
                                phy4[0] * _rho(i, j + rj0 - 2, nz - 5) +
                                phy4[1] * _rho(i, j + rj0 - 1, nz - 5) +
                                phy4[3] * _rho(i, j + rj0 + 1, nz - 5)) +
                 phz4r[k][3] * (phy4[2] * _rho(i, j + rj0, nz - 4) +
                                phy4[0] * _rho(i, j + rj0 - 2, nz - 4) +
                                phy4[1] * _rho(i, j + rj0 - 1, nz - 4) +
                                phy4[3] * _rho(i, j + rj0 + 1, nz - 4)) +
                 phz4r[k][2] * (phy4[2] * _rho(i, j + rj0, nz - 3) +
                                phy4[0] * _rho(i, j + rj0 - 2, nz - 3) +
                                phy4[1] * _rho(i, j + rj0 - 1, nz - 3) +
                                phy4[3] * _rho(i, j + rj0 + 1, nz - 3)) +
                 phz4r[k][1] * (phy4[2] * _rho(i, j + rj0, nz - 2) +
                                phy4[0] * _rho(i, j + rj0 - 2, nz - 2) +
                                phy4[1] * _rho(i, j + rj0 - 1, nz - 2) +
                                phy4[3] * _rho(i, j + rj0 + 1, nz - 2)) +
                 phz4r[k][0] * (phy4[2] * _rho(i, j + rj0, nz - 1) +
                                phy4[0] * _rho(i, j + rj0 - 2, nz - 1) +
                                phy4[1] * _rho(i, j + rj0 - 1, nz - 1) +
                                phy4[3] * _rho(i, j + rj0 + 1, nz - 1));
    float rho2 = phz4r[k][7] * (phx4[2] * _rho(i, j + rj0, nz - 8) +
                                phx4[0] * _rho(i - 2, j + rj0, nz - 8) +
                                phx4[1] * _rho(i - 1, j + rj0, nz - 8) +
                                phx4[3] * _rho(i + 1, j + rj0, nz - 8)) +
                 phz4r[k][6] * (phx4[2] * _rho(i, j + rj0, nz - 7) +
                                phx4[0] * _rho(i - 2, j + rj0, nz - 7) +
                                phx4[1] * _rho(i - 1, j + rj0, nz - 7) +
                                phx4[3] * _rho(i + 1, j + rj0, nz - 7)) +
                 phz4r[k][5] * (phx4[2] * _rho(i, j + rj0, nz - 6) +
                                phx4[0] * _rho(i - 2, j + rj0, nz - 6) +
                                phx4[1] * _rho(i - 1, j + rj0, nz - 6) +
                                phx4[3] * _rho(i + 1, j + rj0, nz - 6)) +
                 phz4r[k][4] * (phx4[2] * _rho(i, j + rj0, nz - 5) +
                                phx4[0] * _rho(i - 2, j + rj0, nz - 5) +
                                phx4[1] * _rho(i - 1, j + rj0, nz - 5) +
                                phx4[3] * _rho(i + 1, j + rj0, nz - 5)) +
                 phz4r[k][3] * (phx4[2] * _rho(i, j + rj0, nz - 4) +
                                phx4[0] * _rho(i - 2, j + rj0, nz - 4) +
                                phx4[1] * _rho(i - 1, j + rj0, nz - 4) +
                                phx4[3] * _rho(i + 1, j + rj0, nz - 4)) +
                 phz4r[k][2] * (phx4[2] * _rho(i, j + rj0, nz - 3) +
                                phx4[0] * _rho(i - 2, j + rj0, nz - 3) +
                                phx4[1] * _rho(i - 1, j + rj0, nz - 3) +
                                phx4[3] * _rho(i + 1, j + rj0, nz - 3)) +
                 phz4r[k][1] * (phx4[2] * _rho(i, j + rj0, nz - 2) +
                                phx4[0] * _rho(i - 2, j + rj0, nz - 2) +
                                phx4[1] * _rho(i - 1, j + rj0, nz - 2) +
                                phx4[3] * _rho(i + 1, j + rj0, nz - 2)) +
                 phz4r[k][0] * (phx4[2] * _rho(i, j + rj0, nz - 1) +
                                phx4[0] * _rho(i - 2, j + rj0, nz - 1) +
                                phx4[1] * _rho(i - 1, j + rj0, nz - 1) +
                                phx4[3] * _rho(i + 1, j + rj0, nz - 1));
    float rho3 = phy4[2] * (phx4[2] * _rho(i, j + rj0, nz - 1 - k) +
                            phx4[0] * _rho(i - 2, j + rj0, nz - 1 - k) +
                            phx4[1] * _rho(i - 1, j + rj0, nz - 1 - k) +
                            phx4[3] * _rho(i + 1, j + rj0, nz - 1 - k)) +
                 phy4[0] * (phx4[2] * _rho(i, j + rj0 - 2, nz - 1 - k) +
                            phx4[0] * _rho(i - 2, j + rj0 - 2, nz - 1 - k) +
                            phx4[1] * _rho(i - 1, j + rj0 - 2, nz - 1 - k) +
                            phx4[3] * _rho(i + 1, j + rj0 - 2, nz - 1 - k)) +
                 phy4[1] * (phx4[2] * _rho(i, j + rj0 - 1, nz - 1 - k) +
                            phx4[0] * _rho(i - 2, j + rj0 - 1, nz - 1 - k) +
                            phx4[1] * _rho(i - 1, j + rj0 - 1, nz - 1 - k) +
                            phx4[3] * _rho(i + 1, j + rj0 - 1, nz - 1 - k)) +
                 phy4[3] * (phx4[2] * _rho(i, j + rj0 + 1, nz - 1 - k) +
                            phx4[0] * _rho(i - 2, j + rj0 + 1, nz - 1 - k) +
                            phx4[1] * _rho(i - 1, j + rj0 + 1, nz - 1 - k) +
                            phx4[3] * _rho(i + 1, j + rj0 + 1, nz - 1 - k));
    float Ai1 = _f_1(i, j + rj0) * _g3_c(nz - 1 - k) * rho1;
    Ai1 = nu * 1.0 / Ai1;
    float Ai2 = _f_2(i, j + rj0) * _g3_c(nz - 1 - k) * rho2;
    Ai2 = nu * 1.0 / Ai2;
    float Ai3 = _f_c(i, j + rj0) * _g3(nz - 1 - k) * rho3;
    Ai3 = nu * 1.0 / Ai3;
    float f_dcrj = _dcrjx(i) * _dcrjy(j + rj0) * _dcrjz(nz - 1 - k);
    _buf_u1(i, j, nz - 1 - k) =
        (a * _u1(i, j + rj0, nz - 1 - k) +
         Ai1 * (dhx4[2] * _f_c(i, j + rj0) * _g3_c(nz - 1 - k) *
                    _s11(i, j + rj0, nz - 1 - k) +
                dhx4[0] * _f_c(i - 2, j + rj0) * _g3_c(nz - 1 - k) *
                    _s11(i - 2, j + rj0, nz - 1 - k) +
                dhx4[1] * _f_c(i - 1, j + rj0) * _g3_c(nz - 1 - k) *
                    _s11(i - 1, j + rj0, nz - 1 - k) +
                dhx4[3] * _f_c(i + 1, j + rj0) * _g3_c(nz - 1 - k) *
                    _s11(i + 1, j + rj0, nz - 1 - k) +
                dhy4[2] * _f(i, j + rj0) * _g3_c(nz - 1 - k) *
                    _s12(i, j + rj0, nz - 1 - k) +
                dhy4[0] * _f(i, j + rj0 - 2) * _g3_c(nz - 1 - k) *
                    _s12(i, j + rj0 - 2, nz - 1 - k) +
                dhy4[1] * _f(i, j + rj0 - 1) * _g3_c(nz - 1 - k) *
                    _s12(i, j + rj0 - 1, nz - 1 - k) +
                dhy4[3] * _f(i, j + rj0 + 1) * _g3_c(nz - 1 - k) *
                    _s12(i, j + rj0 + 1, nz - 1 - k) +
                dhz4r[k][7] * _s13(i, j + rj0, nz - 8) +
                dhz4r[k][6] * _s13(i, j + rj0, nz - 7) +
                dhz4r[k][5] * _s13(i, j + rj0, nz - 6) +
                dhz4r[k][4] * _s13(i, j + rj0, nz - 5) +
                dhz4r[k][3] * _s13(i, j + rj0, nz - 4) +
                dhz4r[k][2] * _s13(i, j + rj0, nz - 3) +
                dhz4r[k][1] * _s13(i, j + rj0, nz - 2) +
                dhz4r[k][0] * _s13(i, j + rj0, nz - 1) -
                _f1_1(i, j + rj0) *
                    (dhpz4r[k][8] * _g_c(nz - 9) *
                         (phx4[2] * _s11(i, j + rj0, nz - 9) +
                          phx4[0] * _s11(i - 2, j + rj0, nz - 9) +
                          phx4[1] * _s11(i - 1, j + rj0, nz - 9) +
                          phx4[3] * _s11(i + 1, j + rj0, nz - 9)) +
                     dhpz4r[k][7] * _g_c(nz - 8) *
                         (phx4[2] * _s11(i, j + rj0, nz - 8) +
                          phx4[0] * _s11(i - 2, j + rj0, nz - 8) +
                          phx4[1] * _s11(i - 1, j + rj0, nz - 8) +
                          phx4[3] * _s11(i + 1, j + rj0, nz - 8)) +
                     dhpz4r[k][6] * _g_c(nz - 7) *
                         (phx4[2] * _s11(i, j + rj0, nz - 7) +
                          phx4[0] * _s11(i - 2, j + rj0, nz - 7) +
                          phx4[1] * _s11(i - 1, j + rj0, nz - 7) +
                          phx4[3] * _s11(i + 1, j + rj0, nz - 7)) +
                     dhpz4r[k][5] * _g_c(nz - 6) *
                         (phx4[2] * _s11(i, j + rj0, nz - 6) +
                          phx4[0] * _s11(i - 2, j + rj0, nz - 6) +
                          phx4[1] * _s11(i - 1, j + rj0, nz - 6) +
                          phx4[3] * _s11(i + 1, j + rj0, nz - 6)) +
                     dhpz4r[k][4] * _g_c(nz - 5) *
                         (phx4[2] * _s11(i, j + rj0, nz - 5) +
                          phx4[0] * _s11(i - 2, j + rj0, nz - 5) +
                          phx4[1] * _s11(i - 1, j + rj0, nz - 5) +
                          phx4[3] * _s11(i + 1, j + rj0, nz - 5)) +
                     dhpz4r[k][3] * _g_c(nz - 4) *
                         (phx4[2] * _s11(i, j + rj0, nz - 4) +
                          phx4[0] * _s11(i - 2, j + rj0, nz - 4) +
                          phx4[1] * _s11(i - 1, j + rj0, nz - 4) +
                          phx4[3] * _s11(i + 1, j + rj0, nz - 4)) +
                     dhpz4r[k][2] * _g_c(nz - 3) *
                         (phx4[2] * _s11(i, j + rj0, nz - 3) +
                          phx4[0] * _s11(i - 2, j + rj0, nz - 3) +
                          phx4[1] * _s11(i - 1, j + rj0, nz - 3) +
                          phx4[3] * _s11(i + 1, j + rj0, nz - 3)) +
                     dhpz4r[k][1] * _g_c(nz - 2) *
                         (phx4[2] * _s11(i, j + rj0, nz - 2) +
                          phx4[0] * _s11(i - 2, j + rj0, nz - 2) +
                          phx4[1] * _s11(i - 1, j + rj0, nz - 2) +
                          phx4[3] * _s11(i + 1, j + rj0, nz - 2)) +
                     dhpz4r[k][0] * _g_c(nz - 1) *
                         (phx4[2] * _s11(i, j + rj0, nz - 1) +
                          phx4[0] * _s11(i - 2, j + rj0, nz - 1) +
                          phx4[1] * _s11(i - 1, j + rj0, nz - 1) +
                          phx4[3] * _s11(i + 1, j + rj0, nz - 1))) -
                _f2_1(i, j + rj0) *
                    (dhpz4r[k][8] * _g_c(nz - 9) *
                         (phy4[2] * _s12(i, j + rj0, nz - 9) +
                          phy4[0] * _s12(i, j + rj0 - 2, nz - 9) +
                          phy4[1] * _s12(i, j + rj0 - 1, nz - 9) +
                          phy4[3] * _s12(i, j + rj0 + 1, nz - 9)) +
                     dhpz4r[k][7] * _g_c(nz - 8) *
                         (phy4[2] * _s12(i, j + rj0, nz - 8) +
                          phy4[0] * _s12(i, j + rj0 - 2, nz - 8) +
                          phy4[1] * _s12(i, j + rj0 - 1, nz - 8) +
                          phy4[3] * _s12(i, j + rj0 + 1, nz - 8)) +
                     dhpz4r[k][6] * _g_c(nz - 7) *
                         (phy4[2] * _s12(i, j + rj0, nz - 7) +
                          phy4[0] * _s12(i, j + rj0 - 2, nz - 7) +
                          phy4[1] * _s12(i, j + rj0 - 1, nz - 7) +
                          phy4[3] * _s12(i, j + rj0 + 1, nz - 7)) +
                     dhpz4r[k][5] * _g_c(nz - 6) *
                         (phy4[2] * _s12(i, j + rj0, nz - 6) +
                          phy4[0] * _s12(i, j + rj0 - 2, nz - 6) +
                          phy4[1] * _s12(i, j + rj0 - 1, nz - 6) +
                          phy4[3] * _s12(i, j + rj0 + 1, nz - 6)) +
                     dhpz4r[k][4] * _g_c(nz - 5) *
                         (phy4[2] * _s12(i, j + rj0, nz - 5) +
                          phy4[0] * _s12(i, j + rj0 - 2, nz - 5) +
                          phy4[1] * _s12(i, j + rj0 - 1, nz - 5) +
                          phy4[3] * _s12(i, j + rj0 + 1, nz - 5)) +
                     dhpz4r[k][3] * _g_c(nz - 4) *
                         (phy4[2] * _s12(i, j + rj0, nz - 4) +
                          phy4[0] * _s12(i, j + rj0 - 2, nz - 4) +
                          phy4[1] * _s12(i, j + rj0 - 1, nz - 4) +
                          phy4[3] * _s12(i, j + rj0 + 1, nz - 4)) +
                     dhpz4r[k][2] * _g_c(nz - 3) *
                         (phy4[2] * _s12(i, j + rj0, nz - 3) +
                          phy4[0] * _s12(i, j + rj0 - 2, nz - 3) +
                          phy4[1] * _s12(i, j + rj0 - 1, nz - 3) +
                          phy4[3] * _s12(i, j + rj0 + 1, nz - 3)) +
                     dhpz4r[k][1] * _g_c(nz - 2) *
                         (phy4[2] * _s12(i, j + rj0, nz - 2) +
                          phy4[0] * _s12(i, j + rj0 - 2, nz - 2) +
                          phy4[1] * _s12(i, j + rj0 - 1, nz - 2) +
                          phy4[3] * _s12(i, j + rj0 + 1, nz - 2)) +
                     dhpz4r[k][0] * _g_c(nz - 1) *
                         (phy4[2] * _s12(i, j + rj0, nz - 1) +
                          phy4[0] * _s12(i, j + rj0 - 2, nz - 1) +
                          phy4[1] * _s12(i, j + rj0 - 1, nz - 1) +
                          phy4[3] * _s12(i, j + rj0 + 1, nz - 1))))) *
        f_dcrj;
    _buf_u2(i, j, nz - 1 - k) =
        (a * _u2(i, j + rj0, nz - 1 - k) +
         Ai2 * (dhz4r[k][7] * _s23(i, j + rj0, nz - 8) +
                dhz4r[k][6] * _s23(i, j + rj0, nz - 7) +
                dhz4r[k][5] * _s23(i, j + rj0, nz - 6) +
                dhz4r[k][4] * _s23(i, j + rj0, nz - 5) +
                dhz4r[k][3] * _s23(i, j + rj0, nz - 4) +
                dhz4r[k][2] * _s23(i, j + rj0, nz - 3) +
                dhz4r[k][1] * _s23(i, j + rj0, nz - 2) +
                dhz4r[k][0] * _s23(i, j + rj0, nz - 1) +
                dx4[1] * _f(i, j + rj0) * _g3_c(nz - 1 - k) *
                    _s12(i, j + rj0, nz - 1 - k) +
                dx4[0] * _f(i - 1, j + rj0) * _g3_c(nz - 1 - k) *
                    _s12(i - 1, j + rj0, nz - 1 - k) +
                dx4[2] * _f(i + 1, j + rj0) * _g3_c(nz - 1 - k) *
                    _s12(i + 1, j + rj0, nz - 1 - k) +
                dx4[3] * _f(i + 2, j + rj0) * _g3_c(nz - 1 - k) *
                    _s12(i + 2, j + rj0, nz - 1 - k) +
                dy4[1] * _f_c(i, j + rj0) * _g3_c(nz - 1 - k) *
                    _s22(i, j + rj0, nz - 1 - k) +
                dy4[0] * _f_c(i, j + rj0 - 1) * _g3_c(nz - 1 - k) *
                    _s22(i, j + rj0 - 1, nz - 1 - k) +
                dy4[2] * _f_c(i, j + rj0 + 1) * _g3_c(nz - 1 - k) *
                    _s22(i, j + rj0 + 1, nz - 1 - k) +
                dy4[3] * _f_c(i, j + rj0 + 2) * _g3_c(nz - 1 - k) *
                    _s22(i, j + rj0 + 2, nz - 1 - k) -
                _f1_2(i, j + rj0) *
                    (dhpz4r[k][8] * _g_c(nz - 9) *
                         (px4[1] * _s12(i, j + rj0, nz - 9) +
                          px4[0] * _s12(i - 1, j + rj0, nz - 9) +
                          px4[2] * _s12(i + 1, j + rj0, nz - 9) +
                          px4[3] * _s12(i + 2, j + rj0, nz - 9)) +
                     dhpz4r[k][7] * _g_c(nz - 8) *
                         (px4[1] * _s12(i, j + rj0, nz - 8) +
                          px4[0] * _s12(i - 1, j + rj0, nz - 8) +
                          px4[2] * _s12(i + 1, j + rj0, nz - 8) +
                          px4[3] * _s12(i + 2, j + rj0, nz - 8)) +
                     dhpz4r[k][6] * _g_c(nz - 7) *
                         (px4[1] * _s12(i, j + rj0, nz - 7) +
                          px4[0] * _s12(i - 1, j + rj0, nz - 7) +
                          px4[2] * _s12(i + 1, j + rj0, nz - 7) +
                          px4[3] * _s12(i + 2, j + rj0, nz - 7)) +
                     dhpz4r[k][5] * _g_c(nz - 6) *
                         (px4[1] * _s12(i, j + rj0, nz - 6) +
                          px4[0] * _s12(i - 1, j + rj0, nz - 6) +
                          px4[2] * _s12(i + 1, j + rj0, nz - 6) +
                          px4[3] * _s12(i + 2, j + rj0, nz - 6)) +
                     dhpz4r[k][4] * _g_c(nz - 5) *
                         (px4[1] * _s12(i, j + rj0, nz - 5) +
                          px4[0] * _s12(i - 1, j + rj0, nz - 5) +
                          px4[2] * _s12(i + 1, j + rj0, nz - 5) +
                          px4[3] * _s12(i + 2, j + rj0, nz - 5)) +
                     dhpz4r[k][3] * _g_c(nz - 4) *
                         (px4[1] * _s12(i, j + rj0, nz - 4) +
                          px4[0] * _s12(i - 1, j + rj0, nz - 4) +
                          px4[2] * _s12(i + 1, j + rj0, nz - 4) +
                          px4[3] * _s12(i + 2, j + rj0, nz - 4)) +
                     dhpz4r[k][2] * _g_c(nz - 3) *
                         (px4[1] * _s12(i, j + rj0, nz - 3) +
                          px4[0] * _s12(i - 1, j + rj0, nz - 3) +
                          px4[2] * _s12(i + 1, j + rj0, nz - 3) +
                          px4[3] * _s12(i + 2, j + rj0, nz - 3)) +
                     dhpz4r[k][1] * _g_c(nz - 2) *
                         (px4[1] * _s12(i, j + rj0, nz - 2) +
                          px4[0] * _s12(i - 1, j + rj0, nz - 2) +
                          px4[2] * _s12(i + 1, j + rj0, nz - 2) +
                          px4[3] * _s12(i + 2, j + rj0, nz - 2)) +
                     dhpz4r[k][0] * _g_c(nz - 1) *
                         (px4[1] * _s12(i, j + rj0, nz - 1) +
                          px4[0] * _s12(i - 1, j + rj0, nz - 1) +
                          px4[2] * _s12(i + 1, j + rj0, nz - 1) +
                          px4[3] * _s12(i + 2, j + rj0, nz - 1))) -
                _f2_2(i, j + rj0) *
                    (dhpz4r[k][8] * _g_c(nz - 9) *
                         (py4[1] * _s22(i, j + rj0, nz - 9) +
                          py4[0] * _s22(i, j + rj0 - 1, nz - 9) +
                          py4[2] * _s22(i, j + rj0 + 1, nz - 9) +
                          py4[3] * _s22(i, j + rj0 + 2, nz - 9)) +
                     dhpz4r[k][7] * _g_c(nz - 8) *
                         (py4[1] * _s22(i, j + rj0, nz - 8) +
                          py4[0] * _s22(i, j + rj0 - 1, nz - 8) +
                          py4[2] * _s22(i, j + rj0 + 1, nz - 8) +
                          py4[3] * _s22(i, j + rj0 + 2, nz - 8)) +
                     dhpz4r[k][6] * _g_c(nz - 7) *
                         (py4[1] * _s22(i, j + rj0, nz - 7) +
                          py4[0] * _s22(i, j + rj0 - 1, nz - 7) +
                          py4[2] * _s22(i, j + rj0 + 1, nz - 7) +
                          py4[3] * _s22(i, j + rj0 + 2, nz - 7)) +
                     dhpz4r[k][5] * _g_c(nz - 6) *
                         (py4[1] * _s22(i, j + rj0, nz - 6) +
                          py4[0] * _s22(i, j + rj0 - 1, nz - 6) +
                          py4[2] * _s22(i, j + rj0 + 1, nz - 6) +
                          py4[3] * _s22(i, j + rj0 + 2, nz - 6)) +
                     dhpz4r[k][4] * _g_c(nz - 5) *
                         (py4[1] * _s22(i, j + rj0, nz - 5) +
                          py4[0] * _s22(i, j + rj0 - 1, nz - 5) +
                          py4[2] * _s22(i, j + rj0 + 1, nz - 5) +
                          py4[3] * _s22(i, j + rj0 + 2, nz - 5)) +
                     dhpz4r[k][3] * _g_c(nz - 4) *
                         (py4[1] * _s22(i, j + rj0, nz - 4) +
                          py4[0] * _s22(i, j + rj0 - 1, nz - 4) +
                          py4[2] * _s22(i, j + rj0 + 1, nz - 4) +
                          py4[3] * _s22(i, j + rj0 + 2, nz - 4)) +
                     dhpz4r[k][2] * _g_c(nz - 3) *
                         (py4[1] * _s22(i, j + rj0, nz - 3) +
                          py4[0] * _s22(i, j + rj0 - 1, nz - 3) +
                          py4[2] * _s22(i, j + rj0 + 1, nz - 3) +
                          py4[3] * _s22(i, j + rj0 + 2, nz - 3)) +
                     dhpz4r[k][1] * _g_c(nz - 2) *
                         (py4[1] * _s22(i, j + rj0, nz - 2) +
                          py4[0] * _s22(i, j + rj0 - 1, nz - 2) +
                          py4[2] * _s22(i, j + rj0 + 1, nz - 2) +
                          py4[3] * _s22(i, j + rj0 + 2, nz - 2)) +
                     dhpz4r[k][0] * _g_c(nz - 1) *
                         (py4[1] * _s22(i, j + rj0, nz - 1) +
                          py4[0] * _s22(i, j + rj0 - 1, nz - 1) +
                          py4[2] * _s22(i, j + rj0 + 1, nz - 1) +
                          py4[3] * _s22(i, j + rj0 + 2, nz - 1))))) *
        f_dcrj;
    _buf_u3(i, j, nz - 1 - k) =
        (a * _u3(i, j + rj0, nz - 1 - k) +
         Ai3 * (dhy4[2] * _f_2(i, j + rj0) * _g3(nz - 1 - k) *
                    _s23(i, j + rj0, nz - 1 - k) +
                dhy4[0] * _f_2(i, j + rj0 - 2) * _g3(nz - 1 - k) *
                    _s23(i, j + rj0 - 2, nz - 1 - k) +
                dhy4[1] * _f_2(i, j + rj0 - 1) * _g3(nz - 1 - k) *
                    _s23(i, j + rj0 - 1, nz - 1 - k) +
                dhy4[3] * _f_2(i, j + rj0 + 1) * _g3(nz - 1 - k) *
                    _s23(i, j + rj0 + 1, nz - 1 - k) +
                dx4[1] * _f_1(i, j + rj0) * _g3(nz - 1 - k) *
                    _s13(i, j + rj0, nz - 1 - k) +
                dx4[0] * _f_1(i - 1, j + rj0) * _g3(nz - 1 - k) *
                    _s13(i - 1, j + rj0, nz - 1 - k) +
                dx4[2] * _f_1(i + 1, j + rj0) * _g3(nz - 1 - k) *
                    _s13(i + 1, j + rj0, nz - 1 - k) +
                dx4[3] * _f_1(i + 2, j + rj0) * _g3(nz - 1 - k) *
                    _s13(i + 2, j + rj0, nz - 1 - k) +
                dz4r[k][6] * _s33(i, j + rj0, nz - 7) +
                dz4r[k][5] * _s33(i, j + rj0, nz - 6) +
                dz4r[k][4] * _s33(i, j + rj0, nz - 5) +
                dz4r[k][3] * _s33(i, j + rj0, nz - 4) +
                dz4r[k][2] * _s33(i, j + rj0, nz - 3) +
                dz4r[k][1] * _s33(i, j + rj0, nz - 2) +
                dz4r[k][0] * _s33(i, j + rj0, nz - 1) -
                _f1_c(i, j + rj0) *
                    (dphz4r[k][8] * _g(nz - 9) *
                         (px4[1] * _s13(i, j + rj0, nz - 9) +
                          px4[0] * _s13(i - 1, j + rj0, nz - 9) +
                          px4[2] * _s13(i + 1, j + rj0, nz - 9) +
                          px4[3] * _s13(i + 2, j + rj0, nz - 9)) +
                     dphz4r[k][7] * _g(nz - 8) *
                         (px4[1] * _s13(i, j + rj0, nz - 8) +
                          px4[0] * _s13(i - 1, j + rj0, nz - 8) +
                          px4[2] * _s13(i + 1, j + rj0, nz - 8) +
                          px4[3] * _s13(i + 2, j + rj0, nz - 8)) +
                     dphz4r[k][6] * _g(nz - 7) *
                         (px4[1] * _s13(i, j + rj0, nz - 7) +
                          px4[0] * _s13(i - 1, j + rj0, nz - 7) +
                          px4[2] * _s13(i + 1, j + rj0, nz - 7) +
                          px4[3] * _s13(i + 2, j + rj0, nz - 7)) +
                     dphz4r[k][5] * _g(nz - 6) *
                         (px4[1] * _s13(i, j + rj0, nz - 6) +
                          px4[0] * _s13(i - 1, j + rj0, nz - 6) +
                          px4[2] * _s13(i + 1, j + rj0, nz - 6) +
                          px4[3] * _s13(i + 2, j + rj0, nz - 6)) +
                     dphz4r[k][4] * _g(nz - 5) *
                         (px4[1] * _s13(i, j + rj0, nz - 5) +
                          px4[0] * _s13(i - 1, j + rj0, nz - 5) +
                          px4[2] * _s13(i + 1, j + rj0, nz - 5) +
                          px4[3] * _s13(i + 2, j + rj0, nz - 5)) +
                     dphz4r[k][3] * _g(nz - 4) *
                         (px4[1] * _s13(i, j + rj0, nz - 4) +
                          px4[0] * _s13(i - 1, j + rj0, nz - 4) +
                          px4[2] * _s13(i + 1, j + rj0, nz - 4) +
                          px4[3] * _s13(i + 2, j + rj0, nz - 4)) +
                     dphz4r[k][2] * _g(nz - 3) *
                         (px4[1] * _s13(i, j + rj0, nz - 3) +
                          px4[0] * _s13(i - 1, j + rj0, nz - 3) +
                          px4[2] * _s13(i + 1, j + rj0, nz - 3) +
                          px4[3] * _s13(i + 2, j + rj0, nz - 3)) +
                     dphz4r[k][1] * _g(nz - 2) *
                         (px4[1] * _s13(i, j + rj0, nz - 2) +
                          px4[0] * _s13(i - 1, j + rj0, nz - 2) +
                          px4[2] * _s13(i + 1, j + rj0, nz - 2) +
                          px4[3] * _s13(i + 2, j + rj0, nz - 2)) +
                     dphz4r[k][0] * _g(nz - 1) *
                         (px4[1] * _s13(i, j + rj0, nz - 1) +
                          px4[0] * _s13(i - 1, j + rj0, nz - 1) +
                          px4[2] * _s13(i + 1, j + rj0, nz - 1) +
                          px4[3] * _s13(i + 2, j + rj0, nz - 1))) -
                _f2_c(i, j + rj0) *
                    (dphz4r[k][8] * _g(nz - 9) *
                         (phy4[2] * _s23(i, j + rj0, nz - 9) +
                          phy4[0] * _s23(i, j + rj0 - 2, nz - 9) +
                          phy4[1] * _s23(i, j + rj0 - 1, nz - 9) +
                          phy4[3] * _s23(i, j + rj0 + 1, nz - 9)) +
                     dphz4r[k][7] * _g(nz - 8) *
                         (phy4[2] * _s23(i, j + rj0, nz - 8) +
                          phy4[0] * _s23(i, j + rj0 - 2, nz - 8) +
                          phy4[1] * _s23(i, j + rj0 - 1, nz - 8) +
                          phy4[3] * _s23(i, j + rj0 + 1, nz - 8)) +
                     dphz4r[k][6] * _g(nz - 7) *
                         (phy4[2] * _s23(i, j + rj0, nz - 7) +
                          phy4[0] * _s23(i, j + rj0 - 2, nz - 7) +
                          phy4[1] * _s23(i, j + rj0 - 1, nz - 7) +
                          phy4[3] * _s23(i, j + rj0 + 1, nz - 7)) +
                     dphz4r[k][5] * _g(nz - 6) *
                         (phy4[2] * _s23(i, j + rj0, nz - 6) +
                          phy4[0] * _s23(i, j + rj0 - 2, nz - 6) +
                          phy4[1] * _s23(i, j + rj0 - 1, nz - 6) +
                          phy4[3] * _s23(i, j + rj0 + 1, nz - 6)) +
                     dphz4r[k][4] * _g(nz - 5) *
                         (phy4[2] * _s23(i, j + rj0, nz - 5) +
                          phy4[0] * _s23(i, j + rj0 - 2, nz - 5) +
                          phy4[1] * _s23(i, j + rj0 - 1, nz - 5) +
                          phy4[3] * _s23(i, j + rj0 + 1, nz - 5)) +
                     dphz4r[k][3] * _g(nz - 4) *
                         (phy4[2] * _s23(i, j + rj0, nz - 4) +
                          phy4[0] * _s23(i, j + rj0 - 2, nz - 4) +
                          phy4[1] * _s23(i, j + rj0 - 1, nz - 4) +
                          phy4[3] * _s23(i, j + rj0 + 1, nz - 4)) +
                     dphz4r[k][2] * _g(nz - 3) *
                         (phy4[2] * _s23(i, j + rj0, nz - 3) +
                          phy4[0] * _s23(i, j + rj0 - 2, nz - 3) +
                          phy4[1] * _s23(i, j + rj0 - 1, nz - 3) +
                          phy4[3] * _s23(i, j + rj0 + 1, nz - 3)) +
                     dphz4r[k][1] * _g(nz - 2) *
                         (phy4[2] * _s23(i, j + rj0, nz - 2) +
                          phy4[0] * _s23(i, j + rj0 - 2, nz - 2) +
                          phy4[1] * _s23(i, j + rj0 - 1, nz - 2) +
                          phy4[3] * _s23(i, j + rj0 + 1, nz - 2)) +
                     dphz4r[k][0] * _g(nz - 1) *
                         (phy4[2] * _s23(i, j + rj0, nz - 1) +
                          phy4[0] * _s23(i, j + rj0 - 2, nz - 1) +
                          phy4[1] * _s23(i, j + rj0 - 1, nz - 1) +
                          phy4[3] * _s23(i, j + rj0 + 1, nz - 1))))) *
        f_dcrj;
  }
#undef _buf_u1
#undef _buf_u2
#undef _buf_u3
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
