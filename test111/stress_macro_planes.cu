#define USE_CONST_ARRAY_ACCESS 0
#if USE_CONST_ARRAY_ACCESS
#define ml d_yline_1
#define ms d_slice_1 
#else
#define ml (2 * align + nz) 
#define ms (2 * ngsl + ny + 4) * (2 * align + nz)
#endif

#define my (2 * align + 2 * ngsl + ny + 4)


#define _f(i, j) f[(j) + align + (i) * my]
#define _f_1(i, j) f_1[(j) + align + (i) * my]
#define _f_2(i, j) f_2[(j) + align + (i) * my]
#define _f2_c(i, j) f2_c[(j) + align + (i) * my]
#define _f1_1(i, j) f1_1[(j) + align + (i) * my]
#define _f2_1(i, j) f2_1[(j) + align + (i) * my]
#define _f2_2(i, j) f2_2[(j) + align + (i) * my]
#define _f_c(i, j) f_c[(j) + align + (i) * my]
#define _f1_c(i, j) f1_c[(j) + align + (i) * my]
#define _f1_2(i, j) f1_2[(j) + align + (i) * my]
#define _g3_c(k) g3_c[(k)]
#define _g_c(k) g_c[(k)]
#define _g(k) g[(k)]
#define _g3(k) g3[(k)]

#define _u1(i, j, k) u1[k +  (i) * ms + ml * (j)]
#define _v1(i, j, k) v1[k +  (i) * ms + ml * (j)]
#define _w1(i, j, k) w1[k +  (i) * ms + ml * (j)]
#define _xx(i, j, k) xx[k +  (i) * ms + ml * (j)]
#define _yy(i, j, k) yy[k +  (i) * ms + ml * (j)]
#define _zz(i, j, k) zz[k +  (i) * ms + ml * (j)]
#define _xy(i, j, k) xy[k +  (i) * ms + ml * (j)]
#define _xz(i, j, k) xz[k +  (i) * ms + ml * (j)]
#define _yz(i, j, k) yz[k +  (i) * ms + ml * (j)]
#define _r1(i, j, k) r1[k +  (i) * ms + ml * (j)]
#define _r2(i, j, k) r2[k +  (i) * ms + ml * (j)]
#define _r3(i, j, k) r3[k +  (i) * ms + ml * (j)]
#define _r4(i, j, k) r4[k +  (i) * ms + ml * (j)]
#define _r5(i, j, k) r5[k +  (i) * ms + ml * (j)]
#define _r6(i, j, k) r6[k +  (i) * ms + ml * (j)]

#define _lam(i, j, k) lam[k +  (i) * ms + ml * (j)]
#define _mu(i, j, k) mu[k +  (i) * ms + ml * (j)]
#define _qp(i, j, k) qp[k +  (i) * ms + ml * (j)]
#define _qs(i, j, k) qs[k +  (i) * ms + ml * (j)]
#define _d_vx1(i, j, k) d_vx1[k +  (i) * ms + ml * (j)]
#define _d_vx2(i, j, k) d_vx2[k +  (i) * ms + ml * (j)]
#define _d_ww(i, j, k) d_ww[k +  (i) * ms + ml * (j)]
#define _d_wwo(i, j, k) d_wwo[k +  (i) * ms + ml * (j)]


#define LDG(x) x


template <int tx, int ty, int na, int nb>
__launch_bounds__ (tx * ty)
__global__ void dtopo_str_111_macro_planes(_prec*  RSTRCT xx, _prec*  RSTRCT yy, _prec*  RSTRCT zz,
           _prec*  RSTRCT xy, _prec*  RSTRCT xz, _prec*  RSTRCT yz,
       _prec*  RSTRCT r1, _prec*  RSTRCT r2,  _prec*  RSTRCT r3, 
       _prec*  RSTRCT r4, _prec*  RSTRCT r5,  _prec*  RSTRCT r6,
       _prec*  RSTRCT u1, 
       _prec*  RSTRCT v1,    
       _prec*  RSTRCT w1,    
       const float *RSTRCT f,
       const float *RSTRCT f1_1, const float *RSTRCT f1_2,
       const float *RSTRCT f1_c, const float *RSTRCT f2_1,
       const float *RSTRCT f2_2, const float *RSTRCT f2_c,
       const float *RSTRCT f_1, const float *RSTRCT f_2,
       const float *RSTRCT f_c, const float *RSTRCT g,
       const float *RSTRCT g3, const float *RSTRCT g3_c,
       const float *RSTRCT g_c,
       const _prec *RSTRCT  lam,   
       const _prec *RSTRCT  mu,     
       const _prec *RSTRCT  qp,
       const _prec *RSTRCT  coeff, 
       const _prec *RSTRCT  qs, 
       const _prec *RSTRCT  dcrjx, 
       const _prec *RSTRCT  dcrjy, 
       const _prec *RSTRCT  dcrjz, 
       const _prec *RSTRCT d_vx1, 
       const _prec *RSTRCT d_vx2, 
       const int *RSTRCT d_ww, 
       const _prec *RSTRCT d_wwo,
       int NX, int ny, int nz, int rankx, int ranky, 
       int nzt, int s_i, int e_i, int s_j, int e_j) 
{ 
  register int   i,  j,  k;
  register int f_ww;
  register _prec vs1, vs2, vs3, a1, tmp, vx1,f_wwo;
  register _prec xl,  xm,  xmu1, xmu2, xmu3;
  register _prec qpa, h,   h1,   h2,   h3;
  register _prec qpaw,hw,h1w,h2w,h3w; 
  register _prec f_vx1, f_vx2,  f_dcrj, f_r,  f_dcrjy, f_dcrjz;
  register _prec f_rtmp;
  _prec f_xx, f_yy, f_zz, f_xy, f_xz, f_yz;

  const float px4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float dhx4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const float phdz4[7] = {-0.0026041666666667, 0.0937500000000000,
                          -0.6796875000000000, -0.0000000000000000,
                          0.6796875000000000,  -0.0937500000000000,
                          0.0026041666666667};
  const float dx4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float phx4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float phy4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const float dhy4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const float dhz4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const float py4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const float dy4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float dz4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const float pdhz4[7] = {-0.0026041666666667, 0.0937500000000000,
                          -0.6796875000000000, -0.0000000000000000,
                          0.6796875000000000,  -0.0937500000000000,
                          0.0026041666666667};

    
  int dm_offset = 3;
  int k0 = na * (blockIdx.x * blockDim.x + threadIdx.x) + align;
  int j0 = nb * (blockIdx.y * blockDim.y + threadIdx.y) + s_j;
  i = s_i;

  float rxx[nb][na], ryy[nb][na], rzz[nb][na];
  float rxy[nb][na], rxz[nb][na], ryz[nb][na];
  float rr1[nb][na], rr2[nb][na], rr3[nb][na];
  float rr4[nb][na], rr5[nb][na], rr6[nb][na];


  if (j0 >= e_j)
    return;

  float ru1[4][7][nb][na];
  float rv1[4][7][nb][na];
  float rw1[4][7][nb][na];

  // Prime the register queue
#pragma unroll
  for (int b = 0; b < nb; ++b) {
          j = j0 + b;
#pragma unroll
  for (int a = 0; a < na; ++a) {
          k = k0 + a;

        # pragma unroll
        for (int q = -3; q < 4; ++q) {
                ru1[0][3 + q][b][a] = _u1(i - 1, j, k + q);
                ru1[1][3 + q][b][a] = _u1(i, j, k + q);
                ru1[2][3 + q][b][a] = _u1(i + 1, j, k + q);

                rv1[0][3 + q][b][a] = _v1(i - 2, j, k + q);
                rv1[1][3 + q][b][a] = _v1(i - 1, j, k + q);
                rv1[2][3 + q][b][a] = _v1(i + 0, j, k + q);
                
                rw1[0][3 + q][b][a] = _w1(i - 2, j, k + q);
                rw1[1][3 + q][b][a] = _w1(i - 1, j, k + q);
                rw1[2][3 + q][b][a] = _w1(i + 0, j, k + q);
        }

        }
        }


  for (int i = s_i; i < e_i; ++i) {

#pragma unroll
  for (int b = 0; b < nb; ++b) {
          j = j0 + b;
#pragma unroll
  for (int a = 0; a < na; ++a) {
          k = k0 + a;
  
   #pragma unroll
   for (int q = -3; q < 4; ++q) {
        ru1[3][3 + q][b][a] = _u1(i + 2, j, k + q);
        rv1[3][3 + q][b][a] = _v1(i + 1, j, k + q);
        rw1[3][3 + q][b][a] = _w1(i + 1, j, k + q);
   }



  f_dcrjz = dcrjz[k];
  f_dcrjy = dcrjy[j];

    f_vx1 = _d_vx1(i, j, k);
    f_vx2 = _d_vx2(i, j, k);
    f_ww  = _d_ww(i, j, k);
    f_wwo = _d_wwo(i, j, k);
    
    f_dcrj   = dcrjx[i]*f_dcrjy*f_dcrjz;


    xl       = 8.0f/(  LDG(_lam(i, j, k))      + LDG(_lam(i+1, j, k)) +
                    LDG(_lam(i, j - 1, k)) + LDG(_lam(i+1,j-1,k))
                       + LDG(_lam(i, j, k - 1))  + LDG(_lam(i+1,j,k-1)) +
                       LDG(_lam(i,j-1,k-1)) + LDG(_lam(i+1,j-1,k-1)) );
    xm       = 16.0f/( LDG(_mu(i,j,k))       + LDG(_mu(i+1,j,k))  +
                    LDG(_mu(i,j-1,k))  + LDG(_mu(i+1,j-1,k))
                       + LDG(_mu(i,j,k-1))   + LDG(_mu(i+1,j,k-1))  +
                       LDG(_mu(i,j-1,k-1))  + LDG(_mu(i+1,j-1,k-1)) );
    xmu1     = 2.0f/(  LDG(_mu(i,j,k))       + LDG(_mu(i,j,k-1)) );
    xmu2     = 2.0/(  LDG(_mu(i,j,k))       + LDG(_mu(i,j-1,k)) );
    xmu3     = 2.0/(  LDG(_mu(i,j,k))       + LDG(_mu(i+1,j,k)) );
    xl       = xl  +  xm;
    qpa      = 0.0625f*( LDG(_qp(i,j,k))     + LDG(_qp(i+1,j,k)) +
                    LDG(_qp(i,j-1,k)) + LDG(_qp(i+1,j-1,k))
                         + LDG(_qp(i,j,k-1)) + LDG(_qp(i+1,j,k-1)) +
                         LDG(_qp(i,j-1,k-1)) + LDG(_qp(i+1,j-1,k-1)) );

    if(1.0f/(qpa*2.0f)<=200.0f)
    {
      qpaw=coeff[f_ww*2-2]*(2.*qpa)*(2.*qpa)+coeff[f_ww*2-1]*(2.*qpa);
    }
    else {
        //suggested by Kyle
	qpaw  = 2.0f*f_wwo*qpa;
        // qpaw  = f_wwo*qpa;
    }
    qpaw=qpaw/f_wwo;


    h        = 0.0625f*( LDG(_qs(i,j,k))     + LDG(_qs(i+1,j,k)) +
                    LDG(_qs(i,j-1,k)) + LDG(_qs(i+1,j-1,k))
                         + LDG(_qs(i,j,k-1)) + LDG(_qs(i+1,j,k-1)) +
                         LDG(_qs(i,j-1,k-1)) + LDG(_qs(i+1,j-1,k-1)) );

    if(1.0f/(h*2.0f)<=200.0f)
    {
      hw=coeff[f_ww*2-2]*(2.0f*h)*(2.0f*h)+coeff[f_ww*2-1]*(2.0f*h);
    }
    else {
      //suggested by Kyle
      hw  = 2.0f*f_wwo*h;
      // hw  = f_wwo*h;
    }
    hw=hw/f_wwo;


    h1       = 0.250f*(  _qs(i,j,k)     + _qs(i,j,k-1) );

    if(1.0f/(h1*2.0f)<=200.0f)
    {
      h1w=coeff[f_ww*2-2]*(2.0f*h1)*(2.0f*h1)+coeff[f_ww*2-1]*(2.0f*h1);
    }
    else {
        //suggested by Kyle
	h1w  = 2.0f*f_wwo*h1;
        // h1w  = f_wwo*h1;
    }
    h1w=h1w/f_wwo;

    h2       = 0.250f*(  _qs(i,j,k)     + _qs(i,j-1,k) );
    if(1.0f/(h2*2.0f)<=200.0f)
    {
      h2w=coeff[f_ww*2-2]*(2.0f*h2)*(2.0f*h2)+coeff[f_ww*2-1]*(2.0f*h2);
    }
    else {
        //suggested by Kyle
        //h2w  = f_wwo*h2;
	h2w  = 2.0f*f_wwo*h2;
    }
    h2w=h2w/f_wwo;


    h3       = 0.250f*(  _qs(i,j,k)     + _qs(i+1,j,k) );
    if(1.0f/(h3*2.0f)<=200.0f)
    {
      h3w=coeff[f_ww*2-2]*(2.0f*h3)*(2.0f*h3)+coeff[f_ww*2-1]*(2.0f*h3);
    }
    else {
      //suggested by Kyle
      h3w  = 2.0f*f_wwo*h3;
      //h3w  = f_wwo*h3;
    }
    h3w=h3w/f_wwo;

    h        = -xm*hw*d_dh1;
    h1       = -xmu1*h1w*d_dh1;
    h2       = -xmu2*h2w*d_dh1;
    h3       = -xmu3*h3w*d_dh1;


    qpa      = -qpaw*xl*d_dh1;

    xm       = xm*d_dth;
    xmu1     = xmu1*d_dth;
    xmu2     = xmu2*d_dth;
    xmu3     = xmu3*d_dth;
    xl       = xl*d_dth;
    h        = h*f_vx1;
    h1       = h1*f_vx1;
    h2       = h2*f_vx1;
    h3       = h3*f_vx1;
    qpa      = qpa*f_vx1;

    xm       = xm+d_DT*h;
    xmu1     = xmu1+d_DT*h1;
    xmu2     = xmu2+d_DT*h2;
    xmu3     = xmu3+d_DT*h3;
    vx1      = d_DT*(1+f_vx2*f_vx1);



    // xx, yy, zz
    float Jii = _f_c(i, j) * _g3_c(k);
          Jii = 1.0 * 1.0 / Jii;
          
    vs1 =
      dx4[1] * ru1[1][3][b][a] + dx4[0] * ru1[0][3][b][a] +
      dx4[2] * ru1[2][3][b][a] + dx4[3] * ru1[3][3][b][a] -
      Jii * _g_c(k) *
          (
           px4[0] * _f1_1(i - 1, j) *
               (
                phdz4[0] * ru1[0][0][b][a] + 
                phdz4[1] * ru1[0][1][b][a] + 
                phdz4[2] * ru1[0][2][b][a] + 
                phdz4[3] * ru1[0][3][b][a] + 
                phdz4[4] * ru1[0][4][b][a] + 
                phdz4[5] * ru1[0][5][b][a] + 
                phdz4[6] * ru1[0][6][b][a]
                ) +
           px4[1] * _f1_1(i, j) *
               (
                phdz4[0] * ru1[1][0][b][a] + 
                phdz4[1] * ru1[1][1][b][a] + 
                phdz4[2] * ru1[1][2][b][a] + 
                phdz4[3] * ru1[1][3][b][a] + 
                phdz4[4] * ru1[1][4][b][a] + 
                phdz4[5] * ru1[1][5][b][a] + 
                phdz4[6] * ru1[1][6][b][a]
                ) +
           px4[2] * _f1_1(i + 1, j) *
               (
                phdz4[0] * ru1[2][0][b][a] + 
                phdz4[1] * ru1[2][1][b][a] + 
                phdz4[2] * ru1[2][2][b][a] + 
                phdz4[3] * ru1[2][3][b][a] + 
                phdz4[4] * ru1[2][4][b][a] + 
                phdz4[5] * ru1[2][5][b][a] + 
                phdz4[6] * ru1[2][6][b][a]
                ) +
           px4[3] * _f1_1(i + 2, j) *
               (
                phdz4[0] * ru1[3][0][b][a] + 
                phdz4[1] * ru1[3][1][b][a] + 
                phdz4[2] * ru1[3][2][b][a] + 
                phdz4[3] * ru1[3][3][b][a] + 
                phdz4[4] * ru1[3][4][b][a] + 
                phdz4[5] * ru1[3][5][b][a] + 
                phdz4[6] * ru1[3][6][b][a]
                )
         );
    vs2 =
      dhy4[2] * rv1[2][3][b][a] + dhy4[0] * _v1(i, j - 2, k) +
      dhy4[1] * _v1(i, j - 1, k) + dhy4[3] * _v1(i, j + 1, k) -
      Jii * _g_c(k) *
          (phy4[2] * _f2_2(i, j) *
               (
                phdz4[0] * rv1[2][0][b][a] +
                phdz4[1] * rv1[2][1][b][a] +
                phdz4[2] * rv1[2][2][b][a] +
                phdz4[3] * rv1[2][3][b][a]+
                phdz4[4] * rv1[2][4][b][a] +
                phdz4[5] * rv1[2][5][b][a] +
                phdz4[6] * rv1[2][6][b][a]
                ) +        
           phy4[0] * _f2_2(i, j - 2) *
                (
                phdz4[0] * _v1(i, j - 2, k - 3) +
                phdz4[1] * _v1(i, j - 2, k - 2) +
                phdz4[2] * _v1(i, j - 2, k - 1) +
                phdz4[3] * _v1(i, j - 2, k) +
                phdz4[4] * _v1(i, j - 2, k + 1) +
                phdz4[5] * _v1(i, j - 2, k + 2) +
                phdz4[6] * _v1(i, j - 2, k + 3)
                ) +
           phy4[1] * _f2_2(i, j - 1) *
               (
                phdz4[0] * _v1(i, j - 1, k - 3) +
                phdz4[1] * _v1(i, j - 1, k - 2) +
                phdz4[2] * _v1(i, j - 1, k - 1) +
                phdz4[3] * _v1(i, j - 1, k) + 
                phdz4[4] * _v1(i, j - 1, k + 1) +
                phdz4[5] * _v1(i, j - 1, k + 2) +
                phdz4[6] * _v1(i, j - 1, k + 3)) +
           phy4[3] * _f2_2(i, j + 1) *
               (
                phdz4[0] * _v1(i, j + 1, k - 3) +
                phdz4[1] * _v1(i, j + 1, k - 2) +
                phdz4[2] * _v1(i, j + 1, k - 1) +
                phdz4[3] * _v1(i, j + 1, k) + 
                phdz4[4] * _v1(i, j + 1, k + 1) +
                phdz4[5] * _v1(i, j + 1, k + 2) +
                phdz4[6] * _v1(i, j + 1, k + 3)
                )
               );
  vs3 =
      Jii * (dhz4[2] * rw1[2][3][b][a] + dhz4[0] * rw1[2][1][b][a] +
             dhz4[1] * rw1[2][2][b][a] + dhz4[3] * rw1[2][4][b][a]);

    tmp      = xl*(vs1+vs2+vs3);

    a1       = qpa*(vs1+vs2+vs3);
    tmp      = tmp+d_DT*a1;

    f_r      = _r1(i, j, k);
    f_rtmp   = -h*(vs2+vs3) + a1; 
    f_xx     = _xx(i, j, k)  + tmp - xm*(vs2+vs3) + vx1*f_r;  
    rr1[b][a]  = f_vx2*f_r + f_wwo*f_rtmp;
    f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1); 
    rxx[b][a]  = (f_xx + d_DT*f_rtmp)*f_dcrj;

    f_r      = _r2(i, j, k);
    f_rtmp   = -h*(vs1+vs3) + a1;  
    f_yy     = (_yy(i,j,k)  + tmp - xm*(vs1+vs3) + vx1*f_r)*f_dcrj;
    rr2[b][a]  = f_vx2*f_r + f_wwo*f_rtmp; 
    f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1); 
    ryy[b][a]  = (f_yy + d_DT*f_rtmp)*f_dcrj;
	
    f_r      = _r3(i,j,k);
    f_rtmp   = -h*(vs1+vs2) + a1;
    f_zz     = (_zz(i,j,k)  + tmp - xm*(vs1+vs2) + vx1*f_r)*f_dcrj;
    rr3[b][a]  = f_vx2*f_r + f_wwo*f_rtmp;
    f_rtmp   = f_rtmp*(f_wwo-1.0f) + f_vx2*f_r*(1.0f-f_vx1);  
    rzz[b][a]  = (f_zz + d_DT*f_rtmp)*f_dcrj;

    // xy
  float J12i = _f(i, j) * _g3_c(k);
  J12i = 1.0 / J12i;

  vs1 =
      dy4[1] * ru1[1][3][b][a] + dy4[0] * _u1(i, j - 1, k) +
      dy4[2] * _u1(i, j + 1, k) + dy4[3] * _u1(i, j + 2, k) -
      J12i * _g_c(k) *
          (py4[1] * _f2_1(i, j) *
               (
               phdz4[0] * ru1[1][0][b][a] +
               phdz4[1] * ru1[1][1][b][a] +
               phdz4[2] * ru1[1][2][b][a] +
               phdz4[3] * ru1[1][3][b][a] +
               phdz4[4] * ru1[1][4][b][a] +
               phdz4[5] * ru1[1][5][b][a] +
               phdz4[6] * ru1[1][6][b][a]
                ) +
           py4[0] * _f2_1(i, j - 1) *
               (phdz4[3] * _u1(i, j - 1, k) + phdz4[0] * _u1(i, j - 1, k - 3) +
                phdz4[1] * _u1(i, j - 1, k - 2) +
                phdz4[2] * _u1(i, j - 1, k - 1) +
                phdz4[4] * _u1(i, j - 1, k + 1) +
                phdz4[5] * _u1(i, j - 1, k + 2) +
                phdz4[6] * _u1(i, j - 1, k + 3)) +
           py4[2] * _f2_1(i, j + 1) *
               (phdz4[3] * _u1(i, j + 1, k) + phdz4[0] * _u1(i, j + 1, k - 3) +
                phdz4[1] * _u1(i, j + 1, k - 2) +
                phdz4[2] * _u1(i, j + 1, k - 1) +
                phdz4[4] * _u1(i, j + 1, k + 1) +
                phdz4[5] * _u1(i, j + 1, k + 2) +
                phdz4[6] * _u1(i, j + 1, k + 3)) +
           py4[3] * _f2_1(i, j + 2) *
               (phdz4[3] * _u1(i, j + 2, k) + phdz4[0] * _u1(i, j + 2, k - 3) +
                phdz4[1] * _u1(i, j + 2, k - 2) +
                phdz4[2] * _u1(i, j + 2, k - 1) +
                phdz4[4] * _u1(i, j + 2, k + 1) +
                phdz4[5] * _u1(i, j + 2, k + 2) +
                phdz4[6] * _u1(i, j + 2, k + 3)));
  vs2 =
      dhx4[2] * _v1(i, j, k) + dhx4[0] * _v1(i - 2, j, k) +
      dhx4[1] * _v1(i - 1, j, k) + dhx4[3] * _v1(i + 1, j, k) -
      J12i * _g_c(k) *
          (phx4[2] * _f1_2(i, j) *
               (
                phdz4[0] * rv1[2][0][b][a] +
                phdz4[1] * rv1[2][1][b][a] +
                phdz4[2] * rv1[2][2][b][a] +
                phdz4[3] * rv1[2][3][b][a] +
                phdz4[4] * rv1[2][4][b][a] +
                phdz4[5] * rv1[2][5][b][a] +
                phdz4[6] * rv1[2][6][b][a]
                ) +
           phx4[0] * _f1_2(i - 2, j) *
               (
                phdz4[0] * rv1[0][0][b][a] +
                phdz4[1] * rv1[0][1][b][a] +
                phdz4[2] * rv1[0][2][b][a] +
                phdz4[3] * rv1[0][3][b][a] +
                phdz4[4] * rv1[0][4][b][a] +
                phdz4[5] * rv1[0][5][b][a] +
                phdz4[6] * rv1[0][6][b][a]
               ) +
           phx4[1] * _f1_2(i - 1, j) *
               (
                phdz4[0] * rv1[1][0][b][a] +
                phdz4[1] * rv1[1][1][b][a] +
                phdz4[2] * rv1[1][2][b][a] +
                phdz4[3] * rv1[1][3][b][a] +
                phdz4[4] * rv1[1][4][b][a] +
                phdz4[5] * rv1[1][5][b][a] +
                phdz4[6] * rv1[1][6][b][a]
                ) +
           phx4[3] * _f1_2(i + 1, j) *
               (
                phdz4[0] * rv1[3][0][b][a] +
                phdz4[1] * rv1[3][1][b][a] +
                phdz4[2] * rv1[3][2][b][a] +
                phdz4[3] * rv1[3][3][b][a] +
                phdz4[4] * rv1[3][4][b][a] +
                phdz4[5] * rv1[3][5][b][a] +
                phdz4[6] * rv1[3][6][b][a]
                )
               );

    f_r      = _r4(i,j,k);
    f_rtmp   = h1*(vs1+vs2); 
    f_xy     = _xy(i,j,k)  + xmu1*(vs1+vs2) + vx1*f_r;
    rr4[b][a]  = f_vx2*f_r + f_wwo*f_rtmp; 
    f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);
    rxy[b][a]  = (f_xy + d_DT*f_rtmp)*f_dcrj;

    // xz

  float J13i = _f_1(i, j) * _g3(k);
  J13i = 1.0 * 1.0 / J13i;

  vs1 = J13i * (dz4[1] * ru1[1][3][b][a] + dz4[0] * ru1[1][2][b][a] +
                      dz4[2] * ru1[1][4][b][a] + dz4[3] * ru1[1][5][b][a]);
  vs2 =
    + dhx4[0] * rw1[0][3][b][a] +
      dhx4[1] * rw1[1][3][b][a] +
      dhx4[2] * rw1[2][3][b][a] +
    + dhx4[3] * rw1[3][3][b][a] -
      J13i * _g(k) *
          (
           phx4[0] * _f1_c(i - 2, j) *
               (
              + pdhz4[0] * rw1[0][0][b][a] +
                pdhz4[1] * rw1[0][1][b][a] +
                pdhz4[2] * rw1[0][2][b][a] +
                pdhz4[3] * rw1[0][3][b][a] +
                pdhz4[4] * rw1[0][4][b][a] +
                pdhz4[5] * rw1[0][5][b][a] +
                pdhz4[6] * rw1[0][6][b][a]) +
           phx4[1] * _f1_c(i - 1, j) *
               (
              + pdhz4[0] * rw1[1][0][b][a] +
                pdhz4[1] * rw1[1][1][b][a] +
                pdhz4[2] * rw1[1][2][b][a] +
                pdhz4[3] * rw1[1][3][b][a] +
                pdhz4[4] * rw1[1][4][b][a] +
                pdhz4[5] * rw1[1][5][b][a] +
                pdhz4[6] * rw1[1][6][b][a]) +
           phx4[2] * _f1_c(i, j) *
               (
                pdhz4[0] * rw1[2][0][b][a] +
                pdhz4[1] * rw1[2][1][b][a] +
                pdhz4[2] * rw1[2][2][b][a] +
                pdhz4[3] * rw1[2][3][b][a] +
                pdhz4[4] * rw1[2][4][b][a] +
                pdhz4[5] * rw1[2][5][b][a] +
                pdhz4[6] * rw1[2][6][b][a]) +
           phx4[3] * _f1_c(i + 1, j) *
           (
              + pdhz4[0] * rw1[3][0][b][a] +
                pdhz4[1] * rw1[3][1][b][a] +
                pdhz4[2] * rw1[3][2][b][a] +
                pdhz4[3] * rw1[3][3][b][a] +
                pdhz4[4] * rw1[3][4][b][a] +
                pdhz4[5] * rw1[3][5][b][a] +
                pdhz4[6] * rw1[3][6][b][a]));

    f_r     = _r5(i,j,k);
    f_rtmp  = h2*(vs1+vs2);
    f_xz    = _xz(i,j,k)  + xmu2*(vs1+vs2) + vx1*f_r; 
    rr5[b][a] = f_vx2*f_r + f_wwo*f_rtmp; 
    f_rtmp  = f_rtmp*(f_wwo-1.0f) + f_vx2*f_r*(1.0f-f_vx1); 
    rxz[b][a] = (f_xz + d_DT*f_rtmp)*f_dcrj;

    // yz

    float J23i = _f_2(i, j) * _g3(k);
    J23i = 1.0 * 1.0 / J23i;
    vs1 = J23i * (dz4[1] * rv1[2][3][b][a] + dz4[0] * rv1[2][2][b][a] +
                  dz4[2] * rv1[2][4][b][a] + dz4[3] * rv1[2][5][b][a]);
    vs2 =
        dy4[0] * _w1(i, j - 1, k) +
        dy4[1] * _w1(i, j, k) +
        dy4[2] * _w1(i, j + 1, k) + 
        dy4[3] * _w1(i, j + 2, k) -
        J23i * _g(k) *
            (py4[1] * _f2_c(i, j) *
                 (
                  pdhz4[0] * rw1[2][0][b][a] +
                  pdhz4[1] * rw1[2][1][b][a] +
                  pdhz4[2] * rw1[2][2][b][a] +
                  pdhz4[3] * rw1[2][3][b][a] +
                  pdhz4[4] * rw1[2][4][b][a] +
                  pdhz4[5] * rw1[2][5][b][a] +
                  pdhz4[6] * rw1[2][6][b][a]) +
             py4[0] * _f2_c(i, j - 1) *
                 (pdhz4[3] * _w1(i, j - 1, k) + pdhz4[0] * _w1(i, j - 1, k - 3) +
                  pdhz4[1] * _w1(i, j - 1, k - 2) +
                  pdhz4[2] * _w1(i, j - 1, k - 1) +
                  pdhz4[4] * _w1(i, j - 1, k + 1) +
                  pdhz4[5] * _w1(i, j - 1, k + 2) +
                  pdhz4[6] * _w1(i, j - 1, k + 3)) +
             py4[2] * _f2_c(i, j + 1) *
                 (pdhz4[3] * _w1(i, j + 1, k) + pdhz4[0] * _w1(i, j + 1, k - 3) +
                  pdhz4[1] * _w1(i, j + 1, k - 2) +
                  pdhz4[2] * _w1(i, j + 1, k - 1) +
                  pdhz4[4] * _w1(i, j + 1, k + 1) +
                  pdhz4[5] * _w1(i, j + 1, k + 2) +
                  pdhz4[6] * _w1(i, j + 1, k + 3)) +
             py4[3] * _f2_c(i, j + 2) *
                 (pdhz4[3] * _w1(i, j + 2, k) + pdhz4[0] * _w1(i, j + 2, k - 3) +
                  pdhz4[1] * _w1(i, j + 2, k - 2) +
                  pdhz4[2] * _w1(i, j + 2, k - 1) +
                  pdhz4[4] * _w1(i, j + 2, k + 1) +
                  pdhz4[5] * _w1(i, j + 2, k + 2) +
                  pdhz4[6] * _w1(i, j + 2, k + 3)));
           
    f_r     = _r6(i,j,k);
    f_rtmp  = h3*(vs1+vs2);
    f_yz    = _yz(i,j,k)  + xmu3*(vs1+vs2) + vx1*f_r;
    rr6[b][a] = f_vx2*f_r + f_wwo*f_rtmp;
    f_rtmp  = f_rtmp*(f_wwo-1.0f) + f_vx2*f_r*(1.0f-f_vx1); 
    ryz[b][a] = (f_yz + d_DT*f_rtmp)*f_dcrj; 

   # pragma unroll
   for (int q = -3; q < 4; ++q) {
        ru1[0][3+q][b][a] = ru1[1][3+q][b][a];
        ru1[1][3+q][b][a] = ru1[2][3+q][b][a];
        ru1[2][3+q][b][a] = ru1[3][3+q][b][a];

        rv1[0][3+q][b][a] = rv1[1][3+q][b][a];
        rv1[1][3+q][b][a] = rv1[2][3+q][b][a];
        rv1[2][3+q][b][a] = rv1[3][3+q][b][a];

        rw1[0][3+q][b][a] = rw1[1][3+q][b][a];
        rw1[1][3+q][b][a] = rw1[2][3+q][b][a];
        rw1[2][3+q][b][a] = rw1[3][3+q][b][a];
   }

  }
  }

#pragma unroll
  for (int b = 0; b < nb; ++b) {
          j = j0 + b;
     if (j >= e_j)
       return;
#pragma unroll
  for (int a = 0; a < na; ++a) {
     k = k0 + a;
     if (k < dm_offset + align)
       continue;
     if (k >= nz - 6 + align)
       continue;

        _xx(i,j,k) =  rxx[b][a];
        _yy(i,j,k) =  ryy[b][a];
        _zz(i,j,k) =  rzz[b][a];
        _xy(i,j,k) =  rxy[b][a];
        _xz(i,j,k) =  rxz[b][a];
        _yz(i,j,k) =  ryz[b][a];
        
        _r1(i,j,k) =  rr1[b][a];
        _r2(i,j,k) =  rr2[b][a];
        _r3(i,j,k) =  rr3[b][a];
        _r4(i,j,k) =  rr4[b][a];
        _r5(i,j,k) =  rr5[b][a];
        _r6(i,j,k) =  rr6[b][a];

        }
  }
  
  }




}

#undef _dcrjx
#undef _dcrjy
#undef _dcrjz
#undef _u1     
#undef _v1     
#undef _w1     
#undef _lam
#undef _qp
#undef _qs
#undef _mu
#undef _s11
#undef _s12
#undef _s13
#undef _s22
#undef _s23
#undef _s33
#undef _u1
#undef _u2
#undef _u3
#undef _d_vx1
#undef _d_vx2
#undef _d_ww
#undef _d_wwo
#undef _r1
#undef _r2
#undef _r3
#undef _r4
#undef _r5
#undef _r6

#undef _f
#undef _f_1
#undef _f_2
#undef _f2_c
#undef _f1_1
#undef _f2_1
#undef _f2_2
#undef _f_c
#undef _f1_c
#undef _f1_2
#undef _g3_c
#undef _g_c
#undef _g
#undef _g3

#undef ml
#undef ms
#undef my
#undef USE_CONST_ARRAY_ACCESS
