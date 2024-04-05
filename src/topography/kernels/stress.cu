#include <cuda_runtime.h>
#include <stdio.h>

#include <awp/definitions.h>
#include <test/test.h>
#include <stdio.h>
#define CURVILINEAR
#define _f(i, j) f[(j) + align + (i) * (2 * align + 2 * ngsl + ny + 4)]
#define _f_1(i, j) f_1[(j) + align + (i) * (2 * align + 2 * ngsl + ny + 4)]
#define _f_2(i, j) f_2[(j) + align + (i) * (2 * align + 2 * ngsl + ny + 4)]
#define _f2_c(i, j) f2_c[(j) + align + (i) * (2 * align + 2 * ngsl + ny + 4)]
#define _f1_1(i, j) f1_1[(j) + align + (i) * (2 * align + 2 * ngsl + ny + 4)]
#define _f2_1(i, j) f2_1[(j) + align + (i) * (2 * align + 2 * ngsl + ny + 4)]
#define _f2_2(i, j) f2_2[(j) + align + (i) * (2 * align + 2 * ngsl + ny + 4)]
#define _f_c(i, j) f_c[(j) + align + (i) * (2 * align + 2 * ngsl + ny + 4)]
#define _f1_c(i, j) f1_c[(j) + align + (i) * (2 * align + 2 * ngsl + ny + 4)]
#define _f1_2(i, j) f1_2[(j) + align + (i) * (2 * align + 2 * ngsl + ny + 4)]
#define _g3_c(k) g3_c[(k)]
#define _g_c(k) g_c[(k)]
#define _g(k) g[(k)]
#define _g3(k) g3[(k)]

__constant__ _prec d_c1;
__constant__ _prec d_c2;
__constant__ _prec d_dth;
__constant__ _prec d_dt1;
__constant__ _prec d_dh1;
__constant__ _prec d_DT;
__constant__ _prec d_DH;
__constant__ int   d_nxt;
__constant__ int   d_nyt;
__constant__ int   d_nzt;
__constant__ int   d_slice_1;
__constant__ int   d_slice_2;
__constant__ int   d_yline_1;
__constant__ int   d_yline_2;

void set_constants(const _prec dh, const _prec dt, const int nxt, const int
                nyt, const int nzt)
{
    _prec h_c1, h_c2, h_dth, h_dt1, h_dh1;

    h_c1  = 9.0/8.0;
    h_c2  = -1.0/24.0;
    h_dt1 = 1.0/dt;

    h_dth = dt/dh;
    h_dh1 = 1.0/dh;
    int slice_1  = (nyt+4+ngsl2)*(nzt+2*align);
    int slice_2  = (nyt+4+ngsl2)*(nzt+2*align)*2;
    int yline_1  = nzt+2*align;
    int yline_2  = (nzt+2*align)*2;


    CUCHK(cudaMemcpyToSymbol(d_c1,      &h_c1,    sizeof(_prec)));
    CUCHK(cudaMemcpyToSymbol(d_c2,      &h_c2,    sizeof(_prec)));
    CUCHK(cudaMemcpyToSymbol(d_dt1,     &h_dt1,   sizeof(_prec)));
    CUCHK(cudaMemcpyToSymbol(d_DT,      &dt,      sizeof(_prec)));
    CUCHK(cudaMemcpyToSymbol(d_dth,     &h_dth,   sizeof(_prec)));
    CUCHK(cudaMemcpyToSymbol(d_dh1,     &h_dh1,   sizeof(_prec)));
    CUCHK(cudaMemcpyToSymbol(d_DH,      &dh,      sizeof(_prec)));
    CUCHK(cudaMemcpyToSymbol(d_nxt,     &nxt,     sizeof(int)));
    CUCHK(cudaMemcpyToSymbol(d_nyt,     &nyt,     sizeof(int)));
    CUCHK(cudaMemcpyToSymbol(d_nzt,     &nzt,     sizeof(int)));
    CUCHK(cudaMemcpyToSymbol(d_slice_1, &slice_1, sizeof(int)));
    CUCHK(cudaMemcpyToSymbol(d_slice_2, &slice_2, sizeof(int)));
    CUCHK(cudaMemcpyToSymbol(d_yline_1, &yline_1, sizeof(int)));
    CUCHK(cudaMemcpyToSymbol(d_yline_2, &yline_2, sizeof(int)));
}

#define LDG(x) x
__launch_bounds__(512)
__global__ void 
dtopo_str_111(_prec*  __restrict__ xx, _prec*  __restrict__ yy, _prec*  __restrict__ zz,
           _prec*  __restrict__ xy, _prec*  __restrict__ xz, _prec*  __restrict__ yz,
       _prec*  __restrict__ r1, _prec*  __restrict__ r2,  _prec*  __restrict__ r3, 
       _prec*  __restrict__ r4, _prec*  __restrict__ r5,  _prec*  __restrict__ r6,
       _prec*  __restrict__ u1, 
       _prec*  __restrict__ v1,    
       _prec*  __restrict__ w1,    
       const _prec *__restrict__ f,
       const _prec *__restrict__ f1_1, const _prec *__restrict__ f1_2,
       const _prec *__restrict__ f1_c, const _prec *__restrict__ f2_1,
       const _prec *__restrict__ f2_2, const _prec *__restrict__ f2_c,
       const _prec *__restrict__ f_1, const _prec *__restrict__ f_2,
       const _prec *__restrict__ f_c, const _prec *__restrict__ g,
       const _prec *__restrict__ g3, const _prec *__restrict__ g3_c,
       const _prec *__restrict__ g_c,
       const _prec *__restrict__  lam,   
       const _prec *__restrict__  mu,     
       const _prec *__restrict__  qp,
       const _prec *__restrict__  coeff, 
       const _prec *__restrict__  qs, 
       const _prec *__restrict__  dcrjx, 
       const _prec *__restrict__  dcrjy, 
       const _prec *__restrict__  dcrjz, 
       const _prec *__restrict__ d_vx1, 
       const _prec *__restrict__ d_vx2, 
       const int *__restrict__ d_ww, 
       const _prec *__restrict__ d_wwo,
       int NX, int ny, int nz, int rankx, int ranky, 
       int nzt, int s_i, int e_i, int s_j, int e_j) 
{ 
  register int   i,  j,  k;
  register int   pos,     pos_ip1, pos_im2, pos_im1;
  register int   pos_km2, pos_km1, pos_kp1, pos_kp2;
  register int   pos_jm2, pos_jm1, pos_jp1, pos_jp2;
  register int   pos_ik1, pos_jk1, pos_ijk, pos_ijk1,f_ww;
  register _prec vs1, vs2, vs3, a1, tmp, vx1,f_wwo;
  register _prec xl,  xm,  xmu1, xmu2, xmu3;
  register _prec qpa, h,   h1,   h2,   h3;
  register _prec qpaw,hw,h1w,h2w,h3w; 
  register _prec f_vx1, f_vx2,  f_dcrj, f_r,  f_dcrjy, f_dcrjz;
  register _prec f_rtmp;
  register _prec f_u1, u1_ip1, u1_ip2, u1_im1;
  register _prec f_v1, v1_im1, v1_ip1, v1_im2;
  register _prec f_w1, w1_im1, w1_im2, w1_ip1;
  _prec f_xx, f_yy, f_zz, f_xy, f_xz, f_yz;
  int maxk, mink;

  const _prec px4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const _prec dhx4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const _prec phdz4[7] = {-0.0026041666666667, 0.0937500000000000,
                          -0.6796875000000000, -0.0000000000000000,
                          0.6796875000000000,  -0.0937500000000000,
                          0.0026041666666667};
  const _prec dx4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const _prec phx4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const _prec phy4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const _prec dhy4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const _prec dhz4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const _prec py4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const _prec dy4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const _prec dz4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const _prec pdhz4[7] = {-0.0026041666666667, 0.0937500000000000,
                          -0.6796875000000000, -0.0000000000000000,
                          0.6796875000000000,  -0.0937500000000000,
                          0.0026041666666667};

#undef _u1
#undef _v1
#undef _w1
#define _u1(i, j, k)                                                           \
  u1[k + (2 * align + nz) * (i) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * (j)]
#define _v1(i, j, k)                                                           \
  v1[(k) + (2 * align + nz) * (i) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * (j)]
#define _w1(i, j, k)                                                           \
  w1[(k) + (2 * align + nz) * (i) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * (j)]
    
  int dm_offset = 3;
  k    = blockIdx.x*blockDim.x+threadIdx.x+align;
  j    = blockIdx.y*blockDim.y+threadIdx.y+s_j;

  if (j >= e_j)
    return;
  if (k < dm_offset + align)
    return;
  if (k >= nz - 6 + align)
    return;

  

  i    = e_i - 1;
  pos  = i*d_slice_1+j*d_yline_1+k;



  u1_ip1 = u1[pos+d_slice_2];
  f_u1   = u1[pos+d_slice_1];
  u1_im1 = u1[pos];    
  f_v1   = v1[pos+d_slice_1];
  v1_im1 = v1[pos];
  v1_im2 = v1[pos-d_slice_1];
  f_w1   = w1[pos+d_slice_1];
  w1_im1 = w1[pos];
  w1_im2 = w1[pos-d_slice_1];
  f_dcrjz = dcrjz[k];
  f_dcrjy = dcrjy[j];

  for(i=e_i-1;i>=s_i;i--)
  {         
    f_vx1 = d_vx1[pos];
    f_vx2 = d_vx2[pos];
    f_ww  = d_ww[pos];
    f_wwo = d_wwo[pos];
    
    f_dcrj   = dcrjx[i]*f_dcrjy*f_dcrjz;


    pos_km2  = pos-2;
    pos_km1  = pos-1;
    pos_kp1  = pos+1;
    pos_kp2  = pos+2;
    pos_jm2  = pos-d_yline_2;
    pos_jm1  = pos-d_yline_1;
    pos_jp1  = pos+d_yline_1;
    pos_jp2  = pos+d_yline_2;
    pos_im2  = pos-d_slice_2;
    pos_im1  = pos-d_slice_1;
    pos_ip1  = pos+d_slice_1;
    pos_jk1  = pos-d_yline_1-1;
    pos_ik1  = pos+d_slice_1-1;
    pos_ijk  = pos+d_slice_1-d_yline_1;
    pos_ijk1 = pos+d_slice_1-d_yline_1-1;

    xl       = 8.0f/(  LDG(lam[pos])      + LDG(lam[pos_ip1]) + LDG(lam[pos_jm1]) + LDG(lam[pos_ijk])
                       + LDG(lam[pos_km1])  + LDG(lam[pos_ik1]) + LDG(lam[pos_jk1]) + LDG(lam[pos_ijk1]) );
    xm       = 16.0f/( LDG(mu[pos])       + LDG(mu[pos_ip1])  + LDG(mu[pos_jm1])  + LDG(mu[pos_ijk])
                       + LDG(mu[pos_km1])   + LDG(mu[pos_ik1])  + LDG(mu[pos_jk1])  + LDG(mu[pos_ijk1]) );
    xmu1     = 2.0f/(  LDG(mu[pos])       + LDG(mu[pos_km1]) );
    xmu2     = 2.0/(  LDG(mu[pos])       + LDG(mu[pos_jm1]) );
    xmu3     = 2.0/(  LDG(mu[pos])       + LDG(mu[pos_ip1]) );
    xl       = xl  +  xm;
    qpa      = 0.0625f*( LDG(qp[pos])     + LDG(qp[pos_ip1]) + LDG(qp[pos_jm1]) + LDG(qp[pos_ijk])
                         + LDG(qp[pos_km1]) + LDG(qp[pos_ik1]) + LDG(qp[pos_jk1]) + LDG(qp[pos_ijk1]) );

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


    h        = 0.0625f*( LDG(qs[pos])     + LDG(qs[pos_ip1]) + LDG(qs[pos_jm1]) + LDG(qs[pos_ijk])
                         + LDG(qs[pos_km1]) + LDG(qs[pos_ik1]) + LDG(qs[pos_jk1]) + LDG(qs[pos_ijk1]) );

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


    h1       = 0.250f*(  qs[pos]     + qs[pos_km1] );

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

    h2       = 0.250f*(  qs[pos]     + qs[pos_jm1] );
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


    h3       = 0.250f*(  qs[pos]     + qs[pos_ip1] );
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
        
    u1_ip2   = u1_ip1;
    u1_ip1   = f_u1;
    f_u1     = u1_im1;
    u1_im1   = u1[pos_im1];
    v1_ip1   = f_v1;
    f_v1     = v1_im1;
    v1_im1   = v1_im2;
    v1_im2   = v1[pos_im2];
    w1_ip1   = f_w1;
    f_w1     = w1_im1;
    w1_im1   = w1_im2;
    w1_im2   = w1[pos_im2];



    // xx, yy, zz

#ifdef CURVILINEAR

    _prec Jii = _f_c(i, j) * _g3_c(k);
          Jii = 1.0 * 1.0 / Jii;
          
    vs1 =
      dx4[1] * _u1(i, j, k) + dx4[0] * _u1(i - 1, j, k) +
      dx4[2] * _u1(i + 1, j, k) + dx4[3] * _u1(i + 2, j, k) -
      Jii * _g_c(k) *
          (
           px4[0] * _f1_1(i - 1, j) *
               (
                phdz4[0] * _u1(i - 1, j, k - 3) +
                phdz4[1] * _u1(i - 1, j, k - 2) +
                phdz4[2] * _u1(i - 1, j, k - 1) +
                phdz4[3] * _u1(i - 1, j, k) +
                phdz4[4] * _u1(i - 1, j, k + 1) +
                phdz4[5] * _u1(i - 1, j, k + 2) +
                phdz4[6] * _u1(i - 1, j, k + 3)
                ) +
           px4[1] * _f1_1(i, j) *
               (
                phdz4[0] * _u1(i, j, k - 3) +
                phdz4[1] * _u1(i, j, k - 2) +
                phdz4[2] * _u1(i, j, k - 1) +
                phdz4[3] * _u1(i, j, k) +
                phdz4[4] * _u1(i, j, k + 1) + 
                phdz4[5] * _u1(i, j, k + 2) +
                phdz4[6] * _u1(i, j, k + 3)
                ) +
           px4[2] * _f1_1(i + 1, j) *
               (
                phdz4[0] * _u1(i + 1, j, k - 3) +
                phdz4[1] * _u1(i + 1, j, k - 2) +
                phdz4[2] * _u1(i + 1, j, k - 1) +
                phdz4[3] * _u1(i + 1, j, k) +
                phdz4[4] * _u1(i + 1, j, k + 1) +
                phdz4[5] * _u1(i + 1, j, k + 2) +
                phdz4[6] * _u1(i + 1, j, k + 3)
                ) +
           px4[3] * _f1_1(i + 2, j) *
               (
                phdz4[0] * _u1(i + 2, j, k - 3) +
                phdz4[1] * _u1(i + 2, j, k - 2) +
                phdz4[2] * _u1(i + 2, j, k - 1) +
                phdz4[3] * _u1(i + 2, j, k) +
                phdz4[4] * _u1(i + 2, j, k + 1) +
                phdz4[5] * _u1(i + 2, j, k + 2) +
                phdz4[6] * _u1(i + 2, j, k + 3)
                )
         );
    vs2 =
      dhy4[2] * _v1(i, j, k) + dhy4[0] * _v1(i, j - 2, k) +
      dhy4[1] * _v1(i, j - 1, k) + dhy4[3] * _v1(i, j + 1, k) -
      Jii * _g_c(k) *
          (phy4[2] * _f2_2(i, j) *
               (
                phdz4[0] * _v1(i, j, k - 3) +
                phdz4[1] * _v1(i, j, k - 2) +
                phdz4[2] * _v1(i, j, k - 1) +
                phdz4[3] * _v1(i, j, k) +
                phdz4[4] * _v1(i, j, k + 1) +
                phdz4[5] * _v1(i, j, k + 2) +
                phdz4[6] * _v1(i, j, k + 3)
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
      Jii * (dhz4[2] * _w1(i, j, k) + dhz4[0] * _w1(i, j, k - 2) +
             dhz4[1] * _w1(i, j, k - 1) + dhz4[3] * _w1(i, j, k + 1));
#else
    // Cartesian      
    vs1      = d_c1*(u1_ip1 - f_u1)        + d_c2*(u1_ip2      - u1_im1);
    vs2      = d_c1*(f_v1   - v1[pos_jm1]) + d_c2*(v1[pos_jp1] - v1[pos_jm2]);
    vs3      = d_c1*(f_w1   - w1[pos_km1]) + d_c2*(w1[pos_kp1] - w1[pos_km2]);
#endif

    tmp      = xl*(vs1+vs2+vs3);

    a1       = qpa*(vs1+vs2+vs3);
    tmp      = tmp+d_DT*a1;

    f_r      = r1[pos];
    f_rtmp   = -h*(vs2+vs3) + a1; 
    f_xx     = xx[pos]  + tmp - xm*(vs2+vs3) + vx1*f_r;  
    r1[pos]  = f_vx2*f_r + f_wwo*f_rtmp;
    f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1); 
    xx[pos]  = (f_xx + d_DT*f_rtmp)*f_dcrj;

    f_r      = r2[pos];
    f_rtmp   = -h*(vs1+vs3) + a1;  
    f_yy     = (yy[pos]  + tmp - xm*(vs1+vs3) + vx1*f_r)*f_dcrj;
    r2[pos]  = f_vx2*f_r + f_wwo*f_rtmp; 
    f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1); 
    yy[pos]  = (f_yy + d_DT*f_rtmp)*f_dcrj;
	
    f_r      = r3[pos];
    f_rtmp   = -h*(vs1+vs2) + a1;
    f_zz     = (zz[pos]  + tmp - xm*(vs1+vs2) + vx1*f_r)*f_dcrj;
    r3[pos]  = f_vx2*f_r + f_wwo*f_rtmp;
    f_rtmp   = f_rtmp*(f_wwo-1.0f) + f_vx2*f_r*(1.0f-f_vx1);  
    zz[pos]  = (f_zz + d_DT*f_rtmp)*f_dcrj;

    // xy
#ifdef CURVILINEAR
  _prec J12i = _f(i, j) * _g3_c(k + 6);
  J12i = 1.0 / J12i;

  vs1 =
      dy4[1] * _u1(i, j, k) + dy4[0] * _u1(i, j - 1, k) +
      dy4[2] * _u1(i, j + 1, k) + dy4[3] * _u1(i, j + 2, k) -
      J12i * _g_c(k) *
          (py4[1] * _f2_1(i, j) *
               (phdz4[3] * _u1(i, j, k) + phdz4[0] * _u1(i, j, k - 3) +
                phdz4[1] * _u1(i, j, k - 2) + phdz4[2] * _u1(i, j, k - 1) +
                phdz4[4] * _u1(i, j, k + 1) + phdz4[5] * _u1(i, j, k + 2) +
                phdz4[6] * _u1(i, j, k + 3)) +
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
               (phdz4[3] * _v1(i, j, k) + phdz4[0] * _v1(i, j, k - 3) +
                phdz4[1] * _v1(i, j, k - 2) + phdz4[2] * _v1(i, j, k - 1) +
                phdz4[4] * _v1(i, j, k + 1) + phdz4[5] * _v1(i, j, k + 2) +
                phdz4[6] * _v1(i, j, k + 3)) +
           phx4[0] * _f1_2(i - 2, j) *
               (phdz4[3] * _v1(i - 2, j, k) + phdz4[0] * _v1(i - 2, j, k - 3) +
                phdz4[1] * _v1(i - 2, j, k - 2) +
                phdz4[2] * _v1(i - 2, j, k - 1) +
                phdz4[4] * _v1(i - 2, j, k + 1) +
                phdz4[5] * _v1(i - 2, j, k + 2) +
                phdz4[6] * _v1(i - 2, j, k + 3)) +
           phx4[1] * _f1_2(i - 1, j) *
               (phdz4[3] * _v1(i - 1, j, k) + phdz4[0] * _v1(i - 1, j, k - 3) +
                phdz4[1] * _v1(i - 1, j, k - 2) +
                phdz4[2] * _v1(i - 1, j, k - 1) +
                phdz4[4] * _v1(i - 1, j, k + 1) +
                phdz4[5] * _v1(i - 1, j, k + 2) +
                phdz4[6] * _v1(i - 1, j, k + 3)) +
           phx4[3] * _f1_2(i + 1, j) *
               (phdz4[3] * _v1(i + 1, j, k) + phdz4[0] * _v1(i + 1, j, k - 3) +
                phdz4[1] * _v1(i + 1, j, k - 2) +
                phdz4[2] * _v1(i + 1, j, k - 1) +
                phdz4[4] * _v1(i + 1, j, k + 1) +
                phdz4[5] * _v1(i + 1, j, k + 2) +
                phdz4[6] * _v1(i + 1, j, k + 3)));
#else
    // Cartesian
    vs1      = d_c1*(u1[pos_jp1] - f_u1)   + d_c2*(u1[pos_jp2] - u1[pos_jm1]);
    vs2      = d_c1*(f_v1        - v1_im1) + d_c2*(v1_ip1      - v1_im2);
#endif

    f_r      = r4[pos];
    f_rtmp   = h1*(vs1+vs2); 
    f_xy     = xy[pos]  + xmu1*(vs1+vs2) + vx1*f_r;
    r4[pos]  = f_vx2*f_r + f_wwo*f_rtmp; 
    f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);
    xy[pos]  = (f_xy + d_DT*f_rtmp)*f_dcrj;

    // xz
#ifdef CURVILINEAR

  _prec J13i = _f_1(i, j) * _g3(k);
  J13i = 1.0 * 1.0 / J13i;

  vs1 = J13i * (dz4[1] * _u1(i, j, k) + dz4[0] * _u1(i, j, k - 1) +
                      dz4[2] * _u1(i, j, k + 1) + dz4[3] * _u1(i, j, k + 2));
  vs2 =
      dhx4[2] * _w1(i, j, k) + dhx4[0] * _w1(i - 2, j, k) +
      dhx4[1] * _w1(i - 1, j, k) + dhx4[3] * _w1(i + 1, j, k) -
      J13i * _g(k) *
          (phx4[2] * _f1_c(i, j) *
               (pdhz4[3] * _w1(i, j, k) + pdhz4[0] * _w1(i, j, k - 3) +
                pdhz4[1] * _w1(i, j, k - 2) + pdhz4[2] * _w1(i, j, k - 1) +
                pdhz4[4] * _w1(i, j, k + 1) + pdhz4[5] * _w1(i, j, k + 2) +
                pdhz4[6] * _w1(i, j, k + 3)) +
           phx4[0] * _f1_c(i - 2, j) *
               (pdhz4[3] * _w1(i - 2, j, k) + pdhz4[0] * _w1(i - 2, j, k - 3) +
                pdhz4[1] * _w1(i - 2, j, k - 2) +
                pdhz4[2] * _w1(i - 2, j, k - 1) +
                pdhz4[4] * _w1(i - 2, j, k + 1) +
                pdhz4[5] * _w1(i - 2, j, k + 2) +
                pdhz4[6] * _w1(i - 2, j, k + 3)) +
           phx4[1] * _f1_c(i - 1, j) *
               (pdhz4[3] * _w1(i - 1, j, k) + pdhz4[0] * _w1(i - 1, j, k - 3) +
                pdhz4[1] * _w1(i - 1, j, k - 2) +
                pdhz4[2] * _w1(i - 1, j, k - 1) +
                pdhz4[4] * _w1(i - 1, j, k + 1) +
                pdhz4[5] * _w1(i - 1, j, k + 2) +
                pdhz4[6] * _w1(i - 1, j, k + 3)) +
           phx4[3] * _f1_c(i + 1, j) *
               (pdhz4[3] * _w1(i + 1, j, k) + pdhz4[0] * _w1(i + 1, j, k - 3) +
                pdhz4[1] * _w1(i + 1, j, k - 2) +
                pdhz4[2] * _w1(i + 1, j, k - 1) +
                pdhz4[4] * _w1(i + 1, j, k + 1) +
                pdhz4[5] * _w1(i + 1, j, k + 2) +
                pdhz4[6] * _w1(i + 1, j, k + 3)));

#else
    vs1     = d_c1*(u1[pos_kp1] - f_u1)   + d_c2*(u1[pos_kp2] - u1[pos_km1]);
    vs2     = d_c1*(f_w1        - w1_im1) + d_c2*(w1_ip1      - w1_im2);
#endif
    f_r     = r5[pos];
    f_rtmp  = h2*(vs1+vs2);
    f_xz    = xz[pos]  + xmu2*(vs1+vs2) + vx1*f_r; 
    r5[pos] = f_vx2*f_r + f_wwo*f_rtmp; 
    f_rtmp  = f_rtmp*(f_wwo-1.0f) + f_vx2*f_r*(1.0f-f_vx1); 
    xz[pos] = (f_xz + d_DT*f_rtmp)*f_dcrj;

    // yz

#ifdef CURVILINEAR
    _prec J23i = _f_2(i, j) * _g3(k);
    J23i = 1.0 * 1.0 / J23i;
    vs1 = J23i * (dz4[1] * _v1(i, j, k) + dz4[0] * _v1(i, j, k - 1) +
                        dz4[2] * _v1(i, j, k + 1) + dz4[3] * _v1(i, j, k + 2));
    vs2 =
        dy4[1] * _w1(i, j, k) + dy4[0] * _w1(i, j - 1, k) +
        dy4[2] * _w1(i, j + 1, k) + dy4[3] * _w1(i, j + 2, k) -
        J23i * _g(k) *
            (py4[1] * _f2_c(i, j) *
                 (pdhz4[3] * _w1(i, j, k) + pdhz4[0] * _w1(i, j, k - 3) +
                  pdhz4[1] * _w1(i, j, k - 2) + pdhz4[2] * _w1(i, j, k - 1) +
                  pdhz4[4] * _w1(i, j, k + 1) + pdhz4[5] * _w1(i, j, k + 2) +
                  pdhz4[6] * _w1(i, j, k + 3)) +
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
#else
    // Cartesian
    vs1     = d_c1*(v1[pos_kp1] - f_v1) + d_c2*(v1[pos_kp2] - v1[pos_km1]);
    vs2     = d_c1*(w1[pos_jp1] - f_w1) + d_c2*(w1[pos_jp2] - w1[pos_jm1]);
#endif
           
    f_r     = r6[pos];
    f_rtmp  = h3*(vs1+vs2);
    f_yz    = yz[pos]  + xmu3*(vs1+vs2) + vx1*f_r;
    r6[pos] = f_vx2*f_r + f_wwo*f_rtmp;
    f_rtmp  = f_rtmp*(f_wwo-1.0f) + f_vx2*f_r*(1.0f-f_vx1); 
    yz[pos] = (f_yz + d_DT*f_rtmp)*f_dcrj; 


    pos     = pos_im1;
  }

#undef _dcrjx
#undef _dcrjy
#undef _dcrjz
#undef _lami
#undef _mui
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


#define LDG(x) x
__launch_bounds__(512)
__global__ void 
dtopo_str_112(_prec*  __restrict__ xx, _prec*  __restrict__ yy, _prec*  __restrict__ zz,
           _prec*  __restrict__ xy, _prec*  __restrict__ xz, _prec*  __restrict__ yz,
       _prec*  __restrict__ r1, _prec*  __restrict__ r2,  _prec*  __restrict__ r3, 
       _prec*  __restrict__ r4, _prec*  __restrict__ r5,  _prec*  __restrict__ r6,
       _prec*  __restrict__ u1, 
       _prec*  __restrict__ v1,    
       _prec*  __restrict__ w1,    
       const _prec *__restrict__ f,
       const _prec *__restrict__ f1_1, const _prec *__restrict__ f1_2,
       const _prec *__restrict__ f1_c, const _prec *__restrict__ f2_1,
       const _prec *__restrict__ f2_2, const _prec *__restrict__ f2_c,
       const _prec *__restrict__ f_1, const _prec *__restrict__ f_2,
       const _prec *__restrict__ f_c, const _prec *__restrict__ g,
       const _prec *__restrict__ g3, const _prec *__restrict__ g3_c,
       const _prec *__restrict__ g_c,
       const _prec *__restrict__  lam,   
       const _prec *__restrict__  mu,     
       const _prec *__restrict__  qp,
       const _prec *__restrict__  coeff, 
       const _prec *__restrict__  qs, 
       const _prec *__restrict__  dcrjx, 
       const _prec *__restrict__  dcrjy, 
       const _prec *__restrict__  dcrjz, 
       const _prec *__restrict__ d_vx1, 
       const _prec *__restrict__ d_vx2, 
       const int *__restrict__ d_ww, 
       const _prec *__restrict__ d_wwo,
       int NX, int ny, int nz, int rankx, int ranky, 
       int nzt, int s_i, int e_i, int s_j, int e_j) 
{ 
  register int   i,  j,  k;
  register int   pos,     pos_ip1, pos_im2, pos_im1;
  register int   pos_km2, pos_km1, pos_kp1, pos_kp2;
  register int   pos_jm2, pos_jm1, pos_jp1, pos_jp2;
  register int   pos_ik1, pos_jk1, pos_ijk, pos_ijk1,f_ww;
  register _prec vs1, vs2, vs3, a1, tmp, vx1,f_wwo;
  register _prec xl,  xm,  xmu1, xmu2, xmu3;
  register _prec qpa, h,   h1,   h2,   h3;
  register _prec qpaw,hw,h1w,h2w,h3w; 
  register _prec f_vx1, f_vx2,  f_dcrj, f_r,  f_dcrjy, f_dcrjz;
  register _prec f_rtmp;
  register _prec f_u1, u1_ip1, u1_ip2, u1_im1;
  register _prec f_v1, v1_im1, v1_ip1, v1_im2;
  register _prec f_w1, w1_im1, w1_im2, w1_ip1;
  _prec f_xx, f_yy, f_zz, f_xy, f_xz, f_yz;
  int maxk, mink;

  const _prec px4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const _prec dhx4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const _prec phdz4[7] = {-0.0026041666666667, 0.0937500000000000,
                          -0.6796875000000000, -0.0000000000000000,
                          0.6796875000000000,  -0.0937500000000000,
                          0.0026041666666667};
  const _prec phdz4r[6][9] = {
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
  const _prec dz4r[6][7] = {
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
  const _prec dx4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const _prec phx4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const _prec phy4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const _prec dhy4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const _prec dhz4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const _prec py4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const _prec dy4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const _prec dz4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const _prec pdhz4[7] = {-0.0026041666666667, 0.0937500000000000,
                          -0.6796875000000000, -0.0000000000000000,
                          0.6796875000000000,  -0.0937500000000000,
                          0.0026041666666667};
  const _prec dhz4r[6][8] = {
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
  const _prec pdhz4r[6][9] = {
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

    
  int k0   = blockIdx.x*blockDim.x + threadIdx.x;
  k    = k0 + align + nz - 6;
  // This index is used to access the array coefficients directly
  int kb = 5 - k0;
  // This index maps to the macros 
  int kc   = -1 - k0;
  j    = blockIdx.y*blockDim.y+threadIdx.y+s_j;

  if (j >= e_j)
    return;

  if (k0 >= 6)
    return;

  i    = e_i - 1;
  pos  = i*d_slice_1+j*d_yline_1+k;

  u1_ip1 = u1[pos+d_slice_2];
  f_u1   = u1[pos+d_slice_1];
  u1_im1 = u1[pos];    
  f_v1   = v1[pos+d_slice_1];
  v1_im1 = v1[pos];
  v1_im2 = v1[pos-d_slice_1];
  f_w1   = w1[pos+d_slice_1];
  w1_im1 = w1[pos];
  w1_im2 = w1[pos-d_slice_1];
  f_dcrjz = dcrjz[k];
  f_dcrjy = dcrjy[j];

  for(i=e_i-1;i>=s_i;i--)
  {         
    f_vx1 = d_vx1[pos];
    f_vx2 = d_vx2[pos];
    f_ww  = d_ww[pos];
    f_wwo = d_wwo[pos];
    
    f_dcrj   = dcrjx[i]*f_dcrjy*f_dcrjz;


    pos_km2  = pos-2;
    pos_km1  = pos-1;
    pos_kp1  = pos+1;
    pos_kp2  = pos+2;
    pos_jm2  = pos-d_yline_2;
    pos_jm1  = pos-d_yline_1;
    pos_jp1  = pos+d_yline_1;
    pos_jp2  = pos+d_yline_2;
    pos_im2  = pos-d_slice_2;
    pos_im1  = pos-d_slice_1;
    pos_ip1  = pos+d_slice_1;
    pos_jk1  = pos-d_yline_1-1;
    pos_ik1  = pos+d_slice_1-1;
    pos_ijk  = pos+d_slice_1-d_yline_1;
    pos_ijk1 = pos+d_slice_1-d_yline_1-1;

    xl       = 8.0f/(  LDG(lam[pos])      + LDG(lam[pos_ip1]) + LDG(lam[pos_jm1]) + LDG(lam[pos_ijk])
                       + LDG(lam[pos_km1])  + LDG(lam[pos_ik1]) + LDG(lam[pos_jk1]) + LDG(lam[pos_ijk1]) );
    xm       = 16.0f/( LDG(mu[pos])       + LDG(mu[pos_ip1])  + LDG(mu[pos_jm1])  + LDG(mu[pos_ijk])
                       + LDG(mu[pos_km1])   + LDG(mu[pos_ik1])  + LDG(mu[pos_jk1])  + LDG(mu[pos_ijk1]) );
    xmu1     = 2.0f/(  LDG(mu[pos])       + LDG(mu[pos_km1]) );
    xmu2     = 2.0/(  LDG(mu[pos])       + LDG(mu[pos_jm1]) );
    xmu3     = 2.0/(  LDG(mu[pos])       + LDG(mu[pos_ip1]) );
    xl       = xl  +  xm;
    qpa      = 0.0625f*( LDG(qp[pos])     + LDG(qp[pos_ip1]) + LDG(qp[pos_jm1]) + LDG(qp[pos_ijk])
                         + LDG(qp[pos_km1]) + LDG(qp[pos_ik1]) + LDG(qp[pos_jk1]) + LDG(qp[pos_ijk1]) );

    if(1.0f/(qpa*2.0f)<=200.0f)
    {
      qpaw=coeff[f_ww*2-2]*(2.*qpa)*(2.*qpa)+coeff[f_ww*2-1]*(2.*qpa);
    }
    else {
        //suggested by Kyle
        // qpaw  = f_wwo*qpa;
	qpaw  = 2.0f*f_wwo*qpa;
    }
    qpaw=qpaw/f_wwo;


    h        = 0.0625f*( LDG(qs[pos])     + LDG(qs[pos_ip1]) + LDG(qs[pos_jm1]) + LDG(qs[pos_ijk])
                         + LDG(qs[pos_km1]) + LDG(qs[pos_ik1]) + LDG(qs[pos_jk1]) + LDG(qs[pos_ijk1]) );

    if(1.0f/(h*2.0f)<=200.0f)
    {
      hw=coeff[f_ww*2-2]*(2.0f*h)*(2.0f*h)+coeff[f_ww*2-1]*(2.0f*h);
    }
    else {
      //suggested by Kyle
      // hw  = f_wwo*h;
      hw  = 2.0f*f_wwo*h;
    }
    hw=hw/f_wwo;


    h1       = 0.250f*(  qs[pos]     + qs[pos_km1] );

    if(1.0f/(h1*2.0f)<=200.0f)
    {
      h1w=coeff[f_ww*2-2]*(2.0f*h1)*(2.0f*h1)+coeff[f_ww*2-1]*(2.0f*h1);
    }
    else {
        //suggested by Kyle
        // h1w  = f_wwo*h1;
	h1w  = 2.0f*f_wwo*h1;
    }
    h1w=h1w/f_wwo;

    h2       = 0.250f*(  qs[pos]     + qs[pos_jm1] );
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


    h3       = 0.250f*(  qs[pos]     + qs[pos_ip1] );
    if(1.0f/(h3*2.0f)<=200.0f)
    {
      h3w=coeff[f_ww*2-2]*(2.0f*h3)*(2.0f*h3)+coeff[f_ww*2-1]*(2.0f*h3);
    }
    else {
      //suggested by Kyle
      //h3w  = f_wwo*h3;
      h3w  = 2.0f*f_wwo*h3;
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
        
    u1_ip2   = u1_ip1;
    u1_ip1   = f_u1;
    f_u1     = u1_im1;
    u1_im1   = u1[pos_im1];
    v1_ip1   = f_v1;
    f_v1     = v1_im1;
    v1_im1   = v1_im2;
    v1_im2   = v1[pos_im2];
    w1_ip1   = f_w1;
    f_w1     = w1_im1;
    w1_im1   = w1_im2;
    w1_im2   = w1[pos_im2];

    // xx, yy, zz

#undef _u1
#undef _v1
#undef _w1
#undef _g3_c
#undef _g_c
#undef _g
#undef _g3

#define _u1(i, j, k)                                                           \
  u1[(k) + align + (2 * align + nz) * (i) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * (j)]
#define _v1(i, j, k)                                                           \
  v1[(k) + align + (2 * align + nz) * (i) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * (j)]
#define _w1(i, j, k)                                                           \
  w1[(k) + align + (2 * align + nz) * (i) * (2 * ngsl + ny + 4) + \
     (2 * align + nz) * (j)]
#define _g3_c(k) g3_c[(k) + align]
#define _g_c(k) g_c[(k) + align]
#define _g(k) g[(k) + align]
#define _g3(k) g3[(k) + align]

#ifdef CURVILINEAR
  _prec Jii = _f_c(i, j) * _g3_c(nz - 1 - kc - 6);
  Jii = 1.0 * 1.0 / Jii;
  // xx, yy, zz
  _prec vs1 = dx4[1] * _u1(i, j, nz - 1 - kc - 6) +
              dx4[0] * _u1(i - 1, j, nz - 1 - kc - 6) +
              dx4[2] * _u1(i + 1, j, nz - 1 - kc - 6) +
              dx4[3] * _u1(i + 2, j, nz - 1 - kc - 6) -
              Jii * _g_c(nz - 1 - kc - 6) *
                  (px4[1] * _f1_1(i, j) *
                       (phdz4r[kb][8] * _u1(i, j, nz - 9) +
                        phdz4r[kb][7] * _u1(i, j, nz - 8) +
                        phdz4r[kb][6] * _u1(i, j, nz - 7) +
                        phdz4r[kb][5] * _u1(i, j, nz - 6) +
                        phdz4r[kb][4] * _u1(i, j, nz - 5) +
                        phdz4r[kb][3] * _u1(i, j, nz - 4) +
                        phdz4r[kb][2] * _u1(i, j, nz - 3) +
                        phdz4r[kb][1] * _u1(i, j, nz - 2) +
                        phdz4r[kb][0] * _u1(i, j, nz - 1)) +
                   px4[0] * _f1_1(i - 1, j) *
                       (phdz4r[kb][8] * _u1(i - 1, j, nz - 9) +
                        phdz4r[kb][7] * _u1(i - 1, j, nz - 8) +
                        phdz4r[kb][6] * _u1(i - 1, j, nz - 7) +
                        phdz4r[kb][5] * _u1(i - 1, j, nz - 6) +
                        phdz4r[kb][4] * _u1(i - 1, j, nz - 5) +
                        phdz4r[kb][3] * _u1(i - 1, j, nz - 4) +
                        phdz4r[kb][2] * _u1(i - 1, j, nz - 3) +
                        phdz4r[kb][1] * _u1(i - 1, j, nz - 2) +
                        phdz4r[kb][0] * _u1(i - 1, j, nz - 1)) +
                   px4[2] * _f1_1(i + 1, j) *
                       (phdz4r[kb][8] * _u1(i + 1, j, nz - 9) +
                        phdz4r[kb][7] * _u1(i + 1, j, nz - 8) +
                        phdz4r[kb][6] * _u1(i + 1, j, nz - 7) +
                        phdz4r[kb][5] * _u1(i + 1, j, nz - 6) +
                        phdz4r[kb][4] * _u1(i + 1, j, nz - 5) +
                        phdz4r[kb][3] * _u1(i + 1, j, nz - 4) +
                        phdz4r[kb][2] * _u1(i + 1, j, nz - 3) +
                        phdz4r[kb][1] * _u1(i + 1, j, nz - 2) +
                        phdz4r[kb][0] * _u1(i + 1, j, nz - 1)) +
                   px4[3] * _f1_1(i + 2, j) *
                       (phdz4r[kb][8] * _u1(i + 2, j, nz - 9) +
                        phdz4r[kb][7] * _u1(i + 2, j, nz - 8) +
                        phdz4r[kb][6] * _u1(i + 2, j, nz - 7) +
                        phdz4r[kb][5] * _u1(i + 2, j, nz - 6) +
                        phdz4r[kb][4] * _u1(i + 2, j, nz - 5) +
                        phdz4r[kb][3] * _u1(i + 2, j, nz - 4) +
                        phdz4r[kb][2] * _u1(i + 2, j, nz - 3) +
                        phdz4r[kb][1] * _u1(i + 2, j, nz - 2) +
                        phdz4r[kb][0] * _u1(i + 2, j, nz - 1)));
  _prec vs2 = dhy4[2] * _v1(i, j, nz - 1 - kc - 6) +
              dhy4[0] * _v1(i, j - 2, nz - 1 - kc - 6) +
              dhy4[1] * _v1(i, j - 1, nz - 1 - kc - 6) +
              dhy4[3] * _v1(i, j + 1, nz - 1 - kc - 6) -
              Jii * _g_c(nz - 1 - kc - 6) *
                  (phy4[2] * _f2_2(i, j) *
                       (phdz4r[kb][8] * _v1(i, j, nz - 9) +
                        phdz4r[kb][7] * _v1(i, j, nz - 8) +
                        phdz4r[kb][6] * _v1(i, j, nz - 7) +
                        phdz4r[kb][5] * _v1(i, j, nz - 6) +
                        phdz4r[kb][4] * _v1(i, j, nz - 5) +
                        phdz4r[kb][3] * _v1(i, j, nz - 4) +
                        phdz4r[kb][2] * _v1(i, j, nz - 3) +
                        phdz4r[kb][1] * _v1(i, j, nz - 2) +
                        phdz4r[kb][0] * _v1(i, j, nz - 1)) +
                   phy4[0] * _f2_2(i, j - 2) *
                       (phdz4r[kb][8] * _v1(i, j - 2, nz - 9) +
                        phdz4r[kb][7] * _v1(i, j - 2, nz - 8) +
                        phdz4r[kb][6] * _v1(i, j - 2, nz - 7) +
                        phdz4r[kb][5] * _v1(i, j - 2, nz - 6) +
                        phdz4r[kb][4] * _v1(i, j - 2, nz - 5) +
                        phdz4r[kb][3] * _v1(i, j - 2, nz - 4) +
                        phdz4r[kb][2] * _v1(i, j - 2, nz - 3) +
                        phdz4r[kb][1] * _v1(i, j - 2, nz - 2) +
                        phdz4r[kb][0] * _v1(i, j - 2, nz - 1)) +
                   phy4[1] * _f2_2(i, j - 1) *
                       (phdz4r[kb][8] * _v1(i, j - 1, nz - 9) +
                        phdz4r[kb][7] * _v1(i, j - 1, nz - 8) +
                        phdz4r[kb][6] * _v1(i, j - 1, nz - 7) +
                        phdz4r[kb][5] * _v1(i, j - 1, nz - 6) +
                        phdz4r[kb][4] * _v1(i, j - 1, nz - 5) +
                        phdz4r[kb][3] * _v1(i, j - 1, nz - 4) +
                        phdz4r[kb][2] * _v1(i, j - 1, nz - 3) +
                        phdz4r[kb][1] * _v1(i, j - 1, nz - 2) +
                        phdz4r[kb][0] * _v1(i, j - 1, nz - 1)) +
                   phy4[3] * _f2_2(i, j + 1) *
                       (phdz4r[kb][8] * _v1(i, j + 1, nz - 9) +
                        phdz4r[kb][7] * _v1(i, j + 1, nz - 8) +
                        phdz4r[kb][6] * _v1(i, j + 1, nz - 7) +
                        phdz4r[kb][5] * _v1(i, j + 1, nz - 6) +
                        phdz4r[kb][4] * _v1(i, j + 1, nz - 5) +
                        phdz4r[kb][3] * _v1(i, j + 1, nz - 4) +
                        phdz4r[kb][2] * _v1(i, j + 1, nz - 3) +
                        phdz4r[kb][1] * _v1(i, j + 1, nz - 2) +
                        phdz4r[kb][0] * _v1(i, j + 1, nz - 1)));
  _prec vs3 =
      Jii * (dhz4r[kb][7] * _w1(i, j, nz - 8) + dhz4r[kb][6] * _w1(i, j, nz - 7) +
             dhz4r[kb][5] * _w1(i, j, nz - 6) + dhz4r[kb][4] * _w1(i, j, nz - 5) +
             dhz4r[kb][3] * _w1(i, j, nz - 4) + dhz4r[kb][2] * _w1(i, j, nz - 3) +
             dhz4r[kb][1] * _w1(i, j, nz - 2) + dhz4r[kb][0] * _w1(i, j, nz - 1));
#else
    // Cartesian      
    //TODO: Implement
#endif

    tmp      = xl*(vs1+vs2+vs3);

    a1       = qpa*(vs1+vs2+vs3);
    tmp      = tmp+d_DT*a1;

    f_r      = r1[pos];
    f_rtmp   = -h*(vs2+vs3) + a1; 
    f_xx     = xx[pos]  + tmp - xm*(vs2+vs3) + vx1*f_r;  
    r1[pos]  = f_vx2*f_r + f_wwo*f_rtmp;
    f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1); 
    xx[pos]  = (f_xx + d_DT*f_rtmp)*f_dcrj;

    f_r      = r2[pos];
    f_rtmp   = -h*(vs1+vs3) + a1;  
    f_yy     = (yy[pos]  + tmp - xm*(vs1+vs3) + vx1*f_r)*f_dcrj;
    r2[pos]  = f_vx2*f_r + f_wwo*f_rtmp; 
    f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1); 
    yy[pos]  = (f_yy + d_DT*f_rtmp)*f_dcrj;
	
    f_r      = r3[pos];
    f_rtmp   = -h*(vs1+vs2) + a1;
    f_zz     = (zz[pos]  + tmp - xm*(vs1+vs2) + vx1*f_r)*f_dcrj;
    r3[pos]  = f_vx2*f_r + f_wwo*f_rtmp;
    f_rtmp   = f_rtmp*(f_wwo-1.0f) + f_vx2*f_r*(1.0f-f_vx1);  
    zz[pos]  = (f_zz + d_DT*f_rtmp)*f_dcrj;

    // xy
#ifdef CURVILINEAR
  _prec J12i = _f(i, j) * _g3_c(nz - 1 - kc - 6);
  J12i = 1.0 * 1.0 / J12i;
  vs1 = dy4[1] * _u1(i, j, nz - 1 - kc - 6) +
              dy4[0] * _u1(i, j - 1, nz - 1 - kc - 6) +
              dy4[2] * _u1(i, j + 1, nz - 1 - kc - 6) +
              dy4[3] * _u1(i, j + 2, nz - 1 - kc - 6) -
              J12i * _g_c(nz - 1 - kc - 6) *
                  (py4[1] * _f2_1(i, j) *
                       (phdz4r[kb][8] * _u1(i, j, nz - 9) +
                        phdz4r[kb][7] * _u1(i, j, nz - 8) +
                        phdz4r[kb][6] * _u1(i, j, nz - 7) +
                        phdz4r[kb][5] * _u1(i, j, nz - 6) +
                        phdz4r[kb][4] * _u1(i, j, nz - 5) +
                        phdz4r[kb][3] * _u1(i, j, nz - 4) +
                        phdz4r[kb][2] * _u1(i, j, nz - 3) +
                        phdz4r[kb][1] * _u1(i, j, nz - 2) +
                        phdz4r[kb][0] * _u1(i, j, nz - 1)) +
                   py4[0] * _f2_1(i, j - 1) *
                       (phdz4r[kb][8] * _u1(i, j - 1, nz - 9) +
                        phdz4r[kb][7] * _u1(i, j - 1, nz - 8) +
                        phdz4r[kb][6] * _u1(i, j - 1, nz - 7) +
                        phdz4r[kb][5] * _u1(i, j - 1, nz - 6) +
                        phdz4r[kb][4] * _u1(i, j - 1, nz - 5) +
                        phdz4r[kb][3] * _u1(i, j - 1, nz - 4) +
                        phdz4r[kb][2] * _u1(i, j - 1, nz - 3) +
                        phdz4r[kb][1] * _u1(i, j - 1, nz - 2) +
                        phdz4r[kb][0] * _u1(i, j - 1, nz - 1)) +
                   py4[2] * _f2_1(i, j + 1) *
                       (phdz4r[kb][8] * _u1(i, j + 1, nz - 9) +
                        phdz4r[kb][7] * _u1(i, j + 1, nz - 8) +
                        phdz4r[kb][6] * _u1(i, j + 1, nz - 7) +
                        phdz4r[kb][5] * _u1(i, j + 1, nz - 6) +
                        phdz4r[kb][4] * _u1(i, j + 1, nz - 5) +
                        phdz4r[kb][3] * _u1(i, j + 1, nz - 4) +
                        phdz4r[kb][2] * _u1(i, j + 1, nz - 3) +
                        phdz4r[kb][1] * _u1(i, j + 1, nz - 2) +
                        phdz4r[kb][0] * _u1(i, j + 1, nz - 1)) +
                   py4[3] * _f2_1(i, j + 2) *
                       (phdz4r[kb][8] * _u1(i, j + 2, nz - 9) +
                        phdz4r[kb][7] * _u1(i, j + 2, nz - 8) +
                        phdz4r[kb][6] * _u1(i, j + 2, nz - 7) +
                        phdz4r[kb][5] * _u1(i, j + 2, nz - 6) +
                        phdz4r[kb][4] * _u1(i, j + 2, nz - 5) +
                        phdz4r[kb][3] * _u1(i, j + 2, nz - 4) +
                        phdz4r[kb][2] * _u1(i, j + 2, nz - 3) +
                        phdz4r[kb][1] * _u1(i, j + 2, nz - 2) +
                        phdz4r[kb][0] * _u1(i, j + 2, nz - 1)));
  vs2 = dhx4[2] * _v1(i, j, nz - 1 - kc - 6) +
              dhx4[0] * _v1(i - 2, j, nz - 1 - kc - 6) +
              dhx4[1] * _v1(i - 1, j, nz - 1 - kc - 6) +
              dhx4[3] * _v1(i + 1, j, nz - 1 - kc - 6) -
              J12i * _g_c(nz - 1 - kc - 6) *
                  (phx4[2] * _f1_2(i, j) *
                       (phdz4r[kb][8] * _v1(i, j, nz - 9) +
                        phdz4r[kb][7] * _v1(i, j, nz - 8) +
                        phdz4r[kb][6] * _v1(i, j, nz - 7) +
                        phdz4r[kb][5] * _v1(i, j, nz - 6) +
                        phdz4r[kb][4] * _v1(i, j, nz - 5) +
                        phdz4r[kb][3] * _v1(i, j, nz - 4) +
                        phdz4r[kb][2] * _v1(i, j, nz - 3) +
                        phdz4r[kb][1] * _v1(i, j, nz - 2) +
                        phdz4r[kb][0] * _v1(i, j, nz - 1)) +
                   phx4[0] * _f1_2(i - 2, j) *
                       (phdz4r[kb][8] * _v1(i - 2, j, nz - 9) +
                        phdz4r[kb][7] * _v1(i - 2, j, nz - 8) +
                        phdz4r[kb][6] * _v1(i - 2, j, nz - 7) +
                        phdz4r[kb][5] * _v1(i - 2, j, nz - 6) +
                        phdz4r[kb][4] * _v1(i - 2, j, nz - 5) +
                        phdz4r[kb][3] * _v1(i - 2, j, nz - 4) +
                        phdz4r[kb][2] * _v1(i - 2, j, nz - 3) +
                        phdz4r[kb][1] * _v1(i - 2, j, nz - 2) +
                        phdz4r[kb][0] * _v1(i - 2, j, nz - 1)) +
                   phx4[1] * _f1_2(i - 1, j) *
                       (phdz4r[kb][8] * _v1(i - 1, j, nz - 9) +
                        phdz4r[kb][7] * _v1(i - 1, j, nz - 8) +
                        phdz4r[kb][6] * _v1(i - 1, j, nz - 7) +
                        phdz4r[kb][5] * _v1(i - 1, j, nz - 6) +
                        phdz4r[kb][4] * _v1(i - 1, j, nz - 5) +
                        phdz4r[kb][3] * _v1(i - 1, j, nz - 4) +
                        phdz4r[kb][2] * _v1(i - 1, j, nz - 3) +
                        phdz4r[kb][1] * _v1(i - 1, j, nz - 2) +
                        phdz4r[kb][0] * _v1(i - 1, j, nz - 1)) +
                   phx4[3] * _f1_2(i + 1, j) *
                       (phdz4r[kb][8] * _v1(i + 1, j, nz - 9) +
                        phdz4r[kb][7] * _v1(i + 1, j, nz - 8) +
                        phdz4r[kb][6] * _v1(i + 1, j, nz - 7) +
                        phdz4r[kb][5] * _v1(i + 1, j, nz - 6) +
                        phdz4r[kb][4] * _v1(i + 1, j, nz - 5) +
                        phdz4r[kb][3] * _v1(i + 1, j, nz - 4) +
                        phdz4r[kb][2] * _v1(i + 1, j, nz - 3) +
                        phdz4r[kb][1] * _v1(i + 1, j, nz - 2) +
                        phdz4r[kb][0] * _v1(i + 1, j, nz - 1)));
#else
    // Cartesian
    //TODO: Implement
#endif
    f_r      = r4[pos];
    f_rtmp   = h1*(vs1+vs2); 
    f_xy     = xy[pos]  + xmu1*(vs1+vs2) + vx1*f_r;
    r4[pos]  = f_vx2*f_r + f_wwo*f_rtmp; 
    f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);
    xy[pos]  = (f_xy + d_DT*f_rtmp)*f_dcrj;

    // xz
#ifdef CURVILINEAR
  _prec J13i = _f_1(i, j) * _g3(nz - 1 - kc - 6);
  J13i = 1.0 * 1.0 / J13i;
  vs1 =
      J13i * (dz4r[kb][6] * _u1(i, j, nz - 7) + dz4r[kb][5] * _u1(i, j, nz - 6) +
              dz4r[kb][4] * _u1(i, j, nz - 5) + dz4r[kb][3] * _u1(i, j, nz - 4) +
              dz4r[kb][2] * _u1(i, j, nz - 3) + dz4r[kb][1] * _u1(i, j, nz - 2) +
              dz4r[kb][0] * _u1(i, j, nz - 1));
  vs2 = dhx4[2] * _w1(i, j, nz - 1 - kc - 6) +
              dhx4[0] * _w1(i - 2, j, nz - 1 - kc - 6) +
              dhx4[1] * _w1(i - 1, j, nz - 1 - kc - 6) +
              dhx4[3] * _w1(i + 1, j, nz - 1 - kc - 6) -
              J13i * _g(nz - 1 - kc - 6) *
                  (phx4[2] * _f1_c(i, j) *
                       (pdhz4r[kb][8] * _w1(i, j, nz - 9) +
                        pdhz4r[kb][7] * _w1(i, j, nz - 8) +
                        pdhz4r[kb][6] * _w1(i, j, nz - 7) +
                        pdhz4r[kb][5] * _w1(i, j, nz - 6) +
                        pdhz4r[kb][4] * _w1(i, j, nz - 5) +
                        pdhz4r[kb][3] * _w1(i, j, nz - 4) +
                        pdhz4r[kb][2] * _w1(i, j, nz - 3) +
                        pdhz4r[kb][1] * _w1(i, j, nz - 2) +
                        pdhz4r[kb][0] * _w1(i, j, nz - 1)) +
                   phx4[0] * _f1_c(i - 2, j) *
                       (pdhz4r[kb][8] * _w1(i - 2, j, nz - 9) +
                        pdhz4r[kb][7] * _w1(i - 2, j, nz - 8) +
                        pdhz4r[kb][6] * _w1(i - 2, j, nz - 7) +
                        pdhz4r[kb][5] * _w1(i - 2, j, nz - 6) +
                        pdhz4r[kb][4] * _w1(i - 2, j, nz - 5) +
                        pdhz4r[kb][3] * _w1(i - 2, j, nz - 4) +
                        pdhz4r[kb][2] * _w1(i - 2, j, nz - 3) +
                        pdhz4r[kb][1] * _w1(i - 2, j, nz - 2) +
                        pdhz4r[kb][0] * _w1(i - 2, j, nz - 1)) +
                   phx4[1] * _f1_c(i - 1, j) *
                       (pdhz4r[kb][8] * _w1(i - 1, j, nz - 9) +
                        pdhz4r[kb][7] * _w1(i - 1, j, nz - 8) +
                        pdhz4r[kb][6] * _w1(i - 1, j, nz - 7) +
                        pdhz4r[kb][5] * _w1(i - 1, j, nz - 6) +
                        pdhz4r[kb][4] * _w1(i - 1, j, nz - 5) +
                        pdhz4r[kb][3] * _w1(i - 1, j, nz - 4) +
                        pdhz4r[kb][2] * _w1(i - 1, j, nz - 3) +
                        pdhz4r[kb][1] * _w1(i - 1, j, nz - 2) +
                        pdhz4r[kb][0] * _w1(i - 1, j, nz - 1)) +
                   phx4[3] * _f1_c(i + 1, j) *
                       (pdhz4r[kb][8] * _w1(i + 1, j, nz - 9) +
                        pdhz4r[kb][7] * _w1(i + 1, j, nz - 8) +
                        pdhz4r[kb][6] * _w1(i + 1, j, nz - 7) +
                        pdhz4r[kb][5] * _w1(i + 1, j, nz - 6) +
                        pdhz4r[kb][4] * _w1(i + 1, j, nz - 5) +
                        pdhz4r[kb][3] * _w1(i + 1, j, nz - 4) +
                        pdhz4r[kb][2] * _w1(i + 1, j, nz - 3) +
                        pdhz4r[kb][1] * _w1(i + 1, j, nz - 2) +
                        pdhz4r[kb][0] * _w1(i + 1, j, nz - 1)));
#else
    //TODO: Implement
#endif
    f_r     = r5[pos];
    f_rtmp  = h2*(vs1+vs2);
    f_xz    = xz[pos]  + xmu2*(vs1+vs2) + vx1*f_r; 
    r5[pos] = f_vx2*f_r + f_wwo*f_rtmp; 
    f_rtmp  = f_rtmp*(f_wwo-1.0f) + f_vx2*f_r*(1.0f-f_vx1); 
    xz[pos] = (f_xz + d_DT*f_rtmp)*f_dcrj;

    // yz

#ifdef CURVILINEAR
  _prec J23i = _f_2(i, j) * _g3(nz - 1 - kc - 6);
  J23i = 1.0 * 1.0 / J23i;
  vs1 =
      J23i * (dz4r[kb][6] * _v1(i, j, nz - 7) + dz4r[kb][5] * _v1(i, j, nz - 6) +
              dz4r[kb][4] * _v1(i, j, nz - 5) + dz4r[kb][3] * _v1(i, j, nz - 4) +
              dz4r[kb][2] * _v1(i, j, nz - 3) + dz4r[kb][1] * _v1(i, j, nz - 2) +
              dz4r[kb][0] * _v1(i, j, nz - 1));
  vs2 = dy4[1] * _w1(i, j, nz - 1 - kc - 6) +
              dy4[0] * _w1(i, j - 1, nz - 1 - kc - 6) +
              dy4[2] * _w1(i, j + 1, nz - 1 - kc - 6) +
              dy4[3] * _w1(i, j + 2, nz - 1 - kc - 6) -
              J23i * _g(nz - 1 - kc - 6) *
                  (py4[1] * _f2_c(i, j) *
                       (pdhz4r[kb][8] * _w1(i, j, nz - 9) +
                        pdhz4r[kb][7] * _w1(i, j, nz - 8) +
                        pdhz4r[kb][6] * _w1(i, j, nz - 7) +
                        pdhz4r[kb][5] * _w1(i, j, nz - 6) +
                        pdhz4r[kb][4] * _w1(i, j, nz - 5) +
                        pdhz4r[kb][3] * _w1(i, j, nz - 4) +
                        pdhz4r[kb][2] * _w1(i, j, nz - 3) +
                        pdhz4r[kb][1] * _w1(i, j, nz - 2) +
                        pdhz4r[kb][0] * _w1(i, j, nz - 1)) +
                   py4[0] * _f2_c(i, j - 1) *
                       (pdhz4r[kb][8] * _w1(i, j - 1, nz - 9) +
                        pdhz4r[kb][7] * _w1(i, j - 1, nz - 8) +
                        pdhz4r[kb][6] * _w1(i, j - 1, nz - 7) +
                        pdhz4r[kb][5] * _w1(i, j - 1, nz - 6) +
                        pdhz4r[kb][4] * _w1(i, j - 1, nz - 5) +
                        pdhz4r[kb][3] * _w1(i, j - 1, nz - 4) +
                        pdhz4r[kb][2] * _w1(i, j - 1, nz - 3) +
                        pdhz4r[kb][1] * _w1(i, j - 1, nz - 2) +
                        pdhz4r[kb][0] * _w1(i, j - 1, nz - 1)) +
                   py4[2] * _f2_c(i, j + 1) *
                       (pdhz4r[kb][8] * _w1(i, j + 1, nz - 9) +
                        pdhz4r[kb][7] * _w1(i, j + 1, nz - 8) +
                        pdhz4r[kb][6] * _w1(i, j + 1, nz - 7) +
                        pdhz4r[kb][5] * _w1(i, j + 1, nz - 6) +
                        pdhz4r[kb][4] * _w1(i, j + 1, nz - 5) +
                        pdhz4r[kb][3] * _w1(i, j + 1, nz - 4) +
                        pdhz4r[kb][2] * _w1(i, j + 1, nz - 3) +
                        pdhz4r[kb][1] * _w1(i, j + 1, nz - 2) +
                        pdhz4r[kb][0] * _w1(i, j + 1, nz - 1)) +
                   py4[3] * _f2_c(i, j + 2) *
                       (pdhz4r[kb][8] * _w1(i, j + 2, nz - 9) +
                        pdhz4r[kb][7] * _w1(i, j + 2, nz - 8) +
                        pdhz4r[kb][6] * _w1(i, j + 2, nz - 7) +
                        pdhz4r[kb][5] * _w1(i, j + 2, nz - 6) +
                        pdhz4r[kb][4] * _w1(i, j + 2, nz - 5) +
                        pdhz4r[kb][3] * _w1(i, j + 2, nz - 4) +
                        pdhz4r[kb][2] * _w1(i, j + 2, nz - 3) +
                        pdhz4r[kb][1] * _w1(i, j + 2, nz - 2) +
                        pdhz4r[kb][0] * _w1(i, j + 2, nz - 1)));
#else
    // Cartesian
    //TODO: Implement
#endif
           
    f_r     = r6[pos];
    f_rtmp  = h3*(vs1+vs2);
    f_yz    = yz[pos]  + xmu3*(vs1+vs2) + vx1*f_r;
    r6[pos] = f_vx2*f_r + f_wwo*f_rtmp;
    f_rtmp  = f_rtmp*(f_wwo-1.0f) + f_vx2*f_r*(1.0f-f_vx1); 
    yz[pos] = (f_yz + d_DT*f_rtmp)*f_dcrj; 

    pos     = pos_im1;
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
#undef _lami
#undef _mui
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


// Kernel functions without attenuation
__global__ void dtopo_str_110(
    _prec *__restrict__ s11, _prec *__restrict__ s12, _prec *__restrict__ s13,
    _prec *__restrict__ s22, _prec *__restrict__ s23, _prec *__restrict__ s33,
    _prec *__restrict__ u1, _prec *__restrict__ u2, _prec *__restrict__ u3,
    const _prec *__restrict__ dcrjx, const _prec *__restrict__ dcrjy,
    const _prec *__restrict__ dcrjz, const _prec *__restrict__ f,
    const _prec *__restrict__ f1_1, const _prec *__restrict__ f1_2,
    const _prec *__restrict__ f1_c, const _prec *__restrict__ f2_1,
    const _prec *__restrict__ f2_2, const _prec *__restrict__ f2_c,
    const _prec *__restrict__ f_1, const _prec *__restrict__ f_2,
    const _prec *__restrict__ f_c, const _prec *__restrict__ g,
    const _prec *__restrict__ g3, const _prec *__restrict__ g3_c,
    const _prec *__restrict__ g_c, const _prec *__restrict__ lami,
    const _prec *__restrict__ mui, const _prec a, const _prec nu, const int nx,
    const int ny, const int nz, const int bi, const int bj, const int ei,
    const int ej) {
  const _prec phz4l[6][7] = {
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
  const _prec phy4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const _prec px4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const _prec dhz4l[6][7] = {
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
  const _prec phdz4l[6][9] = {
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
  const _prec dx4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const _prec dhy4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const _prec phx4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const _prec py4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const _prec dy4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const _prec dhx4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const _prec dz4l[6][8] = {
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
  const _prec pdhz4l[6][9] = {
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
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ngsl + ny)
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
#define _lami(i, j, k)                                                         \
  lami[(k) + align +                                                           \
       (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +             \
       (2 * align + nz) * ((j) + ngsl + 2)]
#define _mui(i, j, k)                                                          \
  mui[(k) + align +                                                            \
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
    _prec Jii = _f_c(i, j) * _g3_c(k);
    Jii = 1.0 * 1.0 / Jii;
    _prec J12i = _f(i, j) * _g3_c(k);
    J12i = 1.0 * 1.0 / J12i;
    _prec J13i = _f_1(i, j) * _g3(k);
    J13i = 1.0 * 1.0 / J13i;
    _prec J23i = _f_2(i, j) * _g3(k);
    J23i = 1.0 * 1.0 / J23i;
    _prec lam =
        nu * 1.0 /
        (phz4l[k][0] *
             (phy4[2] *
                  (px4[1] * _lami(i, j, 0) + px4[0] * _lami(i - 1, j, 0) +
                   px4[2] * _lami(i + 1, j, 0) + px4[3] * _lami(i + 2, j, 0)) +
              phy4[0] * (px4[1] * _lami(i, j - 2, 0) +
                         px4[0] * _lami(i - 1, j - 2, 0) +
                         px4[2] * _lami(i + 1, j - 2, 0) +
                         px4[3] * _lami(i + 2, j - 2, 0)) +
              phy4[1] * (px4[1] * _lami(i, j - 1, 0) +
                         px4[0] * _lami(i - 1, j - 1, 0) +
                         px4[2] * _lami(i + 1, j - 1, 0) +
                         px4[3] * _lami(i + 2, j - 1, 0)) +
              phy4[3] * (px4[1] * _lami(i, j + 1, 0) +
                         px4[0] * _lami(i - 1, j + 1, 0) +
                         px4[2] * _lami(i + 1, j + 1, 0) +
                         px4[3] * _lami(i + 2, j + 1, 0))) +
         phz4l[k][1] *
             (phy4[2] *
                  (px4[1] * _lami(i, j, 1) + px4[0] * _lami(i - 1, j, 1) +
                   px4[2] * _lami(i + 1, j, 1) + px4[3] * _lami(i + 2, j, 1)) +
              phy4[0] * (px4[1] * _lami(i, j - 2, 1) +
                         px4[0] * _lami(i - 1, j - 2, 1) +
                         px4[2] * _lami(i + 1, j - 2, 1) +
                         px4[3] * _lami(i + 2, j - 2, 1)) +
              phy4[1] * (px4[1] * _lami(i, j - 1, 1) +
                         px4[0] * _lami(i - 1, j - 1, 1) +
                         px4[2] * _lami(i + 1, j - 1, 1) +
                         px4[3] * _lami(i + 2, j - 1, 1)) +
              phy4[3] * (px4[1] * _lami(i, j + 1, 1) +
                         px4[0] * _lami(i - 1, j + 1, 1) +
                         px4[2] * _lami(i + 1, j + 1, 1) +
                         px4[3] * _lami(i + 2, j + 1, 1))) +
         phz4l[k][2] *
             (phy4[2] *
                  (px4[1] * _lami(i, j, 2) + px4[0] * _lami(i - 1, j, 2) +
                   px4[2] * _lami(i + 1, j, 2) + px4[3] * _lami(i + 2, j, 2)) +
              phy4[0] * (px4[1] * _lami(i, j - 2, 2) +
                         px4[0] * _lami(i - 1, j - 2, 2) +
                         px4[2] * _lami(i + 1, j - 2, 2) +
                         px4[3] * _lami(i + 2, j - 2, 2)) +
              phy4[1] * (px4[1] * _lami(i, j - 1, 2) +
                         px4[0] * _lami(i - 1, j - 1, 2) +
                         px4[2] * _lami(i + 1, j - 1, 2) +
                         px4[3] * _lami(i + 2, j - 1, 2)) +
              phy4[3] * (px4[1] * _lami(i, j + 1, 2) +
                         px4[0] * _lami(i - 1, j + 1, 2) +
                         px4[2] * _lami(i + 1, j + 1, 2) +
                         px4[3] * _lami(i + 2, j + 1, 2))) +
         phz4l[k][3] *
             (phy4[2] *
                  (px4[1] * _lami(i, j, 3) + px4[0] * _lami(i - 1, j, 3) +
                   px4[2] * _lami(i + 1, j, 3) + px4[3] * _lami(i + 2, j, 3)) +
              phy4[0] * (px4[1] * _lami(i, j - 2, 3) +
                         px4[0] * _lami(i - 1, j - 2, 3) +
                         px4[2] * _lami(i + 1, j - 2, 3) +
                         px4[3] * _lami(i + 2, j - 2, 3)) +
              phy4[1] * (px4[1] * _lami(i, j - 1, 3) +
                         px4[0] * _lami(i - 1, j - 1, 3) +
                         px4[2] * _lami(i + 1, j - 1, 3) +
                         px4[3] * _lami(i + 2, j - 1, 3)) +
              phy4[3] * (px4[1] * _lami(i, j + 1, 3) +
                         px4[0] * _lami(i - 1, j + 1, 3) +
                         px4[2] * _lami(i + 1, j + 1, 3) +
                         px4[3] * _lami(i + 2, j + 1, 3))) +
         phz4l[k][4] *
             (phy4[2] *
                  (px4[1] * _lami(i, j, 4) + px4[0] * _lami(i - 1, j, 4) +
                   px4[2] * _lami(i + 1, j, 4) + px4[3] * _lami(i + 2, j, 4)) +
              phy4[0] * (px4[1] * _lami(i, j - 2, 4) +
                         px4[0] * _lami(i - 1, j - 2, 4) +
                         px4[2] * _lami(i + 1, j - 2, 4) +
                         px4[3] * _lami(i + 2, j - 2, 4)) +
              phy4[1] * (px4[1] * _lami(i, j - 1, 4) +
                         px4[0] * _lami(i - 1, j - 1, 4) +
                         px4[2] * _lami(i + 1, j - 1, 4) +
                         px4[3] * _lami(i + 2, j - 1, 4)) +
              phy4[3] * (px4[1] * _lami(i, j + 1, 4) +
                         px4[0] * _lami(i - 1, j + 1, 4) +
                         px4[2] * _lami(i + 1, j + 1, 4) +
                         px4[3] * _lami(i + 2, j + 1, 4))) +
         phz4l[k][5] *
             (phy4[2] *
                  (px4[1] * _lami(i, j, 5) + px4[0] * _lami(i - 1, j, 5) +
                   px4[2] * _lami(i + 1, j, 5) + px4[3] * _lami(i + 2, j, 5)) +
              phy4[0] * (px4[1] * _lami(i, j - 2, 5) +
                         px4[0] * _lami(i - 1, j - 2, 5) +
                         px4[2] * _lami(i + 1, j - 2, 5) +
                         px4[3] * _lami(i + 2, j - 2, 5)) +
              phy4[1] * (px4[1] * _lami(i, j - 1, 5) +
                         px4[0] * _lami(i - 1, j - 1, 5) +
                         px4[2] * _lami(i + 1, j - 1, 5) +
                         px4[3] * _lami(i + 2, j - 1, 5)) +
              phy4[3] * (px4[1] * _lami(i, j + 1, 5) +
                         px4[0] * _lami(i - 1, j + 1, 5) +
                         px4[2] * _lami(i + 1, j + 1, 5) +
                         px4[3] * _lami(i + 2, j + 1, 5))) +
         phz4l[k][6] *
             (phy4[2] *
                  (px4[1] * _lami(i, j, 6) + px4[0] * _lami(i - 1, j, 6) +
                   px4[2] * _lami(i + 1, j, 6) + px4[3] * _lami(i + 2, j, 6)) +
              phy4[0] * (px4[1] * _lami(i, j - 2, 6) +
                         px4[0] * _lami(i - 1, j - 2, 6) +
                         px4[2] * _lami(i + 1, j - 2, 6) +
                         px4[3] * _lami(i + 2, j - 2, 6)) +
              phy4[1] * (px4[1] * _lami(i, j - 1, 6) +
                         px4[0] * _lami(i - 1, j - 1, 6) +
                         px4[2] * _lami(i + 1, j - 1, 6) +
                         px4[3] * _lami(i + 2, j - 1, 6)) +
              phy4[3] * (px4[1] * _lami(i, j + 1, 6) +
                         px4[0] * _lami(i - 1, j + 1, 6) +
                         px4[2] * _lami(i + 1, j + 1, 6) +
                         px4[3] * _lami(i + 2, j + 1, 6))));
    _prec twomu =
        2 * nu * 1.0 /
        (phz4l[k][0] *
             (phy4[2] *
                  (px4[1] * _mui(i, j, 0) + px4[0] * _mui(i - 1, j, 0) +
                   px4[2] * _mui(i + 1, j, 0) + px4[3] * _mui(i + 2, j, 0)) +
              phy4[0] *
                  (px4[1] * _mui(i, j - 2, 0) + px4[0] * _mui(i - 1, j - 2, 0) +
                   px4[2] * _mui(i + 1, j - 2, 0) +
                   px4[3] * _mui(i + 2, j - 2, 0)) +
              phy4[1] *
                  (px4[1] * _mui(i, j - 1, 0) + px4[0] * _mui(i - 1, j - 1, 0) +
                   px4[2] * _mui(i + 1, j - 1, 0) +
                   px4[3] * _mui(i + 2, j - 1, 0)) +
              phy4[3] *
                  (px4[1] * _mui(i, j + 1, 0) + px4[0] * _mui(i - 1, j + 1, 0) +
                   px4[2] * _mui(i + 1, j + 1, 0) +
                   px4[3] * _mui(i + 2, j + 1, 0))) +
         phz4l[k][1] *
             (phy4[2] *
                  (px4[1] * _mui(i, j, 1) + px4[0] * _mui(i - 1, j, 1) +
                   px4[2] * _mui(i + 1, j, 1) + px4[3] * _mui(i + 2, j, 1)) +
              phy4[0] *
                  (px4[1] * _mui(i, j - 2, 1) + px4[0] * _mui(i - 1, j - 2, 1) +
                   px4[2] * _mui(i + 1, j - 2, 1) +
                   px4[3] * _mui(i + 2, j - 2, 1)) +
              phy4[1] *
                  (px4[1] * _mui(i, j - 1, 1) + px4[0] * _mui(i - 1, j - 1, 1) +
                   px4[2] * _mui(i + 1, j - 1, 1) +
                   px4[3] * _mui(i + 2, j - 1, 1)) +
              phy4[3] *
                  (px4[1] * _mui(i, j + 1, 1) + px4[0] * _mui(i - 1, j + 1, 1) +
                   px4[2] * _mui(i + 1, j + 1, 1) +
                   px4[3] * _mui(i + 2, j + 1, 1))) +
         phz4l[k][2] *
             (phy4[2] *
                  (px4[1] * _mui(i, j, 2) + px4[0] * _mui(i - 1, j, 2) +
                   px4[2] * _mui(i + 1, j, 2) + px4[3] * _mui(i + 2, j, 2)) +
              phy4[0] *
                  (px4[1] * _mui(i, j - 2, 2) + px4[0] * _mui(i - 1, j - 2, 2) +
                   px4[2] * _mui(i + 1, j - 2, 2) +
                   px4[3] * _mui(i + 2, j - 2, 2)) +
              phy4[1] *
                  (px4[1] * _mui(i, j - 1, 2) + px4[0] * _mui(i - 1, j - 1, 2) +
                   px4[2] * _mui(i + 1, j - 1, 2) +
                   px4[3] * _mui(i + 2, j - 1, 2)) +
              phy4[3] *
                  (px4[1] * _mui(i, j + 1, 2) + px4[0] * _mui(i - 1, j + 1, 2) +
                   px4[2] * _mui(i + 1, j + 1, 2) +
                   px4[3] * _mui(i + 2, j + 1, 2))) +
         phz4l[k][3] *
             (phy4[2] *
                  (px4[1] * _mui(i, j, 3) + px4[0] * _mui(i - 1, j, 3) +
                   px4[2] * _mui(i + 1, j, 3) + px4[3] * _mui(i + 2, j, 3)) +
              phy4[0] *
                  (px4[1] * _mui(i, j - 2, 3) + px4[0] * _mui(i - 1, j - 2, 3) +
                   px4[2] * _mui(i + 1, j - 2, 3) +
                   px4[3] * _mui(i + 2, j - 2, 3)) +
              phy4[1] *
                  (px4[1] * _mui(i, j - 1, 3) + px4[0] * _mui(i - 1, j - 1, 3) +
                   px4[2] * _mui(i + 1, j - 1, 3) +
                   px4[3] * _mui(i + 2, j - 1, 3)) +
              phy4[3] *
                  (px4[1] * _mui(i, j + 1, 3) + px4[0] * _mui(i - 1, j + 1, 3) +
                   px4[2] * _mui(i + 1, j + 1, 3) +
                   px4[3] * _mui(i + 2, j + 1, 3))) +
         phz4l[k][4] *
             (phy4[2] *
                  (px4[1] * _mui(i, j, 4) + px4[0] * _mui(i - 1, j, 4) +
                   px4[2] * _mui(i + 1, j, 4) + px4[3] * _mui(i + 2, j, 4)) +
              phy4[0] *
                  (px4[1] * _mui(i, j - 2, 4) + px4[0] * _mui(i - 1, j - 2, 4) +
                   px4[2] * _mui(i + 1, j - 2, 4) +
                   px4[3] * _mui(i + 2, j - 2, 4)) +
              phy4[1] *
                  (px4[1] * _mui(i, j - 1, 4) + px4[0] * _mui(i - 1, j - 1, 4) +
                   px4[2] * _mui(i + 1, j - 1, 4) +
                   px4[3] * _mui(i + 2, j - 1, 4)) +
              phy4[3] *
                  (px4[1] * _mui(i, j + 1, 4) + px4[0] * _mui(i - 1, j + 1, 4) +
                   px4[2] * _mui(i + 1, j + 1, 4) +
                   px4[3] * _mui(i + 2, j + 1, 4))) +
         phz4l[k][5] *
             (phy4[2] *
                  (px4[1] * _mui(i, j, 5) + px4[0] * _mui(i - 1, j, 5) +
                   px4[2] * _mui(i + 1, j, 5) + px4[3] * _mui(i + 2, j, 5)) +
              phy4[0] *
                  (px4[1] * _mui(i, j - 2, 5) + px4[0] * _mui(i - 1, j - 2, 5) +
                   px4[2] * _mui(i + 1, j - 2, 5) +
                   px4[3] * _mui(i + 2, j - 2, 5)) +
              phy4[1] *
                  (px4[1] * _mui(i, j - 1, 5) + px4[0] * _mui(i - 1, j - 1, 5) +
                   px4[2] * _mui(i + 1, j - 1, 5) +
                   px4[3] * _mui(i + 2, j - 1, 5)) +
              phy4[3] *
                  (px4[1] * _mui(i, j + 1, 5) + px4[0] * _mui(i - 1, j + 1, 5) +
                   px4[2] * _mui(i + 1, j + 1, 5) +
                   px4[3] * _mui(i + 2, j + 1, 5))) +
         phz4l[k][6] *
             (phy4[2] *
                  (px4[1] * _mui(i, j, 6) + px4[0] * _mui(i - 1, j, 6) +
                   px4[2] * _mui(i + 1, j, 6) + px4[3] * _mui(i + 2, j, 6)) +
              phy4[0] *
                  (px4[1] * _mui(i, j - 2, 6) + px4[0] * _mui(i - 1, j - 2, 6) +
                   px4[2] * _mui(i + 1, j - 2, 6) +
                   px4[3] * _mui(i + 2, j - 2, 6)) +
              phy4[1] *
                  (px4[1] * _mui(i, j - 1, 6) + px4[0] * _mui(i - 1, j - 1, 6) +
                   px4[2] * _mui(i + 1, j - 1, 6) +
                   px4[3] * _mui(i + 2, j - 1, 6)) +
              phy4[3] *
                  (px4[1] * _mui(i, j + 1, 6) + px4[0] * _mui(i - 1, j + 1, 6) +
                   px4[2] * _mui(i + 1, j + 1, 6) +
                   px4[3] * _mui(i + 2, j + 1, 6))));
    _prec mu12 = nu * 1.0 /
                 (phz4l[k][0] * _mui(i, j, 0) + phz4l[k][1] * _mui(i, j, 1) +
                  phz4l[k][2] * _mui(i, j, 2) + phz4l[k][3] * _mui(i, j, 3) +
                  phz4l[k][4] * _mui(i, j, 4) + phz4l[k][5] * _mui(i, j, 5) +
                  phz4l[k][6] * _mui(i, j, 6));
    _prec mu13 = nu * 1.0 /
                 (phy4[2] * _mui(i, j, k) + phy4[0] * _mui(i, j - 2, k) +
                  phy4[1] * _mui(i, j - 1, k) + phy4[3] * _mui(i, j + 1, k));
    _prec mu23 = nu * 1.0 /
                 (px4[1] * _mui(i, j, k) + px4[0] * _mui(i - 1, j, k) +
                  px4[2] * _mui(i + 1, j, k) + px4[3] * _mui(i + 2, j, k));
    _prec div =
        dhy4[2] * _u2(i, j, k) + dhy4[0] * _u2(i, j - 2, k) +
        dhy4[1] * _u2(i, j - 1, k) + dhy4[3] * _u2(i, j + 1, k) +
        dx4[1] * _u1(i, j, k) + dx4[0] * _u1(i - 1, j, k) +
        dx4[2] * _u1(i + 1, j, k) + dx4[3] * _u1(i + 2, j, k) +
        Jii * (dhz4l[k][0] * _u3(i, j, 0) + dhz4l[k][1] * _u3(i, j, 1) +
               dhz4l[k][2] * _u3(i, j, 2) + dhz4l[k][3] * _u3(i, j, 3) +
               dhz4l[k][4] * _u3(i, j, 4) + dhz4l[k][5] * _u3(i, j, 5) +
               dhz4l[k][6] * _u3(i, j, 6)) -
        Jii * _g_c(k) *
            (phy4[2] * _f2_2(i, j) *
                 (phdz4l[k][0] * _u2(i, j, 0) + phdz4l[k][1] * _u2(i, j, 1) +
                  phdz4l[k][2] * _u2(i, j, 2) + phdz4l[k][3] * _u2(i, j, 3) +
                  phdz4l[k][4] * _u2(i, j, 4) + phdz4l[k][5] * _u2(i, j, 5) +
                  phdz4l[k][6] * _u2(i, j, 6) + phdz4l[k][7] * _u2(i, j, 7) +
                  phdz4l[k][8] * _u2(i, j, 8)) +
             phy4[0] * _f2_2(i, j - 2) *
                 (phdz4l[k][0] * _u2(i, j - 2, 0) +
                  phdz4l[k][1] * _u2(i, j - 2, 1) +
                  phdz4l[k][2] * _u2(i, j - 2, 2) +
                  phdz4l[k][3] * _u2(i, j - 2, 3) +
                  phdz4l[k][4] * _u2(i, j - 2, 4) +
                  phdz4l[k][5] * _u2(i, j - 2, 5) +
                  phdz4l[k][6] * _u2(i, j - 2, 6) +
                  phdz4l[k][7] * _u2(i, j - 2, 7) +
                  phdz4l[k][8] * _u2(i, j - 2, 8)) +
             phy4[1] * _f2_2(i, j - 1) *
                 (phdz4l[k][0] * _u2(i, j - 1, 0) +
                  phdz4l[k][1] * _u2(i, j - 1, 1) +
                  phdz4l[k][2] * _u2(i, j - 1, 2) +
                  phdz4l[k][3] * _u2(i, j - 1, 3) +
                  phdz4l[k][4] * _u2(i, j - 1, 4) +
                  phdz4l[k][5] * _u2(i, j - 1, 5) +
                  phdz4l[k][6] * _u2(i, j - 1, 6) +
                  phdz4l[k][7] * _u2(i, j - 1, 7) +
                  phdz4l[k][8] * _u2(i, j - 1, 8)) +
             phy4[3] * _f2_2(i, j + 1) *
                 (phdz4l[k][0] * _u2(i, j + 1, 0) +
                  phdz4l[k][1] * _u2(i, j + 1, 1) +
                  phdz4l[k][2] * _u2(i, j + 1, 2) +
                  phdz4l[k][3] * _u2(i, j + 1, 3) +
                  phdz4l[k][4] * _u2(i, j + 1, 4) +
                  phdz4l[k][5] * _u2(i, j + 1, 5) +
                  phdz4l[k][6] * _u2(i, j + 1, 6) +
                  phdz4l[k][7] * _u2(i, j + 1, 7) +
                  phdz4l[k][8] * _u2(i, j + 1, 8))) -
        Jii * _g_c(k) *
            (px4[1] * _f1_1(i, j) *
                 (phdz4l[k][0] * _u1(i, j, 0) + phdz4l[k][1] * _u1(i, j, 1) +
                  phdz4l[k][2] * _u1(i, j, 2) + phdz4l[k][3] * _u1(i, j, 3) +
                  phdz4l[k][4] * _u1(i, j, 4) + phdz4l[k][5] * _u1(i, j, 5) +
                  phdz4l[k][6] * _u1(i, j, 6) + phdz4l[k][7] * _u1(i, j, 7) +
                  phdz4l[k][8] * _u1(i, j, 8)) +
             px4[0] * _f1_1(i - 1, j) *
                 (phdz4l[k][0] * _u1(i - 1, j, 0) +
                  phdz4l[k][1] * _u1(i - 1, j, 1) +
                  phdz4l[k][2] * _u1(i - 1, j, 2) +
                  phdz4l[k][3] * _u1(i - 1, j, 3) +
                  phdz4l[k][4] * _u1(i - 1, j, 4) +
                  phdz4l[k][5] * _u1(i - 1, j, 5) +
                  phdz4l[k][6] * _u1(i - 1, j, 6) +
                  phdz4l[k][7] * _u1(i - 1, j, 7) +
                  phdz4l[k][8] * _u1(i - 1, j, 8)) +
             px4[2] * _f1_1(i + 1, j) *
                 (phdz4l[k][0] * _u1(i + 1, j, 0) +
                  phdz4l[k][1] * _u1(i + 1, j, 1) +
                  phdz4l[k][2] * _u1(i + 1, j, 2) +
                  phdz4l[k][3] * _u1(i + 1, j, 3) +
                  phdz4l[k][4] * _u1(i + 1, j, 4) +
                  phdz4l[k][5] * _u1(i + 1, j, 5) +
                  phdz4l[k][6] * _u1(i + 1, j, 6) +
                  phdz4l[k][7] * _u1(i + 1, j, 7) +
                  phdz4l[k][8] * _u1(i + 1, j, 8)) +
             px4[3] * _f1_1(i + 2, j) *
                 (phdz4l[k][0] * _u1(i + 2, j, 0) +
                  phdz4l[k][1] * _u1(i + 2, j, 1) +
                  phdz4l[k][2] * _u1(i + 2, j, 2) +
                  phdz4l[k][3] * _u1(i + 2, j, 3) +
                  phdz4l[k][4] * _u1(i + 2, j, 4) +
                  phdz4l[k][5] * _u1(i + 2, j, 5) +
                  phdz4l[k][6] * _u1(i + 2, j, 6) +
                  phdz4l[k][7] * _u1(i + 2, j, 7) +
                  phdz4l[k][8] * _u1(i + 2, j, 8)));
    _prec f_dcrj = _dcrjx(i) * _dcrjy(j) * _dcrjz(k);
    _s11(i, j, k) =
        (a * _s11(i, j, k) + lam * div +
         twomu * (dx4[1] * _u1(i, j, k) + dx4[0] * _u1(i - 1, j, k) +
                  dx4[2] * _u1(i + 1, j, k) + dx4[3] * _u1(i + 2, j, k)) -
         twomu * Jii * _g_c(k) *
             (px4[1] * _f1_1(i, j) *
                  (phdz4l[k][0] * _u1(i, j, 0) + phdz4l[k][1] * _u1(i, j, 1) +
                   phdz4l[k][2] * _u1(i, j, 2) + phdz4l[k][3] * _u1(i, j, 3) +
                   phdz4l[k][4] * _u1(i, j, 4) + phdz4l[k][5] * _u1(i, j, 5) +
                   phdz4l[k][6] * _u1(i, j, 6) + phdz4l[k][7] * _u1(i, j, 7) +
                   phdz4l[k][8] * _u1(i, j, 8)) +
              px4[0] * _f1_1(i - 1, j) *
                  (phdz4l[k][0] * _u1(i - 1, j, 0) +
                   phdz4l[k][1] * _u1(i - 1, j, 1) +
                   phdz4l[k][2] * _u1(i - 1, j, 2) +
                   phdz4l[k][3] * _u1(i - 1, j, 3) +
                   phdz4l[k][4] * _u1(i - 1, j, 4) +
                   phdz4l[k][5] * _u1(i - 1, j, 5) +
                   phdz4l[k][6] * _u1(i - 1, j, 6) +
                   phdz4l[k][7] * _u1(i - 1, j, 7) +
                   phdz4l[k][8] * _u1(i - 1, j, 8)) +
              px4[2] * _f1_1(i + 1, j) *
                  (phdz4l[k][0] * _u1(i + 1, j, 0) +
                   phdz4l[k][1] * _u1(i + 1, j, 1) +
                   phdz4l[k][2] * _u1(i + 1, j, 2) +
                   phdz4l[k][3] * _u1(i + 1, j, 3) +
                   phdz4l[k][4] * _u1(i + 1, j, 4) +
                   phdz4l[k][5] * _u1(i + 1, j, 5) +
                   phdz4l[k][6] * _u1(i + 1, j, 6) +
                   phdz4l[k][7] * _u1(i + 1, j, 7) +
                   phdz4l[k][8] * _u1(i + 1, j, 8)) +
              px4[3] * _f1_1(i + 2, j) *
                  (phdz4l[k][0] * _u1(i + 2, j, 0) +
                   phdz4l[k][1] * _u1(i + 2, j, 1) +
                   phdz4l[k][2] * _u1(i + 2, j, 2) +
                   phdz4l[k][3] * _u1(i + 2, j, 3) +
                   phdz4l[k][4] * _u1(i + 2, j, 4) +
                   phdz4l[k][5] * _u1(i + 2, j, 5) +
                   phdz4l[k][6] * _u1(i + 2, j, 6) +
                   phdz4l[k][7] * _u1(i + 2, j, 7) +
                   phdz4l[k][8] * _u1(i + 2, j, 8)))) *
        f_dcrj;
    _s22(i, j, k) =
        (a * _s22(i, j, k) + lam * div +
         twomu * (dhy4[2] * _u2(i, j, k) + dhy4[0] * _u2(i, j - 2, k) +
                  dhy4[1] * _u2(i, j - 1, k) + dhy4[3] * _u2(i, j + 1, k)) -
         twomu * Jii * _g_c(k) *
             (phy4[2] * _f2_2(i, j) *
                  (phdz4l[k][0] * _u2(i, j, 0) + phdz4l[k][1] * _u2(i, j, 1) +
                   phdz4l[k][2] * _u2(i, j, 2) + phdz4l[k][3] * _u2(i, j, 3) +
                   phdz4l[k][4] * _u2(i, j, 4) + phdz4l[k][5] * _u2(i, j, 5) +
                   phdz4l[k][6] * _u2(i, j, 6) + phdz4l[k][7] * _u2(i, j, 7) +
                   phdz4l[k][8] * _u2(i, j, 8)) +
              phy4[0] * _f2_2(i, j - 2) *
                  (phdz4l[k][0] * _u2(i, j - 2, 0) +
                   phdz4l[k][1] * _u2(i, j - 2, 1) +
                   phdz4l[k][2] * _u2(i, j - 2, 2) +
                   phdz4l[k][3] * _u2(i, j - 2, 3) +
                   phdz4l[k][4] * _u2(i, j - 2, 4) +
                   phdz4l[k][5] * _u2(i, j - 2, 5) +
                   phdz4l[k][6] * _u2(i, j - 2, 6) +
                   phdz4l[k][7] * _u2(i, j - 2, 7) +
                   phdz4l[k][8] * _u2(i, j - 2, 8)) +
              phy4[1] * _f2_2(i, j - 1) *
                  (phdz4l[k][0] * _u2(i, j - 1, 0) +
                   phdz4l[k][1] * _u2(i, j - 1, 1) +
                   phdz4l[k][2] * _u2(i, j - 1, 2) +
                   phdz4l[k][3] * _u2(i, j - 1, 3) +
                   phdz4l[k][4] * _u2(i, j - 1, 4) +
                   phdz4l[k][5] * _u2(i, j - 1, 5) +
                   phdz4l[k][6] * _u2(i, j - 1, 6) +
                   phdz4l[k][7] * _u2(i, j - 1, 7) +
                   phdz4l[k][8] * _u2(i, j - 1, 8)) +
              phy4[3] * _f2_2(i, j + 1) *
                  (phdz4l[k][0] * _u2(i, j + 1, 0) +
                   phdz4l[k][1] * _u2(i, j + 1, 1) +
                   phdz4l[k][2] * _u2(i, j + 1, 2) +
                   phdz4l[k][3] * _u2(i, j + 1, 3) +
                   phdz4l[k][4] * _u2(i, j + 1, 4) +
                   phdz4l[k][5] * _u2(i, j + 1, 5) +
                   phdz4l[k][6] * _u2(i, j + 1, 6) +
                   phdz4l[k][7] * _u2(i, j + 1, 7) +
                   phdz4l[k][8] * _u2(i, j + 1, 8)))) *
        f_dcrj;
    _s33(i, j, k) =
        (a * _s33(i, j, k) + lam * div +
         twomu * Jii *
             (dhz4l[k][0] * _u3(i, j, 0) + dhz4l[k][1] * _u3(i, j, 1) +
              dhz4l[k][2] * _u3(i, j, 2) + dhz4l[k][3] * _u3(i, j, 3) +
              dhz4l[k][4] * _u3(i, j, 4) + dhz4l[k][5] * _u3(i, j, 5) +
              dhz4l[k][6] * _u3(i, j, 6))) *
        f_dcrj;
    _s12(i, j, k) =
        (a * _s12(i, j, k) +
         mu12 * (dhx4[2] * _u2(i, j, k) + dhx4[0] * _u2(i - 2, j, k) +
                 dhx4[1] * _u2(i - 1, j, k) + dhx4[3] * _u2(i + 1, j, k) +
                 dy4[1] * _u1(i, j, k) + dy4[0] * _u1(i, j - 1, k) +
                 dy4[2] * _u1(i, j + 1, k) + dy4[3] * _u1(i, j + 2, k) -
                 J12i * _g_c(k) *
                     (phx4[2] * _f1_2(i, j) *
                          (phdz4l[k][0] * _u2(i, j, 0) +
                           phdz4l[k][1] * _u2(i, j, 1) +
                           phdz4l[k][2] * _u2(i, j, 2) +
                           phdz4l[k][3] * _u2(i, j, 3) +
                           phdz4l[k][4] * _u2(i, j, 4) +
                           phdz4l[k][5] * _u2(i, j, 5) +
                           phdz4l[k][6] * _u2(i, j, 6) +
                           phdz4l[k][7] * _u2(i, j, 7) +
                           phdz4l[k][8] * _u2(i, j, 8)) +
                      phx4[0] * _f1_2(i - 2, j) *
                          (phdz4l[k][0] * _u2(i - 2, j, 0) +
                           phdz4l[k][1] * _u2(i - 2, j, 1) +
                           phdz4l[k][2] * _u2(i - 2, j, 2) +
                           phdz4l[k][3] * _u2(i - 2, j, 3) +
                           phdz4l[k][4] * _u2(i - 2, j, 4) +
                           phdz4l[k][5] * _u2(i - 2, j, 5) +
                           phdz4l[k][6] * _u2(i - 2, j, 6) +
                           phdz4l[k][7] * _u2(i - 2, j, 7) +
                           phdz4l[k][8] * _u2(i - 2, j, 8)) +
                      phx4[1] * _f1_2(i - 1, j) *
                          (phdz4l[k][0] * _u2(i - 1, j, 0) +
                           phdz4l[k][1] * _u2(i - 1, j, 1) +
                           phdz4l[k][2] * _u2(i - 1, j, 2) +
                           phdz4l[k][3] * _u2(i - 1, j, 3) +
                           phdz4l[k][4] * _u2(i - 1, j, 4) +
                           phdz4l[k][5] * _u2(i - 1, j, 5) +
                           phdz4l[k][6] * _u2(i - 1, j, 6) +
                           phdz4l[k][7] * _u2(i - 1, j, 7) +
                           phdz4l[k][8] * _u2(i - 1, j, 8)) +
                      phx4[3] * _f1_2(i + 1, j) *
                          (phdz4l[k][0] * _u2(i + 1, j, 0) +
                           phdz4l[k][1] * _u2(i + 1, j, 1) +
                           phdz4l[k][2] * _u2(i + 1, j, 2) +
                           phdz4l[k][3] * _u2(i + 1, j, 3) +
                           phdz4l[k][4] * _u2(i + 1, j, 4) +
                           phdz4l[k][5] * _u2(i + 1, j, 5) +
                           phdz4l[k][6] * _u2(i + 1, j, 6) +
                           phdz4l[k][7] * _u2(i + 1, j, 7) +
                           phdz4l[k][8] * _u2(i + 1, j, 8))) -
                 J12i * _g_c(k) *
                     (py4[1] * _f2_1(i, j) *
                          (phdz4l[k][0] * _u1(i, j, 0) +
                           phdz4l[k][1] * _u1(i, j, 1) +
                           phdz4l[k][2] * _u1(i, j, 2) +
                           phdz4l[k][3] * _u1(i, j, 3) +
                           phdz4l[k][4] * _u1(i, j, 4) +
                           phdz4l[k][5] * _u1(i, j, 5) +
                           phdz4l[k][6] * _u1(i, j, 6) +
                           phdz4l[k][7] * _u1(i, j, 7) +
                           phdz4l[k][8] * _u1(i, j, 8)) +
                      py4[0] * _f2_1(i, j - 1) *
                          (phdz4l[k][0] * _u1(i, j - 1, 0) +
                           phdz4l[k][1] * _u1(i, j - 1, 1) +
                           phdz4l[k][2] * _u1(i, j - 1, 2) +
                           phdz4l[k][3] * _u1(i, j - 1, 3) +
                           phdz4l[k][4] * _u1(i, j - 1, 4) +
                           phdz4l[k][5] * _u1(i, j - 1, 5) +
                           phdz4l[k][6] * _u1(i, j - 1, 6) +
                           phdz4l[k][7] * _u1(i, j - 1, 7) +
                           phdz4l[k][8] * _u1(i, j - 1, 8)) +
                      py4[2] * _f2_1(i, j + 1) *
                          (phdz4l[k][0] * _u1(i, j + 1, 0) +
                           phdz4l[k][1] * _u1(i, j + 1, 1) +
                           phdz4l[k][2] * _u1(i, j + 1, 2) +
                           phdz4l[k][3] * _u1(i, j + 1, 3) +
                           phdz4l[k][4] * _u1(i, j + 1, 4) +
                           phdz4l[k][5] * _u1(i, j + 1, 5) +
                           phdz4l[k][6] * _u1(i, j + 1, 6) +
                           phdz4l[k][7] * _u1(i, j + 1, 7) +
                           phdz4l[k][8] * _u1(i, j + 1, 8)) +
                      py4[3] * _f2_1(i, j + 2) *
                          (phdz4l[k][0] * _u1(i, j + 2, 0) +
                           phdz4l[k][1] * _u1(i, j + 2, 1) +
                           phdz4l[k][2] * _u1(i, j + 2, 2) +
                           phdz4l[k][3] * _u1(i, j + 2, 3) +
                           phdz4l[k][4] * _u1(i, j + 2, 4) +
                           phdz4l[k][5] * _u1(i, j + 2, 5) +
                           phdz4l[k][6] * _u1(i, j + 2, 6) +
                           phdz4l[k][7] * _u1(i, j + 2, 7) +
                           phdz4l[k][8] * _u1(i, j + 2, 8))))) *
        f_dcrj;
    _s13(i, j, k) =
        (a * _s13(i, j, k) +
         mu13 *
             (dhx4[2] * _u3(i, j, k) + dhx4[0] * _u3(i - 2, j, k) +
              dhx4[1] * _u3(i - 1, j, k) + dhx4[3] * _u3(i + 1, j, k) +
              J13i * (dz4l[k][0] * _u1(i, j, 0) + dz4l[k][1] * _u1(i, j, 1) +
                      dz4l[k][2] * _u1(i, j, 2) + dz4l[k][3] * _u1(i, j, 3) +
                      dz4l[k][4] * _u1(i, j, 4) + dz4l[k][5] * _u1(i, j, 5) +
                      dz4l[k][6] * _u1(i, j, 6) + dz4l[k][7] * _u1(i, j, 7)) -
              J13i * _g(k) *
                  (phx4[2] * _f1_c(i, j) *
                       (pdhz4l[k][0] * _u3(i, j, 0) +
                        pdhz4l[k][1] * _u3(i, j, 1) +
                        pdhz4l[k][2] * _u3(i, j, 2) +
                        pdhz4l[k][3] * _u3(i, j, 3) +
                        pdhz4l[k][4] * _u3(i, j, 4) +
                        pdhz4l[k][5] * _u3(i, j, 5) +
                        pdhz4l[k][6] * _u3(i, j, 6) +
                        pdhz4l[k][7] * _u3(i, j, 7) +
                        pdhz4l[k][8] * _u3(i, j, 8)) +
                   phx4[0] * _f1_c(i - 2, j) *
                       (pdhz4l[k][0] * _u3(i - 2, j, 0) +
                        pdhz4l[k][1] * _u3(i - 2, j, 1) +
                        pdhz4l[k][2] * _u3(i - 2, j, 2) +
                        pdhz4l[k][3] * _u3(i - 2, j, 3) +
                        pdhz4l[k][4] * _u3(i - 2, j, 4) +
                        pdhz4l[k][5] * _u3(i - 2, j, 5) +
                        pdhz4l[k][6] * _u3(i - 2, j, 6) +
                        pdhz4l[k][7] * _u3(i - 2, j, 7) +
                        pdhz4l[k][8] * _u3(i - 2, j, 8)) +
                   phx4[1] * _f1_c(i - 1, j) *
                       (pdhz4l[k][0] * _u3(i - 1, j, 0) +
                        pdhz4l[k][1] * _u3(i - 1, j, 1) +
                        pdhz4l[k][2] * _u3(i - 1, j, 2) +
                        pdhz4l[k][3] * _u3(i - 1, j, 3) +
                        pdhz4l[k][4] * _u3(i - 1, j, 4) +
                        pdhz4l[k][5] * _u3(i - 1, j, 5) +
                        pdhz4l[k][6] * _u3(i - 1, j, 6) +
                        pdhz4l[k][7] * _u3(i - 1, j, 7) +
                        pdhz4l[k][8] * _u3(i - 1, j, 8)) +
                   phx4[3] * _f1_c(i + 1, j) *
                       (pdhz4l[k][0] * _u3(i + 1, j, 0) +
                        pdhz4l[k][1] * _u3(i + 1, j, 1) +
                        pdhz4l[k][2] * _u3(i + 1, j, 2) +
                        pdhz4l[k][3] * _u3(i + 1, j, 3) +
                        pdhz4l[k][4] * _u3(i + 1, j, 4) +
                        pdhz4l[k][5] * _u3(i + 1, j, 5) +
                        pdhz4l[k][6] * _u3(i + 1, j, 6) +
                        pdhz4l[k][7] * _u3(i + 1, j, 7) +
                        pdhz4l[k][8] * _u3(i + 1, j, 8))))) *
        f_dcrj;
    _s23(i, j, k) =
        (a * _s23(i, j, k) +
         mu23 *
             (dy4[1] * _u3(i, j, k) + dy4[0] * _u3(i, j - 1, k) +
              dy4[2] * _u3(i, j + 1, k) + dy4[3] * _u3(i, j + 2, k) +
              J23i * (dz4l[k][0] * _u2(i, j, 0) + dz4l[k][1] * _u2(i, j, 1) +
                      dz4l[k][2] * _u2(i, j, 2) + dz4l[k][3] * _u2(i, j, 3) +
                      dz4l[k][4] * _u2(i, j, 4) + dz4l[k][5] * _u2(i, j, 5) +
                      dz4l[k][6] * _u2(i, j, 6) + dz4l[k][7] * _u2(i, j, 7)) -
              J23i * _g(k) *
                  (py4[1] * _f2_c(i, j) *
                       (pdhz4l[k][0] * _u3(i, j, 0) +
                        pdhz4l[k][1] * _u3(i, j, 1) +
                        pdhz4l[k][2] * _u3(i, j, 2) +
                        pdhz4l[k][3] * _u3(i, j, 3) +
                        pdhz4l[k][4] * _u3(i, j, 4) +
                        pdhz4l[k][5] * _u3(i, j, 5) +
                        pdhz4l[k][6] * _u3(i, j, 6) +
                        pdhz4l[k][7] * _u3(i, j, 7) +
                        pdhz4l[k][8] * _u3(i, j, 8)) +
                   py4[0] * _f2_c(i, j - 1) *
                       (pdhz4l[k][0] * _u3(i, j - 1, 0) +
                        pdhz4l[k][1] * _u3(i, j - 1, 1) +
                        pdhz4l[k][2] * _u3(i, j - 1, 2) +
                        pdhz4l[k][3] * _u3(i, j - 1, 3) +
                        pdhz4l[k][4] * _u3(i, j - 1, 4) +
                        pdhz4l[k][5] * _u3(i, j - 1, 5) +
                        pdhz4l[k][6] * _u3(i, j - 1, 6) +
                        pdhz4l[k][7] * _u3(i, j - 1, 7) +
                        pdhz4l[k][8] * _u3(i, j - 1, 8)) +
                   py4[2] * _f2_c(i, j + 1) *
                       (pdhz4l[k][0] * _u3(i, j + 1, 0) +
                        pdhz4l[k][1] * _u3(i, j + 1, 1) +
                        pdhz4l[k][2] * _u3(i, j + 1, 2) +
                        pdhz4l[k][3] * _u3(i, j + 1, 3) +
                        pdhz4l[k][4] * _u3(i, j + 1, 4) +
                        pdhz4l[k][5] * _u3(i, j + 1, 5) +
                        pdhz4l[k][6] * _u3(i, j + 1, 6) +
                        pdhz4l[k][7] * _u3(i, j + 1, 7) +
                        pdhz4l[k][8] * _u3(i, j + 1, 8)) +
                   py4[3] * _f2_c(i, j + 2) *
                       (pdhz4l[k][0] * _u3(i, j + 2, 0) +
                        pdhz4l[k][1] * _u3(i, j + 2, 1) +
                        pdhz4l[k][2] * _u3(i, j + 2, 2) +
                        pdhz4l[k][3] * _u3(i, j + 2, 3) +
                        pdhz4l[k][4] * _u3(i, j + 2, 4) +
                        pdhz4l[k][5] * _u3(i, j + 2, 5) +
                        pdhz4l[k][6] * _u3(i, j + 2, 6) +
                        pdhz4l[k][7] * _u3(i, j + 2, 7) +
                        pdhz4l[k][8] * _u3(i, j + 2, 8))))) *
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
#undef _lami
#undef _mui
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

__global__ void dtopo_str_111(
    _prec *__restrict__ s11, _prec *__restrict__ s12, _prec *__restrict__ s13,
    _prec *__restrict__ s22, _prec *__restrict__ s23, _prec *__restrict__ s33,
    _prec *__restrict__ u1, _prec *__restrict__ u2, _prec *__restrict__ u3,
    const _prec *__restrict__ dcrjx, const _prec *__restrict__ dcrjy,
    const _prec *__restrict__ dcrjz, const _prec *__restrict__ f,
    const _prec *__restrict__ f1_1, const _prec *__restrict__ f1_2,
    const _prec *__restrict__ f1_c, const _prec *__restrict__ f2_1,
    const _prec *__restrict__ f2_2, const _prec *__restrict__ f2_c,
    const _prec *__restrict__ f_1, const _prec *__restrict__ f_2,
    const _prec *__restrict__ f_c, const _prec *__restrict__ g,
    const _prec *__restrict__ g3, const _prec *__restrict__ g3_c,
    const _prec *__restrict__ g_c, const _prec *__restrict__ lami,
    const _prec *__restrict__ mui, const _prec a, const _prec nu, const int nx,
    const int ny, const int nz, const int bi, const int bj, const int ei,
    const int ej) {
  const _prec phz4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const _prec phy4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const _prec px4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const _prec dhz4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const _prec phdz4[7] = {-0.0026041666666667, 0.0937500000000000,
                          -0.6796875000000000, -0.0000000000000000,
                          0.6796875000000000,  -0.0937500000000000,
                          0.0026041666666667};
  const _prec dx4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const _prec dhy4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const _prec phx4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const _prec py4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const _prec dy4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const _prec dhx4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const _prec dz4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const _prec pdhz4[7] = {-0.0026041666666667, 0.0937500000000000,
                          -0.6796875000000000, -0.0000000000000000,
                          0.6796875000000000,  -0.0937500000000000,
                          0.0026041666666667};
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ngsl + ny)
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
#define _lami(i, j, k)                                                         \
  lami[(k) + align +                                                           \
       (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +             \
       (2 * align + nz) * ((j) + ngsl + 2)]
#define _mui(i, j, k)                                                          \
  mui[(k) + align +                                                            \
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
    _prec Jii = _f_c(i, j) * _g3_c(k + 6);
    Jii = 1.0 * 1.0 / Jii;
    _prec J12i = _f(i, j) * _g3_c(k + 6);
    J12i = 1.0 * 1.0 / J12i;
    _prec J13i = _f_1(i, j) * _g3(k + 6);
    J13i = 1.0 * 1.0 / J13i;
    _prec J23i = _f_2(i, j) * _g3(k + 6);
    J23i = 1.0 * 1.0 / J23i;
    _prec lam = nu * 1.0 /
                (phz4[0] * (phy4[2] * (px4[1] * _lami(i, j, k + 4) +
                                       px4[0] * _lami(i - 1, j, k + 4) +
                                       px4[2] * _lami(i + 1, j, k + 4) +
                                       px4[3] * _lami(i + 2, j, k + 4)) +
                            phy4[0] * (px4[1] * _lami(i, j - 2, k + 4) +
                                       px4[0] * _lami(i - 1, j - 2, k + 4) +
                                       px4[2] * _lami(i + 1, j - 2, k + 4) +
                                       px4[3] * _lami(i + 2, j - 2, k + 4)) +
                            phy4[1] * (px4[1] * _lami(i, j - 1, k + 4) +
                                       px4[0] * _lami(i - 1, j - 1, k + 4) +
                                       px4[2] * _lami(i + 1, j - 1, k + 4) +
                                       px4[3] * _lami(i + 2, j - 1, k + 4)) +
                            phy4[3] * (px4[1] * _lami(i, j + 1, k + 4) +
                                       px4[0] * _lami(i - 1, j + 1, k + 4) +
                                       px4[2] * _lami(i + 1, j + 1, k + 4) +
                                       px4[3] * _lami(i + 2, j + 1, k + 4))) +
                 phz4[1] * (phy4[2] * (px4[1] * _lami(i, j, k + 5) +
                                       px4[0] * _lami(i - 1, j, k + 5) +
                                       px4[2] * _lami(i + 1, j, k + 5) +
                                       px4[3] * _lami(i + 2, j, k + 5)) +
                            phy4[0] * (px4[1] * _lami(i, j - 2, k + 5) +
                                       px4[0] * _lami(i - 1, j - 2, k + 5) +
                                       px4[2] * _lami(i + 1, j - 2, k + 5) +
                                       px4[3] * _lami(i + 2, j - 2, k + 5)) +
                            phy4[1] * (px4[1] * _lami(i, j - 1, k + 5) +
                                       px4[0] * _lami(i - 1, j - 1, k + 5) +
                                       px4[2] * _lami(i + 1, j - 1, k + 5) +
                                       px4[3] * _lami(i + 2, j - 1, k + 5)) +
                            phy4[3] * (px4[1] * _lami(i, j + 1, k + 5) +
                                       px4[0] * _lami(i - 1, j + 1, k + 5) +
                                       px4[2] * _lami(i + 1, j + 1, k + 5) +
                                       px4[3] * _lami(i + 2, j + 1, k + 5))) +
                 phz4[2] * (phy4[2] * (px4[1] * _lami(i, j, k + 6) +
                                       px4[0] * _lami(i - 1, j, k + 6) +
                                       px4[2] * _lami(i + 1, j, k + 6) +
                                       px4[3] * _lami(i + 2, j, k + 6)) +
                            phy4[0] * (px4[1] * _lami(i, j - 2, k + 6) +
                                       px4[0] * _lami(i - 1, j - 2, k + 6) +
                                       px4[2] * _lami(i + 1, j - 2, k + 6) +
                                       px4[3] * _lami(i + 2, j - 2, k + 6)) +
                            phy4[1] * (px4[1] * _lami(i, j - 1, k + 6) +
                                       px4[0] * _lami(i - 1, j - 1, k + 6) +
                                       px4[2] * _lami(i + 1, j - 1, k + 6) +
                                       px4[3] * _lami(i + 2, j - 1, k + 6)) +
                            phy4[3] * (px4[1] * _lami(i, j + 1, k + 6) +
                                       px4[0] * _lami(i - 1, j + 1, k + 6) +
                                       px4[2] * _lami(i + 1, j + 1, k + 6) +
                                       px4[3] * _lami(i + 2, j + 1, k + 6))) +
                 phz4[3] * (phy4[2] * (px4[1] * _lami(i, j, k + 7) +
                                       px4[0] * _lami(i - 1, j, k + 7) +
                                       px4[2] * _lami(i + 1, j, k + 7) +
                                       px4[3] * _lami(i + 2, j, k + 7)) +
                            phy4[0] * (px4[1] * _lami(i, j - 2, k + 7) +
                                       px4[0] * _lami(i - 1, j - 2, k + 7) +
                                       px4[2] * _lami(i + 1, j - 2, k + 7) +
                                       px4[3] * _lami(i + 2, j - 2, k + 7)) +
                            phy4[1] * (px4[1] * _lami(i, j - 1, k + 7) +
                                       px4[0] * _lami(i - 1, j - 1, k + 7) +
                                       px4[2] * _lami(i + 1, j - 1, k + 7) +
                                       px4[3] * _lami(i + 2, j - 1, k + 7)) +
                            phy4[3] * (px4[1] * _lami(i, j + 1, k + 7) +
                                       px4[0] * _lami(i - 1, j + 1, k + 7) +
                                       px4[2] * _lami(i + 1, j + 1, k + 7) +
                                       px4[3] * _lami(i + 2, j + 1, k + 7))));
    _prec twomu = 2 * nu * 1.0 /
                  (phz4[0] * (phy4[2] * (px4[1] * _mui(i, j, k + 4) +
                                         px4[0] * _mui(i - 1, j, k + 4) +
                                         px4[2] * _mui(i + 1, j, k + 4) +
                                         px4[3] * _mui(i + 2, j, k + 4)) +
                              phy4[0] * (px4[1] * _mui(i, j - 2, k + 4) +
                                         px4[0] * _mui(i - 1, j - 2, k + 4) +
                                         px4[2] * _mui(i + 1, j - 2, k + 4) +
                                         px4[3] * _mui(i + 2, j - 2, k + 4)) +
                              phy4[1] * (px4[1] * _mui(i, j - 1, k + 4) +
                                         px4[0] * _mui(i - 1, j - 1, k + 4) +
                                         px4[2] * _mui(i + 1, j - 1, k + 4) +
                                         px4[3] * _mui(i + 2, j - 1, k + 4)) +
                              phy4[3] * (px4[1] * _mui(i, j + 1, k + 4) +
                                         px4[0] * _mui(i - 1, j + 1, k + 4) +
                                         px4[2] * _mui(i + 1, j + 1, k + 4) +
                                         px4[3] * _mui(i + 2, j + 1, k + 4))) +
                   phz4[1] * (phy4[2] * (px4[1] * _mui(i, j, k + 5) +
                                         px4[0] * _mui(i - 1, j, k + 5) +
                                         px4[2] * _mui(i + 1, j, k + 5) +
                                         px4[3] * _mui(i + 2, j, k + 5)) +
                              phy4[0] * (px4[1] * _mui(i, j - 2, k + 5) +
                                         px4[0] * _mui(i - 1, j - 2, k + 5) +
                                         px4[2] * _mui(i + 1, j - 2, k + 5) +
                                         px4[3] * _mui(i + 2, j - 2, k + 5)) +
                              phy4[1] * (px4[1] * _mui(i, j - 1, k + 5) +
                                         px4[0] * _mui(i - 1, j - 1, k + 5) +
                                         px4[2] * _mui(i + 1, j - 1, k + 5) +
                                         px4[3] * _mui(i + 2, j - 1, k + 5)) +
                              phy4[3] * (px4[1] * _mui(i, j + 1, k + 5) +
                                         px4[0] * _mui(i - 1, j + 1, k + 5) +
                                         px4[2] * _mui(i + 1, j + 1, k + 5) +
                                         px4[3] * _mui(i + 2, j + 1, k + 5))) +
                   phz4[2] * (phy4[2] * (px4[1] * _mui(i, j, k + 6) +
                                         px4[0] * _mui(i - 1, j, k + 6) +
                                         px4[2] * _mui(i + 1, j, k + 6) +
                                         px4[3] * _mui(i + 2, j, k + 6)) +
                              phy4[0] * (px4[1] * _mui(i, j - 2, k + 6) +
                                         px4[0] * _mui(i - 1, j - 2, k + 6) +
                                         px4[2] * _mui(i + 1, j - 2, k + 6) +
                                         px4[3] * _mui(i + 2, j - 2, k + 6)) +
                              phy4[1] * (px4[1] * _mui(i, j - 1, k + 6) +
                                         px4[0] * _mui(i - 1, j - 1, k + 6) +
                                         px4[2] * _mui(i + 1, j - 1, k + 6) +
                                         px4[3] * _mui(i + 2, j - 1, k + 6)) +
                              phy4[3] * (px4[1] * _mui(i, j + 1, k + 6) +
                                         px4[0] * _mui(i - 1, j + 1, k + 6) +
                                         px4[2] * _mui(i + 1, j + 1, k + 6) +
                                         px4[3] * _mui(i + 2, j + 1, k + 6))) +
                   phz4[3] * (phy4[2] * (px4[1] * _mui(i, j, k + 7) +
                                         px4[0] * _mui(i - 1, j, k + 7) +
                                         px4[2] * _mui(i + 1, j, k + 7) +
                                         px4[3] * _mui(i + 2, j, k + 7)) +
                              phy4[0] * (px4[1] * _mui(i, j - 2, k + 7) +
                                         px4[0] * _mui(i - 1, j - 2, k + 7) +
                                         px4[2] * _mui(i + 1, j - 2, k + 7) +
                                         px4[3] * _mui(i + 2, j - 2, k + 7)) +
                              phy4[1] * (px4[1] * _mui(i, j - 1, k + 7) +
                                         px4[0] * _mui(i - 1, j - 1, k + 7) +
                                         px4[2] * _mui(i + 1, j - 1, k + 7) +
                                         px4[3] * _mui(i + 2, j - 1, k + 7)) +
                              phy4[3] * (px4[1] * _mui(i, j + 1, k + 7) +
                                         px4[0] * _mui(i - 1, j + 1, k + 7) +
                                         px4[2] * _mui(i + 1, j + 1, k + 7) +
                                         px4[3] * _mui(i + 2, j + 1, k + 7))));
    _prec mu12 = nu * 1.0 /
                 (phz4[0] * _mui(i, j, k + 4) + phz4[1] * _mui(i, j, k + 5) +
                  phz4[2] * _mui(i, j, k + 6) + phz4[3] * _mui(i, j, k + 7));
    _prec mu13 =
        nu * 1.0 /
        (phy4[2] * _mui(i, j, k + 6) + phy4[0] * _mui(i, j - 2, k + 6) +
         phy4[1] * _mui(i, j - 1, k + 6) + phy4[3] * _mui(i, j + 1, k + 6));
    _prec mu23 =
        nu * 1.0 /
        (px4[1] * _mui(i, j, k + 6) + px4[0] * _mui(i - 1, j, k + 6) +
         px4[2] * _mui(i + 1, j, k + 6) + px4[3] * _mui(i + 2, j, k + 6));
    _prec div =
        dhy4[2] * _u2(i, j, k + 6) + dhy4[0] * _u2(i, j - 2, k + 6) +
        dhy4[1] * _u2(i, j - 1, k + 6) + dhy4[3] * _u2(i, j + 1, k + 6) +
        dx4[1] * _u1(i, j, k + 6) + dx4[0] * _u1(i - 1, j, k + 6) +
        dx4[2] * _u1(i + 1, j, k + 6) + dx4[3] * _u1(i + 2, j, k + 6) +
        Jii * (dhz4[0] * _u3(i, j, k + 4) + dhz4[1] * _u3(i, j, k + 5) +
               dhz4[2] * _u3(i, j, k + 6) + dhz4[3] * _u3(i, j, k + 7)) -
        Jii * _g_c(k + 6) *
            (phy4[2] * _f2_2(i, j) *
                 (phdz4[0] * _u2(i, j, k + 3) + phdz4[1] * _u2(i, j, k + 4) +
                  phdz4[2] * _u2(i, j, k + 5) + phdz4[3] * _u2(i, j, k + 6) +
                  phdz4[4] * _u2(i, j, k + 7) + phdz4[5] * _u2(i, j, k + 8) +
                  phdz4[6] * _u2(i, j, k + 9)) +
             phy4[0] * _f2_2(i, j - 2) *
                 (phdz4[0] * _u2(i, j - 2, k + 3) +
                  phdz4[1] * _u2(i, j - 2, k + 4) +
                  phdz4[2] * _u2(i, j - 2, k + 5) +
                  phdz4[3] * _u2(i, j - 2, k + 6) +
                  phdz4[4] * _u2(i, j - 2, k + 7) +
                  phdz4[5] * _u2(i, j - 2, k + 8) +
                  phdz4[6] * _u2(i, j - 2, k + 9)) +
             phy4[1] * _f2_2(i, j - 1) *
                 (phdz4[0] * _u2(i, j - 1, k + 3) +
                  phdz4[1] * _u2(i, j - 1, k + 4) +
                  phdz4[2] * _u2(i, j - 1, k + 5) +
                  phdz4[3] * _u2(i, j - 1, k + 6) +
                  phdz4[4] * _u2(i, j - 1, k + 7) +
                  phdz4[5] * _u2(i, j - 1, k + 8) +
                  phdz4[6] * _u2(i, j - 1, k + 9)) +
             phy4[3] * _f2_2(i, j + 1) *
                 (phdz4[0] * _u2(i, j + 1, k + 3) +
                  phdz4[1] * _u2(i, j + 1, k + 4) +
                  phdz4[2] * _u2(i, j + 1, k + 5) +
                  phdz4[3] * _u2(i, j + 1, k + 6) +
                  phdz4[4] * _u2(i, j + 1, k + 7) +
                  phdz4[5] * _u2(i, j + 1, k + 8) +
                  phdz4[6] * _u2(i, j + 1, k + 9))) -
        Jii * _g_c(k + 6) *
            (px4[1] * _f1_1(i, j) *
                 (phdz4[0] * _u1(i, j, k + 3) + phdz4[1] * _u1(i, j, k + 4) +
                  phdz4[2] * _u1(i, j, k + 5) + phdz4[3] * _u1(i, j, k + 6) +
                  phdz4[4] * _u1(i, j, k + 7) + phdz4[5] * _u1(i, j, k + 8) +
                  phdz4[6] * _u1(i, j, k + 9)) +
             px4[0] * _f1_1(i - 1, j) *
                 (phdz4[0] * _u1(i - 1, j, k + 3) +
                  phdz4[1] * _u1(i - 1, j, k + 4) +
                  phdz4[2] * _u1(i - 1, j, k + 5) +
                  phdz4[3] * _u1(i - 1, j, k + 6) +
                  phdz4[4] * _u1(i - 1, j, k + 7) +
                  phdz4[5] * _u1(i - 1, j, k + 8) +
                  phdz4[6] * _u1(i - 1, j, k + 9)) +
             px4[2] * _f1_1(i + 1, j) *
                 (phdz4[0] * _u1(i + 1, j, k + 3) +
                  phdz4[1] * _u1(i + 1, j, k + 4) +
                  phdz4[2] * _u1(i + 1, j, k + 5) +
                  phdz4[3] * _u1(i + 1, j, k + 6) +
                  phdz4[4] * _u1(i + 1, j, k + 7) +
                  phdz4[5] * _u1(i + 1, j, k + 8) +
                  phdz4[6] * _u1(i + 1, j, k + 9)) +
             px4[3] * _f1_1(i + 2, j) *
                 (phdz4[0] * _u1(i + 2, j, k + 3) +
                  phdz4[1] * _u1(i + 2, j, k + 4) +
                  phdz4[2] * _u1(i + 2, j, k + 5) +
                  phdz4[3] * _u1(i + 2, j, k + 6) +
                  phdz4[4] * _u1(i + 2, j, k + 7) +
                  phdz4[5] * _u1(i + 2, j, k + 8) +
                  phdz4[6] * _u1(i + 2, j, k + 9)));
    _prec f_dcrj = _dcrjx(i) * _dcrjy(j) * _dcrjz(k + 6);
    _s11(i, j, k + 6) =
        (a * _s11(i, j, k + 6) + lam * div +
         twomu *
             (dx4[1] * _u1(i, j, k + 6) + dx4[0] * _u1(i - 1, j, k + 6) +
              dx4[2] * _u1(i + 1, j, k + 6) + dx4[3] * _u1(i + 2, j, k + 6)) -
         twomu * Jii * _g_c(k + 6) *
             (px4[1] * _f1_1(i, j) *
                  (phdz4[0] * _u1(i, j, k + 3) + phdz4[1] * _u1(i, j, k + 4) +
                   phdz4[2] * _u1(i, j, k + 5) + phdz4[3] * _u1(i, j, k + 6) +
                   phdz4[4] * _u1(i, j, k + 7) + phdz4[5] * _u1(i, j, k + 8) +
                   phdz4[6] * _u1(i, j, k + 9)) +
              px4[0] * _f1_1(i - 1, j) *
                  (phdz4[0] * _u1(i - 1, j, k + 3) +
                   phdz4[1] * _u1(i - 1, j, k + 4) +
                   phdz4[2] * _u1(i - 1, j, k + 5) +
                   phdz4[3] * _u1(i - 1, j, k + 6) +
                   phdz4[4] * _u1(i - 1, j, k + 7) +
                   phdz4[5] * _u1(i - 1, j, k + 8) +
                   phdz4[6] * _u1(i - 1, j, k + 9)) +
              px4[2] * _f1_1(i + 1, j) *
                  (phdz4[0] * _u1(i + 1, j, k + 3) +
                   phdz4[1] * _u1(i + 1, j, k + 4) +
                   phdz4[2] * _u1(i + 1, j, k + 5) +
                   phdz4[3] * _u1(i + 1, j, k + 6) +
                   phdz4[4] * _u1(i + 1, j, k + 7) +
                   phdz4[5] * _u1(i + 1, j, k + 8) +
                   phdz4[6] * _u1(i + 1, j, k + 9)) +
              px4[3] * _f1_1(i + 2, j) *
                  (phdz4[0] * _u1(i + 2, j, k + 3) +
                   phdz4[1] * _u1(i + 2, j, k + 4) +
                   phdz4[2] * _u1(i + 2, j, k + 5) +
                   phdz4[3] * _u1(i + 2, j, k + 6) +
                   phdz4[4] * _u1(i + 2, j, k + 7) +
                   phdz4[5] * _u1(i + 2, j, k + 8) +
                   phdz4[6] * _u1(i + 2, j, k + 9)))) *
        f_dcrj;
    _s22(i, j, k + 6) =
        (a * _s22(i, j, k + 6) + lam * div +
         twomu *
             (dhy4[2] * _u2(i, j, k + 6) + dhy4[0] * _u2(i, j - 2, k + 6) +
              dhy4[1] * _u2(i, j - 1, k + 6) + dhy4[3] * _u2(i, j + 1, k + 6)) -
         twomu * Jii * _g_c(k + 6) *
             (phy4[2] * _f2_2(i, j) *
                  (phdz4[0] * _u2(i, j, k + 3) + phdz4[1] * _u2(i, j, k + 4) +
                   phdz4[2] * _u2(i, j, k + 5) + phdz4[3] * _u2(i, j, k + 6) +
                   phdz4[4] * _u2(i, j, k + 7) + phdz4[5] * _u2(i, j, k + 8) +
                   phdz4[6] * _u2(i, j, k + 9)) +
              phy4[0] * _f2_2(i, j - 2) *
                  (phdz4[0] * _u2(i, j - 2, k + 3) +
                   phdz4[1] * _u2(i, j - 2, k + 4) +
                   phdz4[2] * _u2(i, j - 2, k + 5) +
                   phdz4[3] * _u2(i, j - 2, k + 6) +
                   phdz4[4] * _u2(i, j - 2, k + 7) +
                   phdz4[5] * _u2(i, j - 2, k + 8) +
                   phdz4[6] * _u2(i, j - 2, k + 9)) +
              phy4[1] * _f2_2(i, j - 1) *
                  (phdz4[0] * _u2(i, j - 1, k + 3) +
                   phdz4[1] * _u2(i, j - 1, k + 4) +
                   phdz4[2] * _u2(i, j - 1, k + 5) +
                   phdz4[3] * _u2(i, j - 1, k + 6) +
                   phdz4[4] * _u2(i, j - 1, k + 7) +
                   phdz4[5] * _u2(i, j - 1, k + 8) +
                   phdz4[6] * _u2(i, j - 1, k + 9)) +
              phy4[3] * _f2_2(i, j + 1) *
                  (phdz4[0] * _u2(i, j + 1, k + 3) +
                   phdz4[1] * _u2(i, j + 1, k + 4) +
                   phdz4[2] * _u2(i, j + 1, k + 5) +
                   phdz4[3] * _u2(i, j + 1, k + 6) +
                   phdz4[4] * _u2(i, j + 1, k + 7) +
                   phdz4[5] * _u2(i, j + 1, k + 8) +
                   phdz4[6] * _u2(i, j + 1, k + 9)))) *
        f_dcrj;
    _s33(i, j, k + 6) =
        (a * _s33(i, j, k + 6) + lam * div +
         twomu * Jii *
             (dhz4[0] * _u3(i, j, k + 4) + dhz4[1] * _u3(i, j, k + 5) +
              dhz4[2] * _u3(i, j, k + 6) + dhz4[3] * _u3(i, j, k + 7))) *
        f_dcrj;
    _s12(i, j, k + 6) =
        (a * _s12(i, j, k + 6) +
         mu12 *
             (dhx4[2] * _u2(i, j, k + 6) + dhx4[0] * _u2(i - 2, j, k + 6) +
              dhx4[1] * _u2(i - 1, j, k + 6) + dhx4[3] * _u2(i + 1, j, k + 6) +
              dy4[1] * _u1(i, j, k + 6) + dy4[0] * _u1(i, j - 1, k + 6) +
              dy4[2] * _u1(i, j + 1, k + 6) + dy4[3] * _u1(i, j + 2, k + 6) -
              J12i * _g_c(k + 6) *
                  (phx4[2] * _f1_2(i, j) *
                       (phdz4[0] * _u2(i, j, k + 3) +
                        phdz4[1] * _u2(i, j, k + 4) +
                        phdz4[2] * _u2(i, j, k + 5) +
                        phdz4[3] * _u2(i, j, k + 6) +
                        phdz4[4] * _u2(i, j, k + 7) +
                        phdz4[5] * _u2(i, j, k + 8) +
                        phdz4[6] * _u2(i, j, k + 9)) +
                   phx4[0] * _f1_2(i - 2, j) *
                       (phdz4[0] * _u2(i - 2, j, k + 3) +
                        phdz4[1] * _u2(i - 2, j, k + 4) +
                        phdz4[2] * _u2(i - 2, j, k + 5) +
                        phdz4[3] * _u2(i - 2, j, k + 6) +
                        phdz4[4] * _u2(i - 2, j, k + 7) +
                        phdz4[5] * _u2(i - 2, j, k + 8) +
                        phdz4[6] * _u2(i - 2, j, k + 9)) +
                   phx4[1] * _f1_2(i - 1, j) *
                       (phdz4[0] * _u2(i - 1, j, k + 3) +
                        phdz4[1] * _u2(i - 1, j, k + 4) +
                        phdz4[2] * _u2(i - 1, j, k + 5) +
                        phdz4[3] * _u2(i - 1, j, k + 6) +
                        phdz4[4] * _u2(i - 1, j, k + 7) +
                        phdz4[5] * _u2(i - 1, j, k + 8) +
                        phdz4[6] * _u2(i - 1, j, k + 9)) +
                   phx4[3] * _f1_2(i + 1, j) *
                       (phdz4[0] * _u2(i + 1, j, k + 3) +
                        phdz4[1] * _u2(i + 1, j, k + 4) +
                        phdz4[2] * _u2(i + 1, j, k + 5) +
                        phdz4[3] * _u2(i + 1, j, k + 6) +
                        phdz4[4] * _u2(i + 1, j, k + 7) +
                        phdz4[5] * _u2(i + 1, j, k + 8) +
                        phdz4[6] * _u2(i + 1, j, k + 9))) -
              J12i * _g_c(k + 6) *
                  (py4[1] * _f2_1(i, j) *
                       (phdz4[0] * _u1(i, j, k + 3) +
                        phdz4[1] * _u1(i, j, k + 4) +
                        phdz4[2] * _u1(i, j, k + 5) +
                        phdz4[3] * _u1(i, j, k + 6) +
                        phdz4[4] * _u1(i, j, k + 7) +
                        phdz4[5] * _u1(i, j, k + 8) +
                        phdz4[6] * _u1(i, j, k + 9)) +
                   py4[0] * _f2_1(i, j - 1) *
                       (phdz4[0] * _u1(i, j - 1, k + 3) +
                        phdz4[1] * _u1(i, j - 1, k + 4) +
                        phdz4[2] * _u1(i, j - 1, k + 5) +
                        phdz4[3] * _u1(i, j - 1, k + 6) +
                        phdz4[4] * _u1(i, j - 1, k + 7) +
                        phdz4[5] * _u1(i, j - 1, k + 8) +
                        phdz4[6] * _u1(i, j - 1, k + 9)) +
                   py4[2] * _f2_1(i, j + 1) *
                       (phdz4[0] * _u1(i, j + 1, k + 3) +
                        phdz4[1] * _u1(i, j + 1, k + 4) +
                        phdz4[2] * _u1(i, j + 1, k + 5) +
                        phdz4[3] * _u1(i, j + 1, k + 6) +
                        phdz4[4] * _u1(i, j + 1, k + 7) +
                        phdz4[5] * _u1(i, j + 1, k + 8) +
                        phdz4[6] * _u1(i, j + 1, k + 9)) +
                   py4[3] * _f2_1(i, j + 2) *
                       (phdz4[0] * _u1(i, j + 2, k + 3) +
                        phdz4[1] * _u1(i, j + 2, k + 4) +
                        phdz4[2] * _u1(i, j + 2, k + 5) +
                        phdz4[3] * _u1(i, j + 2, k + 6) +
                        phdz4[4] * _u1(i, j + 2, k + 7) +
                        phdz4[5] * _u1(i, j + 2, k + 8) +
                        phdz4[6] * _u1(i, j + 2, k + 9))))) *
        f_dcrj;
    _s13(i, j, k + 6) =
        (a * _s13(i, j, k + 6) +
         mu13 *
             (dhx4[2] * _u3(i, j, k + 6) + dhx4[0] * _u3(i - 2, j, k + 6) +
              dhx4[1] * _u3(i - 1, j, k + 6) + dhx4[3] * _u3(i + 1, j, k + 6) +
              J13i * (dz4[0] * _u1(i, j, k + 5) + dz4[1] * _u1(i, j, k + 6) +
                      dz4[2] * _u1(i, j, k + 7) + dz4[3] * _u1(i, j, k + 8)) -
              J13i * _g(k + 6) *
                  (phx4[2] * _f1_c(i, j) *
                       (pdhz4[0] * _u3(i, j, k + 3) +
                        pdhz4[1] * _u3(i, j, k + 4) +
                        pdhz4[2] * _u3(i, j, k + 5) +
                        pdhz4[3] * _u3(i, j, k + 6) +
                        pdhz4[4] * _u3(i, j, k + 7) +
                        pdhz4[5] * _u3(i, j, k + 8) +
                        pdhz4[6] * _u3(i, j, k + 9)) +
                   phx4[0] * _f1_c(i - 2, j) *
                       (pdhz4[0] * _u3(i - 2, j, k + 3) +
                        pdhz4[1] * _u3(i - 2, j, k + 4) +
                        pdhz4[2] * _u3(i - 2, j, k + 5) +
                        pdhz4[3] * _u3(i - 2, j, k + 6) +
                        pdhz4[4] * _u3(i - 2, j, k + 7) +
                        pdhz4[5] * _u3(i - 2, j, k + 8) +
                        pdhz4[6] * _u3(i - 2, j, k + 9)) +
                   phx4[1] * _f1_c(i - 1, j) *
                       (pdhz4[0] * _u3(i - 1, j, k + 3) +
                        pdhz4[1] * _u3(i - 1, j, k + 4) +
                        pdhz4[2] * _u3(i - 1, j, k + 5) +
                        pdhz4[3] * _u3(i - 1, j, k + 6) +
                        pdhz4[4] * _u3(i - 1, j, k + 7) +
                        pdhz4[5] * _u3(i - 1, j, k + 8) +
                        pdhz4[6] * _u3(i - 1, j, k + 9)) +
                   phx4[3] * _f1_c(i + 1, j) *
                       (pdhz4[0] * _u3(i + 1, j, k + 3) +
                        pdhz4[1] * _u3(i + 1, j, k + 4) +
                        pdhz4[2] * _u3(i + 1, j, k + 5) +
                        pdhz4[3] * _u3(i + 1, j, k + 6) +
                        pdhz4[4] * _u3(i + 1, j, k + 7) +
                        pdhz4[5] * _u3(i + 1, j, k + 8) +
                        pdhz4[6] * _u3(i + 1, j, k + 9))))) *
        f_dcrj;
    _s23(i, j, k + 6) =
        (a * _s23(i, j, k + 6) +
         mu23 *
             (dy4[1] * _u3(i, j, k + 6) + dy4[0] * _u3(i, j - 1, k + 6) +
              dy4[2] * _u3(i, j + 1, k + 6) + dy4[3] * _u3(i, j + 2, k + 6) +
              J23i * (dz4[0] * _u2(i, j, k + 5) + dz4[1] * _u2(i, j, k + 6) +
                      dz4[2] * _u2(i, j, k + 7) + dz4[3] * _u2(i, j, k + 8)) -
              J23i * _g(k + 6) *
                  (py4[1] * _f2_c(i, j) *
                       (pdhz4[0] * _u3(i, j, k + 3) +
                        pdhz4[1] * _u3(i, j, k + 4) +
                        pdhz4[2] * _u3(i, j, k + 5) +
                        pdhz4[3] * _u3(i, j, k + 6) +
                        pdhz4[4] * _u3(i, j, k + 7) +
                        pdhz4[5] * _u3(i, j, k + 8) +
                        pdhz4[6] * _u3(i, j, k + 9)) +
                   py4[0] * _f2_c(i, j - 1) *
                       (pdhz4[0] * _u3(i, j - 1, k + 3) +
                        pdhz4[1] * _u3(i, j - 1, k + 4) +
                        pdhz4[2] * _u3(i, j - 1, k + 5) +
                        pdhz4[3] * _u3(i, j - 1, k + 6) +
                        pdhz4[4] * _u3(i, j - 1, k + 7) +
                        pdhz4[5] * _u3(i, j - 1, k + 8) +
                        pdhz4[6] * _u3(i, j - 1, k + 9)) +
                   py4[2] * _f2_c(i, j + 1) *
                       (pdhz4[0] * _u3(i, j + 1, k + 3) +
                        pdhz4[1] * _u3(i, j + 1, k + 4) +
                        pdhz4[2] * _u3(i, j + 1, k + 5) +
                        pdhz4[3] * _u3(i, j + 1, k + 6) +
                        pdhz4[4] * _u3(i, j + 1, k + 7) +
                        pdhz4[5] * _u3(i, j + 1, k + 8) +
                        pdhz4[6] * _u3(i, j + 1, k + 9)) +
                   py4[3] * _f2_c(i, j + 2) *
                       (pdhz4[0] * _u3(i, j + 2, k + 3) +
                        pdhz4[1] * _u3(i, j + 2, k + 4) +
                        pdhz4[2] * _u3(i, j + 2, k + 5) +
                        pdhz4[3] * _u3(i, j + 2, k + 6) +
                        pdhz4[4] * _u3(i, j + 2, k + 7) +
                        pdhz4[5] * _u3(i, j + 2, k + 8) +
                        pdhz4[6] * _u3(i, j + 2, k + 9))))) *
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
#undef _lami
#undef _mui
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

__global__ void dtopo_str_112(
    _prec *__restrict__ s11, _prec *__restrict__ s12, _prec *__restrict__ s13,
    _prec *__restrict__ s22, _prec *__restrict__ s23, _prec *__restrict__ s33,
    _prec *__restrict__ u1, _prec *__restrict__ u2, _prec *__restrict__ u3,
    const _prec *__restrict__ dcrjx, const _prec *__restrict__ dcrjy,
    const _prec *__restrict__ dcrjz, const _prec *__restrict__ f,
    const _prec *__restrict__ f1_1, const _prec *__restrict__ f1_2,
    const _prec *__restrict__ f1_c, const _prec *__restrict__ f2_1,
    const _prec *__restrict__ f2_2, const _prec *__restrict__ f2_c,
    const _prec *__restrict__ f_1, const _prec *__restrict__ f_2,
    const _prec *__restrict__ f_c, const _prec *__restrict__ g,
    const _prec *__restrict__ g3, const _prec *__restrict__ g3_c,
    const _prec *__restrict__ g_c, const _prec *__restrict__ lami,
    const _prec *__restrict__ mui, const _prec a, const _prec nu, const int nx,
    const int ny, const int nz, const int bi, const int bj, const int ei,
    const int ej) {
  const _prec phz4r[6][8] = {
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
  const _prec phy4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const _prec px4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const _prec dhz4r[6][8] = {
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
  const _prec phdz4r[6][9] = {
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
  const _prec dx4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const _prec dhy4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const _prec phx4[4] = {-0.0625000000000000, 0.5625000000000000,
                         0.5625000000000000, -0.0625000000000000};
  const _prec py4[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  const _prec dy4[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  const _prec dhx4[4] = {0.0416666666666667, -1.1250000000000000,
                         1.1250000000000000, -0.0416666666666667};
  const _prec dz4r[6][7] = {
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
  const _prec pdhz4r[6][9] = {
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
  const int j = threadIdx.y + blockIdx.y * blockDim.y + bj;
  if (j >= ngsl + ny)
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
#define _lami(i, j, k)                                                         \
  lami[(k) + align +                                                           \
       (2 * align + nz) * ((i) + ngsl + 2) * (2 * ngsl + ny + 4) +             \
       (2 * align + nz) * ((j) + ngsl + 2)]
#define _mui(i, j, k)                                                          \
  mui[(k) + align +                                                            \
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
    _prec Jii = _f_c(i, j) * _g3_c(nz - 1 - k);
    Jii = 1.0 * 1.0 / Jii;
    _prec J12i = _f(i, j) * _g3_c(nz - 1 - k);
    J12i = 1.0 * 1.0 / J12i;
    _prec J13i = _f_1(i, j) * _g3(nz - 1 - k);
    J13i = 1.0 * 1.0 / J13i;
    _prec J23i = _f_2(i, j) * _g3(nz - 1 - k);
    J23i = 1.0 * 1.0 / J23i;
    _prec lam =
        nu * 1.0 /
        (phz4r[k][7] * (phy4[2] * (px4[1] * _lami(i, j, nz - 8) +
                                   px4[0] * _lami(i - 1, j, nz - 8) +
                                   px4[2] * _lami(i + 1, j, nz - 8) +
                                   px4[3] * _lami(i + 2, j, nz - 8)) +
                        phy4[0] * (px4[1] * _lami(i, j - 2, nz - 8) +
                                   px4[0] * _lami(i - 1, j - 2, nz - 8) +
                                   px4[2] * _lami(i + 1, j - 2, nz - 8) +
                                   px4[3] * _lami(i + 2, j - 2, nz - 8)) +
                        phy4[1] * (px4[1] * _lami(i, j - 1, nz - 8) +
                                   px4[0] * _lami(i - 1, j - 1, nz - 8) +
                                   px4[2] * _lami(i + 1, j - 1, nz - 8) +
                                   px4[3] * _lami(i + 2, j - 1, nz - 8)) +
                        phy4[3] * (px4[1] * _lami(i, j + 1, nz - 8) +
                                   px4[0] * _lami(i - 1, j + 1, nz - 8) +
                                   px4[2] * _lami(i + 1, j + 1, nz - 8) +
                                   px4[3] * _lami(i + 2, j + 1, nz - 8))) +
         phz4r[k][6] * (phy4[2] * (px4[1] * _lami(i, j, nz - 7) +
                                   px4[0] * _lami(i - 1, j, nz - 7) +
                                   px4[2] * _lami(i + 1, j, nz - 7) +
                                   px4[3] * _lami(i + 2, j, nz - 7)) +
                        phy4[0] * (px4[1] * _lami(i, j - 2, nz - 7) +
                                   px4[0] * _lami(i - 1, j - 2, nz - 7) +
                                   px4[2] * _lami(i + 1, j - 2, nz - 7) +
                                   px4[3] * _lami(i + 2, j - 2, nz - 7)) +
                        phy4[1] * (px4[1] * _lami(i, j - 1, nz - 7) +
                                   px4[0] * _lami(i - 1, j - 1, nz - 7) +
                                   px4[2] * _lami(i + 1, j - 1, nz - 7) +
                                   px4[3] * _lami(i + 2, j - 1, nz - 7)) +
                        phy4[3] * (px4[1] * _lami(i, j + 1, nz - 7) +
                                   px4[0] * _lami(i - 1, j + 1, nz - 7) +
                                   px4[2] * _lami(i + 1, j + 1, nz - 7) +
                                   px4[3] * _lami(i + 2, j + 1, nz - 7))) +
         phz4r[k][5] * (phy4[2] * (px4[1] * _lami(i, j, nz - 6) +
                                   px4[0] * _lami(i - 1, j, nz - 6) +
                                   px4[2] * _lami(i + 1, j, nz - 6) +
                                   px4[3] * _lami(i + 2, j, nz - 6)) +
                        phy4[0] * (px4[1] * _lami(i, j - 2, nz - 6) +
                                   px4[0] * _lami(i - 1, j - 2, nz - 6) +
                                   px4[2] * _lami(i + 1, j - 2, nz - 6) +
                                   px4[3] * _lami(i + 2, j - 2, nz - 6)) +
                        phy4[1] * (px4[1] * _lami(i, j - 1, nz - 6) +
                                   px4[0] * _lami(i - 1, j - 1, nz - 6) +
                                   px4[2] * _lami(i + 1, j - 1, nz - 6) +
                                   px4[3] * _lami(i + 2, j - 1, nz - 6)) +
                        phy4[3] * (px4[1] * _lami(i, j + 1, nz - 6) +
                                   px4[0] * _lami(i - 1, j + 1, nz - 6) +
                                   px4[2] * _lami(i + 1, j + 1, nz - 6) +
                                   px4[3] * _lami(i + 2, j + 1, nz - 6))) +
         phz4r[k][4] * (phy4[2] * (px4[1] * _lami(i, j, nz - 5) +
                                   px4[0] * _lami(i - 1, j, nz - 5) +
                                   px4[2] * _lami(i + 1, j, nz - 5) +
                                   px4[3] * _lami(i + 2, j, nz - 5)) +
                        phy4[0] * (px4[1] * _lami(i, j - 2, nz - 5) +
                                   px4[0] * _lami(i - 1, j - 2, nz - 5) +
                                   px4[2] * _lami(i + 1, j - 2, nz - 5) +
                                   px4[3] * _lami(i + 2, j - 2, nz - 5)) +
                        phy4[1] * (px4[1] * _lami(i, j - 1, nz - 5) +
                                   px4[0] * _lami(i - 1, j - 1, nz - 5) +
                                   px4[2] * _lami(i + 1, j - 1, nz - 5) +
                                   px4[3] * _lami(i + 2, j - 1, nz - 5)) +
                        phy4[3] * (px4[1] * _lami(i, j + 1, nz - 5) +
                                   px4[0] * _lami(i - 1, j + 1, nz - 5) +
                                   px4[2] * _lami(i + 1, j + 1, nz - 5) +
                                   px4[3] * _lami(i + 2, j + 1, nz - 5))) +
         phz4r[k][3] * (phy4[2] * (px4[1] * _lami(i, j, nz - 4) +
                                   px4[0] * _lami(i - 1, j, nz - 4) +
                                   px4[2] * _lami(i + 1, j, nz - 4) +
                                   px4[3] * _lami(i + 2, j, nz - 4)) +
                        phy4[0] * (px4[1] * _lami(i, j - 2, nz - 4) +
                                   px4[0] * _lami(i - 1, j - 2, nz - 4) +
                                   px4[2] * _lami(i + 1, j - 2, nz - 4) +
                                   px4[3] * _lami(i + 2, j - 2, nz - 4)) +
                        phy4[1] * (px4[1] * _lami(i, j - 1, nz - 4) +
                                   px4[0] * _lami(i - 1, j - 1, nz - 4) +
                                   px4[2] * _lami(i + 1, j - 1, nz - 4) +
                                   px4[3] * _lami(i + 2, j - 1, nz - 4)) +
                        phy4[3] * (px4[1] * _lami(i, j + 1, nz - 4) +
                                   px4[0] * _lami(i - 1, j + 1, nz - 4) +
                                   px4[2] * _lami(i + 1, j + 1, nz - 4) +
                                   px4[3] * _lami(i + 2, j + 1, nz - 4))) +
         phz4r[k][2] * (phy4[2] * (px4[1] * _lami(i, j, nz - 3) +
                                   px4[0] * _lami(i - 1, j, nz - 3) +
                                   px4[2] * _lami(i + 1, j, nz - 3) +
                                   px4[3] * _lami(i + 2, j, nz - 3)) +
                        phy4[0] * (px4[1] * _lami(i, j - 2, nz - 3) +
                                   px4[0] * _lami(i - 1, j - 2, nz - 3) +
                                   px4[2] * _lami(i + 1, j - 2, nz - 3) +
                                   px4[3] * _lami(i + 2, j - 2, nz - 3)) +
                        phy4[1] * (px4[1] * _lami(i, j - 1, nz - 3) +
                                   px4[0] * _lami(i - 1, j - 1, nz - 3) +
                                   px4[2] * _lami(i + 1, j - 1, nz - 3) +
                                   px4[3] * _lami(i + 2, j - 1, nz - 3)) +
                        phy4[3] * (px4[1] * _lami(i, j + 1, nz - 3) +
                                   px4[0] * _lami(i - 1, j + 1, nz - 3) +
                                   px4[2] * _lami(i + 1, j + 1, nz - 3) +
                                   px4[3] * _lami(i + 2, j + 1, nz - 3))) +
         phz4r[k][1] * (phy4[2] * (px4[1] * _lami(i, j, nz - 2) +
                                   px4[0] * _lami(i - 1, j, nz - 2) +
                                   px4[2] * _lami(i + 1, j, nz - 2) +
                                   px4[3] * _lami(i + 2, j, nz - 2)) +
                        phy4[0] * (px4[1] * _lami(i, j - 2, nz - 2) +
                                   px4[0] * _lami(i - 1, j - 2, nz - 2) +
                                   px4[2] * _lami(i + 1, j - 2, nz - 2) +
                                   px4[3] * _lami(i + 2, j - 2, nz - 2)) +
                        phy4[1] * (px4[1] * _lami(i, j - 1, nz - 2) +
                                   px4[0] * _lami(i - 1, j - 1, nz - 2) +
                                   px4[2] * _lami(i + 1, j - 1, nz - 2) +
                                   px4[3] * _lami(i + 2, j - 1, nz - 2)) +
                        phy4[3] * (px4[1] * _lami(i, j + 1, nz - 2) +
                                   px4[0] * _lami(i - 1, j + 1, nz - 2) +
                                   px4[2] * _lami(i + 1, j + 1, nz - 2) +
                                   px4[3] * _lami(i + 2, j + 1, nz - 2))) +
         phz4r[k][0] * (phy4[2] * (px4[1] * _lami(i, j, nz - 1) +
                                   px4[0] * _lami(i - 1, j, nz - 1) +
                                   px4[2] * _lami(i + 1, j, nz - 1) +
                                   px4[3] * _lami(i + 2, j, nz - 1)) +
                        phy4[0] * (px4[1] * _lami(i, j - 2, nz - 1) +
                                   px4[0] * _lami(i - 1, j - 2, nz - 1) +
                                   px4[2] * _lami(i + 1, j - 2, nz - 1) +
                                   px4[3] * _lami(i + 2, j - 2, nz - 1)) +
                        phy4[1] * (px4[1] * _lami(i, j - 1, nz - 1) +
                                   px4[0] * _lami(i - 1, j - 1, nz - 1) +
                                   px4[2] * _lami(i + 1, j - 1, nz - 1) +
                                   px4[3] * _lami(i + 2, j - 1, nz - 1)) +
                        phy4[3] * (px4[1] * _lami(i, j + 1, nz - 1) +
                                   px4[0] * _lami(i - 1, j + 1, nz - 1) +
                                   px4[2] * _lami(i + 1, j + 1, nz - 1) +
                                   px4[3] * _lami(i + 2, j + 1, nz - 1))));
    _prec twomu =
        2 * nu * 1.0 /
        (phz4r[k][7] * (phy4[2] * (px4[1] * _mui(i, j, nz - 8) +
                                   px4[0] * _mui(i - 1, j, nz - 8) +
                                   px4[2] * _mui(i + 1, j, nz - 8) +
                                   px4[3] * _mui(i + 2, j, nz - 8)) +
                        phy4[0] * (px4[1] * _mui(i, j - 2, nz - 8) +
                                   px4[0] * _mui(i - 1, j - 2, nz - 8) +
                                   px4[2] * _mui(i + 1, j - 2, nz - 8) +
                                   px4[3] * _mui(i + 2, j - 2, nz - 8)) +
                        phy4[1] * (px4[1] * _mui(i, j - 1, nz - 8) +
                                   px4[0] * _mui(i - 1, j - 1, nz - 8) +
                                   px4[2] * _mui(i + 1, j - 1, nz - 8) +
                                   px4[3] * _mui(i + 2, j - 1, nz - 8)) +
                        phy4[3] * (px4[1] * _mui(i, j + 1, nz - 8) +
                                   px4[0] * _mui(i - 1, j + 1, nz - 8) +
                                   px4[2] * _mui(i + 1, j + 1, nz - 8) +
                                   px4[3] * _mui(i + 2, j + 1, nz - 8))) +
         phz4r[k][6] * (phy4[2] * (px4[1] * _mui(i, j, nz - 7) +
                                   px4[0] * _mui(i - 1, j, nz - 7) +
                                   px4[2] * _mui(i + 1, j, nz - 7) +
                                   px4[3] * _mui(i + 2, j, nz - 7)) +
                        phy4[0] * (px4[1] * _mui(i, j - 2, nz - 7) +
                                   px4[0] * _mui(i - 1, j - 2, nz - 7) +
                                   px4[2] * _mui(i + 1, j - 2, nz - 7) +
                                   px4[3] * _mui(i + 2, j - 2, nz - 7)) +
                        phy4[1] * (px4[1] * _mui(i, j - 1, nz - 7) +
                                   px4[0] * _mui(i - 1, j - 1, nz - 7) +
                                   px4[2] * _mui(i + 1, j - 1, nz - 7) +
                                   px4[3] * _mui(i + 2, j - 1, nz - 7)) +
                        phy4[3] * (px4[1] * _mui(i, j + 1, nz - 7) +
                                   px4[0] * _mui(i - 1, j + 1, nz - 7) +
                                   px4[2] * _mui(i + 1, j + 1, nz - 7) +
                                   px4[3] * _mui(i + 2, j + 1, nz - 7))) +
         phz4r[k][5] * (phy4[2] * (px4[1] * _mui(i, j, nz - 6) +
                                   px4[0] * _mui(i - 1, j, nz - 6) +
                                   px4[2] * _mui(i + 1, j, nz - 6) +
                                   px4[3] * _mui(i + 2, j, nz - 6)) +
                        phy4[0] * (px4[1] * _mui(i, j - 2, nz - 6) +
                                   px4[0] * _mui(i - 1, j - 2, nz - 6) +
                                   px4[2] * _mui(i + 1, j - 2, nz - 6) +
                                   px4[3] * _mui(i + 2, j - 2, nz - 6)) +
                        phy4[1] * (px4[1] * _mui(i, j - 1, nz - 6) +
                                   px4[0] * _mui(i - 1, j - 1, nz - 6) +
                                   px4[2] * _mui(i + 1, j - 1, nz - 6) +
                                   px4[3] * _mui(i + 2, j - 1, nz - 6)) +
                        phy4[3] * (px4[1] * _mui(i, j + 1, nz - 6) +
                                   px4[0] * _mui(i - 1, j + 1, nz - 6) +
                                   px4[2] * _mui(i + 1, j + 1, nz - 6) +
                                   px4[3] * _mui(i + 2, j + 1, nz - 6))) +
         phz4r[k][4] * (phy4[2] * (px4[1] * _mui(i, j, nz - 5) +
                                   px4[0] * _mui(i - 1, j, nz - 5) +
                                   px4[2] * _mui(i + 1, j, nz - 5) +
                                   px4[3] * _mui(i + 2, j, nz - 5)) +
                        phy4[0] * (px4[1] * _mui(i, j - 2, nz - 5) +
                                   px4[0] * _mui(i - 1, j - 2, nz - 5) +
                                   px4[2] * _mui(i + 1, j - 2, nz - 5) +
                                   px4[3] * _mui(i + 2, j - 2, nz - 5)) +
                        phy4[1] * (px4[1] * _mui(i, j - 1, nz - 5) +
                                   px4[0] * _mui(i - 1, j - 1, nz - 5) +
                                   px4[2] * _mui(i + 1, j - 1, nz - 5) +
                                   px4[3] * _mui(i + 2, j - 1, nz - 5)) +
                        phy4[3] * (px4[1] * _mui(i, j + 1, nz - 5) +
                                   px4[0] * _mui(i - 1, j + 1, nz - 5) +
                                   px4[2] * _mui(i + 1, j + 1, nz - 5) +
                                   px4[3] * _mui(i + 2, j + 1, nz - 5))) +
         phz4r[k][3] * (phy4[2] * (px4[1] * _mui(i, j, nz - 4) +
                                   px4[0] * _mui(i - 1, j, nz - 4) +
                                   px4[2] * _mui(i + 1, j, nz - 4) +
                                   px4[3] * _mui(i + 2, j, nz - 4)) +
                        phy4[0] * (px4[1] * _mui(i, j - 2, nz - 4) +
                                   px4[0] * _mui(i - 1, j - 2, nz - 4) +
                                   px4[2] * _mui(i + 1, j - 2, nz - 4) +
                                   px4[3] * _mui(i + 2, j - 2, nz - 4)) +
                        phy4[1] * (px4[1] * _mui(i, j - 1, nz - 4) +
                                   px4[0] * _mui(i - 1, j - 1, nz - 4) +
                                   px4[2] * _mui(i + 1, j - 1, nz - 4) +
                                   px4[3] * _mui(i + 2, j - 1, nz - 4)) +
                        phy4[3] * (px4[1] * _mui(i, j + 1, nz - 4) +
                                   px4[0] * _mui(i - 1, j + 1, nz - 4) +
                                   px4[2] * _mui(i + 1, j + 1, nz - 4) +
                                   px4[3] * _mui(i + 2, j + 1, nz - 4))) +
         phz4r[k][2] * (phy4[2] * (px4[1] * _mui(i, j, nz - 3) +
                                   px4[0] * _mui(i - 1, j, nz - 3) +
                                   px4[2] * _mui(i + 1, j, nz - 3) +
                                   px4[3] * _mui(i + 2, j, nz - 3)) +
                        phy4[0] * (px4[1] * _mui(i, j - 2, nz - 3) +
                                   px4[0] * _mui(i - 1, j - 2, nz - 3) +
                                   px4[2] * _mui(i + 1, j - 2, nz - 3) +
                                   px4[3] * _mui(i + 2, j - 2, nz - 3)) +
                        phy4[1] * (px4[1] * _mui(i, j - 1, nz - 3) +
                                   px4[0] * _mui(i - 1, j - 1, nz - 3) +
                                   px4[2] * _mui(i + 1, j - 1, nz - 3) +
                                   px4[3] * _mui(i + 2, j - 1, nz - 3)) +
                        phy4[3] * (px4[1] * _mui(i, j + 1, nz - 3) +
                                   px4[0] * _mui(i - 1, j + 1, nz - 3) +
                                   px4[2] * _mui(i + 1, j + 1, nz - 3) +
                                   px4[3] * _mui(i + 2, j + 1, nz - 3))) +
         phz4r[k][1] * (phy4[2] * (px4[1] * _mui(i, j, nz - 2) +
                                   px4[0] * _mui(i - 1, j, nz - 2) +
                                   px4[2] * _mui(i + 1, j, nz - 2) +
                                   px4[3] * _mui(i + 2, j, nz - 2)) +
                        phy4[0] * (px4[1] * _mui(i, j - 2, nz - 2) +
                                   px4[0] * _mui(i - 1, j - 2, nz - 2) +
                                   px4[2] * _mui(i + 1, j - 2, nz - 2) +
                                   px4[3] * _mui(i + 2, j - 2, nz - 2)) +
                        phy4[1] * (px4[1] * _mui(i, j - 1, nz - 2) +
                                   px4[0] * _mui(i - 1, j - 1, nz - 2) +
                                   px4[2] * _mui(i + 1, j - 1, nz - 2) +
                                   px4[3] * _mui(i + 2, j - 1, nz - 2)) +
                        phy4[3] * (px4[1] * _mui(i, j + 1, nz - 2) +
                                   px4[0] * _mui(i - 1, j + 1, nz - 2) +
                                   px4[2] * _mui(i + 1, j + 1, nz - 2) +
                                   px4[3] * _mui(i + 2, j + 1, nz - 2))) +
         phz4r[k][0] * (phy4[2] * (px4[1] * _mui(i, j, nz - 1) +
                                   px4[0] * _mui(i - 1, j, nz - 1) +
                                   px4[2] * _mui(i + 1, j, nz - 1) +
                                   px4[3] * _mui(i + 2, j, nz - 1)) +
                        phy4[0] * (px4[1] * _mui(i, j - 2, nz - 1) +
                                   px4[0] * _mui(i - 1, j - 2, nz - 1) +
                                   px4[2] * _mui(i + 1, j - 2, nz - 1) +
                                   px4[3] * _mui(i + 2, j - 2, nz - 1)) +
                        phy4[1] * (px4[1] * _mui(i, j - 1, nz - 1) +
                                   px4[0] * _mui(i - 1, j - 1, nz - 1) +
                                   px4[2] * _mui(i + 1, j - 1, nz - 1) +
                                   px4[3] * _mui(i + 2, j - 1, nz - 1)) +
                        phy4[3] * (px4[1] * _mui(i, j + 1, nz - 1) +
                                   px4[0] * _mui(i - 1, j + 1, nz - 1) +
                                   px4[2] * _mui(i + 1, j + 1, nz - 1) +
                                   px4[3] * _mui(i + 2, j + 1, nz - 1))));
    _prec mu12 =
        nu * 1.0 /
        (phz4r[k][7] * _mui(i, j, nz - 8) + phz4r[k][6] * _mui(i, j, nz - 7) +
         phz4r[k][5] * _mui(i, j, nz - 6) + phz4r[k][4] * _mui(i, j, nz - 5) +
         phz4r[k][3] * _mui(i, j, nz - 4) + phz4r[k][2] * _mui(i, j, nz - 3) +
         phz4r[k][1] * _mui(i, j, nz - 2) + phz4r[k][0] * _mui(i, j, nz - 1));
    _prec mu13 = nu * 1.0 /
                 (phy4[2] * _mui(i, j, nz - 1 - k) +
                  phy4[0] * _mui(i, j - 2, nz - 1 - k) +
                  phy4[1] * _mui(i, j - 1, nz - 1 - k) +
                  phy4[3] * _mui(i, j + 1, nz - 1 - k));
    _prec mu23 =
        nu * 1.0 /
        (px4[1] * _mui(i, j, nz - 1 - k) + px4[0] * _mui(i - 1, j, nz - 1 - k) +
         px4[2] * _mui(i + 1, j, nz - 1 - k) +
         px4[3] * _mui(i + 2, j, nz - 1 - k));
    _prec div =
        dhy4[2] * _u2(i, j, nz - 1 - k) + dhy4[0] * _u2(i, j - 2, nz - 1 - k) +
        dhy4[1] * _u2(i, j - 1, nz - 1 - k) +
        dhy4[3] * _u2(i, j + 1, nz - 1 - k) + dx4[1] * _u1(i, j, nz - 1 - k) +
        dx4[0] * _u1(i - 1, j, nz - 1 - k) +
        dx4[2] * _u1(i + 1, j, nz - 1 - k) +
        dx4[3] * _u1(i + 2, j, nz - 1 - k) +
        Jii *
            (dhz4r[k][7] * _u3(i, j, nz - 8) + dhz4r[k][6] * _u3(i, j, nz - 7) +
             dhz4r[k][5] * _u3(i, j, nz - 6) + dhz4r[k][4] * _u3(i, j, nz - 5) +
             dhz4r[k][3] * _u3(i, j, nz - 4) + dhz4r[k][2] * _u3(i, j, nz - 3) +
             dhz4r[k][1] * _u3(i, j, nz - 2) +
             dhz4r[k][0] * _u3(i, j, nz - 1)) -
        Jii * _g_c(nz - 1 - k) *
            (phy4[2] * _f2_2(i, j) *
                 (phdz4r[k][8] * _u2(i, j, nz - 9) +
                  phdz4r[k][7] * _u2(i, j, nz - 8) +
                  phdz4r[k][6] * _u2(i, j, nz - 7) +
                  phdz4r[k][5] * _u2(i, j, nz - 6) +
                  phdz4r[k][4] * _u2(i, j, nz - 5) +
                  phdz4r[k][3] * _u2(i, j, nz - 4) +
                  phdz4r[k][2] * _u2(i, j, nz - 3) +
                  phdz4r[k][1] * _u2(i, j, nz - 2) +
                  phdz4r[k][0] * _u2(i, j, nz - 1)) +
             phy4[0] * _f2_2(i, j - 2) *
                 (phdz4r[k][8] * _u2(i, j - 2, nz - 9) +
                  phdz4r[k][7] * _u2(i, j - 2, nz - 8) +
                  phdz4r[k][6] * _u2(i, j - 2, nz - 7) +
                  phdz4r[k][5] * _u2(i, j - 2, nz - 6) +
                  phdz4r[k][4] * _u2(i, j - 2, nz - 5) +
                  phdz4r[k][3] * _u2(i, j - 2, nz - 4) +
                  phdz4r[k][2] * _u2(i, j - 2, nz - 3) +
                  phdz4r[k][1] * _u2(i, j - 2, nz - 2) +
                  phdz4r[k][0] * _u2(i, j - 2, nz - 1)) +
             phy4[1] * _f2_2(i, j - 1) *
                 (phdz4r[k][8] * _u2(i, j - 1, nz - 9) +
                  phdz4r[k][7] * _u2(i, j - 1, nz - 8) +
                  phdz4r[k][6] * _u2(i, j - 1, nz - 7) +
                  phdz4r[k][5] * _u2(i, j - 1, nz - 6) +
                  phdz4r[k][4] * _u2(i, j - 1, nz - 5) +
                  phdz4r[k][3] * _u2(i, j - 1, nz - 4) +
                  phdz4r[k][2] * _u2(i, j - 1, nz - 3) +
                  phdz4r[k][1] * _u2(i, j - 1, nz - 2) +
                  phdz4r[k][0] * _u2(i, j - 1, nz - 1)) +
             phy4[3] * _f2_2(i, j + 1) *
                 (phdz4r[k][8] * _u2(i, j + 1, nz - 9) +
                  phdz4r[k][7] * _u2(i, j + 1, nz - 8) +
                  phdz4r[k][6] * _u2(i, j + 1, nz - 7) +
                  phdz4r[k][5] * _u2(i, j + 1, nz - 6) +
                  phdz4r[k][4] * _u2(i, j + 1, nz - 5) +
                  phdz4r[k][3] * _u2(i, j + 1, nz - 4) +
                  phdz4r[k][2] * _u2(i, j + 1, nz - 3) +
                  phdz4r[k][1] * _u2(i, j + 1, nz - 2) +
                  phdz4r[k][0] * _u2(i, j + 1, nz - 1))) -
        Jii * _g_c(nz - 1 - k) *
            (px4[1] * _f1_1(i, j) *
                 (phdz4r[k][8] * _u1(i, j, nz - 9) +
                  phdz4r[k][7] * _u1(i, j, nz - 8) +
                  phdz4r[k][6] * _u1(i, j, nz - 7) +
                  phdz4r[k][5] * _u1(i, j, nz - 6) +
                  phdz4r[k][4] * _u1(i, j, nz - 5) +
                  phdz4r[k][3] * _u1(i, j, nz - 4) +
                  phdz4r[k][2] * _u1(i, j, nz - 3) +
                  phdz4r[k][1] * _u1(i, j, nz - 2) +
                  phdz4r[k][0] * _u1(i, j, nz - 1)) +
             px4[0] * _f1_1(i - 1, j) *
                 (phdz4r[k][8] * _u1(i - 1, j, nz - 9) +
                  phdz4r[k][7] * _u1(i - 1, j, nz - 8) +
                  phdz4r[k][6] * _u1(i - 1, j, nz - 7) +
                  phdz4r[k][5] * _u1(i - 1, j, nz - 6) +
                  phdz4r[k][4] * _u1(i - 1, j, nz - 5) +
                  phdz4r[k][3] * _u1(i - 1, j, nz - 4) +
                  phdz4r[k][2] * _u1(i - 1, j, nz - 3) +
                  phdz4r[k][1] * _u1(i - 1, j, nz - 2) +
                  phdz4r[k][0] * _u1(i - 1, j, nz - 1)) +
             px4[2] * _f1_1(i + 1, j) *
                 (phdz4r[k][8] * _u1(i + 1, j, nz - 9) +
                  phdz4r[k][7] * _u1(i + 1, j, nz - 8) +
                  phdz4r[k][6] * _u1(i + 1, j, nz - 7) +
                  phdz4r[k][5] * _u1(i + 1, j, nz - 6) +
                  phdz4r[k][4] * _u1(i + 1, j, nz - 5) +
                  phdz4r[k][3] * _u1(i + 1, j, nz - 4) +
                  phdz4r[k][2] * _u1(i + 1, j, nz - 3) +
                  phdz4r[k][1] * _u1(i + 1, j, nz - 2) +
                  phdz4r[k][0] * _u1(i + 1, j, nz - 1)) +
             px4[3] * _f1_1(i + 2, j) *
                 (phdz4r[k][8] * _u1(i + 2, j, nz - 9) +
                  phdz4r[k][7] * _u1(i + 2, j, nz - 8) +
                  phdz4r[k][6] * _u1(i + 2, j, nz - 7) +
                  phdz4r[k][5] * _u1(i + 2, j, nz - 6) +
                  phdz4r[k][4] * _u1(i + 2, j, nz - 5) +
                  phdz4r[k][3] * _u1(i + 2, j, nz - 4) +
                  phdz4r[k][2] * _u1(i + 2, j, nz - 3) +
                  phdz4r[k][1] * _u1(i + 2, j, nz - 2) +
                  phdz4r[k][0] * _u1(i + 2, j, nz - 1)));
    _prec f_dcrj = _dcrjx(i) * _dcrjy(j) * _dcrjz(nz - 1 - k);
    _s11(i, j, nz - 1 - k) =
        (a * _s11(i, j, nz - 1 - k) + lam * div +
         twomu * (dx4[1] * _u1(i, j, nz - 1 - k) +
                  dx4[0] * _u1(i - 1, j, nz - 1 - k) +
                  dx4[2] * _u1(i + 1, j, nz - 1 - k) +
                  dx4[3] * _u1(i + 2, j, nz - 1 - k)) -
         twomu * Jii * _g_c(nz - 1 - k) *
             (px4[1] * _f1_1(i, j) *
                  (phdz4r[k][8] * _u1(i, j, nz - 9) +
                   phdz4r[k][7] * _u1(i, j, nz - 8) +
                   phdz4r[k][6] * _u1(i, j, nz - 7) +
                   phdz4r[k][5] * _u1(i, j, nz - 6) +
                   phdz4r[k][4] * _u1(i, j, nz - 5) +
                   phdz4r[k][3] * _u1(i, j, nz - 4) +
                   phdz4r[k][2] * _u1(i, j, nz - 3) +
                   phdz4r[k][1] * _u1(i, j, nz - 2) +
                   phdz4r[k][0] * _u1(i, j, nz - 1)) +
              px4[0] * _f1_1(i - 1, j) *
                  (phdz4r[k][8] * _u1(i - 1, j, nz - 9) +
                   phdz4r[k][7] * _u1(i - 1, j, nz - 8) +
                   phdz4r[k][6] * _u1(i - 1, j, nz - 7) +
                   phdz4r[k][5] * _u1(i - 1, j, nz - 6) +
                   phdz4r[k][4] * _u1(i - 1, j, nz - 5) +
                   phdz4r[k][3] * _u1(i - 1, j, nz - 4) +
                   phdz4r[k][2] * _u1(i - 1, j, nz - 3) +
                   phdz4r[k][1] * _u1(i - 1, j, nz - 2) +
                   phdz4r[k][0] * _u1(i - 1, j, nz - 1)) +
              px4[2] * _f1_1(i + 1, j) *
                  (phdz4r[k][8] * _u1(i + 1, j, nz - 9) +
                   phdz4r[k][7] * _u1(i + 1, j, nz - 8) +
                   phdz4r[k][6] * _u1(i + 1, j, nz - 7) +
                   phdz4r[k][5] * _u1(i + 1, j, nz - 6) +
                   phdz4r[k][4] * _u1(i + 1, j, nz - 5) +
                   phdz4r[k][3] * _u1(i + 1, j, nz - 4) +
                   phdz4r[k][2] * _u1(i + 1, j, nz - 3) +
                   phdz4r[k][1] * _u1(i + 1, j, nz - 2) +
                   phdz4r[k][0] * _u1(i + 1, j, nz - 1)) +
              px4[3] * _f1_1(i + 2, j) *
                  (phdz4r[k][8] * _u1(i + 2, j, nz - 9) +
                   phdz4r[k][7] * _u1(i + 2, j, nz - 8) +
                   phdz4r[k][6] * _u1(i + 2, j, nz - 7) +
                   phdz4r[k][5] * _u1(i + 2, j, nz - 6) +
                   phdz4r[k][4] * _u1(i + 2, j, nz - 5) +
                   phdz4r[k][3] * _u1(i + 2, j, nz - 4) +
                   phdz4r[k][2] * _u1(i + 2, j, nz - 3) +
                   phdz4r[k][1] * _u1(i + 2, j, nz - 2) +
                   phdz4r[k][0] * _u1(i + 2, j, nz - 1)))) *
        f_dcrj;
    _s22(i, j, nz - 1 - k) =
        (a * _s22(i, j, nz - 1 - k) + lam * div +
         twomu * (dhy4[2] * _u2(i, j, nz - 1 - k) +
                  dhy4[0] * _u2(i, j - 2, nz - 1 - k) +
                  dhy4[1] * _u2(i, j - 1, nz - 1 - k) +
                  dhy4[3] * _u2(i, j + 1, nz - 1 - k)) -
         twomu * Jii * _g_c(nz - 1 - k) *
             (phy4[2] * _f2_2(i, j) *
                  (phdz4r[k][8] * _u2(i, j, nz - 9) +
                   phdz4r[k][7] * _u2(i, j, nz - 8) +
                   phdz4r[k][6] * _u2(i, j, nz - 7) +
                   phdz4r[k][5] * _u2(i, j, nz - 6) +
                   phdz4r[k][4] * _u2(i, j, nz - 5) +
                   phdz4r[k][3] * _u2(i, j, nz - 4) +
                   phdz4r[k][2] * _u2(i, j, nz - 3) +
                   phdz4r[k][1] * _u2(i, j, nz - 2) +
                   phdz4r[k][0] * _u2(i, j, nz - 1)) +
              phy4[0] * _f2_2(i, j - 2) *
                  (phdz4r[k][8] * _u2(i, j - 2, nz - 9) +
                   phdz4r[k][7] * _u2(i, j - 2, nz - 8) +
                   phdz4r[k][6] * _u2(i, j - 2, nz - 7) +
                   phdz4r[k][5] * _u2(i, j - 2, nz - 6) +
                   phdz4r[k][4] * _u2(i, j - 2, nz - 5) +
                   phdz4r[k][3] * _u2(i, j - 2, nz - 4) +
                   phdz4r[k][2] * _u2(i, j - 2, nz - 3) +
                   phdz4r[k][1] * _u2(i, j - 2, nz - 2) +
                   phdz4r[k][0] * _u2(i, j - 2, nz - 1)) +
              phy4[1] * _f2_2(i, j - 1) *
                  (phdz4r[k][8] * _u2(i, j - 1, nz - 9) +
                   phdz4r[k][7] * _u2(i, j - 1, nz - 8) +
                   phdz4r[k][6] * _u2(i, j - 1, nz - 7) +
                   phdz4r[k][5] * _u2(i, j - 1, nz - 6) +
                   phdz4r[k][4] * _u2(i, j - 1, nz - 5) +
                   phdz4r[k][3] * _u2(i, j - 1, nz - 4) +
                   phdz4r[k][2] * _u2(i, j - 1, nz - 3) +
                   phdz4r[k][1] * _u2(i, j - 1, nz - 2) +
                   phdz4r[k][0] * _u2(i, j - 1, nz - 1)) +
              phy4[3] * _f2_2(i, j + 1) *
                  (phdz4r[k][8] * _u2(i, j + 1, nz - 9) +
                   phdz4r[k][7] * _u2(i, j + 1, nz - 8) +
                   phdz4r[k][6] * _u2(i, j + 1, nz - 7) +
                   phdz4r[k][5] * _u2(i, j + 1, nz - 6) +
                   phdz4r[k][4] * _u2(i, j + 1, nz - 5) +
                   phdz4r[k][3] * _u2(i, j + 1, nz - 4) +
                   phdz4r[k][2] * _u2(i, j + 1, nz - 3) +
                   phdz4r[k][1] * _u2(i, j + 1, nz - 2) +
                   phdz4r[k][0] * _u2(i, j + 1, nz - 1)))) *
        f_dcrj;
    _s33(i, j, nz - 1 - k) = (a * _s33(i, j, nz - 1 - k) + lam * div +
                              twomu * Jii *
                                  (dhz4r[k][7] * _u3(i, j, nz - 8) +
                                   dhz4r[k][6] * _u3(i, j, nz - 7) +
                                   dhz4r[k][5] * _u3(i, j, nz - 6) +
                                   dhz4r[k][4] * _u3(i, j, nz - 5) +
                                   dhz4r[k][3] * _u3(i, j, nz - 4) +
                                   dhz4r[k][2] * _u3(i, j, nz - 3) +
                                   dhz4r[k][1] * _u3(i, j, nz - 2) +
                                   dhz4r[k][0] * _u3(i, j, nz - 1))) *
                             f_dcrj;
    _s12(i, j, nz - 1 - k) =
        (a * _s12(i, j, nz - 1 - k) +
         mu12 * (dhx4[2] * _u2(i, j, nz - 1 - k) +
                 dhx4[0] * _u2(i - 2, j, nz - 1 - k) +
                 dhx4[1] * _u2(i - 1, j, nz - 1 - k) +
                 dhx4[3] * _u2(i + 1, j, nz - 1 - k) +
                 dy4[1] * _u1(i, j, nz - 1 - k) +
                 dy4[0] * _u1(i, j - 1, nz - 1 - k) +
                 dy4[2] * _u1(i, j + 1, nz - 1 - k) +
                 dy4[3] * _u1(i, j + 2, nz - 1 - k) -
                 J12i * _g_c(nz - 1 - k) *
                     (phx4[2] * _f1_2(i, j) *
                          (phdz4r[k][8] * _u2(i, j, nz - 9) +
                           phdz4r[k][7] * _u2(i, j, nz - 8) +
                           phdz4r[k][6] * _u2(i, j, nz - 7) +
                           phdz4r[k][5] * _u2(i, j, nz - 6) +
                           phdz4r[k][4] * _u2(i, j, nz - 5) +
                           phdz4r[k][3] * _u2(i, j, nz - 4) +
                           phdz4r[k][2] * _u2(i, j, nz - 3) +
                           phdz4r[k][1] * _u2(i, j, nz - 2) +
                           phdz4r[k][0] * _u2(i, j, nz - 1)) +
                      phx4[0] * _f1_2(i - 2, j) *
                          (phdz4r[k][8] * _u2(i - 2, j, nz - 9) +
                           phdz4r[k][7] * _u2(i - 2, j, nz - 8) +
                           phdz4r[k][6] * _u2(i - 2, j, nz - 7) +
                           phdz4r[k][5] * _u2(i - 2, j, nz - 6) +
                           phdz4r[k][4] * _u2(i - 2, j, nz - 5) +
                           phdz4r[k][3] * _u2(i - 2, j, nz - 4) +
                           phdz4r[k][2] * _u2(i - 2, j, nz - 3) +
                           phdz4r[k][1] * _u2(i - 2, j, nz - 2) +
                           phdz4r[k][0] * _u2(i - 2, j, nz - 1)) +
                      phx4[1] * _f1_2(i - 1, j) *
                          (phdz4r[k][8] * _u2(i - 1, j, nz - 9) +
                           phdz4r[k][7] * _u2(i - 1, j, nz - 8) +
                           phdz4r[k][6] * _u2(i - 1, j, nz - 7) +
                           phdz4r[k][5] * _u2(i - 1, j, nz - 6) +
                           phdz4r[k][4] * _u2(i - 1, j, nz - 5) +
                           phdz4r[k][3] * _u2(i - 1, j, nz - 4) +
                           phdz4r[k][2] * _u2(i - 1, j, nz - 3) +
                           phdz4r[k][1] * _u2(i - 1, j, nz - 2) +
                           phdz4r[k][0] * _u2(i - 1, j, nz - 1)) +
                      phx4[3] * _f1_2(i + 1, j) *
                          (phdz4r[k][8] * _u2(i + 1, j, nz - 9) +
                           phdz4r[k][7] * _u2(i + 1, j, nz - 8) +
                           phdz4r[k][6] * _u2(i + 1, j, nz - 7) +
                           phdz4r[k][5] * _u2(i + 1, j, nz - 6) +
                           phdz4r[k][4] * _u2(i + 1, j, nz - 5) +
                           phdz4r[k][3] * _u2(i + 1, j, nz - 4) +
                           phdz4r[k][2] * _u2(i + 1, j, nz - 3) +
                           phdz4r[k][1] * _u2(i + 1, j, nz - 2) +
                           phdz4r[k][0] * _u2(i + 1, j, nz - 1))) -
                 J12i * _g_c(nz - 1 - k) *
                     (py4[1] * _f2_1(i, j) *
                          (phdz4r[k][8] * _u1(i, j, nz - 9) +
                           phdz4r[k][7] * _u1(i, j, nz - 8) +
                           phdz4r[k][6] * _u1(i, j, nz - 7) +
                           phdz4r[k][5] * _u1(i, j, nz - 6) +
                           phdz4r[k][4] * _u1(i, j, nz - 5) +
                           phdz4r[k][3] * _u1(i, j, nz - 4) +
                           phdz4r[k][2] * _u1(i, j, nz - 3) +
                           phdz4r[k][1] * _u1(i, j, nz - 2) +
                           phdz4r[k][0] * _u1(i, j, nz - 1)) +
                      py4[0] * _f2_1(i, j - 1) *
                          (phdz4r[k][8] * _u1(i, j - 1, nz - 9) +
                           phdz4r[k][7] * _u1(i, j - 1, nz - 8) +
                           phdz4r[k][6] * _u1(i, j - 1, nz - 7) +
                           phdz4r[k][5] * _u1(i, j - 1, nz - 6) +
                           phdz4r[k][4] * _u1(i, j - 1, nz - 5) +
                           phdz4r[k][3] * _u1(i, j - 1, nz - 4) +
                           phdz4r[k][2] * _u1(i, j - 1, nz - 3) +
                           phdz4r[k][1] * _u1(i, j - 1, nz - 2) +
                           phdz4r[k][0] * _u1(i, j - 1, nz - 1)) +
                      py4[2] * _f2_1(i, j + 1) *
                          (phdz4r[k][8] * _u1(i, j + 1, nz - 9) +
                           phdz4r[k][7] * _u1(i, j + 1, nz - 8) +
                           phdz4r[k][6] * _u1(i, j + 1, nz - 7) +
                           phdz4r[k][5] * _u1(i, j + 1, nz - 6) +
                           phdz4r[k][4] * _u1(i, j + 1, nz - 5) +
                           phdz4r[k][3] * _u1(i, j + 1, nz - 4) +
                           phdz4r[k][2] * _u1(i, j + 1, nz - 3) +
                           phdz4r[k][1] * _u1(i, j + 1, nz - 2) +
                           phdz4r[k][0] * _u1(i, j + 1, nz - 1)) +
                      py4[3] * _f2_1(i, j + 2) *
                          (phdz4r[k][8] * _u1(i, j + 2, nz - 9) +
                           phdz4r[k][7] * _u1(i, j + 2, nz - 8) +
                           phdz4r[k][6] * _u1(i, j + 2, nz - 7) +
                           phdz4r[k][5] * _u1(i, j + 2, nz - 6) +
                           phdz4r[k][4] * _u1(i, j + 2, nz - 5) +
                           phdz4r[k][3] * _u1(i, j + 2, nz - 4) +
                           phdz4r[k][2] * _u1(i, j + 2, nz - 3) +
                           phdz4r[k][1] * _u1(i, j + 2, nz - 2) +
                           phdz4r[k][0] * _u1(i, j + 2, nz - 1))))) *
        f_dcrj;
    _s13(i, j, nz - 1 - k) =
        (a * _s13(i, j, nz - 1 - k) +
         mu13 * (dhx4[2] * _u3(i, j, nz - 1 - k) +
                 dhx4[0] * _u3(i - 2, j, nz - 1 - k) +
                 dhx4[1] * _u3(i - 1, j, nz - 1 - k) +
                 dhx4[3] * _u3(i + 1, j, nz - 1 - k) +
                 J13i * (dz4r[k][6] * _u1(i, j, nz - 7) +
                         dz4r[k][5] * _u1(i, j, nz - 6) +
                         dz4r[k][4] * _u1(i, j, nz - 5) +
                         dz4r[k][3] * _u1(i, j, nz - 4) +
                         dz4r[k][2] * _u1(i, j, nz - 3) +
                         dz4r[k][1] * _u1(i, j, nz - 2) +
                         dz4r[k][0] * _u1(i, j, nz - 1)) -
                 J13i * _g(nz - 1 - k) *
                     (phx4[2] * _f1_c(i, j) *
                          (pdhz4r[k][8] * _u3(i, j, nz - 9) +
                           pdhz4r[k][7] * _u3(i, j, nz - 8) +
                           pdhz4r[k][6] * _u3(i, j, nz - 7) +
                           pdhz4r[k][5] * _u3(i, j, nz - 6) +
                           pdhz4r[k][4] * _u3(i, j, nz - 5) +
                           pdhz4r[k][3] * _u3(i, j, nz - 4) +
                           pdhz4r[k][2] * _u3(i, j, nz - 3) +
                           pdhz4r[k][1] * _u3(i, j, nz - 2) +
                           pdhz4r[k][0] * _u3(i, j, nz - 1)) +
                      phx4[0] * _f1_c(i - 2, j) *
                          (pdhz4r[k][8] * _u3(i - 2, j, nz - 9) +
                           pdhz4r[k][7] * _u3(i - 2, j, nz - 8) +
                           pdhz4r[k][6] * _u3(i - 2, j, nz - 7) +
                           pdhz4r[k][5] * _u3(i - 2, j, nz - 6) +
                           pdhz4r[k][4] * _u3(i - 2, j, nz - 5) +
                           pdhz4r[k][3] * _u3(i - 2, j, nz - 4) +
                           pdhz4r[k][2] * _u3(i - 2, j, nz - 3) +
                           pdhz4r[k][1] * _u3(i - 2, j, nz - 2) +
                           pdhz4r[k][0] * _u3(i - 2, j, nz - 1)) +
                      phx4[1] * _f1_c(i - 1, j) *
                          (pdhz4r[k][8] * _u3(i - 1, j, nz - 9) +
                           pdhz4r[k][7] * _u3(i - 1, j, nz - 8) +
                           pdhz4r[k][6] * _u3(i - 1, j, nz - 7) +
                           pdhz4r[k][5] * _u3(i - 1, j, nz - 6) +
                           pdhz4r[k][4] * _u3(i - 1, j, nz - 5) +
                           pdhz4r[k][3] * _u3(i - 1, j, nz - 4) +
                           pdhz4r[k][2] * _u3(i - 1, j, nz - 3) +
                           pdhz4r[k][1] * _u3(i - 1, j, nz - 2) +
                           pdhz4r[k][0] * _u3(i - 1, j, nz - 1)) +
                      phx4[3] * _f1_c(i + 1, j) *
                          (pdhz4r[k][8] * _u3(i + 1, j, nz - 9) +
                           pdhz4r[k][7] * _u3(i + 1, j, nz - 8) +
                           pdhz4r[k][6] * _u3(i + 1, j, nz - 7) +
                           pdhz4r[k][5] * _u3(i + 1, j, nz - 6) +
                           pdhz4r[k][4] * _u3(i + 1, j, nz - 5) +
                           pdhz4r[k][3] * _u3(i + 1, j, nz - 4) +
                           pdhz4r[k][2] * _u3(i + 1, j, nz - 3) +
                           pdhz4r[k][1] * _u3(i + 1, j, nz - 2) +
                           pdhz4r[k][0] * _u3(i + 1, j, nz - 1))))) *
        f_dcrj;
    _s23(i, j, nz - 1 - k) =
        (a * _s23(i, j, nz - 1 - k) +
         mu23 * (dy4[1] * _u3(i, j, nz - 1 - k) +
                 dy4[0] * _u3(i, j - 1, nz - 1 - k) +
                 dy4[2] * _u3(i, j + 1, nz - 1 - k) +
                 dy4[3] * _u3(i, j + 2, nz - 1 - k) +
                 J23i * (dz4r[k][6] * _u2(i, j, nz - 7) +
                         dz4r[k][5] * _u2(i, j, nz - 6) +
                         dz4r[k][4] * _u2(i, j, nz - 5) +
                         dz4r[k][3] * _u2(i, j, nz - 4) +
                         dz4r[k][2] * _u2(i, j, nz - 3) +
                         dz4r[k][1] * _u2(i, j, nz - 2) +
                         dz4r[k][0] * _u2(i, j, nz - 1)) -
                 J23i * _g(nz - 1 - k) *
                     (py4[1] * _f2_c(i, j) *
                          (pdhz4r[k][8] * _u3(i, j, nz - 9) +
                           pdhz4r[k][7] * _u3(i, j, nz - 8) +
                           pdhz4r[k][6] * _u3(i, j, nz - 7) +
                           pdhz4r[k][5] * _u3(i, j, nz - 6) +
                           pdhz4r[k][4] * _u3(i, j, nz - 5) +
                           pdhz4r[k][3] * _u3(i, j, nz - 4) +
                           pdhz4r[k][2] * _u3(i, j, nz - 3) +
                           pdhz4r[k][1] * _u3(i, j, nz - 2) +
                           pdhz4r[k][0] * _u3(i, j, nz - 1)) +
                      py4[0] * _f2_c(i, j - 1) *
                          (pdhz4r[k][8] * _u3(i, j - 1, nz - 9) +
                           pdhz4r[k][7] * _u3(i, j - 1, nz - 8) +
                           pdhz4r[k][6] * _u3(i, j - 1, nz - 7) +
                           pdhz4r[k][5] * _u3(i, j - 1, nz - 6) +
                           pdhz4r[k][4] * _u3(i, j - 1, nz - 5) +
                           pdhz4r[k][3] * _u3(i, j - 1, nz - 4) +
                           pdhz4r[k][2] * _u3(i, j - 1, nz - 3) +
                           pdhz4r[k][1] * _u3(i, j - 1, nz - 2) +
                           pdhz4r[k][0] * _u3(i, j - 1, nz - 1)) +
                      py4[2] * _f2_c(i, j + 1) *
                          (pdhz4r[k][8] * _u3(i, j + 1, nz - 9) +
                           pdhz4r[k][7] * _u3(i, j + 1, nz - 8) +
                           pdhz4r[k][6] * _u3(i, j + 1, nz - 7) +
                           pdhz4r[k][5] * _u3(i, j + 1, nz - 6) +
                           pdhz4r[k][4] * _u3(i, j + 1, nz - 5) +
                           pdhz4r[k][3] * _u3(i, j + 1, nz - 4) +
                           pdhz4r[k][2] * _u3(i, j + 1, nz - 3) +
                           pdhz4r[k][1] * _u3(i, j + 1, nz - 2) +
                           pdhz4r[k][0] * _u3(i, j + 1, nz - 1)) +
                      py4[3] * _f2_c(i, j + 2) *
                          (pdhz4r[k][8] * _u3(i, j + 2, nz - 9) +
                           pdhz4r[k][7] * _u3(i, j + 2, nz - 8) +
                           pdhz4r[k][6] * _u3(i, j + 2, nz - 7) +
                           pdhz4r[k][5] * _u3(i, j + 2, nz - 6) +
                           pdhz4r[k][4] * _u3(i, j + 2, nz - 5) +
                           pdhz4r[k][3] * _u3(i, j + 2, nz - 4) +
                           pdhz4r[k][2] * _u3(i, j + 2, nz - 3) +
                           pdhz4r[k][1] * _u3(i, j + 2, nz - 2) +
                           pdhz4r[k][0] * _u3(i, j + 2, nz - 1))))) *
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
#undef _lami
#undef _mui
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

__global__ void dtopo_init_material_111(_prec *__restrict__ lami,
                                        _prec *__restrict__ mui,
                                        _prec *__restrict__ rho, const int nx,
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
