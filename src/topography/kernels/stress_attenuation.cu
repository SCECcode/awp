#include <cuda_runtime.h>
#include <stdio.h>

#include <awp/definitions.h>
#include <test/test.h>
#include <topography/kernels/stress_attenuation.cuh>


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
__launch_bounds__(512,2)
__global__ void 
dtopo_str_111(_prec*  __restrict__ xx, _prec*  __restrict__ yy, _prec*  __restrict__ zz,
           _prec*  __restrict__ xy, _prec*  __restrict__ xz, _prec*  __restrict__ yz,
       _prec*  __restrict__ r1, _prec*  __restrict__ r2,  _prec*  __restrict__ r3, 
       _prec*  __restrict__ r4, _prec*  __restrict__ r5,  _prec*  __restrict__ r6,
       _prec*  __restrict__ u1, 
       _prec*  __restrict__ v1,    
       _prec*  __restrict__ w1,    
       const float *__restrict__ f,
       const float *__restrict__ f1_1, const float *__restrict__ f1_2,
       const float *__restrict__ f1_c, const float *__restrict__ f2_1,
       const float *__restrict__ f2_2, const float *__restrict__ f2_c,
       const float *__restrict__ f_1, const float *__restrict__ f_2,
       const float *__restrict__ f_c, const float *__restrict__ g,
       const float *__restrict__ g3, const float *__restrict__ g3_c,
       const float *__restrict__ g_c,
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
       int NX, int rankx, int ranky, 
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
  int maxk, mink=align+3;
    
  k    = blockIdx.x*blockDim.x+threadIdx.x+align;
  j    = blockIdx.y*blockDim.y+threadIdx.y+s_j;

  maxk = nzt + align -1;

  if (k < mink || k > maxk || j > e_j) return;
  

  i    = e_i;
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

  for(i=e_i;i>=s_i;i--)
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
      qpaw  = f_wwo*qpa;
    }
    qpaw=qpaw/f_wwo;


    h        = 0.0625f*( LDG(qs[pos])     + LDG(qs[pos_ip1]) + LDG(qs[pos_jm1]) + LDG(qs[pos_ijk])
                         + LDG(qs[pos_km1]) + LDG(qs[pos_ik1]) + LDG(qs[pos_jk1]) + LDG(qs[pos_ijk1]) );

    if(1.0f/(h*2.0f)<=200.0f)
    {
      hw=coeff[f_ww*2-2]*(2.0f*h)*(2.0f*h)+coeff[f_ww*2-1]*(2.0f*h);
    }
    else {
      hw  = f_wwo*h;
    }
    hw=hw/f_wwo;


    h1       = 0.250f*(  qs[pos]     + qs[pos_km1] );

    if(1.0f/(h1*2.0f)<=200.0f)
    {
      h1w=coeff[f_ww*2-2]*(2.0f*h1)*(2.0f*h1)+coeff[f_ww*2-1]*(2.0f*h1);
    }
    else {
      h1w  = f_wwo*h1;
    }
    h1w=h1w/f_wwo;

    h2       = 0.250f*(  qs[pos]     + qs[pos_jm1] );
    if(1.0f/(h2*2.0f)<=200.0f)
    {
      h2w=coeff[f_ww*2-2]*(2.0f*h2)*(2.0f*h2)+coeff[f_ww*2-1]*(2.0f*h2);
    }
    else {
      h2w  = f_wwo*h2;
    }
    h2w=h2w/f_wwo;


    h3       = 0.250f*(  qs[pos]     + qs[pos_ip1] );
    if(1.0f/(h3*2.0f)<=200.0f)
    {
      h3w=coeff[f_ww*2-2]*(2.0f*h3)*(2.0f*h3)+coeff[f_ww*2-1]*(2.0f*h3);
    }
    else {
      h3w  = f_wwo*h3;
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


    vs1      = d_c1*(u1_ip1 - f_u1)        + d_c2*(u1_ip2      - u1_im1);
    vs2      = d_c1*(f_v1   - v1[pos_jm1]) + d_c2*(v1[pos_jp1] - v1[pos_jm2]);
    vs3      = d_c1*(f_w1   - w1[pos_km1]) + d_c2*(w1[pos_kp1] - w1[pos_km2]);

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


    vs1      = d_c1*(u1[pos_jp1] - f_u1)   + d_c2*(u1[pos_jp2] - u1[pos_jm1]);
    vs2      = d_c1*(f_v1        - v1_im1) + d_c2*(v1_ip1      - v1_im2);
    f_r      = r4[pos];
    f_rtmp   = h1*(vs1+vs2); 
    f_xy     = xy[pos]  + xmu1*(vs1+vs2) + vx1*f_r;
    r4[pos]  = f_vx2*f_r + f_wwo*f_rtmp; 
    f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);
    xy[pos]  = (f_xy + d_DT*f_rtmp)*f_dcrj;


    vs1     = d_c1*(u1[pos_kp1] - f_u1)   + d_c2*(u1[pos_kp2] - u1[pos_km1]);
    vs2     = d_c1*(f_w1        - w1_im1) + d_c2*(w1_ip1      - w1_im2);
    f_r     = r5[pos];
    f_rtmp  = h2*(vs1+vs2);
    f_xz    = xz[pos]  + xmu2*(vs1+vs2) + vx1*f_r; 
    r5[pos] = f_vx2*f_r + f_wwo*f_rtmp; 
    f_rtmp  = f_rtmp*(f_wwo-1.0f) + f_vx2*f_r*(1.0f-f_vx1); 
    xz[pos] = (f_xz + d_DT*f_rtmp)*f_dcrj;
	 
    vs1     = d_c1*(v1[pos_kp1] - f_v1) + d_c2*(v1[pos_kp2] - v1[pos_km1]);
    vs2     = d_c1*(w1[pos_jp1] - f_w1) + d_c2*(w1[pos_jp2] - w1[pos_jm1]);
    f_r     = r6[pos];
    f_rtmp  = h3*(vs1+vs2);
    f_yz    = yz[pos]  + xmu3*(vs1+vs2) + vx1*f_r;
    r6[pos] = f_vx2*f_r + f_wwo*f_rtmp;
    f_rtmp  = f_rtmp*(f_wwo-1.0f) + f_vx2*f_r*(1.0f-f_vx1); 
    yz[pos] = (f_yz + d_DT*f_rtmp)*f_dcrj; 

  if (k == mink && i == s_i) {
        printf("xx = %g vs1 = %g  \n", xx[pos], vx1);
  }

    pos     = pos_im1;
  }
}

