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

#define LDG(x) x

template <int tx, int ty, int tz>
__launch_bounds__ (tx*ty*tz)
__global__ void dtopo_str_111_index(_prec*  RSTRCT xx, _prec*  RSTRCT yy, _prec*  RSTRCT zz,
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
  // i - 1, j, k - 3: k + 3
  int m2p0m3 = pos - d_slice_2 - 3;
  int m2p0m2 = pos - d_slice_2 - 2;
  int m2p0m1 = pos - d_slice_2 - 1;
  int m2p0p0 = pos - d_slice_2 + 0;
  int m2p0p1 = pos - d_slice_2 + 1;
  int m2p0p2 = pos - d_slice_2 + 2;
  int m2p0p3 = pos - d_slice_2 + 3;


  // i - 1, j, k - 3: k + 3
  int m1p0m3 = pos - d_slice_1 - 3;
  int m1p0m2 = pos - d_slice_1 - 2;
  int m1p0m1 = pos - d_slice_1 - 1;
  int m1p0p0 = pos - d_slice_1 + 0;
  int m1p0p1 = pos - d_slice_1 + 1;
  int m1p0p2 = pos - d_slice_1 + 2;
  int m1p0p3 = pos - d_slice_1 + 3;

  // i, j, k - 3: k + 3
  int p0p0m3 = pos - 3;
  int p0p0m2 = pos - 2;
  int p0p0m1 = pos - 1;
  int p0p0p0 = pos + 0;
  int p0p0p1 = pos + 1;
  int p0p0p2 = pos + 2;
  int p0p0p3 = pos + 3;

  // i + 1, j, k - 3: k + 3
  int p1p0m3 = pos + d_slice_1 - 3;
  int p1p0m2 = pos + d_slice_1 - 2;
  int p1p0m1 = pos + d_slice_1 - 1;
  int p1p0p0 = pos + d_slice_1 + 0;
  int p1p0p1 = pos + d_slice_1 + 1;
  int p1p0p2 = pos + d_slice_1 + 2;
  int p1p0p3 = pos + d_slice_1 + 3;

  // i + 2, j, k - 3: k + 3
  int p2p0m3 = pos + d_slice_2 - 3;
  int p2p0m2 = pos + d_slice_2 - 2;
  int p2p0m1 = pos + d_slice_2 - 1;
  int p2p0p0 = pos + d_slice_2 + 0;
  int p2p0p1 = pos + d_slice_2 + 1;
  int p2p0p2 = pos + d_slice_2 + 2;
  int p2p0p3 = pos + d_slice_2 + 3;

  // i, j - 2, k - 3: k + 3
  int p0m2m3 = pos - d_yline_2 - 3;
  int p0m2m2 = pos - d_yline_2 - 2;
  int p0m2m1 = pos - d_yline_2 - 1;
  int p0m2p0 = pos - d_yline_2 + 0;
  int p0m2p1 = pos - d_yline_2 + 1;
  int p0m2p2 = pos - d_yline_2 + 2;
  int p0m2p3 = pos - d_yline_2 + 3;

  // i, j - 1, k - 3: k + 3
  int p0m1m3 = pos - d_yline_1 - 3;
  int p0m1m2 = pos - d_yline_1 - 2;
  int p0m1m1 = pos - d_yline_1 - 1;
  int p0m1p0 = pos - d_yline_1 + 0;
  int p0m1p1 = pos - d_yline_1 + 1;
  int p0m1p2 = pos - d_yline_1 + 2;
  int p0m1p3 = pos - d_yline_1 + 3;

  // i, j + 1, k - 3: k + 3
  int p0p1m3 = pos + d_yline_1 - 3;
  int p0p1m2 = pos + d_yline_1 - 2;
  int p0p1m1 = pos + d_yline_1 - 1;
  int p0p1p0 = pos + d_yline_1 + 0;
  int p0p1p1 = pos + d_yline_1 + 1;
  int p0p1p2 = pos + d_yline_1 + 2;
  int p0p1p3 = pos + d_yline_1 + 3;

  // i, j + 2, k - 3: k + 3
  int p0p2m3 = pos + d_yline_2 - 3;
  int p0p2m2 = pos + d_yline_2 - 2;
  int p0p2m1 = pos + d_yline_2 - 1;
  int p0p2p0 = pos + d_yline_2 + 0;
  int p0p2p1 = pos + d_yline_2 + 1;
  int p0p2p2 = pos + d_yline_2 + 2;
  int p0p2p3 = pos + d_yline_2 + 3;


  // i - 2 : i + 1, j
  //int m2p0 = fpos - d_fline_2;
  //int m1p0 = fpos - d_fline_1;
  //int p0p0 = fpos;
  //int p1p0 = fpos + d_fline_1;
  //int p2p0 = fpos + d_fline_2;


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

    float Jii = _f_c(i, j) * _g3_c(k);
          Jii = 1.0 * 1.0 / Jii;

    vs1 =
      dx4[1] * u1[p0p0p0] + dx4[0] * u1[m1p0p0] +
      dx4[2] * u1[p1p0p0] + dx4[3] * u1[p2p0p0] -
      Jii * _g_c(k) *
          (
           px4[0] * _f1_1(i - 1, j) *
               (
                phdz4[0] * u1[m1p0m3] +
                phdz4[1] * u1[m1p0m2] +
                phdz4[2] * u1[m1p0m1] +
                phdz4[3] * u1[m1p0p0] +
                phdz4[4] * u1[m1p0p1] +
                phdz4[5] * u1[m1p0p2] +
                phdz4[6] * u1[m1p0p3]
                ) 
               +
           px4[1] * _f1_1(i, j) *
               (
                phdz4[0] * u1[p0p0m3] +
                phdz4[1] * u1[p0p0m2] +
                phdz4[2] * u1[p0p0m1] +
                phdz4[3] * u1[p0p0p0] +
                phdz4[4] * u1[p0p0p1] +
                phdz4[5] * u1[p0p0p2] +
                phdz4[6] * u1[p0p0p3]
                ) +
           px4[2] * _f1_1(i + 1, j) *
               (
                phdz4[0] * u1[p1p0m3] +
                phdz4[1] * u1[p1p0m2] +
                phdz4[2] * u1[p1p0m1] +
                phdz4[3] * u1[p1p0p0] +
                phdz4[4] * u1[p1p0p1] +
                phdz4[5] * u1[p1p0p2] +
                phdz4[6] * u1[p1p0p3]
                ) +
           px4[3] * _f1_1(i + 2, j) *
               (
                phdz4[0] * u1[p2p0m3] +
                phdz4[1] * u1[p2p0m2] +
                phdz4[2] * u1[p2p0m1] +
                phdz4[3] * u1[p2p0p0] +
                phdz4[4] * u1[p2p0p1] +
                phdz4[5] * u1[p2p0p2] +
                phdz4[6] * u1[p2p0p3]
                )
         );
    vs2 =
      dhy4[2] * v1[p0p0p0] + dhy4[0] * v1[p0m2p0] +
      dhy4[1] * v1[p0m1p0] + dhy4[3] * v1[p0p1p0] -
      Jii * _g_c(k) *
           (phy4[0] * _f2_2(i, j - 2) *
                (
                phdz4[0] * v1[p0m2m3] +
                phdz4[1] * v1[p0m2m2] +
                phdz4[2] * v1[p0m2m1] +
                phdz4[3] * v1[p0m2p0] +
                phdz4[4] * v1[p0m2p1] +
                phdz4[5] * v1[p0m2p2] +
                phdz4[6] * v1[p0m2p3]
                ) +
           phy4[1] * _f2_2(i, j - 1) *
               (
                phdz4[0] * v1[p0m1m3] +
                phdz4[1] * v1[p0m1m2] +
                phdz4[2] * v1[p0m1m1] +
                phdz4[3] * v1[p0m1p0] +
                phdz4[4] * v1[p0m1p1] +
                phdz4[5] * v1[p0m1p2] +
                phdz4[6] * v1[p0m1p3]
               ) +
          phy4[2] * _f2_2(i, j) *
               (
                phdz4[0] * v1[p0p0m3] +
                phdz4[1] * v1[p0p0m2] +
                phdz4[2] * v1[p0p0m1] +
                phdz4[3] * v1[p0p0p0] +
                phdz4[4] * v1[p0p0p1] +
                phdz4[5] * v1[p0p0p2] +
                phdz4[6] * v1[p0p0p3]
                ) +
           phy4[3] * _f2_2(i, j + 1) *
               (
                phdz4[0] * v1[p0p1m3] +
                phdz4[1] * v1[p0p1m2] +
                phdz4[2] * v1[p0p1m1] +
                phdz4[3] * v1[p0p1p0] +
                phdz4[4] * v1[p0p1p1] +
                phdz4[5] * v1[p0p1p2] +
                phdz4[6] * v1[p0p1p3]
                )
               );
  vs3 =
      Jii * (dhz4[2] * w1[p0p0p0] + dhz4[0] * w1[p0p0m2] +
             dhz4[1] * w1[p0p0m1] + dhz4[3] * w1[p0p0p1]);

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
  float J12i = _f(i, j) * _g3_c(k);
  J12i = 1.0 / J12i;

  vs1 =
      dy4[1] * u1[p0p0p0] + dy4[0] * u1[p0m1p0] +
      dy4[2] * u1[p0p1p0] + dy4[3] * u1[p0p2p0] -
      J12i * _g_c(k) *
          (
           py4[0] * _f2_1(i, j - 1) *
               (
                phdz4[0] * u1[p0m1m3] +
                phdz4[1] * u1[p0m1m2] +
                phdz4[2] * u1[p0m1m1] +
                phdz4[3] * u1[p0m1p0] +
                phdz4[4] * u1[p0m1p1] +
                phdz4[5] * u1[p0m1p2] +
                phdz4[6] * u1[p0m1p3]) +
           py4[1] * _f2_1(i, j) *
               (
                phdz4[0] * u1[p0p0m3] +
                phdz4[1] * u1[p0p0m2] +
                phdz4[2] * u1[p0p0m1] +
                phdz4[3] * u1[p0p0p0] +
                phdz4[4] * u1[p0p0p1] +
                phdz4[5] * u1[p0p0p2] +
                phdz4[6] * u1[p0p0p3]) +
           py4[2] * _f2_1(i, j + 1) *
               (
                phdz4[0] * u1[p0p1m3] +
                phdz4[1] * u1[p0p1m2] +
                phdz4[2] * u1[p0p1m1] +
                phdz4[3] * u1[p0p1p0] +
                phdz4[4] * u1[p0p1p1] +
                phdz4[5] * u1[p0p1p2] +
                phdz4[6] * u1[p0p1p3]) +
           py4[3] * _f2_1(i, j + 2) *
               (
                phdz4[0] * u1[p0p2m3] +
                phdz4[1] * u1[p0p2m2] +
                phdz4[2] * u1[p0p2m1] +
                phdz4[3] * u1[p0p2p0] +
                phdz4[4] * u1[p0p2p1] +
                phdz4[5] * u1[p0p2p2] +
                phdz4[6] * u1[p0p2p3]) 
                );
  vs2 =
      dhx4[2] * v1[p0p0p0] + dhx4[0] * v1[m2p0p0] +
      dhx4[1] * v1[m1p0p0] + dhx4[3] * v1[p1p0p0] -
      J12i * _g_c(k) *
          (
           phx4[0] * _f1_2(i - 2, j) *
               (
                phdz4[0] * v1[m2p0m3] +
                phdz4[1] * v1[m2p0m2] +
                phdz4[2] * v1[m2p0m1] +
                phdz4[3] * v1[m2p0p0] +
                phdz4[4] * v1[m2p0p1] +
                phdz4[5] * v1[m2p0p2] +
                phdz4[6] * v1[m2p0p3]
                ) +
           phx4[1] * _f1_2(i - 1, j) *
               (
                phdz4[0] * v1[m1p0m3] +
                phdz4[1] * v1[m1p0m2] +
                phdz4[2] * v1[m1p0m1] +
                phdz4[3] * v1[m1p0p0] +
                phdz4[4] * v1[m1p0p1] +
                phdz4[5] * v1[m1p0p2] +
                phdz4[6] * v1[m1p0p3]
                ) +
           phx4[2] * _f1_2(i, j) *
               (
                phdz4[0] * v1[p0p0m3] +
                phdz4[1] * v1[p0p0m2] +
                phdz4[2] * v1[p0p0m1] +
                phdz4[3] * v1[p0p0p0] +
                phdz4[4] * v1[p0p0p1] +
                phdz4[5] * v1[p0p0p2] +
                phdz4[6] * v1[p0p0p3]
                ) +
           phx4[3] * _f1_2(i + 1, j) *
               (
                phdz4[0] * v1[p1p0m3] +
                phdz4[1] * v1[p1p0m2] +
                phdz4[2] * v1[p1p0m1] +
                phdz4[3] * v1[p1p0p0] +
                phdz4[4] * v1[p1p0p1] +
                phdz4[5] * v1[p1p0p2] +
                phdz4[6] * v1[p1p0p3]
                ));

    f_r      = r4[pos];
    f_rtmp   = h1*(vs1+vs2); 
    f_xy     = xy[pos]  + xmu1*(vs1+vs2) + vx1*f_r;
    r4[pos]  = f_vx2*f_r + f_wwo*f_rtmp; 
    f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);
    xy[pos]  = (f_xy + d_DT*f_rtmp)*f_dcrj;

    // xz

  float J13i = _f_1(i, j) * _g3(k);
  J13i = 1.0 * 1.0 / J13i;

  vs1 = J13i * (dz4[1] * u1[p0p0p0] + dz4[0] * u1[p0p0m1] +
                dz4[2] * u1[p0p0p1] + dz4[3] * u1[p0p0p2]);
  vs2 =
      dhx4[2] * w1[p0p0p0] + dhx4[0] * w1[m2p0p0] +
      dhx4[1] * w1[m1p0p0] + dhx4[3] * w1[p1p0p0] -
      J13i * _g(k) *
          (     
           phx4[0] * _f1_c(i - 2, j) *
               (
                pdhz4[0] * w1[m2p0m3] +
                pdhz4[1] * w1[m2p0m2] +
                pdhz4[2] * w1[m2p0m1] +
                pdhz4[3] * w1[m2p0p0] +
                pdhz4[4] * w1[m2p0p1] +
                pdhz4[5] * w1[m2p0p2] +
                pdhz4[6] * w1[m2p0p3]
               ) + 
           phx4[1] * _f1_c(i - 1, j) *
                (
                pdhz4[0] * w1[m1p0m3] +
                pdhz4[1] * w1[m1p0m2] +
                pdhz4[2] * w1[m1p0m1] +
                pdhz4[3] * w1[m1p0p0] +
                pdhz4[4] * w1[m1p0p1] +
                pdhz4[5] * w1[m1p0p2] +
                pdhz4[6] * w1[m1p0p3]) +
           phx4[2] * _f1_c(i, j) *
               (pdhz4[0] * w1[p0p0m3] +
                pdhz4[1] * w1[p0p0m2] +
                pdhz4[2] * w1[p0p0m1] +
                pdhz4[3] * w1[p0p0p0] +
                pdhz4[4] * w1[p0p0p1] +
                pdhz4[5] * w1[p0p0p2] +
                pdhz4[6] * w1[p0p0p3]) +
           phx4[3] * _f1_c(i + 1, j) *
               (pdhz4[0] * w1[p1p0m3] +
                pdhz4[1] * w1[p1p0m2] +
                pdhz4[2] * w1[p1p0m1] +
                pdhz4[3] * w1[p1p0p0] +
                pdhz4[4] * w1[p1p0p1] +
                pdhz4[5] * w1[p1p0p2] +
                pdhz4[6] * w1[p1p0p3]
                ));
    f_r     = r5[pos];
    f_rtmp  = h2*(vs1+vs2);
    f_xz    = xz[pos]  + xmu2*(vs1+vs2) + vx1*f_r; 
    r5[pos] = f_vx2*f_r + f_wwo*f_rtmp; 
    f_rtmp  = f_rtmp*(f_wwo-1.0f) + f_vx2*f_r*(1.0f-f_vx1); 
    xz[pos] = (f_xz + d_DT*f_rtmp)*f_dcrj;

    // yz

    float J23i = _f_2(i, j) * _g3(k);
    J23i = 1.0 * 1.0 / J23i;
    vs1 = J23i * (dz4[1] * v1[p0p0p0] + dz4[0] * v1[p0p0m1] +
                  dz4[2] * v1[p0p0p1] + dz4[3] * v1[p0p0p2]);
    vs2 =
        dy4[1] * w1[p0p0p0] + dy4[0] * w1[p0m1p0] +
        dy4[2] * w1[p0p1p0] + dy4[3] * w1[p0p2p0] -
        J23i * _g(k) *
            (
             py4[0] * _f2_c(i, j - 1) *
                 (
                  pdhz4[0] * w1[p0m1m3] +
                  pdhz4[1] * w1[p0m1m2] +
                  pdhz4[2] * w1[p0m1m1] +
                  pdhz4[3] * w1[p0m1p0] +
                  pdhz4[4] * w1[p0m1p1] +
                  pdhz4[5] * w1[p0m1p2] +
                  pdhz4[6] * w1[p0m1p3]
                  ) +
             py4[1] * _f2_c(i, j) *
                 (
                  pdhz4[0] * w1[p0p0m3] +
                  pdhz4[1] * w1[p0p0m2] +
                  pdhz4[2] * w1[p0p0m1] +
                  pdhz4[3] * w1[p0p0p0] +
                  pdhz4[4] * w1[p0p0p1] +
                  pdhz4[5] * w1[p0p0p2] +
                  pdhz4[6] * w1[p0p0p3]
                  ) +
             py4[2] * _f2_c(i, j + 1) *
                 (
                  pdhz4[0] * w1[p0p1m3] +
                  pdhz4[1] * w1[p0p1m2] +
                  pdhz4[2] * w1[p0p1m1] +
                  pdhz4[3] * w1[p0p1p0] +
                  pdhz4[4] * w1[p0p1p1] +
                  pdhz4[5] * w1[p0p1p2] +
                  pdhz4[6] * w1[p0p1p3]
                  ) +
             py4[3] * _f2_c(i, j + 2) *
                 (
                  pdhz4[0] * w1[p0p2m3] +
                  pdhz4[1] * w1[p0p2m2] +
                  pdhz4[2] * w1[p0p2m1] +
                  pdhz4[3] * w1[p0p2p0] +
                  pdhz4[4] * w1[p0p2p1] +
                  pdhz4[5] * w1[p0p2p2] +
                  pdhz4[6] * w1[p0p2p3]
                  ));
           
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




