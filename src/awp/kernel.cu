#include <stdio.h>
#include <math.h>
#include "awp/kernel.h"
#include "awp/pmcl3d_cons.h"
#include "cuPrintf.cu"
#include <cuda.h>

__constant__ _prec d_c1;
__constant__ _prec d_c2;
__constant__ _prec d_dth[MAXGRIDS];
__constant__ _prec d_dt1;
__constant__ _prec d_dh1[MAXGRIDS];
__constant__ _prec d_DT;
__constant__ _prec d_DH[MAXGRIDS];
__constant__ int   d_nxt[MAXGRIDS];
__constant__ int   d_nyt[MAXGRIDS];
__constant__ int   d_nzt[MAXGRIDS];
__constant__ int   d_slice_1[MAXGRIDS];
__constant__ int   d_slice_2[MAXGRIDS];
__constant__ int   d_yline_1[MAXGRIDS];
__constant__ int   d_yline_2[MAXGRIDS];

/*
texture<float, 1, cudaReadModeElementType> p_vx1;
texture<float, 1, cudaReadModeElementType> p_vx2;
texture<int, 1, cudaReadModeElementType> p_ww;
texture<float, 1, cudaReadModeElementType> p_wwo;
*/
//Parameters used for STF filtering (Daniel)
__constant__ int d_filtorder;
__constant__ double d_srcfilt_b[MAXFILT], d_srcfilt_a[MAXFILT];

//#define LDG(x) __ldg(&x)
#define LDG(x) x

//Compute initial stress on GPU (Daniel)
__constant__ _prec d_fmajor;
__constant__ _prec d_fminor;
__constant__ _prec d_Rz[9];
__constant__ _prec d_RzT[9];

__device__ void matmul3(register _prec *a, register _prec *b, register _prec *c){
   register int i, j, k;
   
   for (i=0; i<3; i++)
      for (j=0; j<3; j++)
         for (k=0; k<3; k++) c[i*3+j]+=a[i*3+k]*b[k*3+j];
}

__device__ void rotate_principal(register _prec sigma2, register _prec pfluid, register _prec *ssp){

      register _prec ss[9], tmp[9];
      register int k;

      for (k=0; k<9; k++) ss[k] = tmp[k] = ssp[k] = 0.;

      ss[0] = (sigma2 + pfluid) * d_fmajor;
      ss[4] = sigma2 + pfluid;
      ss[8] = (sigma2 + pfluid) * d_fminor;

      matmul3(d_RzT, ss, tmp);
      matmul3(tmp, d_Rz, ssp);

      ssp[0] -= pfluid;
      ssp[4] -= pfluid;
      ssp[8] -= pfluid;

      /*cuPrintf("ssp:%5.2e %5.2e %5.2e %5.2e %5.2e %5.2e\n", 
         ssp[0], ssp[4], ssp[8], 
         ssp[1], ssp[2], ssp[5]);*/
}

//end of routines for on-GPU initial stress computation (Daniel)

extern "C"
void SetDeviceConstValue(_prec *DH, _prec DT, int *nxt, int *nyt, int *nzt, int ngrids,
   _prec fmajor, _prec fminor, _prec *Rz, _prec *RzT)
{
    _prec h_c1, h_c2, *h_dth, h_dt1, *h_dh1;
    int   *slice_1,  *slice_2,  *yline_1,  *yline_2;
    int k;
    int num_bytes;

    h_c1  = 9.0/8.0;
    h_c2  = -1.0/24.0;
    h_dt1 = 1.0/DT;

    h_dth=(float*) calloc(ngrids, sizeof(float));
    h_dh1=(float*) calloc(ngrids, sizeof(float));
    slice_1=(int*) calloc(ngrids, sizeof(float));
    slice_2=(int*) calloc(ngrids, sizeof(float));
    yline_1=(int*) calloc(ngrids, sizeof(float));
    yline_2=(int*) calloc(ngrids, sizeof(float));

    for (k=0; k<ngrids; k++){
       h_dth[k] = DT/DH[k];
       h_dh1[k] = 1.0/DH[k];
       slice_1[k]  = (nyt[k]+4+ngsl2)*(nzt[k]+2*align);
       slice_2[k]  = (nyt[k]+4+ngsl2)*(nzt[k]+2*align)*2;
       yline_1[k]  = nzt[k]+2*align;
       yline_2[k]  = (nzt[k]+2*align)*2;
    }

    CUCHK(cudaMemcpyToSymbol(d_c1,      &h_c1,    sizeof(float)));
    CUCHK(cudaMemcpyToSymbol(d_c2,      &h_c2,    sizeof(float)));
    CUCHK(cudaMemcpyToSymbol(d_dt1,     &h_dt1,   sizeof(float)));
    CUCHK(cudaMemcpyToSymbol(d_DT,      &DT,      sizeof(float)));
    CUCHK(cudaMemcpyToSymbol(d_dth,     h_dth,   sizeof(float) * ngrids));
    CUCHK(cudaMemcpyToSymbol(d_dh1,     h_dh1,   sizeof(float) * ngrids));
    CUCHK(cudaMemcpyToSymbol(d_DH,      DH,      sizeof(float) * ngrids));
    CUCHK(cudaMemcpyToSymbol(d_nxt,     nxt,     sizeof(int) * ngrids));
    CUCHK(cudaMemcpyToSymbol(d_nyt,     nyt,     sizeof(int) * ngrids));
    CUCHK(cudaMemcpyToSymbol(d_nzt,     nzt,     sizeof(int) * ngrids));
    CUCHK(cudaMemcpyToSymbol(d_slice_1, slice_1, sizeof(int) * ngrids));
    CUCHK(cudaMemcpyToSymbol(d_slice_2, slice_2, sizeof(int) * ngrids));
    CUCHK(cudaMemcpyToSymbol(d_yline_1, yline_1, sizeof(int) * ngrids));
    CUCHK(cudaMemcpyToSymbol(d_yline_2, yline_2, sizeof(int) * ngrids));

    //Compute initial stress on GPU (Daniel)
    CUCHK(cudaMemcpyToSymbol(d_fmajor, &fmajor, sizeof(float)));
    CUCHK(cudaMemcpyToSymbol(d_fminor, &fminor, sizeof(float)));
    CUCHK(cudaMemcpyToSymbol(d_Rz, Rz, 9*sizeof(float)));
    CUCHK(cudaMemcpyToSymbol(d_RzT, RzT, 9*sizeof(float)));
    return;
}
/*
extern "C"
void BindArrayToTexture(float* vx1, float* vx2,int* ww, float* wwo, int memsize)   
{
   cudaBindTexture(0, p_vx1,  vx1,  memsize);
   cudaBindTexture(0, p_vx2,  vx2,  memsize);
   cudaBindTexture(0, p_ww,   ww,   memsize);
   cudaBindTexture(0, p_wwo,   wwo,   memsize);
   cudaDeviceSynchronize ();
   return;
}

extern "C"
void UnBindArrayFromTexture()
{
   cudaUnbindTexture(p_vx1);
   cudaUnbindTexture(p_vx2);
   cudaUnbindTexture(p_ww);
   cudaUnbindTexture(p_wwo);
   return;
}
*/
extern "C"
void SetDeviceFilterParameters(int filtorder, double *srcfilt_b, double *srcfilt_a){
    CUCHK(cudaMemcpyToSymbol(d_filtorder, &filtorder, sizeof(int)));
    CUCHK(cudaMemcpyToSymbol(d_srcfilt_b, srcfilt_b, (filtorder+1)*sizeof(double)));
    CUCHK(cudaMemcpyToSymbol(d_srcfilt_a, srcfilt_a, (filtorder+1)*sizeof(double)));
}

template <int BLOCK_Z, int BLOCK_Y>
__global__ void dvelcx_opt(_prec * __restrict__ u1,
                           _prec * __restrict__ v1, 
                           _prec * __restrict__ w1,
                           const _prec *xx,    const _prec *yy,    const _prec *zz,
                           const _prec *xy,    const _prec *xz,    const _prec *yz, 
                           const _prec *dcrjx, const _prec *dcrjy, const _prec *dcrjz,
                           const _prec *d_1,   
                           const int s_i,
                           const int e_i,
                           const int d_i,
                           const int ngrids) {
    register _prec f_xx,    xx_im1,  xx_ip1,  xx_im2;
    register _prec f_xy,    xy_ip1,  xy_ip2,  xy_im1;
    register _prec f_xz,    xz_ip1,  xz_ip2,  xz_im1;
    register _prec f_dcrj, f_dcrjy, f_dcrjz, f_yz;

    const int k = blockIdx.x*BLOCK_SIZE_Z+threadIdx.x+align;
    const int j = blockIdx.y*BLOCK_SIZE_Y+threadIdx.y+2+ngsl;
    int pos;
   
    if (k > d_nzt[d_i]+align-3 && d_i > 0) return;
    if (k < align+3 && d_i<(ngrids-1)) return;

    pos  = e_i*d_slice_1[d_i]+j*d_yline_1[d_i]+k;

    f_xx    = xx[pos+d_slice_1[d_i]];
    xx_im1  = xx[pos];
    xx_im2  = xx[pos-d_slice_1[d_i]]; 
    xy_ip1  = xy[pos+d_slice_2[d_i]];
    f_xy    = xy[pos+d_slice_1[d_i]];
    xy_im1  = xy[pos];
    xz_ip1  = xz[pos+d_slice_2[d_i]];
    f_xz    = xz[pos+d_slice_1[d_i]];
    xz_im1  = xz[pos];
    f_dcrjz = dcrjz[k];
    f_dcrjy = dcrjy[j];
    _prec f_d_1 = d_1[pos+d_slice_1[d_i]]; //f_d_1_ip1 will get this value
    _prec f_d_1_km1 = d_1[pos+d_slice_1[d_i]-1]; //f_d_1_ik1 will get this value
    for(int i=e_i; i >= s_i; i--)   
    {
        pos = i*d_slice_1[d_i]+j*d_yline_1[d_i]+k;
        const int pos_km2  = pos-2;
        const int pos_km1  = pos-1;
        const int pos_kp1  = pos+1;
        const int pos_kp2  = pos+2;
        const int pos_jm2  = pos-d_yline_2[d_i];
        const int pos_jm1  = pos-d_yline_1[d_i];
        const int pos_jp1  = pos+d_yline_1[d_i];
        const int pos_jp2  = pos+d_yline_2[d_i];
        const int pos_im1  = pos-d_slice_1[d_i];
        const int pos_im2  = pos-d_slice_2[d_i];
        //const int pos_ip1  = pos+d_slice_1[d_i];
        const int pos_jk1  = pos-d_yline_1[d_i]-1;
        //const int pos_ik1  = pos+d_slice_1[d_i]-1;
        const int pos_ijk  = pos+d_slice_1[d_i]-d_yline_1[d_i];

        // xx pipeline
        xx_ip1   = f_xx;
        f_xx     = xx_im1;
        xx_im1   = xx_im2;
        xx_im2   = xx[pos_im2];

        // xy pipeline
        xy_ip2   = xy_ip1;
        xy_ip1   = f_xy;
        f_xy     = xy_im1;
        xy_im1   = xy[pos_im1];

        // xz pipeline
        xz_ip2   = xz_ip1;
        xz_ip1   = f_xz;
        f_xz     = xz_im1;
        xz_im1   = xz[pos_im1];

        f_yz     = yz[pos];

        // d_1[pos] pipeline
        _prec f_d_1_ip1 = f_d_1;
        f_d_1 = d_1[pos];

        // d_1[pos-1] pipeline
        _prec f_d_1_ik1 = f_d_1_km1;
        f_d_1_km1 = d_1[pos_km1];

        f_dcrj   = dcrjx[i]*f_dcrjy*f_dcrjz;
        //f_d1     = 0.25*(d_1[pos] + d_1[pos_jm1] + d_1[pos_km1] + d_1[pos_jk1]);
        //f_d2     = 0.25*(d_1[pos] + d_1[pos_ip1] + d_1[pos_km1] + d_1[pos_ik1]);
        //f_d3     = 0.25*(d_1[pos] + d_1[pos_ip1] + d_1[pos_jm1] + d_1[pos_ijk]);
        _prec f_d1     = 0.25*(f_d_1 + d_1[pos_jm1] + f_d_1_km1 + d_1[pos_jk1]);
        _prec f_d2     = 0.25*(f_d_1 + f_d_1_ip1    + f_d_1_km1 + f_d_1_ik1);
        _prec f_d3     = 0.25*(f_d_1 + f_d_1_ip1    + d_1[pos_jm1] + d_1[pos_ijk]);

        f_d1     = d_dth[d_i]/f_d1;
        f_d2     = d_dth[d_i]/f_d2;
	f_d3     = d_dth[d_i]/f_d3;

    	u1[pos]  = (u1[pos] + f_d1*( d_c1*(f_xx        - xx_im1)      + d_c2*(xx_ip1      - xx_im2) 
                                   + d_c1*(f_xy        - xy[pos_jm1]) + d_c2*(xy[pos_jp1] - xy[pos_jm2])
                                   + d_c1*(f_xz        - xz[pos_km1]) + d_c2*(xz[pos_kp1] - xz[pos_km2]) ))*f_dcrj; 
        /*if ((d_i==0) && (k==32) && (i==94) && (j==97)) {
           cuPrintf("after update: u1[%d]=%e, f_d1=%e, xx=%.20g, %20g, %20g, %20g\n", 
              pos, u1[pos], f_d1, f_xx, xx_im1, xx_ip1, xx_im2);
           cuPrintf("xy=%.20g %.20g %.20g %.20g\n", f_xy, xy[pos_jm1], xy[pos_jp1], xy[pos_jm2]);
           cuPrintf("yz=%.20g %.20g %.20g %.20g\n", f_xz, xz[pos_km1], xz[pos_kp1], xz[pos_km2]);
           cuPrintf("f_dcrj=%e, d_c1=%e, d_c2=%e\n", f_dcrj, d_c1, d_c2);
        }*/
        /*if ((k==43) && (i==102) && (j==100)) cuPrintf("before update: v1[%d]=%e\n", pos, v1[pos]);*/
        v1[pos]  = (v1[pos] + f_d2*( d_c1*(xy_ip1      - f_xy)        + d_c2*(xy_ip2      - xy_im1)
                                   + d_c1*(yy[pos_jp1] - yy[pos])     + d_c2*(yy[pos_jp2] - yy[pos_jm1])
                                   + d_c1*(f_yz        - yz[pos_km1]) + d_c2*(yz[pos_kp1] - yz[pos_km2]) ))*f_dcrj;
        /*if ((k==51) && (i==102) && (j==100)) {
           cuPrintf("after update: v1[%d]=%e, f_d2=%e, xy=%.20g, %.20g, %.20g, %.20g\n", 
              pos, v1[pos], f_d2, xy_ip1, f_xy, xy_ip2, xy_im1);
           cuPrintf("yy=%.20g %.20g %.20g %.20g\n", yy[pos_jp1], yy[pos], yy[pos_jp2], yy[pos_jm1]);
           cuPrintf("yz=%.20g %.20g %.20g %.20g\n", f_yz, yz[pos_km1], yz[pos_kp1], yz[pos_km2]);
           cuPrintf("f_dcrj=%e, d_c1=%e, d_c2=%e\n", f_dcrj, d_c1, d_c2);
        }*/

        //if ((k==39) && (i==102) && (j==104)) cuPrintf("before update: w1[%d]=%e\n", pos, w1[pos]);

        w1[pos]  = (w1[pos] + f_d3*( d_c1*(xz_ip1      - f_xz)        + d_c2*(xz_ip2      - xz_im1)
                                   + d_c1*(f_yz        - yz[pos_jm1]) + d_c2*(yz[pos_jp1] - yz[pos_jm2])
                                   + d_c1*(zz[pos_kp1] - zz[pos])     + d_c2*(zz[pos_kp2] - zz[pos_km1]) ))*f_dcrj;

        /*if ((k==39) && (i==102) && (j==104)) {
           cuPrintf("after update: w1[%d]=%e, f_d3=%e, f_xz=%e, xz_ip1=%e, xz_ip2=%e, xz_im1=%e\n", 
              pos, w1[pos], f_d3, f_xz, xz_ip1, xz_ip2, xz_im1);
           cuPrintf("f_yz=%e, yz[%d]=%e, yz[%d]=%e, yz[%d]=%e\n", f_yz, pos_jm1, yz[pos_jm1], pos_jp1, yz[pos_jp1], 
             pos_jm2, yz[pos_jm2]);
           cuPrintf("zz[%d]=%e, zz[%d]=%e, zz[%d]=%e, zz[%d]=%e\n", pos_kp1, zz[pos_kp1], pos, zz[pos], 
             pos_kp2, zz[pos_kp2], pos_km1, zz[pos_km1]);
           cuPrintf("f_dcrj=%e, d_c1=%e, d_c2=%e\n", f_dcrj, d_c1, d_c2);
        }*/
    }

    return;

}

__global__ void print_const(int ngrids)
{
    int p;
    for (p=0; p<ngrids; p++){
       cuPrintf("device constants[%d]:\nd_yline_=%d,%d, d_slice=%d,%d,nxt,nyt,nzt=%d,%d,%d\n",
	     p, d_yline_1[p], d_yline_2[p], d_slice_1[p], d_slice_2[p], d_nxt[p], d_nyt[p], d_nzt[p]);
       cuPrintf("d_DH=%e, d_dth=%e, d_dh1=%e\n", d_DH[p], d_dth[p], d_dh1[p]); 
    }
    /*cuPrintf("d_filtorder=%d\n", d_filtorder);
    if (d_filtorder > 0){
       for (p=0; p<d_filtorder+1; p++){
           cuPrintf("d_srcfilt_b[%d] = %le, d_srcfilt_a[%d] = %le\n", p, d_srcfilt_b[p], p, 
               d_srcfilt_a[p]);
       }
    }*/
}

extern "C"
void print_const_H(int ngrids)
{
    dim3 block (1, 1, 1);
    dim3 grid (1, 1, 1);
    CUCHK(cudaPrintfInit());
    print_const<<<grid, block, 0>>>(ngrids);
    CUCHK(cudaPrintfDisplay(stdout, 1));
    cudaPrintfEnd();
    return;
}

extern "C"
void dvelcx_H_opt(float* u1,    float* v1,    float* w1,    
                  float* xx,  float* yy, float* zz, float* xy,      float* xz, float* yz,
                  float* dcrjx, float* dcrjy, float* dcrjz,
                  float* d_1, int nyt,   int nzt,  
                  cudaStream_t St, int s_i,   int e_i, int d_i, int ngrids)
{
    dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1);
    dim3 grid ((nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, (nyt+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,1);
    CUCHK(cudaFuncSetCacheConfig(dvelcx_opt<BLOCK_SIZE_Z, BLOCK_SIZE_Y>, cudaFuncCachePreferL1));
    CUCHK(cudaGetLastError());
    /*CUCHK(cudaPrintfInit());*/
    //fprintf(stdout, "launching dvelcx_opt\n");
    dvelcx_opt<BLOCK_SIZE_Z, BLOCK_SIZE_Y><<<grid, block, 0, St>>>(u1, v1, w1, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d_1, 
         s_i, e_i, d_i, ngrids);
    /*CUCHK(cudaPrintfDisplay(stdout, 1));
    cudaPrintfEnd();*/
    CUCHK(cudaGetLastError());
    return;
}
extern "C"
void dvelcy_H(float* u1,       float* v1,    float* w1,    float* xx,  float* yy, float* zz, float* xy,   float* xz,   float* yz,
              float* dcrjx,    float* dcrjy, float* dcrjz, float* d_1, int nxt,   int nzt,   float* s_u1, float* s_v1, float* s_w1,  
              cudaStream_t St, int s_j,      int e_j,      int rank, int d_i)
{
    if(rank==-1) return;
    dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1);
    dim3 grid ((nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, (nxt+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,1);
    cudaFuncSetCacheConfig(dvelcy, cudaFuncCachePreferL1);
    CUCHK(cudaGetLastError());
    dvelcy<<<grid, block, 0, St>>>(u1, v1, w1, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d_1, s_u1, s_v1, s_w1, s_j, e_j, d_i);
    CUCHK(cudaGetLastError());
    return;
}

extern "C"
void update_bound_y_H(float* u1,   float* v1, float* w1, float* f_u1,      float* f_v1,      float* f_w1,  float* b_u1, float* b_v1, 
                      float* b_w1, int nxt,   int nzt,   cudaStream_t St1, cudaStream_t St2, int rank_f,  int rank_b, int d_i)
{
     if(rank_f==-1 && rank_b==-1) return;
     dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1);
     dim3 grid ((nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, (nxt+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,1);
     cudaFuncSetCacheConfig(update_boundary_y, cudaFuncCachePreferL1);
     update_boundary_y<<<grid, block, 0, St1>>>(u1, v1, w1, f_u1, f_v1, f_w1, rank_f, Front, d_i);
     update_boundary_y<<<grid, block, 0, St2>>>(u1, v1, w1, b_u1, b_v1, b_w1, rank_b, Back, d_i);
     return;
}

extern "C"
void dstrqc_H(float* xx,       float* yy,     float* zz,    float* xy,    float* xz, float* yz,
              float* r1,       float* r2,     float* r3,    float* r4,    float* r5, float* r6,
              float* u1,       float* v1,     float* w1,    float* lam,   float* mu, float* qp,float* coeff, 
              float* qs,       float* dcrjx,  float* dcrjy, float* dcrjz, int nyt,   int nzt, 
              cudaStream_t St, float* lam_mu, 
              _prec *vx1, _prec *vx2, int *ww, _prec *wwo,
              int NX,       int NPC, int rankx,    int ranky, int  s_i,  
              int e_i,         int s_j,       int e_j, int d_i)
{
    dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1);
    dim3 grid ((nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, (e_j-s_j+1+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,1);
    cudaFuncSetCacheConfig(dstrqc, cudaFuncCachePreferL1);
    dstrqc<<<grid, block, 0, St>>>(xx,    yy,    zz,  xy,  xz, yz, r1, r2,    r3,    r4,    r5,     r6, 
                                   u1,    v1,    w1,  lam, mu, qp,coeff, qs, dcrjx, dcrjy, dcrjz, lam_mu, 
                                   vx1, vx2, ww, wwo, 
                                   NX, NPC, rankx, ranky, nzt, s_i, e_i, s_j, e_j, d_i);
    return;
}

template<int BLOCKX, int BLOCKY>
__global__ void 
__launch_bounds__(512,2)
dstrqc_new(float* __restrict__ xx, float* __restrict__ yy, float* __restrict__ zz,
           float* __restrict__ xy, float* __restrict__ xz, float* __restrict__ yz,
       float* __restrict__ r1, float* __restrict__ r2,  float* __restrict__ r3, 
       float* __restrict__ r4, float* __restrict__ r5,  float* __restrict__ r6,
       float* __restrict__ u1, 
       float* __restrict__ v1,    
       float* __restrict__ w1,    
       float* lam,   
       float* mu,     
       float* qp,
       float* coeff, 
       float* qs, 
       float* dcrjx, float* dcrjy, float* dcrjz, float* lam_mu, 
       //_prec *d_vx1, _prec *d_vx2, _prec *d_ww, _prec *d_wwo, //pengs version
       _prec *d_vx1, _prec *d_vx2, int *d_ww, _prec *d_wwo,
       int NX, int NPC, int rankx, int ranky, int nzt, int s_i, int e_i, int s_j, int e_j, int d_i) 
{ 
//#define SMEM  /*this does not work with DM.  Do not use. */
//#define REGQ
  register int   i,  j,  k,  g_i;
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
#ifdef REGQ
  _prec mu_i, mu_ip1, lam_i, lam_ip1, qp_i, qp_ip1, qs_i, qs_ip1;
  _prec mu_jk1, mu_ijk1, lam_jk1, lam_ijk1, qp_jk1, qp_ijk1, qs_jk1, qs_ijk1;
  _prec mu_jm1, mu_ijk, lam_jm1, lam_ijk, qp_jm1, qp_ijk, qs_jm1, qs_ijk;
  _prec mu_km1, mu_ik1, lam_km1, lam_ik1, qp_km1, qp_ik1, qs_km1, qs_ik1;
#endif
  int maxk, mink=align+3;
    
  k    = blockIdx.x*blockDim.x+threadIdx.x+align;
  j    = blockIdx.y*blockDim.y+threadIdx.y+s_j;

  if (d_i == 0) {
     maxk = nzt + align -1;
  }
  else maxk = nzt + align -3;

  if (k < mink || k > maxk || j > e_j) return;
  
  //if (k==align) cuPrintf("inside dstrqc_new(): j=%d\n", j);

#ifdef SMEM
  __shared__ _prec s_u1[BLOCKX+3][BLOCKY+3], s_v1[BLOCKX+3][BLOCKY+3], s_w1[BLOCKX+3][BLOCKY+3];
#endif

  i    = e_i;
  pos  = i*d_slice_1[d_i]+j*d_yline_1[d_i]+k;

  u1_ip1 = u1[pos+d_slice_2[d_i]];
  f_u1   = u1[pos+d_slice_1[d_i]];
  u1_im1 = u1[pos];    
  f_v1   = v1[pos+d_slice_1[d_i]];
  v1_im1 = v1[pos];
  v1_im2 = v1[pos-d_slice_1[d_i]];
  f_w1   = w1[pos+d_slice_1[d_i]];
  w1_im1 = w1[pos];
  w1_im2 = w1[pos-d_slice_1[d_i]];
  f_dcrjz = dcrjz[k];
  f_dcrjy = dcrjy[j];

#ifdef REGQ  
  mu_i  = mu [pos+d_slice_1[d_i]];
  lam_i = lam[pos+d_slice_1[d_i]];
  qp_i  = qp [pos+d_slice_1[d_i]];
  qs_i  = qs [pos+d_slice_1[d_i]];
  
  mu_jk1  = mu [pos+d_slice_1[d_i]-d_yline_1[d_i]-1];
  lam_jk1 = lam[pos+d_slice_1[d_i]-d_yline_1[d_i]-1];
  qp_jk1  = qp [pos+d_slice_1[d_i]-d_yline_1[d_i]-1];
  qs_jk1  = qs [pos+d_slice_1[d_i]-d_yline_1[d_i]-1];

  mu_jm1  = mu [pos+d_slice_1[d_i]-d_yline_1[d_i]];
  lam_jm1 = lam[pos+d_slice_1[d_i]-d_yline_1[d_i]];
  qp_jm1  = qp [pos+d_slice_1[d_i]-d_yline_1[d_i]];
  qs_jm1  = qs [pos+d_slice_1[d_i]-d_yline_1[d_i]];

  mu_km1  = mu [pos+d_slice_1[d_i]-1];
  lam_km1 = lam[pos+d_slice_1[d_i]-1];
  qp_km1  = qp [pos+d_slice_1[d_i]-1];
  qs_km1  = qs [pos+d_slice_1[d_i]-1];
#endif
  for(i=e_i;i>=s_i;i--)
  {
    f_vx1 = d_vx1[pos];
    f_vx2 = d_vx2[pos];
    f_ww  = d_ww[pos];
    f_wwo = d_wwo[pos];
    
#ifdef REGQ
    mu_ip1   = mu_i;
    lam_ip1  = lam_i;
    qp_ip1   = qp_i;
    qs_ip1   = qs_i;
    mu_ijk1  = mu_jk1;
    lam_ijk1 = lam_jk1;
    qp_ijk1  = qp_jk1;
    qs_ijk1  = qs_jk1;
    mu_ijk   = mu_jm1;
    lam_ijk  = lam_jm1;
    qp_ijk   = qp_jm1;
    qs_ijk   = qs_jm1;
    mu_ik1   = mu_km1;
    lam_ik1  = lam_km1;
    qp_ik1   = qp_km1;
    qs_ik1   = qs_km1;

    mu_i    = LDG(mu [pos]);
    lam_i   = LDG(lam[pos]);
    qp_i    = LDG(qp [pos]);
    qs_i    = LDG(qs [pos]);
    mu_jk1  = LDG(mu [pos-d_yline_1[d_i]-1]);
    lam_jk1 = LDG(lam[pos-d_yline_1[d_i]-1]);
    qp_jk1  = LDG(qp [pos-d_yline_1[d_i]-1]);
    qs_jk1  = LDG(qs [pos-d_yline_1[d_i]-1]);
    mu_jm1  = LDG(mu [pos-d_yline_1[d_i]]);
    lam_jm1 = LDG(lam[pos-d_yline_1[d_i]]);
    qp_jm1  = LDG(qp [pos-d_yline_1[d_i]]);
    qs_jm1  = LDG(qs [pos-d_yline_1[d_i]]);
    mu_km1  = LDG(mu [pos-1]);
    lam_km1 = LDG(lam[pos-1]);
    qp_km1  = LDG(qp [pos-1]);
    qs_km1  = LDG(qs [pos-1]);
#endif

/*
  if(f_wwo!=f_wwo){
  xx[pos] = yy[pos] = zz[pos] = xy[pos] = xz[pos] = yz[pos] = 1.0;
  r1[pos] = r2[pos] = r3[pos] = r4[pos] = r5[pos] = r6[pos] = 1.0;
  return;
  }
*/
    f_dcrj   = dcrjx[i]*f_dcrjy*f_dcrjz;

    pos_km2  = pos-2;
    pos_km1  = pos-1;
    pos_kp1  = pos+1;
    pos_kp2  = pos+2;
    pos_jm2  = pos-d_yline_2[d_i];
    pos_jm1  = pos-d_yline_1[d_i];
    pos_jp1  = pos+d_yline_1[d_i];
    pos_jp2  = pos+d_yline_2[d_i];
    pos_im2  = pos-d_slice_2[d_i];
    pos_im1  = pos-d_slice_1[d_i];
    pos_ip1  = pos+d_slice_1[d_i];
    pos_jk1  = pos-d_yline_1[d_i]-1;
    pos_ik1  = pos+d_slice_1[d_i]-1;
    pos_ijk  = pos+d_slice_1[d_i]-d_yline_1[d_i];
    pos_ijk1 = pos+d_slice_1[d_i]-d_yline_1[d_i]-1;

#ifdef REGQ
    xl       = 8.0f/(  lam_i      + lam_ip1 + lam_jm1 + lam_ijk
                       + lam_km1  + lam_ik1 + lam_jk1 + lam_ijk1 );
    xm       = 16.0f/( mu_i       + mu_ip1  + mu_jm1  + mu_ijk
                      + mu_km1   + mu_ik1  + mu_jk1  + mu_ijk1 );
    xmu1     = 2.0f/(  mu_i       + mu_km1 );
    xmu2     = 2.0/(  mu_i       + mu_jm1 );
    xmu3     = 2.0/(  mu_i       + mu_ip1 );
    xl       = xl  +  xm;
    qpa      = 0.0625f*( qp_i     + qp_ip1 + qp_jm1 + qp_ijk
                        + qp_km1 + qp_ik1 + qp_jk1 + qp_ijk1 );
#else
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
#endif
//                        www=f_ww;
    if(1.0f/(qpa*2.0f)<=200.0f)
    {
//      printf("coeff[f_ww*2-2] %g\n",coeff[f_ww*2-2]);
      qpaw=coeff[f_ww*2-2]*(2.*qpa)*(2.*qpa)+coeff[f_ww*2-1]*(2.*qpa);
//              qpaw=coeff[www*2-2]*(2.*qpa)*(2.*qpa)+coeff[www*2-1]*(2.*qpa);
//                qpaw=qpaw/2.;
    }
    else {
      qpaw  = f_wwo*qpa;
    }
//                 printf("qpaw %f\n",qpaw);
//              printf("qpaw1 %g\n",qpaw);
    qpaw=qpaw/f_wwo;
//      printf("qpaw2 %g\n",qpaw);


#ifdef REGQ
    h        = 0.0625f*( qs_i     + qs_ip1 + qs_jm1 + qs_ijk
                         + qs_km1 + qs_ik1 + qs_jk1 + qs_ijk1 );
#else
    h        = 0.0625f*( LDG(qs[pos])     + LDG(qs[pos_ip1]) + LDG(qs[pos_jm1]) + LDG(qs[pos_ijk])
                         + LDG(qs[pos_km1]) + LDG(qs[pos_ik1]) + LDG(qs[pos_jk1]) + LDG(qs[pos_ijk1]) );
#endif
    if(1.0f/(h*2.0f)<=200.0f)
    {
      hw=coeff[f_ww*2-2]*(2.0f*h)*(2.0f*h)+coeff[f_ww*2-1]*(2.0f*h);
      //                  hw=hw/2.0f;
    }
    else {
      hw  = f_wwo*h;
    }
    hw=hw/f_wwo;


    h1       = 0.250f*(  qs[pos]     + qs[pos_km1] );

    if(1.0f/(h1*2.0f)<=200.0f)
    {
      h1w=coeff[f_ww*2-2]*(2.0f*h1)*(2.0f*h1)+coeff[f_ww*2-1]*(2.0f*h1);
      //                  h1w=h1w/2.0f;
    }
    else {
      h1w  = f_wwo*h1;
    }
    h1w=h1w/f_wwo;



    h2       = 0.250f*(  qs[pos]     + qs[pos_jm1] );
    if(1.0f/(h2*2.0f)<=200.0f)
    {
      h2w=coeff[f_ww*2-2]*(2.0f*h2)*(2.0f*h2)+coeff[f_ww*2-1]*(2.0f*h2);
      //                  h2w=h2w/2.;
    }
    else {
      h2w  = f_wwo*h2;
    }
    h2w=h2w/f_wwo;


    h3       = 0.250f*(  qs[pos]     + qs[pos_ip1] );
    if(1.0f/(h3*2.0f)<=200.0f)
    {
      h3w=coeff[f_ww*2-2]*(2.0f*h3)*(2.0f*h3)+coeff[f_ww*2-1]*(2.0f*h3);
      //                  h3w=h3w/2.0f;
    }
    else {
      h3w  = f_wwo*h3;
    }
    h3w=h3w/f_wwo;

    h        = -xm*hw*d_dh1[d_i];
    h1       = -xmu1*h1w*d_dh1[d_i];
    h2       = -xmu2*h2w*d_dh1[d_i];
    h3       = -xmu3*h3w*d_dh1[d_i];


    //        h1       = -xmu1*hw1*d_dh1[d_i];
    //h2       = -xmu2*hw2*d_dh1[d_i];
    //h3       = -xmu3*hw3*d_dh1[d_i];


    qpa      = -qpaw*xl*d_dh1[d_i];
    //        qpa      = -qpaw*xl*d_dh1[d_i];

    xm       = xm*d_dth[d_i];
    xmu1     = xmu1*d_dth[d_i];
    xmu2     = xmu2*d_dth[d_i];
    xmu3     = xmu3*d_dth[d_i];
    xl       = xl*d_dth[d_i];
    //  f_vx2    = f_vx2*f_vx1;
    h        = h*f_vx1;
    h1       = h1*f_vx1;
    h2       = h2*f_vx1;
    h3       = h3*f_vx1;
    qpa      = qpa*f_vx1;

    #ifndef ELA
    xm       = xm+d_DT*h;
    xmu1     = xmu1+d_DT*h1;
    xmu2     = xmu2+d_DT*h2;
    xmu3     = xmu3+d_DT*h3;
    vx1      = d_DT*(1+f_vx2*f_vx1);
    #endif
        
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

    if (d_i == 0){ /* apply FS condition on uppermost grid only */
       if(k == d_nzt[d_i]+align-1)
       {
	 u1[pos_kp1] = f_u1 - (f_w1        - w1_im1);
	 v1[pos_kp1] = f_v1 - (w1[pos_jp1] - f_w1);

	 g_i  = d_nxt[d_i]*rankx + i - ngsl - 1;
    
	 if(g_i<NX || NPC ==2) // NVE==2 means periodic BCs - Daniel
	   vs1	= u1_ip1 - (w1_ip1    - f_w1);
	 else
	   vs1	= 0.0f;

	 g_i  = d_nyt[d_i]*ranky + j - ngsl - 1;
	 if(g_i>1 || NPC ==2) // periodic BCs
	   vs2	= v1[pos_jm1] - (f_w1 - w1[pos_jm1]);
	 else
	   vs2	= 0.0f;

	 w1[pos_kp1]	= w1[pos_km1] - lam_mu[i*(d_nyt[d_i]+4+ngsl2) + j]*((vs1         - u1[pos_kp1]) + (u1_ip1 - f_u1)
									  +     			                (v1[pos_kp1] - vs2)         + (f_v1   - v1[pos_jm1]) );
       }
       else if(k == d_nzt[d_i]+align-2)
       {
	 u1[pos_kp2] = u1[pos_kp1] - (w1[pos_kp1]   - w1[pos_im1+1]);
	 v1[pos_kp2] = v1[pos_kp1] - (w1[pos_jp1+1] - w1[pos_kp1]);
       }
    }

#ifdef SMEM
    __threadfence_block();
    __syncthreads();
    s_u1[threadIdx.x+1][threadIdx.y+1] = u1[pos];
    s_v1[threadIdx.x+1][threadIdx.y+2] = v1[pos];
    s_w1[threadIdx.x+2][threadIdx.y+1] = w1[pos];
    if (threadIdx.x == 0) { // k halo
      s_u1[0       ][threadIdx.y+1] = u1[pos-1];
      s_u1[BLOCKX+1][threadIdx.y+1] = u1[pos+BLOCKX];
      s_u1[BLOCKX+2][threadIdx.y+1] = u1[pos+BLOCKX+1];

      s_v1[0       ][threadIdx.y+2] = v1[pos-1];
      s_v1[BLOCKX+1][threadIdx.y+2] = v1[pos+BLOCKX];
      s_v1[BLOCKX+2][threadIdx.y+2] = v1[pos+BLOCKX+1];

      s_w1[0       ][threadIdx.y+1] = w1[pos-2];
      s_w1[1       ][threadIdx.y+1] = w1[pos-1];
      s_w1[BLOCKX+2][threadIdx.y+1] = w1[pos+BLOCKX];
    }
    if (threadIdx.y == 0) { // j halo
      s_u1[threadIdx.x+1][0       ] = u1[pos - d_yline_1[d_i]];
      s_u1[threadIdx.x+1][BLOCKY+1] = u1[pos + BLOCKY*d_yline_1[d_i]];
      s_u1[threadIdx.x+1][BLOCKY+2] = u1[pos + (BLOCKY+1)*d_yline_1[d_i]];

      s_v1[threadIdx.x+1][0       ] = v1[pos - 2*d_yline_1[d_i]];
      s_v1[threadIdx.x+1][1       ] = v1[pos - d_yline_1[d_i]];
      s_v1[threadIdx.x+1][BLOCKY+2] = v1[pos + BLOCKY*d_yline_1[d_i]];

      s_w1[threadIdx.x+2][0       ] = w1[pos - d_yline_1[d_i]];
      s_w1[threadIdx.x+2][BLOCKY+1] = w1[pos + BLOCKY*d_yline_1[d_i]];
      s_w1[threadIdx.x+2][BLOCKY+2] = w1[pos + (BLOCKY+1)*d_yline_1[d_i]];
    }
    __syncthreads();
#endif 

    vs1      = d_c1*(u1_ip1 - f_u1)        + d_c2*(u1_ip2      - u1_im1);
#ifdef SMEM
    vs2      = d_c1*(f_v1   - s_v1[threadIdx.x+1][threadIdx.y+1]) 
      + d_c2*(s_v1[threadIdx.x+1][threadIdx.y+3] - s_v1[threadIdx.x+1][threadIdx.y]);
    vs3      = d_c1*(f_w1   - s_w1[threadIdx.x+1][threadIdx.y+1]) + 
      d_c2*(s_w1[threadIdx.x+3][threadIdx.y+1] - s_w1[threadIdx.x][threadIdx.y+1]);
#else
    vs2      = d_c1*(f_v1   - v1[pos_jm1]) + d_c2*(v1[pos_jp1] - v1[pos_jm2]);
    vs3      = d_c1*(f_w1   - w1[pos_km1]) + d_c2*(w1[pos_kp1] - w1[pos_km2]);
#endif

    tmp      = xl*(vs1+vs2+vs3);
    #ifdef ELA
    if (k==41 && i==102 && j==102) cuPrintf("before update xx=%.20g\n", xx[pos]);
    xx[pos]  = (xx[pos] + tmp - xm*(vs2+vs3))*f_dcrj;
    yy[pos]  = (yy[pos] + tmp - xm*(vs1+vs3))*f_dcrj;
    zz[pos]  = (zz[pos] + tmp - xm*(vs1+vs2))*f_dcrj;
    if (k==41 && i==102 && j==102)
       cuPrintf("after update xx=%.30g, xm=%.30g, vs1=%.30g, vs2=%.30g, vs3=%.30g, f_drj=%.30g\n", 
	     xx[pos], xm, vs1, vs2, vs3, f_dcrj);
    #else
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
    #endif

#ifdef SMEM
    vs1      = d_c1*(s_u1[threadIdx.x+1][threadIdx.y+2] - f_u1)   
      + d_c2*(s_u1[threadIdx.x+1][threadIdx.y+3] - s_u1[threadIdx.x+1][threadIdx.y]);
#else
    vs1      = d_c1*(u1[pos_jp1] - f_u1)   + d_c2*(u1[pos_jp2] - u1[pos_jm1]);
#endif

    vs2      = d_c1*(f_v1        - v1_im1) + d_c2*(v1_ip1      - v1_im2);
    #ifdef ELA
    xy[pos]  = (xy[pos] + xmu1*(vs1+vs2))*f_dcrj;
    #else
    f_r      = r4[pos];
    f_rtmp   = h1*(vs1+vs2); 
    f_xy     = xy[pos]  + xmu1*(vs1+vs2) + vx1*f_r;
    r4[pos]  = f_vx2*f_r + f_wwo*f_rtmp; 
    f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);
    xy[pos]  = (f_xy + d_DT*f_rtmp)*f_dcrj;
    #endif
 
    //moved to separate subroutine fstr, to be executed after plasticity (Daniel)
    /*if(k == d_nzt[d_i]+align-1)
      {
      zz[pos+1] = -zz[pos];
      xz[pos]   = 0.0f;
      yz[pos]   = 0.0f;
      }
      else
      {*/
#ifdef SMEM
    vs1     = d_c1*(s_u1[threadIdx.x+2][threadIdx.y+1] - f_u1)   
      + d_c2*(s_u1[threadIdx.x+3][threadIdx.y+1] - s_u1[threadIdx.x][threadIdx.y+1]);
#else
    vs1     = d_c1*(u1[pos_kp1] - f_u1)   + d_c2*(u1[pos_kp2] - u1[pos_km1]);
#endif
    vs2     = d_c1*(f_w1        - w1_im1) + d_c2*(w1_ip1      - w1_im2);
    #ifdef ELA
    xz[pos] = (xz[pos] + xmu2*(vs1+vs2))*f_dcrj;
    #else
    f_r     = r5[pos];
    f_rtmp  = h2*(vs1+vs2);
    f_xz    = xz[pos]  + xmu2*(vs1+vs2) + vx1*f_r; 
    r5[pos] = f_vx2*f_r + f_wwo*f_rtmp; 
    f_rtmp  = f_rtmp*(f_wwo-1.0f) + f_vx2*f_r*(1.0f-f_vx1); 
    xz[pos] = (f_xz + d_DT*f_rtmp)*f_dcrj;
    #endif
    /*if (k==85 && i==38 and j==38){
       cuPrintf("xz=%e, pos=%d, f_rtmp=%e, f_wwo=%e, f_vx2=%e, r5=%e, vs2=%e, vs1=%e\n",
		 xz[pos], pos, f_rtmp, f_wwo, f_vx2, r5[pos], vs2, vs1 );
       cuPrintf("u1[%d]=%e, f_u1=%e, u1[%d]=%e, u1[%d]=%e\n",
                 pos_kp1, u1[pos_kp1], f_u1, pos_kp2, u1[pos_kp2], pos_km1, u1[pos_km1]);
    }*/
	 
#ifdef SMEM
    vs2     = d_c1*(s_w1[threadIdx.x+2][threadIdx.y+2] - f_w1) + 
      d_c2*(s_w1[threadIdx.x+2][threadIdx.y+3] - s_w1[threadIdx.x+2][threadIdx.y]);
    vs1     = d_c1*(s_v1[threadIdx.x+2][threadIdx.y+2] - f_v1) + 
      d_c2*(s_v1[threadIdx.x+3][threadIdx.y+2] - s_v1[threadIdx.x][threadIdx.y+2]);
#else
    vs1     = d_c1*(v1[pos_kp1] - f_v1) + d_c2*(v1[pos_kp2] - v1[pos_km1]);
    vs2     = d_c1*(w1[pos_jp1] - f_w1) + d_c2*(w1[pos_jp2] - w1[pos_jm1]);
#endif
    #ifdef ELA
    yz[pos] = (yz[pos] + xmu3*(vs1+vs2))*f_dcrj; 
    #else
    f_r     = r6[pos];
    f_rtmp  = h3*(vs1+vs2);
    f_yz    = yz[pos]  + xmu3*(vs1+vs2) + vx1*f_r;
    r6[pos] = f_vx2*f_r + f_wwo*f_rtmp;
    f_rtmp  = f_rtmp*(f_wwo-1.0f) + f_vx2*f_r*(1.0f-f_vx1); 
    yz[pos] = (f_yz + d_DT*f_rtmp)*f_dcrj; 
    #endif

    // also moved to fstr (Daniel)
    /*if(k == d_nzt[d_i]+align-2)
      {
      zz[pos+3] = -zz[pos];
      xz[pos+2] = -xz[pos];
      yz[pos+2] = -yz[pos];                                               
      }
      else if(k == d_nzt[d_i]+align-3)
      {
      xz[pos+4] = -xz[pos];
      yz[pos+4] = -yz[pos];
      }*/
    /*}*/
    pos     = pos_im1;
  }
  return;
}



extern "C"
void dstrqc_H_new(float* xx,       float* yy,     float* zz,    float* xy,    float* xz, float* yz,
                  float* r1,       float* r2,     float* r3,    float* r4,    float* r5, float* r6,
                  float* u1,       float* v1,     float* w1,    float* lam,   float* mu, float* qp,float* coeff, 
                  float* qs,       float* dcrjx,  float* dcrjy, float* dcrjz, int nyt,   int nzt, 
                  cudaStream_t St, float* lam_mu, 
                  //_prec *vx1, _prec *vx2, _prec *ww, _prec *wwo, //peng's version
                  _prec *vx1, _prec *vx2, int *ww, _prec *wwo,
                  int NX,          int NPC,       int rankx,    int ranky, int  s_i,  
                  int e_i,         int s_j,       int e_j, int d_i)
{
    //fprintf(stderr, "nzt=%d, e_j=%d, s_j=%d\n", nzt, e_j, s_j);
    /*cudaPrintfInit();*/
    if (0 == (nzt % 64) && 0 == (( e_j-s_j+1) % 8)) {
      const int blockx = 64, blocky = 8;
      dim3 block(blockx, blocky, 1);
      dim3 grid ((nzt+block.x-1)/block.x, (e_j-s_j+1+block.y-1)/block.y,1);
      CUCHK( cudaFuncSetCacheConfig(dstrqc_new<blockx,blocky>, cudaFuncCachePreferShared) );
      dstrqc_new<blockx,blocky><<<grid, block, 0, St>>>(xx, yy, zz, xy,  xz, yz, r1, r2,    r3,    r4,    r5,     r6, 
                                     u1, v1, w1, lam, mu, qp,coeff, qs, dcrjx, dcrjy, dcrjz, lam_mu, 
                                     vx1, vx2, ww, wwo,
                                     NX, NPC, rankx, ranky, nzt, s_i, e_i, s_j, e_j, d_i);
    } else {
      const int blockx = BLOCK_SIZE_Z, blocky = BLOCK_SIZE_Y;
      dim3 block(blockx, blocky, 1);
      dim3 grid ((nzt+block.x-1)/block.x, (e_j-s_j+1+block.y-1)/block.y,1);
      CUCHK( cudaFuncSetCacheConfig(dstrqc_new<blockx,blocky>, cudaFuncCachePreferShared) );
      dstrqc_new<blockx,blocky><<<grid, block, 0, St>>>(xx, yy, zz, xy,  xz, yz, r1, r2,    r3,    r4,    r5,     r6, 
                                     u1, v1, w1, lam, mu, qp,coeff, qs, dcrjx, dcrjy, dcrjz, lam_mu, 
                                     vx1, vx2, ww, wwo,
                                     NX, NPC, rankx, ranky, nzt, s_i, e_i, s_j, e_j, d_i);

    }
    cudaError_t cerr;
    CUCHK(cerr=cudaGetLastError());
    if(cerr!=cudaSuccess) printf("CUDA ERROR: dstrqc_H_new after kernel: %s\n",cudaGetErrorString(cerr));
    /*cudaPrintfDisplay(stdout, 1);
    cudaPrintfEnd();*/
    return;
}


/* kernel function to apply free-surface B.C. to stresses - (Daniel) */
extern "C"
void fstr_H(float* zz, float* xz, float* yz, cudaStream_t St, int s_i, int e_i, int s_j, int e_j)
{
    dim3 block (2, BLOCK_SIZE_Y, 1);
    dim3 grid (1,(e_j-s_j+1+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,1);
    cudaFuncSetCacheConfig(fstr, cudaFuncCachePreferL1);
    fstr<<<grid, block, 0, St>>>(zz, xz, yz, s_i, e_i, s_j);
    return;
}


__global__ void 
__launch_bounds__(512,2)
drprecpc_calc_opt(_prec *xx, _prec *yy, _prec *zz, 
                  const float* __restrict__ xy, 
                  const float* __restrict__ xz, 
                  const float* __restrict__ yz, 
                  _prec *mu, _prec *d1, 
                  _prec *sigma2, 
                  _prec *yldfac,_prec *cohes, _prec *phi,
                  _prec *neta,
                  int nzt, int s_i, int e_i, int s_j, int e_j,  int d_i) { 
  register int i,j,k,pos;
  register int pos_im1,pos_ip1,pos_jm1,pos_km1;
  register int pos_ip1jm1;
  register int pos_ip1km1,pos_jm1km1;
  register _prec Sxx, Syy, Szz, Sxy, Sxz, Syz;
  register _prec Sxxp, Syyp, Szzp, Sxyp, Sxzp, Syzp;
  register _prec depxx, depyy, depzz, depxy, depxz, depyz;
  register _prec SDxx, SDyy, SDzz;
  register _prec iyldfac, Tv, sigma_m, taulim, taulim2, rphi;
  register _prec xm, iixx, iiyy, iizz;
  register _prec mu_, secinv, sqrtSecinv;
  register int   jj,kk;

  // Compute initial stress on GPU (Daniel)
  register _prec ini[9], ini_ip1[9];
  register _prec depth, pfluid;
  register int srfpos;

  k    = blockIdx.x*blockDim.x+threadIdx.x+align;
  j    = blockIdx.y*blockDim.y+threadIdx.y+s_j;
  
  //if (k >= nzt+align || j > e_j) return;
  if (k > nzt+align+1 || j > e_j) return;

  i    = e_i;
  pos  = i*d_slice_1[d_i]+j*d_yline_1[d_i]+k;

  kk   = k - align;
  jj   = j - (2+ngsl);

  srfpos = d_nzt[d_i] + align - 1;
  depth = (float) (srfpos - k) * d_DH[d_i];

  if (depth > 0) pfluid = (depth + d_DH[d_i]*0.5) * 9.81e3;
  else pfluid = d_DH[d_i] / 2. * 9.81e3;
 
  //cuPrintf("k=%d, depth=%f, pfluid=%e\n", k, depth, pfluid);

  _prec sigma2_ip1, sigma2_i;
  _prec xy_ip1, xy_i, xz_ip1, xz_i, yz_ip1, yz_i;
  _prec mu_ip1, mu_i;
  _prec xz_km1, xz_ip1km1, xy_jm1, xy_ip1jm1;
  sigma2_i = sigma2[pos + d_slice_1[d_i]];
  xy_i    = xy   [pos + d_slice_1[d_i]];
  xz_i    = xz   [pos + d_slice_1[d_i]];
  mu_i    = mu   [pos + d_slice_1[d_i]];
  xz_km1  = xz   [pos + d_slice_1[d_i] - 1];
  xy_jm1  = xy   [pos + d_slice_1[d_i] - d_yline_1[d_i]];
  for(i=e_i;i>=s_i;--i){
    sigma2_ip1 = sigma2_i;
    xy_ip1    = xy_i;
    xz_ip1    = xz_i;
    mu_ip1    = mu_i;
    xz_ip1km1 = xz_km1;
    xy_ip1jm1 = xy_jm1;

    pos_im1 = pos - d_slice_1[d_i];
    pos_ip1 = pos + d_slice_1[d_i];
    pos_jm1 = pos - d_yline_1[d_i];
    pos_km1 = pos - 1;
    pos_ip1jm1 = pos_ip1 - d_yline_1[d_i];
    pos_ip1km1 = pos_ip1 - 1;
    pos_jm1km1 = pos_jm1 - 1;

    sigma2_i = sigma2[pos];
    xy_i    = xy   [pos];
    xy_jm1  = xy   [pos_jm1];
    xz_i    = xz   [pos];
    xz_km1  = xz   [pos_km1];
    mu_i    = mu   [pos];

    // mu_ = mu[pos];

// start drprnn
    rotate_principal(sigma2_i, pfluid, ini);
    rotate_principal(sigma2_ip1, pfluid, ini_ip1);
    /*cuPrintf("ini[8] = %5.2e, ini[4]=%5.2e sigma2=%5.2e pfluid=%5.2e\n", 
         ini[8], ini[4], sigma2[pos], pfluid);*/
    /*iixx  = 0.5f*(inixx_i + inixx_ip1);
    iiyy  = 0.5f*(iniyy_i + iniyy_ip1);
    iizz  = 0.5f*(inizz_i + inizz_ip1);*/
    iixx  = 0.5f*(ini[0] + ini_ip1[0]);
    iiyy  = 0.5f*(ini[4] + ini_ip1[4]);
    iizz  = 0.5f*(ini[8] + ini_ip1[8]);

    Sxx   = xx[pos] + iixx;
    Syy   = yy[pos] + iiyy;
    Szz   = zz[pos] + iizz;
    Sxz   = 0.25f*(xz_i + xz_ip1 + xz[pos_km1] + xz[pos_ip1km1])
      //+ 0.5f*(inixz_i + inixz_ip1);
              + 0.5f*(ini[2] + ini_ip1[2]);
    Syz   = 0.25f*(yz[pos] + yz[pos_jm1] + yz[pos_km1] + yz[pos_jm1km1])
      //+ 0.5f*(iniyz_i + iniyz_ip1);
              + 0.5f*(ini[5] + ini_ip1[5]);
    Sxy   = 0.25f*(xy_i + xy_ip1 + xy[pos_jm1] + xy[pos_ip1jm1])
      //+ 0.5f*(inixy_i + inixy_ip1);
              + 0.5f*(ini[1] + ini_ip1[1]);

    Tv = d_DH[d_i]/sqrt(1.0f/(mu_i*d1[pos]));

    Sxxp = Sxx;
    Syyp = Syy;
    Szzp = Szz;
    Sxyp = Sxy;
    Sxzp = Sxz;
    Syzp = Syz;

// drucker_prager function:
    rphi = phi[pos] * 0.017453292519943295f;
    sigma_m = (Sxx + Syy + Szz)/3.0f;
    SDxx = Sxx - sigma_m;
    SDyy = Syy - sigma_m;
    SDzz = Szz - sigma_m;
    secinv  = 0.5f*(SDxx*SDxx + SDyy*SDyy + SDzz*SDzz)
      + Sxz*Sxz + Sxy*Sxy + Syz*Syz;
    sqrtSecinv = sqrt(secinv);
    taulim2 = cohes[pos]*cos(rphi) - (sigma_m + pfluid)*sin(rphi);

    if(taulim2 > 0.0f)  taulim = taulim2;
    else              taulim = 0.0f;
    if(sqrtSecinv > taulim){
      iyldfac = taulim/sqrtSecinv
        + (1.0f-taulim/sqrtSecinv)*exp(-d_DT/Tv);
      Sxx = SDxx*iyldfac + sigma_m;
      Syy = SDyy*iyldfac + sigma_m;
      Szz = SDzz*iyldfac + sigma_m;
      Sxz = Sxz*iyldfac;
      Sxy = Sxy*iyldfac;
      Syz = Syz*iyldfac;
    }
    else  iyldfac = 1.0f;
    yldfac[pos] = iyldfac;
// end drucker_prager function

    if(yldfac[pos]<1.0f){
      xm = 2.0f/(mu_i + mu_ip1);
      depxx = (Sxx - Sxxp) / xm;
      depyy = (Syy - Syyp) / xm;
      depzz = (Szz - Szzp) / xm;
      depxy = (Sxy - Sxyp) / xm;
      depxz = (Sxz - Sxzp) / xm;
      depyz = (Syz - Syzp) / xm;
        
      neta[pos] = neta[pos] + sqrt(0.5f*(depxx*depxx + depyy*depyy + depzz*depzz)
                                   + 2.0f*(depxy*depxy + depxz*depxz + depyz*depyz));
    }
    else  yldfac[pos] = 1.0f;
    // end drprnn 

// DEBUG if neta/EPxx etc are set
    //neta[pos] = 1.E+2;
    //EPxx[pos] = 1.E+2;
// DEBUG - end

    pos = pos_im1;
  }

  return;
}

// drprecpc is for plasticity computation for cerjan and wave propagation
extern "C"
void drprecpc_calc_H_opt(_prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
        _prec *mu, _prec *d1, _prec *sigma2,
        _prec *yldfac,_prec *cohes, _prec *phi,
        _prec *neta,
        int nzt,
        int xls, int xre, int yls, int yre, cudaStream_t St, int d_i){

    dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1);
    dim3 grid ((nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, ((yre-yls+1)+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,1);
    cudaFuncSetCacheConfig(drprecpc_calc_opt, cudaFuncCachePreferL1);

    //split into tho routines, one for the normal, one for shear stress components (Daniel)
    drprecpc_calc_opt<<<grid, block, 0, St>>>(xx,yy,zz,xy,xz,yz,mu,d1,
        sigma2,yldfac,cohes,phi,neta, 
        nzt, xls,xre,yls, yre, d_i);
    CUCHK(cudaGetLastError());

return;
}

extern "C"
void drprecpc_app_H(_prec *xx, _prec *yy, _prec *zz, 
        _prec *xy, _prec *xz, _prec *yz,
        _prec *mu, _prec *sigma2, _prec *yldfac, 
        int nzt, int xls, int xre, int yls, int yre, cudaStream_t St, int d_i){

    dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1);
    dim3 grid ((nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, ((yre-yls+1)+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,1);
    cudaError_t cerr;

    cerr=cudaGetLastError();
    if(cerr!=cudaSuccess) printf("CUDA ERROR: drprecpc_app before kernel: %s\n",cudaGetErrorString(cerr));
    cudaFuncSetCacheConfig(drprecpc_app, cudaFuncCachePreferL1);
    drprecpc_app<<<grid, block, 0, St>>>(xx,yy,zz,xy,xz,yz,mu,
        sigma2,yldfac,xls,xre,yls,d_i);
    cerr=cudaGetLastError();
    if(cerr!=cudaSuccess) printf("CUDA ERROR: drprecpc_app after kernel: %s\n",cudaGetErrorString(cerr));

return;
}

extern "C"
void addsrc_H(int i,      int READ_STEP, int dim,    int* psrc,  int npsrc,  cudaStream_t St,
              float* axx, float* ayy,    float* azz, float* axz, float* ayz, float* axy,
              float* xx,  float* yy,     float* zz,  float* xy,  float* yz,  float* xz, int d_i)
{
    dim3 grid, block;
    if(npsrc < 256)
    {
       block.x = npsrc;
       grid.x = 1;
    }
    else
    {
       block.x = 256;
       grid.x  = int((npsrc+255)/256);
    }
    cudaError_t cerr;
    cerr=cudaGetLastError();
    if(cerr!=cudaSuccess) printf("CUDA ERROR: addsrc before kernel: %s\n",cudaGetErrorString(cerr));
    /*cudaPrintfInit();*/
    addsrc_cu<<<grid, block, 0, St>>>(i,  READ_STEP, dim, psrc, npsrc, axx, ayy, azz, axz, ayz, axy,
                                      xx, yy,        zz,  xy,   yz,  xz, d_i);
    cerr=cudaGetLastError();
    /*cudaPrintfDisplay(stdout, 1);
    cudaPrintfEnd();*/
    if(cerr!=cudaSuccess) printf("CUDA ERROR: addsrc after kernel: %s\n",cudaGetErrorString(cerr));
    return;
}

__global__ void dvelcy(float* u1,    float* v1,    float* w1,    float* xx,  float* yy,   float* zz,   float* xy, float* xz, float* yz,
                       float* dcrjx, float* dcrjy, float* dcrjz, float* d_1, float* s_u1, float* s_v1, float* s_w1, int s_j, int e_j,
                       int d_i)
{
    register int   i, j, k, pos,     j2,      pos2, pos_jm1, pos_jm2;
    register int   pos_km2, pos_km1, pos_kp1, pos_kp2;
    register int   pos_im2, pos_im1, pos_ip1, pos_ip2;
    register int   pos_jk1, pos_ik1, pos_ijk;
    register _prec f_xy,    xy_jp1,  xy_jm1,  xy_jm2;
    register _prec f_yy,    yy_jp2,  yy_jp1,  yy_jm1;
    register _prec f_yz,    yz_jp1,  yz_jm1,  yz_jm2;
    register _prec f_d1,    f_d2,    f_d3,    f_dcrj, f_dcrjx, f_dcrjz, f_xz;

    if (k > d_nzt[d_i]+align-3 && d_i > 0) return;

    k     = blockIdx.x*BLOCK_SIZE_Z+threadIdx.x+align;
    i     = blockIdx.y*BLOCK_SIZE_Y+threadIdx.y+2+ngsl;
    j     = e_j;
    j2    = ngsl-1;
    pos   = i*d_slice_1[d_i]+j*d_yline_1[d_i]+k;
    pos2  = i*ngsl*d_yline_1[d_i]+j2*d_yline_1[d_i]+k; 

    f_xy    = xy[pos+d_yline_1[d_i]];
    xy_jm1  = xy[pos];
    xy_jm2  = xy[pos-d_yline_1[d_i]];
    yy_jp1  = yy[pos+d_yline_2[d_i]];
    f_yy    = yy[pos+d_yline_1[d_i]];
    yy_jm1  = yy[pos];
    f_yz    = yz[pos+d_yline_1[d_i]];
    yz_jm1  = yz[pos];
    yz_jm2  = yz[pos-d_yline_1[d_i]];
    f_dcrjz = dcrjz[k];
    f_dcrjx = dcrjx[i];
    for(j=e_j; j>=s_j; j--)
    {
        pos_km2  = pos-2;
        pos_km1  = pos-1;
        pos_kp1  = pos+1;
        pos_kp2  = pos+2;
        pos_jm2  = pos-d_yline_2[d_i];
        pos_jm1  = pos-d_yline_1[d_i];
        pos_im1  = pos-d_slice_1[d_i];
        pos_im2  = pos-d_slice_2[d_i];
        pos_ip1  = pos+d_slice_1[d_i];
        pos_ip2  = pos+d_slice_2[d_i];
        pos_jk1  = pos-d_yline_1[d_i]-1;
        pos_ik1  = pos+d_slice_1[d_i]-1;
        pos_ijk  = pos+d_slice_1[d_i]-d_yline_1[d_i];

        xy_jp1   = f_xy;
        f_xy     = xy_jm1;
        xy_jm1   = xy_jm2;
        xy_jm2   = xy[pos_jm2];
        yy_jp2   = yy_jp1;
        yy_jp1   = f_yy;
        f_yy     = yy_jm1;
        yy_jm1   = yy[pos_jm1];
        yz_jp1   = f_yz;
        f_yz     = yz_jm1;
        yz_jm1   = yz_jm2;
        yz_jm2   = yz[pos_jm2];
        f_xz     = xz[pos];

        f_dcrj   = f_dcrjx*dcrjy[j]*f_dcrjz;
        f_d1     = 0.25*(d_1[pos] + d_1[pos_jm1] + d_1[pos_km1] + d_1[pos_jk1]);
        f_d2     = 0.25*(d_1[pos] + d_1[pos_ip1] + d_1[pos_km1] + d_1[pos_ik1]);
        f_d3     = 0.25*(d_1[pos] + d_1[pos_ip1] + d_1[pos_jm1] + d_1[pos_ijk]);

        f_d1     = d_dth[d_i]/f_d1;
        f_d2     = d_dth[d_i]/f_d2;
        f_d3     = d_dth[d_i]/f_d3;

        s_u1[pos2] = (u1[pos] + f_d1*( d_c1*(xx[pos]     - xx[pos_im1]) + d_c2*(xx[pos_ip1] - xx[pos_im2])
                                     + d_c1*(f_xy        - xy_jm1)      + d_c2*(xy_jp1      - xy_jm2)
                                     + d_c1*(f_xz        - xz[pos_km1]) + d_c2*(xz[pos_kp1] - xz[pos_km2]) ))*f_dcrj;
        s_v1[pos2] = (v1[pos] + f_d2*( d_c1*(xy[pos_ip1] - f_xy)        + d_c2*(xy[pos_ip2] - xy[pos_im1])
                                     + d_c1*(yy_jp1      - f_yy)        + d_c2*(yy_jp2      - yy_jm1)
                                     + d_c1*(f_yz        - yz[pos_km1]) + d_c2*(yz[pos_kp1] - yz[pos_km2]) ))*f_dcrj;
        s_w1[pos2] = (w1[pos] + f_d3*( d_c1*(xz[pos_ip1] - f_xz)        + d_c2*(xz[pos_ip2] - xz[pos_im1])
                                     + d_c1*(f_yz        - yz_jm1)      + d_c2*(yz_jp1      - yz_jm2)
                                     + d_c1*(zz[pos_kp1] - zz[pos])     + d_c2*(zz[pos_kp2] - zz[pos_km1]) ))*f_dcrj;

        pos        = pos_jm1;
        pos2       = pos2 - d_yline_1[d_i];
    }
    return;
}

__global__ void update_boundary_y(float* u1, float* v1, float* w1, float* s_u1, float* s_v1, float* s_w1, int rank, int flag, int d_i)
{
    register int i, j, k, pos, posj;
    k     = blockIdx.x*BLOCK_SIZE_Z+threadIdx.x+align;
    i     = blockIdx.y*BLOCK_SIZE_Y+threadIdx.y+2+ngsl;

    if(flag==Front && rank!=-1){
	j     = 2;
    	pos   = i*d_slice_1[d_i]+j*d_yline_1[d_i]+k;
        posj  = i*ngsl*d_yline_1[d_i]+k;
	for(j=2;j<2+ngsl;j++){
		u1[pos] = s_u1[posj];
		v1[pos] = s_v1[posj];
		w1[pos] = s_w1[posj];
		pos	= pos  + d_yline_1[d_i];
  		posj	= posj + d_yline_1[d_i];	
	}
    }

    if(flag==Back && rank!=-1){
    	j     = d_nyt[d_i]+ngsl+2;
    	pos   = i*d_slice_1[d_i]+j*d_yline_1[d_i]+k;
        posj  = i*ngsl*d_yline_1[d_i]+k;
	for(j=d_nyt[d_i]+ngsl+2;j<d_nyt[d_i]+ngsl2+2;j++){
	        u1[pos] = s_u1[posj];
                v1[pos] = s_v1[posj];
                w1[pos] = s_w1[posj];
                pos     = pos  + d_yline_1[d_i];
                posj    = posj + d_yline_1[d_i];
	}
    }
    return;
}

/* kernel functions to apply free-surface B.C.s to stress */
__global__ void fstr (float* zz, float* xz, float* yz, int s_i, int e_i, int s_j)
{
    register int i, j, k;
    register int pos, pos_im1; 

    k    = d_nzt[0]+align-1;
    j    = blockIdx.y*BLOCK_SIZE_Y+threadIdx.y+s_j;
    i    = e_i;
    pos  = i*d_slice_1[0]+j*d_yline_1[0]+k;

    for(i=e_i;i>=s_i;i--)
    {
        pos_im1  = pos-d_slice_1[0];

        // asymmetry reflection above free surface
        zz[pos+1] = -zz[pos];
        zz[pos+2] = -zz[pos-1];

        xz[pos+1] = -xz[pos-1];
        xz[pos+2] = -xz[pos-2];

        yz[pos+1] = -yz[pos-1];                                               
        yz[pos+2] = -yz[pos-2];

        // horizontal shear stresses on free surface
        xz[pos]   = 0.0;
        yz[pos]   = 0.0;

        pos     = pos_im1;
    }

}

/* Old dstrqc routine */
__global__ void dstrqc(float* xx, float* yy,    float* zz,    float* xy,    float* xz,     float* yz,
                       float* r1, float* r2,    float* r3,    float* r4,    float* r5,     float* r6,
                       float* u1, float* v1,    float* w1,    float* lam,   float* mu,     float* qp,float* coeff, 
                       float* qs, float* dcrjx, float* dcrjy, float* dcrjz, float* lam_mu, 
                       _prec *d_vx1, _prec *d_vx2, int *d_ww, _prec *d_wwo,
                       int NX, int NPC, int rankx, int ranky, int nzt, int s_i, int e_i, int s_j, int e_j, int d_i)
{
    register int   i,  j,  k,  g_i;
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
    int maxk, mink = align+3;
    
    k    = blockIdx.x*BLOCK_SIZE_Z+threadIdx.x+align;
    j    = blockIdx.y*BLOCK_SIZE_Y+threadIdx.y+s_j;

    if (d_i == 0) {
       maxk = nzt + align -1;
    }
    else maxk = nzt + align -3;

    if (k < mink || k > maxk || j > e_j) return;
 
    i    = e_i;
    pos  = i*d_slice_1[d_i]+j*d_yline_1[d_i]+k;

    u1_ip1 = u1[pos+d_slice_2[d_i]];
    f_u1   = u1[pos+d_slice_1[d_i]];
    u1_im1 = u1[pos];    
    f_v1   = v1[pos+d_slice_1[d_i]];
    v1_im1 = v1[pos];
    v1_im2 = v1[pos-d_slice_1[d_i]];
    f_w1   = w1[pos+d_slice_1[d_i]];
    w1_im1 = w1[pos];
    w1_im2 = w1[pos-d_slice_1[d_i]];
    f_dcrjz = dcrjz[k];
    f_dcrjy = dcrjy[j];
    for(i=e_i;i>=s_i;i--)
    {
        /*f_vx1    = tex1Dfetch(p_vx1, pos);
        f_vx2    = tex1Dfetch(p_vx2, pos);
        f_ww     = tex1Dfetch(p_ww, pos);
        f_wwo     = tex1Dfetch(p_wwo, pos);*/
        f_vx1 = d_vx1[pos];
        f_vx2 = d_vx2[pos];
        f_ww  = d_ww[pos];
        f_wwo = d_wwo[pos];
        /*
        if(f_wwo!=f_wwo){
          xx[pos] = yy[pos] = zz[pos] = xy[pos] = xz[pos] = yz[pos] = 1.0;
          r1[pos] = r2[pos] = r3[pos] = r4[pos] = r5[pos] = r6[pos] = 1.0;
          return;
        }
*/
        f_dcrj   = dcrjx[i]*f_dcrjy*f_dcrjz;

        pos_km2  = pos-2;
        pos_km1  = pos-1;
        pos_kp1  = pos+1;
        pos_kp2  = pos+2;
        pos_jm2  = pos-d_yline_2[d_i];
        pos_jm1  = pos-d_yline_1[d_i];
        pos_jp1  = pos+d_yline_1[d_i];
        pos_jp2  = pos+d_yline_2[d_i];
        pos_im2  = pos-d_slice_2[d_i];
        pos_im1  = pos-d_slice_1[d_i];
        pos_ip1  = pos+d_slice_1[d_i];
        pos_jk1  = pos-d_yline_1[d_i]-1;
        pos_ik1  = pos+d_slice_1[d_i]-1;
        pos_ijk  = pos+d_slice_1[d_i]-d_yline_1[d_i];
        pos_ijk1 = pos+d_slice_1[d_i]-d_yline_1[d_i]-1;

        xl       = 8.0/(  lam[pos]      + lam[pos_ip1] + lam[pos_jm1] + lam[pos_ijk]
                        + lam[pos_km1]  + lam[pos_ik1] + lam[pos_jk1] + lam[pos_ijk1] );
        xm       = 16.0/( mu[pos]       + mu[pos_ip1]  + mu[pos_jm1]  + mu[pos_ijk]
                        + mu[pos_km1]   + mu[pos_ik1]  + mu[pos_jk1]  + mu[pos_ijk1] );
        xmu1     = 2.0/(  mu[pos]       + mu[pos_km1] );
        xmu2     = 2.0/(  mu[pos]       + mu[pos_jm1] );
        xmu3     = 2.0/(  mu[pos]       + mu[pos_ip1] );
        xl       = xl  +  xm;
        qpa      = 0.0625*( qp[pos]     + qp[pos_ip1] + qp[pos_jm1] + qp[pos_ijk]
                          + qp[pos_km1] + qp[pos_ik1] + qp[pos_jk1] + qp[pos_ijk1] );

//                        www=f_ww;
        if(1./(qpa*2.0)<=200.0)
        {
//      printf("coeff[f_ww*2-2] %g\n",coeff[f_ww*2-2]);
                  qpaw=coeff[f_ww*2-2]*(2.*qpa)*(2.*qpa)+coeff[f_ww*2-1]*(2.*qpa);
//              qpaw=coeff[www*2-2]*(2.*qpa)*(2.*qpa)+coeff[www*2-1]*(2.*qpa);
//                qpaw=qpaw/2.;
                  }
               else {
                  qpaw  = 2.0f*f_wwo*qpa;  //Fix for Q(f) suggested by Kyle
		  	}
//                 printf("qpaw %f\n",qpaw);
//              printf("qpaw1 %g\n",qpaw);
        qpaw=qpaw/f_wwo;
//      printf("qpaw2 %g\n",qpaw);



        h        = 0.0625*( qs[pos]     + qs[pos_ip1] + qs[pos_jm1] + qs[pos_ijk]
                          + qs[pos_km1] + qs[pos_ik1] + qs[pos_jk1] + qs[pos_ijk1] );

       if(1./(h*2.0)<=200.0)
        {
                  hw=coeff[f_ww*2-2]*(2.*h)*(2.*h)+coeff[f_ww*2-1]*(2.*h);
                  //                  hw=hw/2.;
                  }
               else {
                  hw  = 2.0f*f_wwo*h;  //Fix for Q(f) suggested by Kyle
                }
        hw=hw/f_wwo;


        h1       = 0.250*(  qs[pos]     + qs[pos_km1] );

        if(1./(h1*2.0)<=200.0)
        {
                  h1w=coeff[f_ww*2-2]*(2.*h1)*(2.*h1)+coeff[f_ww*2-1]*(2.*h1);
                  //                  h1w=h1w/2.;
                  }
                         else {
                  h1w  = 2.0f*f_wwo*h1; //Fix for Q(f) suggested by Kyle
                }
        h1w=h1w/f_wwo;



        h2       = 0.250*(  qs[pos]     + qs[pos_jm1] );
        if(1./(h2*2.0)<=200.0)
        {
                  h2w=coeff[f_ww*2-2]*(2.*h2)*(2.*h2)+coeff[f_ww*2-1]*(2.*h2);
                  //                  h2w=h2w/2.;
                  }
                         else {
                  h2w  = 2.0f*f_wwo*h2; //Fix for Q(f) suggested by Kyle
                }
        h2w=h2w/f_wwo;


        h3       = 0.250*(  qs[pos]     + qs[pos_ip1] );
        if(1./(h3*2.0)<=200.0)
        {
                  h3w=coeff[f_ww*2-2]*(2.*h3)*(2.*h3)+coeff[f_ww*2-1]*(2.*h3);
                  //                  h3w=h3w/2.;
                  }
                         else {
                  h3w  = 2.0f*f_wwo*h3; //Fix for Q(f) suggested by Kyle
                }
        h3w=h3w/f_wwo;

	h        = -xm*hw*d_dh1[d_i];
        h1       = -xmu1*h1w*d_dh1[d_i];
        h2       = -xmu2*h2w*d_dh1[d_i];
        h3       = -xmu3*h3w*d_dh1[d_i];


        //        h1       = -xmu1*hw1*d_dh1[d_i];
        //h2       = -xmu2*hw2*d_dh1[d_i];
        //h3       = -xmu3*hw3*d_dh1[d_i];


        qpa      = -qpaw*xl*d_dh1[d_i];
        //        qpa      = -qpaw*xl*d_dh1[d_i];

        xm       = xm*d_dth[d_i];
        xmu1     = xmu1*d_dth[d_i];
        xmu2     = xmu2*d_dth[d_i];
        xmu3     = xmu3*d_dth[d_i];
        xl       = xl*d_dth[d_i];
      //  f_vx2    = f_vx2*f_vx1;
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

        if (d_i == 0){ /*Apply FS condition on uppermost grid only*/
	  if(k == d_nzt[d_i]+align-1) {
	      u1[pos_kp1] = f_u1 - (f_w1 - w1_im1);
	      v1[pos_kp1] = f_v1 - (w1[pos_jp1] - f_w1);

	      g_i  = d_nxt[d_i]*rankx + i - ngsl - 1;

	      if(g_i<NX || NPC == 2)
		      vs1 = u1_ip1 - (w1_ip1 - f_w1);
	      else
		      vs1 = 0.0;

	      g_i  = d_nyt[d_i]*ranky + j - ngsl - 1;
	      if(g_i>1 || NPC == 2) //periodic BCs
		      vs2 = v1[pos_jm1] - (f_w1 - w1[pos_jm1]);
	      else
		      vs2 = 0.0;

	      w1[pos_kp1] = w1[pos_km1] - lam_mu[i*(d_nyt[d_i]+4+ngsl2) + j]*((vs1 - u1[pos_kp1]) + (u1_ip1 - f_u1)
                           + (v1[pos_kp1] - vs2) + (f_v1   - v1[pos_jm1]) );
	  }
	  else if(k == d_nzt[d_i]+align-2) {
		  u1[pos_kp2] = u1[pos_kp1] - (w1[pos_kp1]   - w1[pos_im1+1]);
		  v1[pos_kp2] = v1[pos_kp1] - (w1[pos_jp1+1] - w1[pos_kp1]);
	  }
        }
 
    	vs1      = d_c1*(u1_ip1 - f_u1)        + d_c2*(u1_ip2      - u1_im1);
        vs2      = d_c1*(f_v1   - v1[pos_jm1]) + d_c2*(v1[pos_jp1] - v1[pos_jm2]);
        vs3      = d_c1*(f_w1   - w1[pos_km1]) + d_c2*(w1[pos_kp1] - w1[pos_km2]);
 
        tmp      = xl*(vs1+vs2+vs3);
        a1       = qpa*(vs1+vs2+vs3);
        tmp      = tmp+d_DT*a1;

        f_r      = r1[pos];
	 f_rtmp   = -h*(vs2+vs3) + a1; 
	 xx[pos]  = xx[pos]  + tmp - xm*(vs2+vs3) + vx1*f_r;  
	 r1[pos]  = f_vx2*f_r + f_wwo*f_rtmp;
	 f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1); 
	  xx[pos]  = (xx[pos] + d_DT*f_rtmp)*f_dcrj;

        f_r      = r2[pos];
	 f_rtmp   = -h*(vs1+vs3) + a1;  
        yy[pos]  = (yy[pos]  + tmp - xm*(vs1+vs3) + vx1*f_r)*f_dcrj;

	 r2[pos]  = f_vx2*f_r + f_wwo*f_rtmp; 
	 f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1); 
	  yy[pos]  = (yy[pos] + d_DT*f_rtmp)*f_dcrj;
	
        f_r      = r3[pos];
	f_rtmp   = -h*(vs1+vs2) + a1;
        zz[pos]  = (zz[pos]  + tmp - xm*(vs1+vs2) + vx1*f_r)*f_dcrj;
	 r3[pos]  = f_vx2*f_r + f_wwo*f_rtmp;
	 f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);  
	 zz[pos]  = (zz[pos] + d_DT*f_rtmp)*f_dcrj;

        vs1      = d_c1*(u1[pos_jp1] - f_u1)   + d_c2*(u1[pos_jp2] - u1[pos_jm1]);
        vs2      = d_c1*(f_v1        - v1_im1) + d_c2*(v1_ip1      - v1_im2);
        f_r      = r4[pos];
 	f_rtmp   = h1*(vs1+vs2); 
	 xy[pos]  = xy[pos]  + xmu1*(vs1+vs2) + vx1*f_r;
	 r4[pos]  = f_vx2*f_r + f_wwo*f_rtmp; 
	 f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);
	 xy[pos]  = (xy[pos] + d_DT*f_rtmp)*f_dcrj;
 
        //moved to separate subroutine fstr, to be executed after plasticity (Daniel)
        /*if(k == d_nzt+align-1)
        {
                zz[pos+1] = -zz[pos];
        	xz[pos]   = 0.0;
                yz[pos]   = 0.0;
        }
        else
        {*/
        	vs1     = d_c1*(u1[pos_kp1] - f_u1)   + d_c2*(u1[pos_kp2] - u1[pos_km1]);
        	vs2     = d_c1*(f_w1        - w1_im1) + d_c2*(w1_ip1      - w1_im2);
        	f_r     = r5[pos];
		 f_rtmp  = h2*(vs1+vs2);
		  xz[pos] = xz[pos]  + xmu2*(vs1+vs2) + vx1*f_r; 
		   r5[pos] = f_vx2*f_r + f_wwo*f_rtmp; 
		   f_rtmp  = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1); 
		   xz[pos] = (xz[pos] + d_DT*f_rtmp)*f_dcrj;
	 

        	vs1     = d_c1*(v1[pos_kp1] - f_v1) + d_c2*(v1[pos_kp2] - v1[pos_km1]);
        	vs2     = d_c1*(w1[pos_jp1] - f_w1) + d_c2*(w1[pos_jp2] - w1[pos_jm1]);
        	f_r     = r6[pos];
		f_rtmp  = h3*(vs1+vs2);
		yz[pos] = yz[pos]  + xmu3*(vs1+vs2) + vx1*f_r;
		 r6[pos] = f_vx2*f_r + f_wwo*f_rtmp;
		  f_rtmp  = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1); 
		  yz[pos] = (yz[pos] + d_DT*f_rtmp)*f_dcrj; 

                // also moved to fstr (Daniel)
                /*if(k == d_nzt+align-2)
                {
                    zz[pos+3] = -zz[pos];
                    xz[pos+2] = -xz[pos];
                    yz[pos+2] = -yz[pos];                                               
		}
		else if(k == d_nzt+align-3)
		{
                    xz[pos+4] = -xz[pos];
                    yz[pos+4] = -yz[pos];
		}*/
 	/*}*/
        pos     = pos_im1;
    }
    return;
}

// treatment of shear stress components moved to separate kernel code (Daniel)
__global__ void drprecpc_app(_prec *xx, _prec *yy, _prec *zz, 
      _prec *xy, _prec *xz, _prec *yz, 
      _prec *mu, _prec *sigma2, 
      _prec *yldfac, int s_i, int e_i, int s_j, int d_i){

    register int i,j,k,pos;
    register int pos_im1,pos_ip1,pos_jp1,pos_kp1;
    register int pos_im1jp1,pos_im1kp1,pos_ip1jp1;
    register int pos_ip1kp1,pos_jp1kp1,pos_ip1jp1kp1;
    register _prec iyldfac; 
    register _prec xm, tst, iist;
    register _prec mu_;
    register _prec Sxx, Syy, Szz;
    register _prec iixx, iiyy, iizz, SDxx, SDyy, SDzz, sigma_m;
    register _prec ini[9], ini_ip1[9], ini_kp1[9], ini_ip1kp1[9];
    register _prec ini_jp1[9], ini_ip1jp1[9], ini_jp1kp1[9], ini_ip1jp1kp1[9];
    register int srfpos;
    register _prec depth, pfluid, depth_kp1, pfluid_kp1;

    k    = blockIdx.x*BLOCK_SIZE_Z+threadIdx.x+align;
    j    = blockIdx.y*BLOCK_SIZE_Y+threadIdx.y+s_j;
    i    = e_i;
    pos  = i*d_slice_1[d_i]+j*d_yline_1[d_i]+k;

    srfpos = d_nzt[d_i] + align - 1;
    depth = (float) (srfpos - k) * d_DH[d_i];
    depth_kp1 = (float) (srfpos - k + 1) * d_DH[d_i];

    if (depth > 0) pfluid = (depth + d_DH[d_i]/2.) * 9.81e3;
    else pfluid = d_DH[d_i] / 2. * 9.81e3;

    if (depth_kp1 > 0) pfluid_kp1 = (depth_kp1 + d_DH[d_i]/2.) * 9.81e3;
    else pfluid_kp1 = d_DH[d_i] / 2. * 9.81e3;

    //cuPrintf("k=%d, depth=%f, pfluid=%e\n", k, depth, pfluid);
    //cuPrintf("k=%d, depth=%f, pfluid=%e\n", k+1, depth_kp1, pfluid_kp1);

    for(i=e_i;i>=s_i;--i){

      pos_im1 = pos - d_slice_1[d_i];
      pos_ip1 = pos + d_slice_1[d_i];
      pos_jp1 = pos + d_yline_1[d_i];
      pos_kp1 = pos + 1;  //changed from -1 to +1 (Daniel)
      pos_im1jp1 = pos_im1 + d_yline_1[d_i];
      pos_im1kp1 = pos_im1 + 1;
      pos_ip1jp1 = pos_ip1 + d_yline_1[d_i];
      pos_ip1kp1 = pos_ip1 + 1;
      pos_jp1kp1 = pos_jp1 + 1;
      pos_ip1jp1kp1 = pos_ip1 + d_yline_1[d_i] + 1;

      mu_ = mu[pos];

//start drprnn
      if(yldfac[pos] < 1.){
         //compute initial stress at pos and pos_ip1
         rotate_principal(sigma2[pos], pfluid, ini);
         rotate_principal(sigma2[pos_ip1], pfluid_kp1, ini_ip1);
         iixx  = 0.5*(ini[0] + ini_ip1[0]);
         iiyy  = 0.5*(ini[4] + ini_ip1[4]);
         iizz  = 0.5*(ini[8] + ini_ip1[8]);

         Sxx   = xx[pos] + iixx;
         Syy   = yy[pos] + iiyy;
         Szz   = zz[pos] + iizz;

         sigma_m = (Sxx + Syy + Szz)/3.;
         SDxx   = xx[pos] + iixx - sigma_m;
         SDyy   = yy[pos] + iiyy - sigma_m;
         SDzz   = zz[pos] + iizz - sigma_m;

         xx[pos] = SDxx*yldfac[pos] + sigma_m - iixx;
         yy[pos] = SDyy*yldfac[pos] + sigma_m - iiyy;
         zz[pos] = SDzz*yldfac[pos] + sigma_m - iizz;
      }

// start drprxz
      iyldfac = 0.25*(yldfac[pos_im1] + yldfac[pos]
            + yldfac[pos_im1kp1] + yldfac[pos_kp1]);
      if(iyldfac<1.){
        //compute initial stress at pos and pos_kp1
        rotate_principal(sigma2[pos], pfluid, ini);
        rotate_principal(sigma2[pos_kp1], pfluid_kp1, ini_kp1);

        iist = 0.5*(ini[2] + ini_kp1[2]);
        tst = xz[pos];
        xz[pos] = (xz[pos] + iist)*iyldfac - iist;
        xm = 2./(mu_+mu[pos_kp1]);
      }
// end drprxz / start drpryz
      iyldfac = 0.25*(yldfac[pos] + yldfac[pos_jp1]
            + yldfac[pos_jp1kp1] + yldfac[pos_kp1]);
      if(iyldfac<1.){
        //compute initial stress at 8 positions
        rotate_principal(sigma2[pos], pfluid, ini);
        rotate_principal(sigma2[pos_ip1], pfluid, ini_ip1);
        rotate_principal(sigma2[pos_kp1], pfluid_kp1, ini_kp1);
        rotate_principal(sigma2[pos_ip1kp1], pfluid_kp1, ini_ip1kp1);
        rotate_principal(sigma2[pos_jp1], pfluid, ini_jp1);
        rotate_principal(sigma2[pos_ip1jp1], pfluid, ini_ip1jp1);
        rotate_principal(sigma2[pos_jp1kp1], pfluid_kp1, ini_jp1kp1);
        rotate_principal(sigma2[pos_ip1jp1kp1], pfluid_kp1, ini_ip1jp1kp1);

        iist = 0.125*(ini[5] + ini_ip1[5]
            + ini_kp1[5] + ini_ip1kp1[5]
            + ini_jp1[5] + ini_ip1jp1[5]
            + ini_jp1kp1[5] + ini_ip1jp1kp1[5]);

        tst = yz[pos];
        yz[pos] = (yz[pos] + iist)*iyldfac - iist;
        xm = 8./(mu_ + mu[pos_ip1] + mu[pos_kp1]
            + mu[pos_ip1kp1] + mu[pos_jp1] + mu[pos_ip1jp1]
            + mu[pos_jp1kp1] + mu[pos_ip1jp1kp1]);
      }
// end drpryz / start drprxy
      iyldfac = 0.25*(yldfac[pos] + yldfac[pos_jp1]
            + yldfac[pos_im1] + yldfac[pos_im1jp1]);
      if(iyldfac<1.){
        rotate_principal(sigma2[pos], pfluid, ini);
        rotate_principal(sigma2[pos_jp1], pfluid, ini_jp1);
        iist = 0.5*(ini[1] + ini_jp1[1]);
        tst = xy[pos];
        xy[pos] = (xy[pos] + iist)*iyldfac - iist;
        xm = 2./(mu_ + mu[pos_jp1]);
      } 

      pos = pos_im1;
}
   return;
}

__global__ void addsrc_cu(int i,      int READ_STEP, int dim,    int* psrc,  int npsrc,
                          float* axx, float* ayy,    float* azz, float* axz, float* ayz, float* axy,
                          float* xx,  float* yy,     float* zz,  float* xy,  float* yz,  float* xz, int d_i)
{
        register _prec vtst;
        register int idx, idy, idz, j, pos;
        j = blockIdx.x*blockDim.x+threadIdx.x;
        if(j >= npsrc) return;
        vtst = (float)d_DT/(d_DH[d_i]*d_DH[d_i]*d_DH[d_i]);

        i   = i - 1;
        idx = psrc[j*dim]   + 1 + ngsl;
        idy = psrc[j*dim+1] + 1 + ngsl;
        idz = psrc[j*dim+2] + align - 1;
        pos = idx*d_slice_1[d_i] + idy*d_yline_1[d_i] + idz;

        /*cuPrintf("addsrc_cu: (%d,%d,%d) (%e,%e,%e,%e,%e,%e)\n", idx, idy, idz, 
           axx[j*READ_STEP+i], ayy[j*READ_STEP+i], azz[j*READ_STEP+i], axz[j*READ_STEP+i], ayz[j*READ_STEP+i], axy[j*READ_STEP+i]);*/

	xx[pos] = xx[pos] - vtst*axx[j*READ_STEP+i];
	yy[pos] = yy[pos] - vtst*ayy[j*READ_STEP+i];
	zz[pos] = zz[pos] - vtst*azz[j*READ_STEP+i];
	xz[pos] = xz[pos] - vtst*axz[j*READ_STEP+i];
	yz[pos] = yz[pos] - vtst*ayz[j*READ_STEP+i];
	xy[pos] = xy[pos] - vtst*axy[j*READ_STEP+i];

        return;
}

extern "C"
void frcvel_H(int i,      int READ_STEP, int dim,    int* psrc,  int npsrc,  int tskp, cudaStream_t St,
              float* axx, float* ayy,    float* azz, float* axz, float* ayz, float* axy,
              float* u1,  float* v1,     float* w1, int ymin, int ymax, int d_i)
{
    dim3 grid, block;
    if(npsrc < 256)
    {
       block.x = npsrc;
       grid.x = 1;
    }
    else
    {
       block.x = 256;
       grid.x  = int((npsrc+255)/256);
    }
    cudaError_t cerr;
    CUCHK(cudaGetLastError());
    //if(cerr!=cudaSuccess) printf("CUDA ERROR: addsrc before kernel: %s\n",cudaGetErrorString(cerr));
    //cudaPrintfInit();
    frcvel_cu<<<grid, block, 0, St>>>(i,  READ_STEP, dim, psrc, npsrc, tskp, axx, ayy, azz, axz, ayz, axy,
                                      u1, v1, w1, ymin, ymax, d_i);
    CUCHK(cudaGetLastError());
    //cudaPrintfDisplay(stdout, 1);
    //cudaPrintfEnd();
    //if(cerr!=cudaSuccess) printf("CUDA ERROR: addsrc after kernel: %s\n",cudaGetErrorString(cerr));
    return;
}

__global__ void frcvel_cu(int i,      int READ_STEP, int dim,    int* psrc,  int npsrc, int tskp,
                          float* axx, float* ayy,    float* azz, float* axz, float* ayz, float* axy,
                          float* u1,  float* v1,     float* w1, int xmin, int xmax, int d_i)
{
        register int idx, idy, idz, j, pos;
        register int i0, i1;
        register _prec u1_p, u1_n, v1_p, v1_n, w1_p, w1_n;
        register _prec u1_i, v1_i, w1_i, pfact;
        /*register int pos_jm1, pos_jp1, pos_jm2;*/
        bool abvmin, blwmax;
 
        j = blockIdx.x*blockDim.x+threadIdx.x;
        if(j >= npsrc) return;

        i   = i - 1;
        i0=int(tskp * floorf(float(i+1.) / float(tskp)));
        i1=int(tskp * ceilf(float(i+1.) / float(tskp)));

        // Linear interpolation
        pfact = float(i + 1 - i0) / float(tskp);
        // Cosine interpolation
        //pfact = -cosf(float(i + 1 - i0) / float(tskp) * M_PI)/2 + 0.5;
        //if (j==0) cuPrintf("inside frcvel_cu: i=%d, i0=%d i1=%d\n", i, i0, i1);

        i0 /= tskp;
        i1 /= tskp;

        idx = psrc[j*dim]   + 1 + ngsl;
        idy = psrc[j*dim+1] + 1 + ngsl;
        idz = psrc[j*dim+2] + align - 1;
        pos = idx*d_slice_1[d_i] + idy*d_yline_1[d_i] + idz;
        //cuPrintf("%d %d %d\n", psrc[j*dim], psrc[j*dim+1], psrc[j*dim+2]);

        /* only add velocities inside a given zone */
        if ((xmin == -1) || (idx >= xmin)) abvmin = 1;
        else abvmin =0;

        if ((xmax == -1) || (idx <= xmax)) blwmax = 1;
        else blwmax =0;
       

        if (abvmin && blwmax){

	   if (i < (READ_STEP*tskp)) {
	      u1_p = axx[i0*npsrc+j];
	      v1_p = ayy[i0*npsrc+j];
	      w1_p = azz[i0*npsrc+j];

	      u1_n = axx[i1*npsrc+j];
	      v1_n = ayy[i1*npsrc+j];
	      w1_n = azz[i1*npsrc+j];

	      u1_i = u1_p + (u1_n - u1_p) * pfact;
	      v1_i = v1_p + (v1_n - v1_p) * pfact;
	      w1_i = w1_p + (w1_n - w1_p) * pfact;

	      /*if (j==0){
		 cuPrintf("u1[%d]=%e, u1[%d]=%e, u1_i=%e\n", i0, u1_p, i1, u1_n, u1_i);
	      }*/

	      if (i == (READ_STEP*tskp-1)){
		 //if (j==0) cuPrintf("inside frcvel_cu: last step at i=%d\n", i);
		 /* copy last value back to beginning of array */
		 axx[j]= axx[i1*npsrc+j];
		 ayy[j]= ayy[i1*npsrc+j];
		 azz[j]= azz[i1*npsrc+j];
    
		 /* save restoring force to source array in case this is the last 
		    time step where velocity is prescribed  */
		 axx[i1*npsrc+j] = (u1_n - u1[pos]) / d_DT;
		 ayy[i1*npsrc+j] = (v1_n - v1[pos]) / d_DT;
		 azz[i1*npsrc+j] = (w1_n - w1[pos]) / d_DT;
	      }

	      u1[pos] = u1_i;
	      v1[pos] = v1_i;
	      w1[pos] = w1_i;

	      /*if (((psrc[j*dim] == 90) && (psrc[j*dim+1] == 10)) && (psrc[j*dim+2] == 180)){
		 cuPrintf("dbg1>> i=%d, pos=%ld, %e, %e, %e\n", i, pos, 
		    axx[i*npsrc+j], ayy[i*npsrc+j], azz[i*npsrc+j]);
	      }*/
	   }

	   /* we keep applying the static force needed to stabilize discontinuity */
	   else {
	      u1[pos] += axx[READ_STEP*npsrc+j] * d_DT;
	      v1[pos] += ayy[READ_STEP*npsrc+j] * d_DT;
	      w1[pos] += azz[READ_STEP*npsrc+j] * d_DT;
	      /*if (((psrc[j*dim] == 90) && (psrc[j*dim+1] == 10)) && (psrc[j*dim+2] == 180)){
		 cuPrintf("i=%d, pos=%ld, %e, %e, %e\n", i, pos, 
		    axx[(READ_STEP-1)*npsrc+j], ayy[(READ_STEP-1)*npsrc+j], azz[(READ_STEP-1)*npsrc+j]);
	      }*/
	   }
        }

        return;
}


/* kernel function to apply free-surface B.C. to velocities - (Daniel) */
extern "C"
void fvel_H(float* u1, float* v1, float* w1, cudaStream_t St, float* lam_mu, int NX, int rankx, int ranky, 
     int s_i, int e_i, int s_j, int e_j)
{
    dim3 block (2, BLOCK_SIZE_Y, 1);
    dim3 grid (1,(e_j-s_j+1+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,1);
    cudaFuncSetCacheConfig(fvel, cudaFuncCachePreferL1);
    fvel<<<grid, block, 0, St>>>(u1, v1, w1, lam_mu, NX, rankx, ranky, s_i, e_i, s_j);
    return;
}


/* kernel functions to apply free-surface B.C.s to velocity */
__global__ void fvel (float* u1, float* v1, float* w1, float* lam_mu, int NX, int rankx, int ranky, int s_i, int e_i, int s_j)
{
    register int i, j, k;
    //register _prec w1_im1, w1_im2, u1_ip1, f_u1, f_v1, f_w1;
    //register int pos, pos_km1, pos_kp1, pos_kp2, pos_jm1, pos_jp1, pos_im1;
    register int g_i;
    register _prec vs1, vs2;

    register int   pos,     pos_ip1, pos_im2, pos_im1;
    register int   pos_km2, pos_km1, pos_kp1, pos_kp2;
    register int   pos_jm2, pos_jm1, pos_jp1, pos_jp2;
    register int   pos_ik1, pos_jk1, pos_ijk, pos_ijk1,f_ww;

    register _prec f_u1, u1_ip1, u1_ip2, u1_im1;
    register _prec f_v1, v1_im1, v1_ip1, v1_im2;
    register _prec f_w1, w1_im1, w1_im2, w1_ip1;

    k    = d_nzt[0]+align-1;
    j    = blockIdx.y*BLOCK_SIZE_Y+threadIdx.y+s_j;
    i    = e_i;
    pos  = i*d_slice_1[0]+j*d_yline_1[0]+k;

    u1_ip1 = u1[pos+d_slice_2[0]];
    f_u1   = u1[pos+d_slice_1[0]];
    u1_im1 = u1[pos];    
    f_v1   = v1[pos+d_slice_1[0]];
    v1_im1 = v1[pos];
    v1_im2 = v1[pos-d_slice_1[0]];
    f_w1   = w1[pos+d_slice_1[0]];
    w1_im1 = w1[pos];
    w1_im2 = w1[pos-d_slice_1[0]];

    for(i=e_i;i>=s_i;i--)
    {
        pos_km2  = pos-2;
        pos_km1  = pos-1;
        pos_kp1  = pos+1;
        pos_kp2  = pos+2;
        pos_jm2  = pos-d_yline_2[0];
        pos_jm1  = pos-d_yline_1[0];
        pos_jp1  = pos+d_yline_1[0];
        pos_jp2  = pos+d_yline_2[0];
        pos_im2  = pos-d_slice_2[0];
        pos_im1  = pos-d_slice_1[0];
        pos_ip1  = pos+d_slice_1[0];
        pos_jk1  = pos-d_yline_1[0]-1;
        pos_ik1  = pos+d_slice_1[0]-1;
        pos_ijk  = pos+d_slice_1[0]-d_yline_1[0];
        pos_ijk1 = pos+d_slice_1[0]-d_yline_1[0]-1;

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

	u1[pos_kp1] = f_u1 - (f_w1        - w1_im1);
    	v1[pos_kp1] = f_v1 - (w1[pos_jp1] - f_w1);

        g_i  = d_nxt[0]*rankx + i - ngsl - 1;
 
    	if(g_i<NX)
        	vs1	= u1_ip1 - (w1_ip1    - f_w1);
   	else
        	vs1	= 0.0;

        g_i  = d_nyt[0]*ranky + j - ngsl - 1;
    	if(g_i>1)
        	vs2	= v1[pos_jm1] - (f_w1 - w1[pos_jm1]);
    	else
        	vs2	= 0.0;

    	w1[pos_kp1]	= w1[pos_km1] - lam_mu[i*(d_nyt[0]+4+ngsl2) + j]*((vs1         - u1[pos_kp1]) + (u1_ip1 - f_u1)
                                      +     			                (v1[pos_kp1] - vs2)         + (f_v1   - v1[pos_jm1]) );

        pos     = pos_im1;
    }

}

extern "C"
void update_yldfac_buffer_x_H(float* yldfac, _prec *buf_L, _prec *buf_R, int nyt, int nzt, cudaStream_t St1, cudaStream_t St2, 
     int rank_L, int rank_R, int d_i) {
     if(rank_L==-1 && rank_R==-1) return;

     /*dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1);
     dim3 grid ((nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, (nyt+ngsl2+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y, 1);*/
     dim3 block (1, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
     dim3 grid (1, (nyt+ngsl2+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y, (nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z);
     //cudaPrintfInit();
     CUCHK(cudaFuncSetCacheConfig(update_yldfac_buffer_x, cudaFuncCachePreferL1));
     update_yldfac_buffer_x<<<grid, block, 0, St1>>>(yldfac, buf_L, rank_L, Left, d_i);
     CUCHK( cudaGetLastError() );
     update_yldfac_buffer_x<<<grid, block, 0, St2>>>(yldfac, buf_R, rank_R, Right, d_i);
     CUCHK( cudaGetLastError() );
     //cudaPrintfDisplay(stdout, 1);
     //cudaPrintfEnd();
     return;
}

/* buffer exchanged for the swap area */
__global__ void update_yldfac_buffer_x(float* yldfac, _prec *buf, int rank, int flag, int d_i)
{
    register int i, j, k, pos, bpos;
    register int b_slice_1, b_yline_1;
    register int xs, xe, zs;
    register int nxt, nyt, nzt, slice_1, yline_1;

    nyt = d_nyt[d_i];
    nxt = d_nxt[d_i];
    nzt = d_nzt[d_i];
    slice_1 = d_slice_1[d_i];
    yline_1 = d_yline_1[d_i];

    b_slice_1  = (nyt+ngsl2)*nzt;
    b_yline_1  = nzt;
     
    j     = blockIdx.y*BLOCK_SIZE_Y+threadIdx.y;
    k     = blockIdx.z*BLOCK_SIZE_Z+threadIdx.z + align;
    zs    = align;

    if(flag==Left){
       xs=2+ngsl;
       xe=xs+ngsl;
    }
    else if (flag == Right){
       xs=nxt+2;
       xe=xs+ngsl;
    }
    if (rank != -1){
	for (i=xs; i < xe; i++){
	   pos   = i*slice_1+j*yline_1+k;

	   bpos = (i-xs)*b_slice_1+j*b_yline_1+(k-zs);
	   buf[bpos] = yldfac[pos];

	   /*if (((flag == Right) && (i==103)) && ((j==46) && (k==132))){
	      cuPrintf("swap send: buf[%d] = %.16g, yldfac[%d] = %.16g\n", bpos, buf[bpos], pos, yldfac[pos]);
	   }*/

	}
    }
    return;
}

extern "C"
void update_yldfac_data_x_H(float* yldfac, _prec *buf_L, _prec *buf_R, int nyt, int nzt, cudaStream_t St1, cudaStream_t St2, 
     int rank_L, int rank_R, int d_i) {
     if(rank_L==-1 && rank_R==-1) return;

     dim3 block (1, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
     dim3 grid (1, (nyt+ngsl2+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,(nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z);
     //cudaPrintfInit();
     CUCHK(cudaFuncSetCacheConfig(update_yldfac_buffer_x, cudaFuncCachePreferL1));
     update_yldfac_data_x<<<grid, block, 0, St1>>>(yldfac, buf_L, rank_L, Left, d_i);
     CUCHK( cudaGetLastError() );
     update_yldfac_data_x<<<grid, block, 0, St2>>>(yldfac, buf_R, rank_R, Right, d_i);
     CUCHK( cudaGetLastError() );
     //cudaPrintfDisplay(stdout, 1);
     //cudaPrintfEnd();
     return;
}

/* copy exchanged buffer data back to swap zone*/
__global__ void update_yldfac_data_x(float* yldfac, _prec *buf, int rank, int flag, int d_i)
{
    register int i, j, k, pos, bpos;
    register int b_slice_1, b_yline_1;
    register int xs, xe, zs;
    register int nxt, nyt, nzt, slice_1, yline_1;

    nyt = d_nyt[d_i];
    nxt = d_nxt[d_i];
    nzt = d_nzt[d_i];
    slice_1 = d_slice_1[d_i];
    yline_1 = d_yline_1[d_i];

    b_slice_1  = (nyt+ngsl2)*nzt;
    b_yline_1  = nzt;
     
    j     = blockIdx.y*BLOCK_SIZE_Y+threadIdx.y;
    k     = blockIdx.z*BLOCK_SIZE_Z+threadIdx.z + align;
    zs = align;

    if(flag==Left){
       xs=2;
       xe=2+ngsl;
    }
    else if (flag == Right){
       xs=nxt+2+ngsl;
       xe=xs+ngsl;
    }
    if (rank != -1){
	for (i=xs; i < xe; i++){
	   pos   = i*slice_1+j*yline_1+k;

	   bpos = (i-xs)*b_slice_1+j*b_yline_1+(k-zs);
	   yldfac[pos] = buf[bpos];

	   /*if (((flag == Left) && (i==3)) && ((j==46) && (k==132))){
	      cuPrintf("swap recv: buf[%d] = %.16g, yldfac[%d] = %.16g\n", bpos, buf[bpos], pos, yldfac[pos]);
	   }*/

	}
    }
    return;
}

extern "C"
void update_yldfac_buffer_y_H(float* yldfac, _prec *buf_F, _prec *buf_B, int nxt, int nzt,
   cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, int d_i) {
     if(rank_F==-1 && rank_B==-1) return;

     dim3 block (BLOCK_SIZE_X, 1, BLOCK_SIZE_Z);
     dim3 grid ((nxt+BLOCK_SIZE_X-1)/BLOCK_SIZE_X, 1, (nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z);
     //cudaPrintfInit();
     CUCHK(cudaFuncSetCacheConfig(update_yldfac_buffer_y, cudaFuncCachePreferL1));
     update_yldfac_buffer_y<<<grid, block, 0, St1>>>(yldfac, buf_F, rank_F, Front, d_i);
     CUCHK( cudaGetLastError() );
     update_yldfac_buffer_y<<<grid, block, 0, St2>>>(yldfac, buf_B, rank_B, Back, d_i);
     CUCHK( cudaGetLastError() );
     /*cudaPrintfDisplay(stdout, 1);
     cudaPrintfEnd();*/
     return;
}

/* buffer exchanged for the swap area along Y*/
__global__ void update_yldfac_buffer_y(float* yldfac, _prec *buf, int rank, int flag, int d_i)
{
    register int i, j, k, pos, bpos;
    register int b_slice_1, b_yline_1;
    register int ys, ye, xs, zs;
    register int nyt, nzt, slice_1, yline_1;

    nyt = d_nyt[d_i];
    nzt = d_nzt[d_i];
    slice_1 = d_slice_1[d_i];
    yline_1 = d_yline_1[d_i];

    b_slice_1  = ngsl*nzt;
    b_yline_1  = nzt;
     
    i     = blockIdx.x*BLOCK_SIZE_X+threadIdx.x+2+ngsl;
    k     = blockIdx.z*BLOCK_SIZE_Z+threadIdx.z+align;

    xs=2+ngsl;
    zs=align;

    if(flag==Front){
       ys=2+ngsl;
       ye=ys+ngsl;
    }
    else if (flag == Back){
       ys=nyt+2;
       ye=ys+ngsl;
    }
    if (rank != -1){
	for (j=ys; j < ye; j++){
	   pos   = i*slice_1+j*yline_1+k;

	   bpos = (i-xs)*b_slice_1+(j-ys)*b_yline_1+(k-zs);
	   buf[bpos] = yldfac[pos];

	   /*if (((flag == Back) && (i==103)) && ((j==33) && (k==132))){
	      cuPrintf("swap send: buf[%d] = %.16g, yldfac[%d] = %.16g\n", bpos, buf[bpos], pos, yldfac[pos]);
	   }*/

	}
    }
    return;
}

extern "C"
void update_yldfac_data_y_H(float* yldfac, _prec *buf_F, _prec *buf_B, int nxt, int nzt,
    cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, int d_i) {
     if(rank_F==-1 && rank_B==-1) return;

     dim3 block (BLOCK_SIZE_X, 1, BLOCK_SIZE_Z);
     dim3 grid ((nxt+BLOCK_SIZE_X-1)/BLOCK_SIZE_X, 1,(nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z);
     //cudaPrintfInit();
     CUCHK(cudaFuncSetCacheConfig(update_yldfac_buffer_y, cudaFuncCachePreferL1));
     update_yldfac_data_y<<<grid, block, 0, St1>>>(yldfac, buf_F, rank_F, Front, d_i);
     CUCHK( cudaGetLastError() );
     update_yldfac_data_y<<<grid, block, 0, St2>>>(yldfac, buf_B, rank_B, Back, d_i);
     CUCHK( cudaGetLastError() );
     /*cudaPrintfDisplay(stdout, 1);
     cudaPrintfEnd();*/
     return;
}

/* copy exchanged buffer data back to swap zone*/
__global__ void update_yldfac_data_y(float* yldfac, _prec *buf, int rank, int flag, int d_i)
{
    register int i, j, k, pos, bpos;
    register int b_slice_1, b_yline_1;
    register int ys, ye, xs, zs;
    register int nxt, nyt, nzt, slice_1, yline_1;

    nyt = d_nyt[d_i];
    nxt = d_nxt[d_i];
    nzt = d_nzt[d_i];
    slice_1 = d_slice_1[d_i];
    yline_1 = d_yline_1[d_i];

    b_slice_1  = ngsl*nzt;
    b_yline_1  = nzt;
     
    i     = blockIdx.x*BLOCK_SIZE_X+threadIdx.x+2+ngsl;
    k     = blockIdx.z*BLOCK_SIZE_Z+threadIdx.z+align;

    xs = 2+ngsl;
    zs = align;

    if(flag==Front){
       ys=2;
       ye=ys+ngsl;
    }
    else if (flag == Back){
       ys=nyt+2+ngsl;
       ye=ys+ngsl;
    }
    if (rank != -1){
	for (j=ys; j < ye; j++){
	   pos   = i*slice_1+j*yline_1+k;

	   bpos = (i-xs)*b_slice_1+(j-ys)*b_yline_1+(k-zs);
	   yldfac[pos] = buf[bpos];

	   /*if (((flag == Front) && (i==103)) && ((j==3) && (k==132))){
	      cuPrintf("swap recv: buf[%d] = %.16g, yldfac[%d] = %.16g\n", bpos, buf[bpos], pos, yldfac[pos]);
	   }*/

	}
    }
    return;
}

__global__ void dvelc2(float* u1,    float* v1,    float* w1,    float* xx,  float* yy, float* zz, float* xy, 
             float* xz, float* yz, float* dcrjx, float* dcrjy, float* dcrjz, float* d_1, int d_i)
{
    register int   i, j, k, pos,     pos_im1;
    register int   pos_km1, pos_kp1;
    register int   pos_jm1, pos_jp1;
    register int   pos_ip1, pos_jk1, pos_ik1, pos_ijk;
    register _prec f_d1,    f_d2,    f_d3,    f_dcrj;

    i = blockIdx.x*blockDim.x+threadIdx.x+2+ngsl;
    j = blockIdx.y*blockDim.y+threadIdx.y+2+ngsl;

    if(1+ngsl<i && i<d_nxt[d_i]+2+ngsl && 1+ngsl<j && j<d_nyt[d_i]+2+ngsl) {
    //cuPrintf(">> in dvelc2: i=%d\n", i);
// w is updated along first line
    k = 2 + align;
    pos = i*d_slice_1[d_i] + j*d_yline_1[d_i] + k;
    pos_ip1 = pos + d_slice_1[d_i];
    pos_jm1 = pos - d_yline_1[d_i];
    pos_kp1 = pos + 1;
    pos_ijk = pos + d_slice_1[d_i]-d_yline_1[d_i];

    f_d3     = 0.25*(d_1[pos] + d_1[pos_ip1] + d_1[pos_jm1] + d_1[pos_ijk]);
    f_d3 = d_dth[d_i]/f_d3;
    f_dcrj = dcrjz[k]*dcrjx[i]*dcrjy[j];

    w1[pos]  = (w1[pos] + f_d3*((xz[pos_ip1] - xz[pos]) + (yz[pos] - yz[pos_jm1]) + (zz[pos_kp1] - zz[pos])))*f_dcrj;

// u v are updated along second line
    k = 2 + align;
    pos = i*d_slice_1[d_i] + j*d_yline_1[d_i] + k;
    pos_im1 = pos - d_slice_1[d_i];
    pos_jm1 = pos - d_yline_1[d_i];
    pos_km1 = pos - 1;
    pos_jp1 = pos + d_yline_1[d_i];
    pos_jk1 = pos - d_yline_1[d_i]-1;
    pos_ip1 = pos + d_slice_1[d_i];
    pos_ik1 = pos + d_slice_1[d_i]-1;

    f_d1     = 0.25*(d_1[pos] + d_1[pos_jm1] + d_1[pos_km1] + d_1[pos_jk1]);
    f_d2     = 0.25*(d_1[pos] + d_1[pos_ip1] + d_1[pos_km1] + d_1[pos_ik1]);

    f_d1 = d_dth[d_i]/f_d1;
    f_d2 = d_dth[d_i]/f_d2;
    f_dcrj = dcrjz[k]*dcrjx[i]*dcrjy[j];
    //if(i == 95 && j == 95 ) printf("dvel2b checkh2 xy = %e  %e yy=%e %e yz=%e %e\n", xy[pos_ip1],xy[pos],yy[pos_jp1],yy[pos],yz[pos],yz[pos_km1]);
    u1[pos]  = (u1[pos] + f_d1*( (xx[pos] - xx[pos_im1]) + (xy[pos] - xy[pos_jm1]) + (xz[pos]-xz[pos_km1])))*f_dcrj;

    v1[pos]  = (v1[pos] + f_d2*( (xy[pos_ip1] - xy[pos]) + (yy[pos_jp1] - yy[pos]) + (yz[pos] - yz[pos_km1])))*f_dcrj;
    //if(i == 95 && j == 95 ) printf("dvel2a checkh2 u1 = %e v1 = %e f_dcrj=%e fd1=%e fd2=%e\n", u1[pos],v1[pos],f_dcrj,f_d1,f_d2);
      }


     return;
}

extern "C"
void dvelc2_H(float* u1,    float* v1,    float* w1,    float* xx,  float* yy, float* zz, float* xy, float* xz, float* yz,
             float* dcrjx, float* dcrjy, float* dcrjz, float* d_1, int nxt,   int nyt, cudaStream_t St, int d_i)
{
    dim3 block (BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 grid ((nxt+BLOCK_SIZE_X-1)/BLOCK_SIZE_X, (nyt+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,1);
    cudaFuncSetCacheConfig(dvelc2, cudaFuncCachePreferL1);

    /*cudaPrintfInit();*/
    dvelc2<<<grid,block,0,St>>>(u1,v1,w1,xx,yy,zz,xy,xz,yz,dcrjx,dcrjy,dcrjz,d_1,d_i);
    /*cudaPrintfDisplay(stdout, 1);
    cudaPrintfEnd();*/
}


__global__ void intp3d(_prec *u1l, float* v1l, _prec *w1l, _prec *xxl, _prec *yyl, _prec *zzl, 
        _prec *xyl, _prec * xzl, float* yzl,
        _prec *u1h, _prec *v1h, float* w1h, _prec *xxh, _prec *yyh, _prec *zzh, 
        _prec *xyh, _prec *xzh, float* yzh, int rank, int d_i)
{
    register int i,j,k,ii,jj,posl;
    register int posl_ip1,posl_jp1,posl_ij1;
    register int ih,jh,kh,posh,index;
    register _prec w[4],var[4];
    register int maxindex;

    w[0]=1.;
    w[1]=2./3.;
    w[2]=1./3.;
    w[3]=0.;

    maxindex=(d_nxt[d_i-1]+4+ngsl2)*(d_nyt[d_i-1]+4+ngsl2)*(d_nzt[d_i-1]+2*align)-1;

    i = blockIdx.x*blockDim.x+threadIdx.x+ngsl;
    j = blockIdx.y*blockDim.y+threadIdx.y+ngsl;

//
    if (ngsl<=i && i<=d_nxt[d_i]+3+ngsl && ngsl<=j && j<=d_nyt[d_i]+3+ngsl)

    //cuPrintf(">> in intp3d: i=%d\n", i);
    {

        k = d_nzt[d_i] + align - 3;
        // first surface of higher portion
        posl  = i*d_slice_1[d_i]+j*d_yline_1[d_i]+k;

        posl_ip1 = posl + d_slice_1[d_i];
        posl_jp1 = posl + d_yline_1[d_i];
        posl_ij1 = posl + d_slice_1[d_i] + d_yline_1[d_i];

        ih = (i-1-1-ngsl)*3+2+ngsl;
        jh = (j-1-1-ngsl)*3+2+ngsl;
        kh = align + 1;
        posh = ih*d_slice_1[d_i-1]+jh*d_yline_1[d_i-1]+kh;

        // interpolate from ul to uh

        var[0] = xzl[posl];
        var[1] = xzl[posl_ip1];
        var[2] = xzl[posl_jp1];
        var[3] = xzl[posl_ij1];
        
        for(jj = 1; jj<=4; jj++ )
        {
            for(ii = 1; ii<=4; ii++ )
            {
                   //This would be the correct way, but the if condition results in different results during 
                   //each run (thread divergence issues?)
                   //if ((ih+ii) < (d_nxt[d_i-1]+4+ngsl2) && (jh+jj) < (d_nyt[d_i-1]+4+ngsl2)){
		   index = min(posh + (ii-1) * d_slice_1[d_i-1] + (jj-1) * d_yline_1[d_i-1], maxindex);
		   xzh[index] =    var[0]*w[ii-1]*w[jj-1] +
				   var[1]*w[4-ii]*w[jj-1] +
				   var[2]*w[ii-1]*w[4-jj] +
				   var[3]*w[4-ii]*w[4-jj];
            }
        }

        // interpolate from v1l to v1h
        var[0] = yzl[posl];
 	var[1] = yzl[posl_ip1];
 	var[2] = yzl[posl_jp1];
 	var[3] = yzl[posl_ij1];

        for(jj = 1; jj<=4; jj++ )
        {
            for(ii = 1; ii<=4; ii++ )
            {
		   index = min(posh + (ii) * d_slice_1[d_i-1] + (jj) * d_yline_1[d_i-1], maxindex);
		   yzh[index] =    var[0]*w[ii-1]*w[jj-1] +
				   var[1]*w[4 - ii]*w[jj-1] +
				   var[2]*w[ii-1]*w[4 - jj] +
				   var[3]*w[4-ii]*w[4 - jj];
            }
        }

        var[0] = w1l[posl];
  	var[1] = w1l[posl_ip1];
 	var[2] = w1l[posl_jp1];
 	var[3] = w1l[posl_ij1];

        for(jj = 1; jj<=4; jj++ )
        {
            for(ii = 1; ii<=4; ii++ )
            {
		   index = min(posh + (ii) * d_slice_1[d_i-1] + (jj-1) * d_yline_1[d_i-1], maxindex);
		   w1h[index] =    var[0]*w[ii-1]*w[jj-1] +
				   var[1]*w[4-ii]*w[jj-1] +
				   var[2]*w[ii-1]*w[4-jj] +
				   var[3]*w[4-ii]*w[4-jj];
            }
        }

    /* xx,yy,zz,u1,v1 and xy can not be interpolated horizontally from kh=align+1 (not vertically aligned) */
    /* Uncommented this code segment.  Daniel Roten, December 6 2018 */

    /*    var[0] = xxl[posl];
  	var[1] = xxl[posl_ip1];
 	var[2] = xxl[posl_jp1];
 	var[3] = xxl[posl_ij1];
        for(jj = 1; jj<=4; jj++ )
        {
            for(ii = 1; ii<=4; ii++ )
            {
		   index = min(posh + (ii) * d_slice_1[d_i-1] + (jj-1) * d_yline_1[d_i-1], maxindex);
		   xxh[index] = var[0]*w[ii-1]*w[jj-1] +
				var[1]*w[4-ii]*w[jj-1] +
				var[2]*w[ii-1]*w[4-jj] +
				var[3]*w[4-ii]*w[4-jj];
            }
        }

        var[0] = yyl[posl];
  	var[1] = yyl[posl_ip1];
 	var[2] = yyl[posl_jp1];
 	var[3] = yyl[posl_ij1];
        for(jj = 1; jj<=4; jj++ )
        {
            for(ii = 1; ii<=4; ii++ )
            {
		   index = min(posh + (ii) * d_slice_1[d_i-1] + (jj-1) * d_yline_1[d_i-1], maxindex);
		   yyh[index] = var[0]*w[ii-1]*w[jj-1] +
				var[1]*w[4-ii]*w[jj-1] +
				var[2]*w[ii-1]*w[4-jj] +
				var[3]*w[4-ii]*w[4-jj];
            }
        }

        var[0] = zzl[posl];
  	var[1] = zzl[posl_ip1];
 	var[2] = zzl[posl_jp1];
 	var[3] = zzl[posl_ij1];
        for(jj = 1; jj<=4; jj++ )
        {
            for(ii = 1; ii<=4; ii++ )
            {
		   index = min(posh + (ii) * d_slice_1[d_i-1] + (jj-1) * d_yline_1[d_i-1], maxindex);
		   zzh[index] = var[0]*w[ii-1]*w[jj-1] +
				var[1]*w[4-ii]*w[jj-1] +
				var[2]*w[ii-1]*w[4-jj] +
				var[3]*w[4-ii]*w[4-jj];
            }
        }

        var[0] = u1l[posl];
        var[1] = u1l[posl_ip1];
        var[2] = u1l[posl_jp1];
        var[3] = u1l[posl_ij1];
        
        for(jj = 1; jj<=4; jj++ )
        {
            for(ii = 1; ii<=4; ii++ )
            {
		   index = min(posh + (ii-1) * d_slice_1[d_i-1] + (jj-1) * d_yline_1[d_i-1], maxindex);
		   u1h[index] = var[0]*w[ii-1]*w[jj-1] +
				var[1]*w[4-ii]*w[jj-1] +
				var[2]*w[ii-1]*w[4-jj] +
				var[3]*w[4-ii]*w[4-jj];
            }
        }

        var[0] = v1l[posl];
 	var[1] = v1l[posl_ip1];
 	var[2] = v1l[posl_jp1];
 	var[3] = v1l[posl_ij1];

        for(jj = 1; jj<=4; jj++ )
        {
            for(ii = 1; ii<=4; ii++ )
            {
		   index = min(posh + (ii) * d_slice_1[d_i-1] + (jj) * d_yline_1[d_i-1], maxindex);
		   v1h[index] = var[0]*w[ii-1]*w[jj-1] +
				var[1]*w[4 - ii]*w[jj-1] +
				var[2]*w[ii-1]*w[4 - jj] +
				var[3]*w[4-ii]*w[4 - jj];
            }
        }

        var[0] = xyl[posl];
 	var[1] = xyl[posl_ip1];
 	var[2] = xyl[posl_jp1];
 	var[3] = xyl[posl_ij1];

        for(jj = 1; jj<=4; jj++ )
        {
            for(ii = 1; ii<=4; ii++ )
            {
		   index = min(posh + (ii-1) * d_slice_1[d_i-1] + (jj) * d_yline_1[d_i-1], maxindex);
		   xyh[index] = var[0]*w[ii-1]*w[jj-1] +
				var[1]*w[4 - ii]*w[jj-1] +
				var[2]*w[ii-1]*w[4 - jj] +
				var[3]*w[4-ii]*w[4 - jj];
            }
        } */
    } // if (1<i && i<d_nxtl+1 && 1<j && j<d_nytl+1)
//

    return;
}

// 2nd order stress update
__global__ void dstrqc2(float* xx, float* yy,    float* zz,    float* xy,    float* xz,  float* yz,
                       float* r1, float* r2,    float* r3,    float* r4,    float* r5,  float* r6,
                       float* u1, float* v1,    float* w1,    float* lam,   float* mu,  float* qp,
                       float* qs, float* dcrjx, float* dcrjy, float* dcrjz, 
                       float* coeff, _prec *d_vx1, _prec *d_vx2, int *d_ww, _prec *d_wwo,
                       int s_i, int e_i, int s_j, int e_j, int d_i)
{
    register int   i,  j,  k;
    register int   pos,     pos_ip1, pos_im1;
    register int   pos_km1, pos_kp1;
    register int   pos_jm1, pos_jp1;
    register int   pos_ik1, pos_jk1, pos_ijk, pos_ijk1;
    register _prec vs1, vs2, vs3, a1, tmp, vx1;
    register _prec xl,  xm,  xmu1, xmu2, xmu3;
    register _prec qpa, h,   h1,   h2,   h3;
    register _prec f_vx1, f_vx2,  f_dcrj, f_r;
    register _prec f_rtmp;
    register int f_ww;
    register _prec qpaw, hw, h1w, h2w, h3w, f_wwo;

	i = blockIdx.x*blockDim.x+threadIdx.x+2;
	j = blockIdx.y*blockDim.y+threadIdx.y+2;

      if(s_i<=i && i<=e_i && s_j<=j && j<=e_j) {
      k = align + 2;
      pos = i*d_slice_1[d_i] + j*d_yline_1[d_i] + k;
      f_dcrj = dcrjz[k]*dcrjx[i]*dcrjy[j];

        pos_kp1  = pos+1;
        pos_km1  = pos-1;
        pos_jm1  = pos-d_yline_1[d_i];
        pos_jp1  = pos+d_yline_1[d_i];
        pos_im1  = pos-d_slice_1[d_i];
        pos_ip1  = pos+d_slice_1[d_i];
        pos_jk1  = pos-d_yline_1[d_i]-1;
        pos_ik1  = pos+d_slice_1[d_i]-1;
        pos_ijk  = pos+d_slice_1[d_i]-d_yline_1[d_i];
        pos_ijk1 = pos+d_slice_1[d_i]-d_yline_1[d_i]-1;

        #ifndef ELA2
        f_vx1 = d_vx1[pos];
        f_vx2 = d_vx2[pos];
        f_ww  = d_ww[pos];
        f_wwo = d_wwo[pos];
        #endif

        xl       = 8.0/(  lam[pos]      + lam[pos_ip1] + lam[pos_jm1] + lam[pos_ijk]
                        + lam[pos_km1]  + lam[pos_ik1] + lam[pos_jk1] + lam[pos_ijk1] );
        xm       = 16.0/( mu[pos]       + mu[pos_ip1]  + mu[pos_jm1]  + mu[pos_ijk]
                        + mu[pos_km1]   + mu[pos_ik1]  + mu[pos_jk1]  + mu[pos_ijk1] );
        xmu1     = 2.0/(  mu[pos]       + mu[pos_km1] );
        xmu2     = 2.0/(  mu[pos]       + mu[pos_jm1] );
        xmu3     = 2.0/(  mu[pos]       + mu[pos_ip1] );
        xl       = xl  +  xm;

        #ifndef ELA2
        qpa      = 0.0625*( qp[pos]     + qp[pos_ip1] + qp[pos_jm1] + qp[pos_ijk]
                          + qp[pos_km1] + qp[pos_ik1] + qp[pos_jk1] + qp[pos_ijk1] );

	if(1.0f/(qpa*2.0f)<=200.0f)
  	   qpaw=coeff[f_ww*2-2]*(2.*qpa)*(2.*qpa)+coeff[f_ww*2-1]*(2.*qpa);
	else qpaw  = f_wwo*qpa;
	
	qpaw=qpaw/f_wwo;

	h = 0.0625f*(qs[pos] + qs[pos_ip1] + qs[pos_jm1] + qs[pos_ijk]
                 + qs[pos_km1] + qs[pos_ik1] + qs[pos_jk1] + qs[pos_ijk1] );

        if(1./(h*2.0)<=200.0) 
          hw=coeff[f_ww*2-2]*(2.*h)*(2.*h)+coeff[f_ww*2-1]*(2.*h);
        else hw  = f_wwo*h;
        hw=hw/f_wwo;

        h1 = 0.25*(qs[pos] + qs[pos_km1]);

        if(1./(h1*2.0)<=200.0)
                  h1w=coeff[f_ww*2-2]*(2.*h1)*(2.*h1)+coeff[f_ww*2-1]*(2.*h1);
        else h1w  = f_wwo*h1;

        h1w=h1w/f_wwo;

        h2 = 0.250*(qs[pos] + qs[pos_jm1]);
        if(1./(h2*2.0)<=200.0)
                  h2w=coeff[f_ww*2-2]*(2.*h2)*(2.*h2)+coeff[f_ww*2-1]*(2.*h2);
        else h2w  = f_wwo*h2;

        h2w=h2w/f_wwo;

        h3       = 0.250*(  qs[pos]     + qs[pos_ip1] );
        if(1./(h3*2.0)<=200.0) 
                  h3w=coeff[f_ww*2-2]*(2.*h3)*(2.*h3)+coeff[f_ww*2-1]*(2.*h3);
        else h3w  = f_wwo*h3;

        h3w=h3w/f_wwo;

	h        = -xm*hw*d_dh1[d_i];
        h1       = -xmu1*h1w*d_dh1[d_i];
        h2       = -xmu2*h2w*d_dh1[d_i];
        h3       = -xmu3*h3w*d_dh1[d_i];

        qpa      = -qpaw*xl*d_dh1[d_i];
        #endif

        xm       = xm*d_dth[d_i];
        xmu1     = xmu1*d_dth[d_i];
        xmu2     = xmu2*d_dth[d_i];
        xmu3     = xmu3*d_dth[d_i];
        xl       = xl*d_dth[d_i];

        #ifndef ELA2
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
        #endif

        vs1     = (u1[pos_kp1] - u1[pos]);
        vs2     = (w1[pos] - w1[pos_im1]);
        #ifdef ELA2
        xz[pos] = (xz[pos]  + xmu2*(vs1+vs2) )*f_dcrj;
        #else
        f_r     = r5[pos];
	f_rtmp  = h2*(vs1+vs2);
	xz[pos] = xz[pos]  + xmu2*(vs1+vs2) + vx1*f_r; 
	r5[pos] = f_vx2*f_r + f_wwo*f_rtmp; 
	f_rtmp  = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1); 
	xz[pos] = (xz[pos] + d_DT*f_rtmp)*f_dcrj;
        #endif

        vs1     = (v1[pos_kp1] - v1[pos]);
        vs2     = (w1[pos_jp1] - w1[pos]);
        #ifdef ELA2
        yz[pos] = (yz[pos]  + xmu3*(vs1+vs2) )*f_dcrj;
        #else
       	f_r     = r6[pos];
	f_rtmp  = h3*(vs1+vs2);
	yz[pos] = yz[pos]  + xmu3*(vs1+vs2) + vx1*f_r;
        r6[pos] = f_vx2*f_r + f_wwo*f_rtmp;
	f_rtmp  = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1); 
	yz[pos] = (yz[pos] + d_DT*f_rtmp)*f_dcrj; 
        #endif

        //if(i == 95 && j == 95) printf("dstr2 checkh1 xmu3= %e xz = %e yz = %e \n", xmu3,xz[pos],yz[pos]);

  	vs1      = (u1[pos_ip1] - u1[pos]);
        vs2      = (v1[pos] - v1[pos_jm1]);
        vs3      = (w1[pos] - w1[pos_km1]);

        tmp      = xl*(vs1+vs2+vs3);
        a1       = qpa*(vs1+vs2+vs3);
        tmp      = tmp+d_DT*a1;

        #ifdef ELA2
        xx[pos]  = (xx[pos]  + tmp - xm*(vs2+vs3) )*f_dcrj;
        yy[pos]  = (yy[pos]  + tmp - xm*(vs1+vs3) )*f_dcrj;
        zz[pos]  = (zz[pos]  + tmp - xm*(vs1+vs2) )*f_dcrj;

        #else
        f_r      = r1[pos];
	f_rtmp   = -h*(vs2+vs3) + a1; 
	xx[pos]  = xx[pos]  + tmp - xm*(vs2+vs3) + vx1*f_r;  
	r1[pos]  = f_vx2*f_r + f_wwo*f_rtmp;
	f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1); 
	xx[pos]  = (xx[pos] + d_DT*f_rtmp)*f_dcrj;

        f_r      = r2[pos];
	f_rtmp   = -h*(vs1+vs3) + a1;  
        yy[pos]  = (yy[pos]  + tmp - xm*(vs1+vs3) + vx1*f_r)*f_dcrj;
	r2[pos]  = f_vx2*f_r + f_wwo*f_rtmp; 
	f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1); 
	yy[pos]  = (yy[pos] + d_DT*f_rtmp)*f_dcrj;
	
        f_r      = r3[pos];
	f_rtmp   = -h*(vs1+vs2) + a1;
        zz[pos]  = (zz[pos]  + tmp - xm*(vs1+vs2) + vx1*f_r)*f_dcrj;
	r3[pos]  = f_vx2*f_r + f_wwo*f_rtmp;
	f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);  
	zz[pos]  = (zz[pos] + d_DT*f_rtmp)*f_dcrj;
        #endif

        vs1      = (u1[pos_jp1] - u1[pos]) ;
        vs2      = (v1[pos] - v1[pos_im1]) ;
        #ifdef ELA2
        xy[pos]  = (xy[pos]  + xmu1*(vs1+vs2) )*f_dcrj;
        #else
        f_r      = r4[pos];
 	f_rtmp   = h1*(vs1+vs2); 
	xy[pos]  = xy[pos] + xmu1*(vs1+vs2) + vx1*f_r;
	r4[pos]  = f_vx2*f_r + f_wwo*f_rtmp; 
	f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);
	xy[pos]  = (xy[pos] + d_DT*f_rtmp)*f_dcrj;
        #endif

        /*if(i == 95 && j == 96) printf("dstr2 checkh2 xmu1= %e xy = %e xx = %e yy = %e zz = %e\n", xmu1,xy[pos],xx[pos],yy[pos],zz[pos]);*/
      }
	return;
}

extern "C"
void dstrqc2_H(float* xx, float* yy,    float* zz,    float* xy,    float* xz,  float* yz,
              float* r1, float* r2,    float* r3,    float* r4,    float* r5,  float* r6,
              float* u1, float* v1,    float* w1,    float* lam,   float* mu,  float* qp,
              float* qs, float* dcrjx, float* dcrjy, float* dcrjz, int nxt,    int nyt,
              cudaStream_t St, 
              float* coeff, _prec *vx1, _prec *vx2, int *ww, _prec *wwo,
              int s_i, int e_i, int s_j, int e_j, int d_i) {
    dim3 block (BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 grid ((nxt+BLOCK_SIZE_X+ngsl2-1)/BLOCK_SIZE_X, (nyt+BLOCK_SIZE_Y+ngsl2-1)/BLOCK_SIZE_Y,1);
    cudaFuncSetCacheConfig(dstrqc2, cudaFuncCachePreferL1);
    dstrqc2<<<grid, block, 0, St>>>(xx, yy, zz, xy, xz, yz, r1, r2, r3, r4, r5, r6, u1, v1, w1,
                            lam, mu, qp, qs, dcrjx, dcrjy, dcrjz, coeff, vx1, vx2, ww, wwo, 
                            s_i, e_i, s_j, e_j, d_i);

    return;
}

extern "C"
void intp3d_H(_prec *u1l, float* v1l, _prec *w1l, _prec *xxl, _prec *yyl, _prec *zzl, 
        _prec *xyl, _prec * xzl, float* yzl,
        _prec *u1h, _prec *v1h, float* w1h, _prec *xxh, _prec *yyh, _prec *zzh, 
        _prec *xyh, _prec *xzh, float* yzh,
        int nxtl, int nytl, int rank, cudaStream_t St, int d_i) {

    /* here, d_i is the grid number of the "low" grid, to which xzl, yzl, and w1l pertain */

    dim3 block (BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 grid ((nxtl+BLOCK_SIZE_X+ngsl2-1)/BLOCK_SIZE_X, (nytl+BLOCK_SIZE_Y+ngsl2-1)/BLOCK_SIZE_Y,1);
    /*cudaEvent_t start, stop;
    _prec duration = 0;*/
    cudaFuncSetCacheConfig(intp3d, cudaFuncCachePreferL1);
    //cudaPrintfInit();
   
    /*cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);*/
    intp3d<<<grid, block, 0, St>>>(u1l,v1l,w1l,xxl,yyl,zzl,xyl,xzl,yzl,
                                   u1h,v1h,w1h,xxh,yyh,zzh,xyh,xzh,yzh,
                                   rank,d_i); 
    /*cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    fprintf(stdout, "Time for intp3d: %f ms\n", duration);*/
    /*cudaPrintfDisplay(stdout, 1);
    cudaPrintfEnd();*/
    return;
}

/* This helper functions is for accessing data that may be located beyond the ghost cell and padding region
   of a subdomain, but that is stored in buffers already copied to the GPU.
   It is used by the swap kernel if the transpose matrix swap is invoked */
__device__ _prec bgcaccess(_prec *dsub, int varpos, _prec *buf_L, _prec *buf_R, _prec *buf_F, _prec *buf_B,
      int ipos, int jpos, int kpos, int d_i){
   register long int blr_slice_1, blr_yline_1;
   register long int bfb_slice_1, bfb_yline_1;
   register long int blr_offset, bfb_offset;
   register long int bpos, posh;
   /*define extent of swap area buffer relative to local grid*/
   register int xs_left = -WWL, xs_right = d_nxt[d_i]+4+ngsl2-6;  /*check this ! */
   register int ys_front = -WWL, ys_back = d_nyt[d_i]+4+ngsl2-6; 
   register int ys_lr = -WWL;
   register int zs=align+1, ze=align+8;
   _prec nval;

   blr_slice_1  = (d_nyt[d_i]+4+ngsl2+2*WWL)*(ze-zs+1);
   blr_yline_1  = ze-zs+1;
   blr_offset   = (2+ngsl+WWL)*blr_slice_1;

   bfb_slice_1  = (2+ngsl+WWL)*(ze-zs+1);
   bfb_yline_1  = ze-zs+1;
   bfb_offset   = (d_nxt[d_i]+4+ngsl2)*bfb_slice_1;

   if (ipos < 0 || ipos < 2+ngsl && jpos < 0 || ipos < 2+ngsl && jpos >= d_nyt[d_i]+4+ngsl2){
        bpos = (ipos-xs_left)*blr_slice_1+(jpos-ys_lr)*blr_yline_1+(kpos-zs);
        nval=buf_L[bpos + varpos*blr_offset];
   }
   else if (ipos >= d_nxt[d_i]+4+ngsl2 || ipos >= d_nxt[d_i]+2+ngsl && jpos < 0 ||
       ipos >= d_nxt[d_i]+2+ngsl && jpos >= d_nyt[d_i]+4+ngsl2 ){
        bpos = (ipos-xs_right)*blr_slice_1+(jpos-ys_lr)*blr_yline_1+(kpos-zs);
        nval=buf_R[bpos + varpos*blr_offset];
   }
   else if (jpos < 0){
        bpos = ipos*bfb_slice_1+(jpos-ys_front)*bfb_yline_1+(kpos-zs);
        nval=buf_F[bpos + varpos*bfb_offset];
   }
   else if( jpos >= d_nyt[d_i]+4+ngsl2){
        bpos = ipos*bfb_slice_1+(jpos-ys_back)*bfb_yline_1+(kpos-zs);
        nval=buf_B[bpos + varpos*bfb_offset];
   }
   else {
        posh = ipos*d_slice_1[d_i]+jpos*d_yline_1[d_i]+kpos;
        nval=dsub[posh];
   }
   return(nval);
}

__global__ void swap(_prec * xxl, float* yyl, float* zzl, float* xyl,float* xzl,float* yzl,float* u1l, float* v1l, float* w1l,
                     _prec * xxh, float* yyh, float* zzh, float* xyh, float* xzh, float* yzh,float* u1h, float* v1h, float* w1h, 
                     _prec *buf_L, _prec *buf_R, _prec *buf_F, _prec *buf_B, int rank, int d_i) {
    register int i,j,k,ih,jh,kh,posl,posh,ii,jj,poshij;
    register _prec sum1, sum2, sum3;
    //register int b_slice_1, b_yline_1;
    register long int b_offset, bpos, bposij;
    //register int zs=2, ze=7;
    register double ttlwght=0.;

    /*b_slice_1  = (2+ngsl+WWL)*(ze-zs+1);
    b_yline_1  = ze-zs+1;
    b_offset   = (d_nxt[d_i]+4+ngsl2)*b_slice_1;*/

    i = blockIdx.x*blockDim.x+threadIdx.x+ngsl;
    j = blockIdx.y*blockDim.y+threadIdx.y+ngsl;

    for(jj=-WWL;jj<=WWL;jj++)
      for(ii=-WWL;ii<=WWL;ii++)
           ttlwght+=(1.+WWL-abs(jj))*(1.+WWL-abs(ii));

//    transpose matrix swap      

      if(ngsl-1<i && i<d_nxt[d_i+1]+ngsl+5 && ngsl-1<j && j<d_nyt[d_i+1]+ngsl+5) {

        k = align + d_nzt[d_i+1] - 2;
        posl  = i*d_slice_1[d_i+1]+j*d_yline_1[d_i+1]+k;

        sum1 = 0.;
        sum2 = 0.;
        sum3 = 0.;
        for(jj=-WWL;jj<=WWL;jj++)
	   for(ii=-WWL;ii<=WWL;ii++){
	      ih = (i-1-1-ngsl)*3+2+ngsl;
	      jh = (j-1-1-ngsl)*3+2+ngsl;
	      kh = align + 3;
	      sum1 = sum1 + 1/ttlwght*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))* 
                     bgcaccess(u1h, sbvpos_u1, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i);

	      ih = (i-1-1-ngsl)*3+3+ngsl;
	      jh = (j-1-1-ngsl)*3+3+ngsl;
	      kh = align + 3;
	      sum2 = sum2 + 1/ttlwght*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*
                     bgcaccess(v1h, sbvpos_v1, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i);

	      ih = (i-1-1-ngsl)*3+2+ngsl;
	      jh = (j-1-1-ngsl)*3+3+ngsl;
	      kh = align + 3;
	      sum3 = sum3 + 1/ttlwght*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*
                     bgcaccess(xyh, sbvpos_xy, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i);

	   }
        u1l[posl] = sum1;
        v1l[posl] = sum2;
        xyl[posl] = sum3;


        sum1 = 0.;
        sum2 = 0.;
        sum3 = 0.;
        for(jj=-WWL;jj<=WWL;jj++)
	   for(ii=-WWL;ii<=WWL;ii++){
	      ih = (i-1-1-ngsl)*3+3+ngsl;
	      jh = (j-1-1-ngsl)*3+2+ngsl;
	      kh = align + 3;
	      sum1 = sum1 + 1/ttlwght*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*
                     bgcaccess(xxh, sbvpos_xx, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i);
	      sum2 = sum2 + 1/ttlwght*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*
                     bgcaccess(yyh, sbvpos_yy, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i);
	      sum3 = sum3 + 1/ttlwght*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*
                     bgcaccess(zzh, sbvpos_zz, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i);
	      /*if (i==4 && j==4 && rank == 3)
		 cuPrintf("%d: ii=%d, jj=%d, xxh=%e\n", rank, ii, jj, 
		    bgcaccess(xxh, sbvpos_xx, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i));
	      if (i==36 && j==36 && rank == 0)
		 cuPrintf("%d: ii=%d, jj=%d, xxh=%e\n", rank, ii, jj, 
		    bgcaccess(xxh, sbvpos_xx, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i));*/
	   }
        xxl[posl] = sum1;
        yyl[posl] = sum2;
        zzl[posl] = sum3;

        sum1 = 0.;
        sum2 = 0.;
        sum3 = 0.;
        for(jj=-WWL;jj<=WWL;jj++)
	   for(ii=-WWL;ii<=WWL;ii++){
	      ih = (i-1-1-ngsl)*3+2+ngsl;
	      jh = (j-1-1-ngsl)*3+2+ngsl;
	      kh = align + 4;
	      sum1 = sum1 + 1/ttlwght*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*
                     bgcaccess(xzh, sbvpos_xz, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i);
	      //if (i==36 && j==4 && rank == 1){
		 //cuPrintf("%d: ii=%d, jj=%d, poshij=%d, xzh=%e, xzh in buf=%e\n", rank, ii, jj, poshij, 
		 //   xzh[poshij], newxz);
		 //}
	      //if (i==36 && j==36 && rank == 0){
		 //cuPrintf("%d: ii=%d, jj=%d, poshij=%d, xzh=%e\n", rank, ii, jj, poshij, 
		  //  xzh[poshij]);
		 //}

	      ih = (i-1-1-ngsl)*3+3+ngsl;
	      jh = (j-1-1-ngsl)*3+2+ngsl;
	      kh = align + 4;
	      sum2 = sum2 + 1/ttlwght*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*
                     bgcaccess(w1h, sbvpos_w1, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i);

	      ih = (i-1-1-ngsl)*3+3+ngsl;
	      jh = (j-1-1-ngsl)*3+3+ngsl;
	      kh = align + 4;
       	      sum3 = sum3 + 1/ttlwght*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*
                     bgcaccess(yzh, sbvpos_yz, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i);

       	   }
        xzl[posl] = sum1;
        w1l[posl] = sum2;
        yzl[posl] = sum3;

        k = align + d_nzt[d_i+1]-1;
        posl  = i*d_slice_1[d_i+1]+j*d_yline_1[d_i+1]+k;
        sum1 = 0.;
        sum2 = 0.;
        sum3 = 0.;
        for(jj=-WWL;jj<=WWL;jj++)
	   for(ii=-WWL;ii<=WWL;ii++){
	      ih = (i-1-1-ngsl)*3+2+ngsl;
	      jh = (j-1-1-ngsl)*3+2+ngsl;
	      kh = align + 6;
	      sum1 = sum1 + 1/ttlwght*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*
                     bgcaccess(u1h, sbvpos_u1, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i);

	      ih = (i-1-1-ngsl)*3+3+ngsl;
	      jh = (j-1-1-ngsl)*3+3+ngsl;
	      kh = align + 6;
	      sum2 = sum2 + 1/ttlwght*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*
                     bgcaccess(v1h, sbvpos_v1, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i);

	      ih = (i-1-1-ngsl)*3+2+ngsl;
	      jh = (j-1-1-ngsl)*3+3+ngsl;
	      kh = align + 6;
	      sum3 = sum3 + 1/ttlwght*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*
                     bgcaccess(xyh, sbvpos_xy, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i);
	   }
        u1l[posl] = sum1;
        v1l[posl] = sum2;
        xyl[posl] = sum3;

        sum1 = 0.;
        sum2 = 0.;
        sum3 = 0.;
        for(jj=-WWL;jj<=WWL;jj++)
	   for(ii=-WWL;ii<=WWL;ii++){
	      ih = (i-1-1-ngsl)*3+3+ngsl;
	      jh = (j-1-1-ngsl)*3+2+ngsl;
	      kh = align + 6;
	      sum1 = sum1 + 1/ttlwght*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*
                     bgcaccess(xxh, sbvpos_xx, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i);
	      sum2 = sum2 + 1/ttlwght*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*
                     bgcaccess(yyh, sbvpos_yy, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i);
	      sum3 = sum3 + 1/ttlwght*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*
                     bgcaccess(zzh, sbvpos_zz, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i);
	      /*if (i==37 && j==39 && rank == 0)
		 cuPrintf("%d: ii=%d, jj=%d, xxh=%e\n", rank, ii, jj, 
		    bgcaccess(xxh, sbvpos_xx, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i));*/
	      /*if (i==38 && j==5 && rank == 1)
		 cuPrintf("%d: ii=%d, jj=%d, xxh=%e\n", rank, ii, jj, 
		    bgcaccess(xxh, sbvpos_xx, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i));*/
	   }
        xxl[posl] = sum1;
        yyl[posl] = sum2;
        zzl[posl] = sum3;
	/*if (i==4 && j==4 && rank == 0) cuPrintf("%d: xxl=%e\n", rank, xxl[posl]);
	if (i==39 && j==4 && rank == 1) cuPrintf("%d: xxl=%e\n", rank, xxl[posl]);*/
    }

    return;
}

__global__ void swap3(_prec * xxl, float* yyl, float* zzl, float* xyl,float* xzl,float* yzl,float* u1l, float* v1l, float* w1l,
                     _prec * xxh, float* yyh, float* zzh, float* xyh, float* xzh, float* yzh,float* u1h, float* v1h, float* w1h, 
                     _prec *buf_L, _prec *buf_R, _prec *buf_F, _prec *buf_B, int rank, int d_i) {
    register int i,j,k,ih,jh,kh,posl,posh,ii,jj,kk,poshij;
    register _prec sum1, sum2, sum3;
    //register int b_slice_1, b_yline_1;
    register long int b_offset, bpos, bposij;
    //register int zs=2, ze=7;
    register double ttlwght2=0., ttlwght3=0., ttlwght4=0.;
    register int wwl_kk2=1, wwl_kk3, wwl_kk4;

    /*b_slice_1  = (2+ngsl+WWL)*(ze-zs+1);
    b_yline_1  = ze-zs+1;
    b_offset   = (d_nxt[d_i]+4+ngsl2)*b_slice_1;*/

    i = blockIdx.x*blockDim.x+threadIdx.x+ngsl;
    j = blockIdx.y*blockDim.y+threadIdx.y+ngsl;

    if (WWL >= 3) wwl_kk3 = 3; else wwl_kk3=WWL;
    if (WWL >= 4) wwl_kk4 = 4; else wwl_kk4=WWL;

    for(jj=-WWL;jj<=WWL;jj++)
      for(ii=-WWL;ii<=WWL;ii++){
         for(kk=-wwl_kk2;kk<=wwl_kk2;kk++)
            ttlwght2+=(1.+WWL-abs(jj))*(1.+WWL-abs(ii))*(1.+WWL-abs(kk));
         for(kk=-wwl_kk3;kk<=wwl_kk3;kk++)
            ttlwght3+=(1.+WWL-abs(jj))*(1.+WWL-abs(ii))*(1.+WWL-abs(kk));
         for(kk=-wwl_kk4;kk<=wwl_kk4;kk++)
            ttlwght4+=(1.+WWL-abs(jj))*(1.+WWL-abs(ii))*(1.+WWL-abs(kk));
      }

//    transpose matrix swap      

      if(ngsl-1<i && i<d_nxt[d_i+1]+ngsl+5 && ngsl-1<j && j<d_nyt[d_i+1]+ngsl+5) {

        k = align + d_nzt[d_i+1] - 2;
        posl  = i*d_slice_1[d_i+1]+j*d_yline_1[d_i+1]+k;

        sum1 = 0.;
        sum2 = 0.;
        sum3 = 0.;
	kh = align + 3;
        for(jj=-WWL;jj<=WWL;jj++)
	   for(ii=-WWL;ii<=WWL;ii++)
   	      for(kk=-wwl_kk2;kk<=wwl_kk2;kk++){
		 ih = (i-1-1-ngsl)*3+2+ngsl;
		 jh = (j-1-1-ngsl)*3+2+ngsl;
		 sum1 = sum1 + 1/ttlwght2*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*(1.+WWL-abs(kk))* 
			bgcaccess(u1h, sbvpos_u1, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh+kk, d_i);

		 ih = (i-1-1-ngsl)*3+3+ngsl;
		 jh = (j-1-1-ngsl)*3+3+ngsl;
		 sum2 = sum2 + 1/ttlwght2*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*(1.+WWL-abs(kk))*
			bgcaccess(v1h, sbvpos_v1, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh+kk, d_i);

		 ih = (i-1-1-ngsl)*3+2+ngsl;
		 jh = (j-1-1-ngsl)*3+3+ngsl;
		 sum3 = sum3 + 1/ttlwght2*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*(1.+WWL-abs(kk))*
			bgcaccess(xyh, sbvpos_xy, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh+kk, d_i);

	   }
        u1l[posl] = sum1;
        v1l[posl] = sum2;
        xyl[posl] = sum3;


        sum1 = 0.;
        sum2 = 0.;
        sum3 = 0.;
	kh = align + 3;
        for(jj=-WWL;jj<=WWL;jj++)
	   for(ii=-WWL;ii<=WWL;ii++)
	      for(kk=-wwl_kk2;kk<=wwl_kk2;kk++){
		 ih = (i-1-1-ngsl)*3+3+ngsl;
		 jh = (j-1-1-ngsl)*3+2+ngsl;
		 sum1 = sum1 + 1/ttlwght2*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*(1.+WWL-abs(kk))*
			bgcaccess(xxh, sbvpos_xx, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh+kk, d_i);
		 sum2 = sum2 + 1/ttlwght2*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*(1.+WWL-abs(kk))*
			bgcaccess(yyh, sbvpos_yy, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh+kk, d_i);
		 sum3 = sum3 + 1/ttlwght2*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*(1.+WWL-abs(kk))*
			bgcaccess(zzh, sbvpos_zz, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh+kk, d_i);
		 /*if (i==4 && j==4 && rank == 3)
		    cuPrintf("%d: ii=%d, jj=%d, xxh=%e\n", rank, ii, jj, 
		       bgcaccess(xxh, sbvpos_xx, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i));
		 if (i==36 && j==36 && rank == 0)
		    cuPrintf("%d: ii=%d, jj=%d, xxh=%e\n", rank, ii, jj, 
		       bgcaccess(xxh, sbvpos_xx, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i));*/
	   }
        xxl[posl] = sum1;
        yyl[posl] = sum2;
        zzl[posl] = sum3;

        sum1 = 0.;
        sum2 = 0.;
        sum3 = 0.;
	kh = align + 4;
        for(jj=-WWL;jj<=WWL;jj++)
	   for(ii=-WWL;ii<=WWL;ii++)
   	      for(kk=-wwl_kk3;kk<=wwl_kk3;kk++){
		 ih = (i-1-1-ngsl)*3+2+ngsl;
		 jh = (j-1-1-ngsl)*3+2+ngsl;
		 sum1 = sum1 + 1/ttlwght3*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*(1.+WWL-abs(kk))*
			bgcaccess(xzh, sbvpos_xz, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh+kk, d_i);
		 //if (i==36 && j==4 && rank == 1){
		    //cuPrintf("%d: ii=%d, jj=%d, poshij=%d, xzh=%e, xzh in buf=%e\n", rank, ii, jj, poshij, 
		    //   xzh[poshij], newxz);
		    //}
		 //if (i==36 && j==36 && rank == 0){
		    //cuPrintf("%d: ii=%d, jj=%d, poshij=%d, xzh=%e\n", rank, ii, jj, poshij, 
		     //  xzh[poshij]);
		    //}

		 ih = (i-1-1-ngsl)*3+3+ngsl;
		 jh = (j-1-1-ngsl)*3+2+ngsl;
		 sum2 = sum2 + 1/ttlwght3*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*(1.+WWL-abs(kk))*
			bgcaccess(w1h, sbvpos_w1, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh+kk, d_i);

		 ih = (i-1-1-ngsl)*3+3+ngsl;
		 jh = (j-1-1-ngsl)*3+3+ngsl;
		 sum3 = sum3 + 1/ttlwght3*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*(1.+WWL-abs(kk))*
			bgcaccess(yzh, sbvpos_yz, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh+kk, d_i);

       	   }
        xzl[posl] = sum1;
        w1l[posl] = sum2;
        yzl[posl] = sum3;

        k = align + d_nzt[d_i+1]-1;
        posl  = i*d_slice_1[d_i+1]+j*d_yline_1[d_i+1]+k;
        sum1 = 0.;
        sum2 = 0.;
        sum3 = 0.;
	kh = align + 6;
        for(jj=-WWL;jj<=WWL;jj++)
	   for(ii=-WWL;ii<=WWL;ii++)
   	      for(kk=-wwl_kk4;kk<=wwl_kk4;kk++){
		 ih = (i-1-1-ngsl)*3+2+ngsl;
		 jh = (j-1-1-ngsl)*3+2+ngsl;
		 sum1 = sum1 + 1/ttlwght4*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*(1.+WWL-abs(kk))*
			bgcaccess(u1h, sbvpos_u1, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh+kk, d_i);

		 ih = (i-1-1-ngsl)*3+3+ngsl;
		 jh = (j-1-1-ngsl)*3+3+ngsl;
		 sum2 = sum2 + 1/ttlwght4*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*(1.+WWL-abs(kk))*
			bgcaccess(v1h, sbvpos_v1, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh+kk, d_i);

		 ih = (i-1-1-ngsl)*3+2+ngsl;
		 jh = (j-1-1-ngsl)*3+3+ngsl;
		 sum3 = sum3 + 1/ttlwght4*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*(1.+WWL-abs(kk))*
			bgcaccess(xyh, sbvpos_xy, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh+kk, d_i);
	   }
        u1l[posl] = sum1;
        v1l[posl] = sum2;
        xyl[posl] = sum3;

        sum1 = 0.;
        sum2 = 0.;
        sum3 = 0.;
        kh = align + 6;
        for(jj=-WWL;jj<=WWL;jj++)
	   for(ii=-WWL;ii<=WWL;ii++)
	      for(kk=-wwl_kk4;kk<=wwl_kk4;kk++){
		 ih = (i-1-1-ngsl)*3+3+ngsl;
		 jh = (j-1-1-ngsl)*3+2+ngsl;
		 sum1 = sum1 + 1/ttlwght4*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*(1.+WWL-abs(kk))*
			bgcaccess(xxh, sbvpos_xx, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh+kk, d_i);
		 sum2 = sum2 + 1/ttlwght4*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*(1.+WWL-abs(kk))*
			bgcaccess(yyh, sbvpos_yy, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh+kk, d_i);
		 sum3 = sum3 + 1/ttlwght4*(1.+WWL-abs(ii))*(1.+WWL-abs(jj))*(1.+WWL-abs(kk))*
			bgcaccess(zzh, sbvpos_zz, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh+kk, d_i);
		 /*if (i==37 && j==39 && rank == 0)
		    cuPrintf("%d: ii=%d, jj=%d, xxh=%e\n", rank, ii, jj, 
		       bgcaccess(xxh, sbvpos_xx, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i));*/
		 /*if (i==38 && j==5 && rank == 1)
		    cuPrintf("%d: ii=%d, jj=%d, xxh=%e\n", rank, ii, jj, 
		       bgcaccess(xxh, sbvpos_xx, buf_L, buf_R, buf_F, buf_B, ih+ii, jh+jj, kh, d_i));*/
	   }
        xxl[posl] = sum1;
        yyl[posl] = sum2;
        zzl[posl] = sum3;
	/*if (i==4 && j==4 && rank == 0) cuPrintf("%d: xxl=%e\n", rank, xxl[posl]);
	if (i==39 && j==4 && rank == 1) cuPrintf("%d: xxl=%e\n", rank, xxl[posl]);*/
    }

    return;
}

extern "C"
void swap_H(_prec * xxl, float* yyl, float* zzl, float* xyl,float* xzl,float* yzl,float* u1l, float* v1l, float* w1l,
            _prec * xxh, float* yyh, float* zzh, float* xyh, float* xzh, float* yzh,float* u1h, float* v1h, float* w1h, 
            int nxtl,int nytl, _prec *buf_L, _prec *buf_R, _prec *buf_F, _prec *buf_B, int rank, cudaStream_t St, int d_i) {

    /* here, d_i is the grid number of the "high" grid, to which xxh, yyh, ... pertain */

    dim3 block (BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 grid ((nxtl+BLOCK_SIZE_X+ngsl-1)/(BLOCK_SIZE_X), (nytl+BLOCK_SIZE_Y+ngsl-1)/(BLOCK_SIZE_Y),1);
    cudaFuncSetCacheConfig(swap, cudaFuncCachePreferL1);
    /*cudaPrintfInit();*/
    swap3<<<grid,block,0,St>>>(xxl,yyl,zzl,xyl,xzl,yzl,u1l,v1l,w1l,xxh,yyh,zzh,xyh,xzh,yzh,u1h,v1h,w1h,
                         buf_L, buf_R, buf_F, buf_B, rank, d_i);
    /*cudaPrintfDisplay(stdout, 1);
    cudaPrintfEnd();*/
    return;
}

__global__ void print_nonzero(_prec *array, int nx, int ny, int nz, int d_i)
{
    int ix, iy, iz;

    cuPrintf("nonzeros in grid %d: =====================================================\n", d_i);
    for (iz=0; iz<nz; iz++)
    {
        for (iy=0; iy<ny; iy++)
        {
            for (ix=0; ix<nx; ix++)
            {
                int idx = ix*ny*nz + iy*nz + iz;
                // if (array[idx] != 0.0)
                // {
                //     printf("|   [%d][%d][%d] = %+le\n", ix, iy, iz, array[idx]);
                // }
                if (array[idx] != 0.0)
                   cuPrintf("%d %d %d %e\n", ix, iy, iz, array[idx]);
            }
        }
    }
}

void print_nonzero_H(_prec *array, int nx, int ny, int nz, int d_i)
{
    cudaPrintfInit();
    print_nonzero<<<1,1>>>(array, nx, ny, nz, d_i);
    cudaPrintfDisplay(stdout, 1);
    cudaPrintfEnd();
}

__global__ void print_nonzero_mat(_prec *array, int nx, int ny, int nz, int d_i,
    _prec *d1, _prec *mu, _prec *lam, _prec *qp, _prec *qs, int rank)
{
    int ix, iy, iz;

    cuPrintf("nonzeros in grid %d: =====================================================\n", d_i);
    for (iz=0; iz<nz; iz++) {
        for (iy=0; iy<ny; iy++) {
            for (ix=0; ix<nx; ix++) {
                int idx = ix*ny*nz + iy*nz + iz;
                if (array[idx] != 0.0)
                   cuPrintf("%d: mat @ %d %d %d: %e, %e, %e, %e, %e; val=%e\n", rank, ix, iy, iz, d1[idx], mu[idx], lam[idx], qp[idx], 
                      qs[idx], array[idx]);
            }
        }
    }
}

extern "C"
void print_nonzero_mat_H(_prec *array, int nx, int ny, int nz, int d_i, 
     _prec *d1, _prec *mu, _prec *lam, _prec *qp, _prec *qs, int rank)
{
    cudaPrintfInit();
    print_nonzero_mat<<<1,1>>>(array, nx, ny, nz, d_i, d1, mu, lam, qp, qs, rank);
    cudaPrintfDisplay(stdout, 1);
    cudaPrintfEnd();
}

__global__ void print_nan(_prec *array, int nx, int ny, int nz, char *vname)
{
    int ix, iy, iz;

    for (iz=0; iz<nz; iz++)
    {
        for (iy=0; iy<ny; iy++)
        {
            for (ix=0; ix<nx; ix++)
            {
                int idx = ix*ny*nz + iy*nz + iz;
                if (array[idx] != array[idx])
                   cuPrintf("%s(%d,%d,%d)=%e\n", vname, ix, iy, iz, array[idx]);
            }
        }
    }
}

extern "C"
void print_nan_H(_prec *array, int nx, int ny, int nz, char *vname)
{
    cudaPrintfInit();
    print_nan<<<1,1>>>(array, nx, ny, nz, vname);
    cudaPrintfDisplay(stdout, 1);
    cudaPrintfEnd();
}

extern "C"
void update_swapzone_buffer_x_H(float* u1, float* v1, float* w1, float* xx, float* yy, float* zz, _prec *xy, _prec *xz, _prec *yz, 
   _prec *buf_L, _prec *buf_R, int nyt, cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R, int zs, int ze, int d_i) {
     if(rank_L==-1 && rank_R==-1) return;

     dim3 block (1, BLOCK_SIZE_Y, 1);
     dim3 grid (1, (nyt+4+ngsl2+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,1);
     /*cudaPrintfInit();*/
     CUCHK(cudaFuncSetCacheConfig(update_swapzone_buffer_x, cudaFuncCachePreferL1));
     update_swapzone_buffer_x<<<grid, block, 0, St1>>>(u1, v1, w1, xx, yy, zz, xy, xz, yz, buf_L, rank_L, Left, zs, ze, d_i);
     CUCHK( cudaGetLastError() );
     update_swapzone_buffer_x<<<grid, block, 0, St2>>>(u1, v1, w1, xx, yy, zz, xy, xz, yz, buf_R, rank_R, Right, zs, ze, d_i);
     CUCHK( cudaGetLastError() );
     /*cudaPrintfDisplay(stdout, 1);
     cudaPrintfEnd();*/
     return;
}

/* buffer exchanged for the swap area */
__global__ void update_swapzone_buffer_x(float* u1, float* v1, float* w1, _prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
   _prec *buf, int rank, int flag, int zs, int ze, int d_i)
{
    register int i, j, k, pos, bpos;
    register long int b_offset;
    register int b_slice_1, b_yline_1;
    register int xs, xe; /*ys=-2*/;
    register int nxt, nyt, slice_1, yline_1;

    nyt = d_nyt[d_i];
    nxt = d_nxt[d_i];
    slice_1 = d_slice_1[d_i];
    yline_1 = d_yline_1[d_i];

    b_slice_1  = (nyt+4+ngsl2+2*WWL)*(ze-zs+1);
    b_yline_1  = ze-zs+1;
    b_offset   = (2+ngsl+WWL)*b_slice_1;
     
    j     = blockIdx.y*BLOCK_SIZE_Y+threadIdx.y;

    if(flag==Left){
       xs=6;
       xe=xs+2+ngsl+WWL; //14 if WWL=2
    }
    else if (flag == Right){
       xs=nxt-WWL;
       xe=xs+2+ngsl+WWL; /*nxt+6*/
    }
    if (rank != -1){
        for (k=zs; k<=ze; k++){
           for (i=xs; i < xe; i++){
	      pos   = i*slice_1+j*yline_1+k;

              bpos = (i-xs)*b_slice_1+(j+WWL)*b_yline_1+(k-zs);
              buf[bpos] = u1[pos];

              bpos = (i-xs)*b_slice_1+(j+WWL)*b_yline_1+(k-zs) + b_offset;
              buf[bpos] = v1[pos];

              bpos = (i-xs)*b_slice_1+(j+WWL)*b_yline_1+(k-zs) + 2*b_offset;
              buf[bpos] = w1[pos];

              bpos = (i-xs)*b_slice_1+(j+WWL)*b_yline_1+(k-zs) + 3*b_offset;
              buf[bpos] = xx[pos];

              bpos = (i-xs)*b_slice_1+(j+WWL)*b_yline_1+(k-zs) + 4*b_offset;
              buf[bpos] = yy[pos];

              bpos = (i-xs)*b_slice_1+(j+WWL)*b_yline_1+(k-zs) + 5*b_offset;
              buf[bpos] = zz[pos];

              bpos = (i-xs)*b_slice_1+(j+WWL)*b_yline_1+(k-zs) + 6*b_offset;
              buf[bpos] = xy[pos];

              bpos = (i-xs)*b_slice_1+(j+WWL)*b_yline_1+(k-zs) + 7*b_offset;
              buf[bpos] = xz[pos];
         
              bpos = (i-xs)*b_slice_1+(j+WWL)*b_yline_1+(k-zs) + 8*b_offset;
              buf[bpos] = yz[pos];

              /*if (((flag == Right) && (i==37)) && ((j==36) && (k==64))){
                 cuPrintf("swap send: buf[%d] = %.16g, yz[%d] = %.16g\n", bpos, buf[bpos], pos, yz[pos]);
              }*/

           }
        }
    }
    return;
}

extern "C"
void update_swapzone_data_x_H(float* u1, float* v1, float* w1, float* xx, float* yy, float* zz, _prec *xy, _prec *xz, _prec *yz, 
   _prec *buf_L, _prec *buf_R, int nyt, cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R, int zs, int ze, int d_i) {
     if(rank_L==-1 && rank_R==-1) return;

     dim3 block (1, BLOCK_SIZE_Y, 1);
     dim3 grid (1, (nyt+4+ngsl2+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,1);
     /*cudaPrintfInit();*/
     CUCHK(cudaFuncSetCacheConfig(update_swapzone_buffer_x, cudaFuncCachePreferL1));
     update_swapzone_data_x<<<grid, block, 0, St1>>>(u1, v1, w1, xx, yy, zz, xy, xz, yz, buf_L, rank_L, Left, zs, ze, d_i);
     CUCHK( cudaGetLastError() );
     update_swapzone_data_x<<<grid, block, 0, St2>>>(u1, v1, w1, xx, yy, zz, xy, xz, yz, buf_R, rank_R, Right, zs, ze, d_i);
     CUCHK( cudaGetLastError() );
     /*cudaPrintfDisplay(stdout, 1);
     cudaPrintfEnd();*/
     return;
}

/* copy exchanged buffer data back to swap zone*/
__global__ void update_swapzone_data_x(float* u1, float* v1, float* w1, _prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
   _prec *buf, int rank, int flag, int zs, int ze, int d_i)
{
    register int i, j, k, pos, bpos;
    register long int b_offset;
    register int b_slice_1, b_yline_1;
    register int xs, xe, xoff; /*ys=-2*/;
    register int nxt, nyt, slice_1, yline_1;

    nyt = d_nyt[d_i];
    nxt = d_nxt[d_i];
    slice_1 = d_slice_1[d_i];
    yline_1 = d_yline_1[d_i];

    b_slice_1  = (nyt+4+ngsl2+2*WWL)*(ze-zs+1);
    b_yline_1  = ze-zs+1;
    b_offset   = (2+ngsl+WWL)*b_slice_1;
     
    //j     = blockIdx.y*BLOCK_SIZE_Y+threadIdx.y+2+ngsl;
    j     = blockIdx.y*BLOCK_SIZE_Y+threadIdx.y;

    if(flag==Left){
       xs=0;
       xe=2+ngsl;
       xoff=WWL;
    }
    else if (flag == Right){
       xs=nxt+2+ngsl;
       xe=nxt+4+ngsl2;
       xoff=0;
    }
    if (rank != -1){
        for (k=zs; k<=ze; k++){
           for (i=xs; i < xe; i++){
	      pos   = i*slice_1+j*yline_1+k;

              bpos = (i-xs+xoff)*b_slice_1+(j+WWL)*b_yline_1+(k-zs);
              u1[pos] = buf[bpos];

              bpos = (i-xs+xoff)*b_slice_1+(j+WWL)*b_yline_1+(k-zs) + b_offset;
              v1[pos] = buf[bpos];

              bpos = (i-xs+xoff)*b_slice_1+(j+WWL)*b_yline_1+(k-zs) + 2*b_offset;
              w1[pos] = buf[bpos];

              bpos = (i-xs+xoff)*b_slice_1+(j+WWL)*b_yline_1+(k-zs) + 3*b_offset;
              xx[pos] = buf[bpos];

              bpos = (i-xs+xoff)*b_slice_1+(j+WWL)*b_yline_1+(k-zs) + 4*b_offset;
              yy[pos] = buf[bpos];

              bpos = (i-xs+xoff)*b_slice_1+(j+WWL)*b_yline_1+(k-zs) + 5*b_offset;
              zz[pos] = buf[bpos];

              bpos = (i-xs+xoff)*b_slice_1+(j+WWL)*b_yline_1+(k-zs) + 6*b_offset;
              xy[pos] = buf[bpos];

              bpos = (i-xs+xoff)*b_slice_1+(j+WWL)*b_yline_1+(k-zs) + 7*b_offset;
              xz[pos] = buf[bpos];

              bpos = (i-xs+xoff)*b_slice_1+(j+WWL)*b_yline_1+(k-zs) + 8*b_offset;
              yz[pos] = buf[bpos];

              /*if (((flag == Left) && (i==5)) && ((j==36) && (k==64))){
                 cuPrintf("swap recv: buf[%d] = %.16g, yz[%d] = %.16g\n", bpos, buf[bpos], pos, yz[pos]);
              }*/

           }
        }
    }
    return;
}

extern "C"
void update_swapzone_buffer_y_H(float* u1, float* v1, float* w1, float* xx, float* yy, float* zz, _prec *xy, _prec *xz, _prec *yz, 
   _prec *buf_F, _prec *buf_B, int nxt, cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, int zs, int ze, int d_i) {
     if(rank_F==-1 && rank_B==-1) return;

     dim3 block (BLOCK_SIZE_X, 1, 1);
     dim3 grid ((nxt+BLOCK_SIZE_X-1)/BLOCK_SIZE_X, 1,1);
     /*cudaPrintfInit();*/
     CUCHK(cudaFuncSetCacheConfig(update_swapzone_buffer_y, cudaFuncCachePreferL1));
     update_swapzone_buffer_y<<<grid, block, 0, St1>>>(u1, v1, w1, xx, yy, zz, xy, xz, yz, buf_F, rank_F, Front, zs, ze, d_i);
     CUCHK( cudaGetLastError() );
     update_swapzone_buffer_y<<<grid, block, 0, St2>>>(u1, v1, w1, xx, yy, zz, xy, xz, yz, buf_B, rank_B, Back, zs, ze, d_i);
     CUCHK( cudaGetLastError() );
     /*cudaPrintfDisplay(stdout, 1);
     cudaPrintfEnd();*/
     return;
}

/* buffer exchanged for the swap area along Y*/
__global__ void update_swapzone_buffer_y(float* u1, float* v1, float* w1, _prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
   _prec *buf, int rank, int flag, int zs, int ze, int d_i)
{
    register int i, j, k, pos, bpos;
    register long int b_offset;
    register int b_slice_1, b_yline_1;
    register int ys, ye;
    register int nxt, nyt, slice_1, yline_1;

    nyt = d_nyt[d_i];
    nxt = d_nxt[d_i];
    slice_1 = d_slice_1[d_i];
    yline_1 = d_yline_1[d_i];

    b_slice_1  = (2+ngsl+WWL)*(ze-zs+1);
    b_yline_1  = ze-zs+1;
    b_offset   = (nxt+4+ngsl2)*b_slice_1;
     
    i     = blockIdx.x*BLOCK_SIZE_X+threadIdx.x+2+ngsl;

    if(flag==Front){
       ys=2+ngsl;
       ye=ys+2+ngsl+WWL; //14 if WWL=2
    }
    else if (flag == Back){
       ys=nyt-WWL;
       ye=ys+2+ngsl+WWL; //nyt+6
    }
    if (rank != -1){
        for (k=zs; k<=ze; k++){
           for (j=ys; j < ye; j++){
	      pos   = i*slice_1+j*yline_1+k;

              bpos = i*b_slice_1+(j-ys)*b_yline_1+(k-zs);
              buf[bpos] = u1[pos];

              bpos = i*b_slice_1+(j-ys)*b_yline_1+(k-zs) + b_offset;
              buf[bpos] = v1[pos];

              bpos = i*b_slice_1+(j-ys)*b_yline_1+(k-zs) + 2*b_offset;
              buf[bpos] = w1[pos];

              bpos = i*b_slice_1+(j-ys)*b_yline_1+(k-zs) + 3*b_offset;
              buf[bpos] = xx[pos];

              bpos = i*b_slice_1+(j-ys)*b_yline_1+(k-zs) + 4*b_offset;
              buf[bpos] = yy[pos];

              bpos = i*b_slice_1+(j-ys)*b_yline_1+(k-zs) + 5*b_offset;
              buf[bpos] = zz[pos];

              bpos = i*b_slice_1+(j-ys)*b_yline_1+(k-zs) + 6*b_offset;
              buf[bpos] = xy[pos];

              bpos = i*b_slice_1+(j-ys)*b_yline_1+(k-zs) + 7*b_offset;
              buf[bpos] = xz[pos];
         
              bpos = i*b_slice_1+(j-ys)*b_yline_1+(k-zs) + 8*b_offset;
              buf[bpos] = yz[pos];

              /*if (((flag == Back) && (i==96)) && ((j==97) && (k==3))){
                 cuPrintf("swap send: buf[%d] = %.16g, yz[%d] = %.16g\n", bpos, buf[bpos], pos, yz[pos]);
              }*/

           }
        }
    }
    return;
}

extern "C"
void update_swapzone_data_y_H(float* u1, float* v1, float* w1, float* xx, float* yy, float* zz, _prec *xy, _prec *xz, _prec *yz, 
   _prec *buf_F, _prec *buf_B, int nxt, cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, int zs, int ze, int d_i) {
     if(rank_F==-1 && rank_B==-1) return;

     dim3 block (BLOCK_SIZE_X, 1, 1);
     dim3 grid ((nxt+BLOCK_SIZE_X-1)/BLOCK_SIZE_X, 1,1);
     /*cudaPrintfInit();*/
     CUCHK(cudaFuncSetCacheConfig(update_swapzone_buffer_y, cudaFuncCachePreferL1));
     update_swapzone_data_y<<<grid, block, 0, St1>>>(u1, v1, w1, xx, yy, zz, xy, xz, yz, buf_F, rank_F, Front, zs, ze, d_i);
     CUCHK( cudaGetLastError() );
     update_swapzone_data_y<<<grid, block, 0, St2>>>(u1, v1, w1, xx, yy, zz, xy, xz, yz, buf_B, rank_B, Back, zs, ze, d_i);
     CUCHK( cudaGetLastError() );
     /*cudaPrintfDisplay(stdout, 1);
     cudaPrintfEnd();*/
     return;
}

/* copy exchanged buffer data back to swap zone*/
__global__ void update_swapzone_data_y(float* u1, float* v1, float* w1, _prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
   _prec *buf, int rank, int flag, int zs, int ze, int d_i)
{
    register int i, j, k, pos, bpos;
    register long int b_offset;
    register int b_slice_1, b_yline_1;
    register int ys, ye, yoff;
    register int nxt, nyt, slice_1, yline_1;

    nyt = d_nyt[d_i];
    nxt = d_nxt[d_i];
    slice_1 = d_slice_1[d_i];
    yline_1 = d_yline_1[d_i];

    b_slice_1  = (2+ngsl+WWL)*(ze-zs+1);
    b_yline_1  = ze-zs+1;
    b_offset   = (nxt+4+ngsl2)*b_slice_1;
     
    i     = blockIdx.x*BLOCK_SIZE_X+threadIdx.x+2+ngsl;

    if(flag==Front){
       ys=0;
       ye=2+ngsl;
       yoff=WWL;
    }
    else if (flag == Back){
       ys=nyt+2+ngsl;
       ye=nyt+4+ngsl2;
       yoff=0;
    }
    if (rank != -1){
        for (k=zs; k<=ze; k++){
           for (j=ys; j < ye; j++){
	      pos   = i*slice_1+j*yline_1+k;

              bpos = i*b_slice_1+(j-ys+yoff)*b_yline_1+(k-zs);
              u1[pos] = buf[bpos];

              bpos = i*b_slice_1+(j-ys+yoff)*b_yline_1+(k-zs) + b_offset;
              v1[pos] = buf[bpos];

              bpos = i*b_slice_1+(j-ys+yoff)*b_yline_1+(k-zs) + 2*b_offset;
              w1[pos] = buf[bpos];

              bpos = i*b_slice_1+(j-ys+yoff)*b_yline_1+(k-zs) + 3*b_offset;
              xx[pos] = buf[bpos];

              bpos = i*b_slice_1+(j-ys+yoff)*b_yline_1+(k-zs) + 4*b_offset;
              yy[pos] = buf[bpos];

              bpos = i*b_slice_1+(j-ys+yoff)*b_yline_1+(k-zs) + 5*b_offset;
              zz[pos] = buf[bpos];

              bpos = i*b_slice_1+(j-ys+yoff)*b_yline_1+(k-zs) + 6*b_offset;
              xy[pos] = buf[bpos];

              bpos = i*b_slice_1+(j-ys+yoff)*b_yline_1+(k-zs) + 7*b_offset;
              xz[pos] = buf[bpos];

              bpos = i*b_slice_1+(j-ys+yoff)*b_yline_1+(k-zs) + 8*b_offset;
              yz[pos] = buf[bpos];

              /*if (((flag == Front) && (i==96)) && ((j==1) && (k==3))){
                 cuPrintf("swap recv: buf[%d] = %.16g, yz[%d] = %.16g\n", bpos, buf[bpos], pos, yz[pos]);
              }*/

           }
        }
    }
    return;
}

extern "C"
void addkinsrc_H(int i,   int dim,    int* psrc,  int npsrc,  cudaStream_t St, float* mu,
              float* axx, float* ayy,    float* azz, float* axz, float* ayz, float* axy,
              float* xx,  float* yy,     float* zz,  float* xy,  float* yz,  float* xz, 
              float* mom, double *srcfilt_d, int d_i)
{
    dim3 grid, block;
    if(npsrc < 256)
    {
       block.x = npsrc;
       grid.x = 1;
    }
    else
    {
       block.x = 256;
       grid.x  = int((npsrc+255)/256);
    }
    cudaError_t cerr;
    cerr=cudaGetLastError();
    if(cerr!=cudaSuccess) printf("CUDA ERROR: addkinsrc before kernel: %s\n",cudaGetErrorString(cerr));
    //cudaPrintfInit();
    addkinsrc_cu<<<grid, block, 0, St>>>(i, dim, psrc, npsrc, mu, axx, ayy, azz, axz, ayz, axy,
                                      xx, yy, zz,  xy,   yz,  xz, mom, srcfilt_d, d_i);
    cerr=cudaGetLastError();
    /*cudaPrintfDisplay(stdout, 1);
    cudaPrintfEnd();*/
    if(cerr!=cudaSuccess) printf("CUDA ERROR: addkinsrc after kernel: %s\n",cudaGetErrorString(cerr));
    return;
}

__device__ void compmt(_prec str, _prec dip, _prec rake, 
        _prec *xx, _prec *yy, _prec *zz, _prec *xz, _prec *yz, _prec *xy ){

      //angles must be provided in rads

      *yy= -(sinf(dip)*cosf(rake)*sinf(2.*str)+ 
           sinf(2.*dip)*sinf(rake)*sinf(str)*sinf(str));

      *xy= sinf(dip)*cosf(rake)*cosf(2.*str)+ 
           0.5*(sinf(2.*dip)*sinf(rake)*sinf(2.*str));

      *yz= (cosf(dip)*cosf(rake)*cosf(str)+ 
           cosf(2.*dip)*sinf(rake)*sinf(str));

      *xx= sinf(dip)*cosf(rake)*sinf(2.*str)- 
           sinf(2.*dip)*sinf(rake)*cosf(str)*cosf(str);

      *xz= (cosf(dip)*cosf(rake)*sinf(str)- 
           cosf(2.*dip)*sinf(rake)*cosf(str));

      *zz= sinf(2.*dip)*sinf(rake);
}

__device__ _prec brune(_prec freq, _prec time){
   register _prec stf, omega;
   omega=freq * M_PI * 2.;
   if (time > 0.)
      stf = powf(omega, 2.) * time * expf(-omega*time);
   else
      stf = 0.;
   return(stf);
}

/* Liu et al. (2006) source time function.  tau = risetime */
__device__ _prec liu(_prec tau, _prec time){
   register _prec tau1, tau2, CN, stf;

   tau1 = 0.13 * tau;
   tau2 = tau-tau1;

   CN=M_PI / (1.4*M_PI*tau1 + 1.2*tau1 + 0.3*M_PI*tau2);

   if (time < tau1)
      stf = CN*(0.7 - 0.7*cosf(M_PI*time/tau1) + 0.6*sinf(0.5*M_PI*time/tau1));
   else if (time < 2*tau1)
      stf = CN*(1.0 - 0.7*cosf(M_PI*time/tau1) + 0.3*cosf(M_PI*(time-tau1)/tau2));
   else if (time < tau)
      stf = CN*(0.3 + 0.3*cosf(M_PI*(time-tau1) / tau2));
   else
      stf = 0.;

   return(stf);
}

/* GP 2010, revised based on Liu et al. (2006) source time function.  tau = risetime */
__device__ _prec gp10(_prec tau, _prec time){
   register _prec tau1, tau2, CN, stf;

   tau1 = 0.13 * tau;
   tau2 = tau-tau1;

   CN=M_PI / (1.5 * M_PI*tau1 + 1.2*tau1 + 0.2 * M_PI * tau2);
   if (time < 0.)
      stf = 0.;
   else if (time < tau1)
      stf = CN*(0.7 - 0.7*cosf(M_PI*time/tau1) + 0.6*sinf(0.5*M_PI*time/tau1));
   else if (time < 2*tau1)
      stf = CN*(1.0 - 0.8*cosf(M_PI*time/tau1) + 0.2*cosf(M_PI*(time-tau1)/tau2));
   else if (time < tau)
      stf = CN*(0.2 + 0.2*cosf(M_PI*(time-tau1) / tau2));
   else
      stf = 0.;

   return(stf);
}

/* 1-D FIR or IIR filter, modeled after scipy implementation of lfilter (Daniel) */
__device__ double lfilter(int order, double *b, double *a, double x, double *d){
   register int n; 
   register double y;

   y = (b[0]*x + d[0]) / a[0];

   for (n=0; n<order-1; n++)
      d[n] = b[n+1]*x - a[n+1]*y + d[n+1];

   d[order-1] = b[order]*x - a[order]*y;

   return y;
}

__global__ void addkinsrc_cu(int i, int dim,    int* psrc,  int npsrc, float* mu,
                          float* axx, float* ayy,    float* azz, float* axz, float* ayz, float* axy,
                          float* xx,  float* yy,     float* zz,  float* xy,  float* yz,  float* xz,  
                          float* mom, double *d_srcfilt_d, int d_i)
{

        register _prec vtst;
        register int idx, idy, idz, j, pos;

        register _prec atime, freq, stf;
        register _prec axxt, ayyt, azzt, axzt, ayzt, axyt; 
        register int stf_type;
        register _prec slip, ruptime, risetime, strike, dip, rake, area;

        register int READ_STEP = 2;
        register double *stff[MAXFILT];
        int n;

        j = blockIdx.x*blockDim.x+threadIdx.x;
        if(j >= npsrc) return;

        // For a kinematic source, the moment-rate is computed at run-time from given subfault parameters,
        // which are stored inside the arrays axx...axz 
        stf_type = (int) axx[j*READ_STEP]; // type of source time function.  1=Brune
        slip = ayy[j*READ_STEP];     // total slip
        ruptime = azz[j*READ_STEP];  // rupture time
        risetime = axz[j*READ_STEP]; // rise time

        atime = i*d_DT;
        
        if (atime > ruptime) {
	   area = ayz[j*READ_STEP];   // subfault area

	   strike = axx[j*READ_STEP+1] / 180. * M_PI;   // strike angle (given in degrees)
	   dip = ayy[j*READ_STEP+1] / 180. * M_PI;      // dip angle
	   rake = azz[j*READ_STEP+1] / 180. * M_PI;     // rake

           compmt(strike, dip, rake, &axxt, &ayyt, &azzt, &axzt, &ayzt, &axyt);
 
	   if (stf_type == 1.0f){
	      freq = 1./risetime;
	      stf = brune(freq, atime - ruptime);
	   }
	   else if (stf_type == 2.0f)
	      stf = liu(risetime, atime - ruptime);
	   else if (stf_type == 3.0f)
        stf = gp10(risetime, atime - ruptime);
     else
	      stf = 0.;               

           if (d_filtorder > 0)
	      stf = (float) lfilter(d_filtorder, d_srcfilt_b, d_srcfilt_a, (double) stf, 
		   d_srcfilt_d+j*(d_filtorder+1));
	   
	   vtst = (float)d_DT/(d_DH[d_i]*d_DH[d_i]*d_DH[d_i]);

	   idx = psrc[j*dim]   + 1 + ngsl;
	   idy = psrc[j*dim+1] + 1 + ngsl;
	   idz = psrc[j*dim+2] + align - 1;
	   pos = idx*d_slice_1[d_i] + idy*d_yline_1[d_i] + idz;

           stf *= slip*area/mu[pos];
           mom[j] += stf * d_DT;

           //cuPrintf("stf: %d %e %e %e %e\n", j, atime, stf, slip, area);
           //cuPrintf("mom: %d %e\n", j, mom[j]);

           stf *= vtst;

           /*if (j == 0)
   	      cuPrintf("addkinsrc_cu: (%d,%d,%d) (%e, %e,%e,%e,%e,%e,%e)\n", idx, idy, idz, 
	         stf, axxt, ayyt, azzt, axzt, ayzt, axyt);
	      cuPrintf("addkinsrc_cu: (%d,%d,%d) (%e, %e, %e m^2, %f m)\n", idx, idy, idz, 
	         stf, 1./mu[pos], area, slip);*/

	   xx[pos] = xx[pos] - stf*axxt;
	   yy[pos] = yy[pos] - stf*ayyt;
	   zz[pos] = zz[pos] - stf*azzt;
	   xz[pos] = xz[pos] - stf*axzt;
	   yz[pos] = yz[pos] - stf*ayzt;
	   xy[pos] = xy[pos] - stf*axyt;

        }

        return;
}

extern "C"
void addplanesrc_H(int i,  int dim,   int NST,  cudaStream_t St,
              _prec *mu, _prec *lambda, int ND, int nxt, int nyt, 
              float* axx, float* ayy,    float* azz,
              float* xx,  float* yy,     float* zz,  float* xy,  float* yz,  float* xz, int d_i){

    dim3 grid, block;
    int nx, ny;

    //dim3 block (BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    //dim3 grid ((nxt+BLOCK_SIZE_X-1)/BLOCK_SIZE_X, (nyt+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,1);

    nx = nxt + 4;
    ny = nyt + 4;

    block.x = BLOCK_SIZE_X;
    grid.x = (nx+BLOCK_SIZE_X-1)/BLOCK_SIZE_X;

    block.y = BLOCK_SIZE_Y;
    grid.y = (ny+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y;

    block.z = 1;
    grid.z = 1;

    //fprintf(stdout, "Inside addplanesrc_H: %d,%d,%d,%d\n", block.x, block.y, grid.x, grid.y);
    cudaError_t cerr;
    cerr=cudaGetLastError();
    if(cerr!=cudaSuccess) printf("CUDA ERROR: addplanesrc before kernel: %s\n",cudaGetErrorString(cerr));
    /*CUCHK(cudaPrintfInit());*/
    addplanesrc_cu<<<grid, block, 0, St>>>(i, dim, NST, mu, lambda, ND, axx, ayy, azz, 
                                      xx, yy, zz, xy, yz, xz, d_i);
    /*CUCHK(cudaPrintfDisplay(stdout, 1));
    cudaPrintfEnd();*/
    cerr=cudaGetLastError();
    if(cerr!=cudaSuccess) printf("CUDA ERROR: addplanesrc after kernel: %s\n",cudaGetErrorString(cerr));
}


__global__ void addplanesrc_cu(int n, int dim,  int NST, float* mu, float* lambda, int ND,
                          float* axx, float* ayy,    float* azz,
                          float* xx,  float* yy,     float* zz,  float* xy,  float* yz,  float* xz,  
                          int d_i)
{
        register int j, i, k, pos;
        register _prec vtst;

        i = blockIdx.x*blockDim.x+threadIdx.x + 4;
        j = blockIdx.y*blockDim.y+threadIdx.y + 4;

        vtst = (float) d_DT/d_DH[d_i];

        k = align + ND + 9; //chosing value consistent with CPU code

        pos = i*d_slice_1[d_i] + j*d_yline_1[d_i] + k;

        //cuPrintf("i=%d,j=%d,k=%d, pos=%d\n", i, j, k, pos);
        //cuPrintf("i=%d,j=%d,k=%d, pos=%d, xz[%d]=%e, axx[%d]=%e\n", i, j, k, pos, pos, xz[pos], n, axx[n]);
	xz[pos] -= vtst*2./mu[pos]*axx[n];
	yz[pos] -= vtst*2./mu[pos]*ayy[n];

        // I'm not entirely sure if that's correct - still to be double-checked 
        xx[pos] -= vtst*2./lambda[pos]*azz[n];
        yy[pos] -= vtst*2./lambda[pos]*azz[n];
        zz[pos] -= vtst*2./(lambda[pos]+2.*mu[pos])*azz[n];

        return;
}

extern "C"
void velbuffer_H(const _prec *u1, const _prec *v1, const _prec *w1, const _prec *neta,
       _prec *Bufx, _prec *Bufy, _prec *Bufz, _prec *Bufeta, int NVE, 
       int nbgx, int nedx, int nskpx, int nbgy, int nedy, int nskpy, int nbgz, int nedz, int nskpz,
       int rec_nxt, int rec_nyt, int rec_nzt, cudaStream_t St, int FOLLOWBATHY, const int* bathy, int d_i){

    dim3 block (BLOCK_SIZE_Y, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 grid ((rec_nxt+BLOCK_SIZE_X-1)/BLOCK_SIZE_X, 
               (rec_nyt+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,
               (rec_nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z);

    cudaFuncSetCacheConfig(velbuffer, cudaFuncCachePreferL1);
    CUCHK(cudaGetLastError());
    velbuffer <<<grid, block, 0, St>>>(u1, v1, w1, neta, Bufx, Bufy, Bufz, Bufeta, NVE, 
         nbgx, nedx, nskpx, nbgy, nedy, nskpy, nbgz, nedz, nskpz, rec_nxt, rec_nyt, FOLLOWBATHY, bathy, d_i);

    cudaError_t cerr;
    CUCHK(cerr=cudaGetLastError());

    if(cerr!=cudaSuccess) printf("CUDA ERROR: velbuffer_H after kernel: %s\n",cudaGetErrorString(cerr));
}

__global__ void velbuffer(const _prec *u1, const _prec *v1, const _prec *w1, const _prec *neta,
       _prec *Bufx, _prec *Bufy, _prec *Bufz, _prec *Bufeta, int NVE, 
       int nbgx, int nedx, int nskpx, int nbgy, int nedy, int nskpy, int nbgz, int nedz, int nskpz,
       int rec_nxt, int rec_nyt, int FOLLOWBATHY, const int *bathy, int d_i)
{
    register int i, j, k, ko;

    int tmpInd, pos, bpos;

    i = 2+ngsl+nbgx + (blockIdx.x*blockDim.x+threadIdx.x) * nskpx;
    j = 2+ngsl+nbgy + (blockIdx.y*blockDim.y+threadIdx.y) * nskpy;
    k = nbgz + (blockIdx.z*blockDim.z+threadIdx.z) * nskpz;

    if (i > 2+ngsl+nedx) return;
    if (j > 2+ngsl+nedy) return;
    if (k > nedz) return;
    //This implementation assumes topography is turned on.
    //Vx and Vy at k=d_nzt[0]+align-1 can be used.
    //Since Vz at k=d_nzt[0]+align-1 is always zero and shoud be avoided. The Vz right below will be output instead.
    register int koxy, koz, posxy, posz, sfcidx;
    if (FOLLOWBATHY && d_i == 0)
    {
    bpos=j*(d_nxt[0]+4+ngsl2)+i;
    sfcidx=bathy[bpos];
    }
    else
    {
    sfcidx=d_nzt[d_i]+align-1;
    }

    koxy = sfcidx - k;
    koz = sfcidx - k - 1;

    posxy = i*d_slice_1[d_i]+j*d_yline_1[d_i]+koxy;
    posz = i*d_slice_1[d_i]+j*d_yline_1[d_i]+koz;

    tmpInd =  (k - nbgz)/nskpz*rec_nxt*rec_nyt +
           (j-2-ngsl-nbgy)/nskpy*rec_nxt +
           (i-2-ngsl-nbgx)/nskpx;
    Bufx[tmpInd] = u1[posxy];
    Bufy[tmpInd] = v1[posxy];
    Bufz[tmpInd] = w1[posz];

    if (NVE == 3) Bufeta[tmpInd] = neta[posz];
   
}





