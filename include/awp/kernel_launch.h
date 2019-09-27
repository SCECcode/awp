#ifndef _KERNEL_LAUNCH_H
#define _KERNEL_LAUNCH_H
#include <cuda.h>
#include <cuda_runtime.h>
#include "pmcl3d_cons.h"
// Host kernel launch functions
//
#ifdef __cplusplus
extern "C" {
#endif
        void print_nonzero_H(_prec *array, int nx, int ny, int nz, int d_i, int
                        rank);

        void print_nonzero_mat_H(_prec *array, int nx, int ny, int nz, int d_i,
                        _prec *d1, _prec *mu, _prec *lam, _prec *qp, _prec *qs,
                        int rank);

        void print_nan_H(_prec *array, int nx, int ny, int nz, char *vname);

        void print_const_H(int ngrids);

        void SetDeviceConstValue(_prec *DH, _prec DT, int *nxt, int *nyt, int *nzt, int ngrids,
           _prec fmajor, _prec fminor, _prec *Rz, _prec *RzT);
        
        void SetDeviceFilterParameters(int filtorder, double *srcfilt_b, double *srcfilt_a);
        
        void dvelcx_H_opt(_prec*  u1,    _prec*  v1,    _prec*  w1,
                          _prec*  xx,  _prec*  yy, _prec*  zz, _prec*  xy,
                          _prec*  xz, _prec*  yz,
                          _prec*  dcrjx, _prec*  dcrjy, _prec*  dcrjz,
                          _prec*  d_1, int nyt,   int nzt,
                          cudaStream_t St, int s_i,   int e_i, int d_i, int ngrids);
        
        void dvelcy_H(_prec*  u1,       _prec*  v1,    _prec*  w1,    _prec*  xx,  _prec*  yy, _prec*  zz, _prec*  xy,   _prec*  xz,   _prec*  yz,
                      _prec*  dcrjx,    _prec*  dcrjy, _prec*  dcrjz, _prec*  d_1, int nxt,   int nzt,   _prec*  s_u1, _prec*  s_v1, _prec*  s_w1,
                      cudaStream_t St, int s_j,      int e_j,      int rank, int d_i);
        
        void dstrqc_H_new(_prec*  xx,       _prec*  yy,     _prec*  zz,    _prec*  xy,    _prec*  xz, _prec*  yz,
                          _prec*  r1,       _prec*  r2,     _prec*  r3,    _prec*  r4,    _prec*  r5, _prec*  r6,
                          _prec*  u1,       _prec*  v1,     _prec*  w1,    _prec*  lam,   _prec*  mu, _prec*  qp,_prec*  coeff,
                          _prec*  qs,       _prec*  dcrjx,  _prec*  dcrjy, _prec*  dcrjz, int nyt,   int nzt,
                          cudaStream_t St, _prec*  lam_mu,
                          _prec *vx1, _prec *vx2, int *ww, _prec *wwo,
                          int NX,          int NPC,       int rankx,    int ranky, int  s_i,
                          int e_i,         int s_j,       int e_j, int d_i);
        
        void fstr_H(_prec*  zz, _prec*  xz, _prec*  yz, cudaStream_t St, int s_i, int e_i, int s_j, int e_j);
        
        void drprecpc_calc_H_opt(_prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
                _prec *mu, _prec *d1, _prec *sigma2, 
                _prec *yldfac,_prec *cohes, _prec *phi,
                _prec *neta,
                int nzt,
                int xls, int xre, int yls, int yre, cudaStream_t St, int d_i);
        
        void drprecpc_app_H(_prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
                _prec *mu, 
                _prec *sigma2, _prec *yldfac, 
                int nzt, int xls, int xre, int yls, int yre, cudaStream_t St, int d_i);
        
        void addsrc_H(int i,      int READ_STEP, int dim,    int* psrc,  int npsrc,  cudaStream_t St,
                      _prec*  axx, _prec*  ayy,    _prec*  azz, _prec*  axz, _prec*  ayz, _prec*  axy,
                      _prec*  xx,  _prec*  yy,     _prec*  zz,  _prec*  xy,  _prec*  yz,  _prec*  xz, int d_i);
        
        void frcvel_H(int i,      int READ_STEP, int dim,    int* psrc,  int npsrc,  int tskp, cudaStream_t St,
                      _prec*  axx, _prec*  ayy,    _prec*  azz, _prec*  axz, _prec*  ayz, _prec*  axy,
                      _prec*  u1,  _prec*  v1,     _prec*  w1, int ymin, int ymax, int d_i);
        
        void fvel_H(_prec*  u1, _prec*  v1, _prec*  w1, cudaStream_t St, _prec*  lam_mu, int NX, int rankx, int ranky, 
             int s_i, int e_i, int s_j, int e_j);
        
        void update_yldfac_buffer_y_H(_prec*  yldfac, _prec *buf_F, _prec *buf_B, int nxt, int nzt,
           cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, int d_i);
        
        void update_yldfac_buffer_x_H(_prec*  yldfac, _prec *buf_L, _prec *buf_R, int nyt, int nzt, cudaStream_t St1, 
             cudaStream_t St2, int rank_L, int rank_R, int d_i);
        
        void dvelc2_H(_prec*  u1,    _prec*  v1,    _prec*  w1,    _prec*  xx,  _prec*  yy, _prec*  zz, _prec*  xy, _prec*  xz, _prec*  yz,
                     _prec*  dcrjx, _prec*  dcrjy, _prec*  dcrjz, _prec*  d_1, int nxt, int nyt, cudaStream_t St, int d_i);

        void dstrqc_H(float* xx,       float* yy,     float* zz,    float* xy,    float* xz, float* yz,
              float* r1,       float* r2,     float* r3,    float* r4,    float* r5, float* r6,
              float* u1,       float* v1,     float* w1,    float* lam,   float* mu, float* qp,float* coeff, 
              float* qs,       float* dcrjx,  float* dcrjy, float* dcrjz, int nyt,   int nzt, 
              cudaStream_t St, float* lam_mu, 
              float *vx1, float *vx2, int *ww, float *wwo,
              int NX,       int NPC, int rankx,    int ranky, int  s_i,  
              int e_i,         int s_j,       int e_j, int d_i);
        
        void dstrqc2_H(_prec*  xx, _prec*  yy,    _prec*  zz,    _prec*  xy,    _prec*  xz,  _prec*  yz,
                      _prec*  r1, _prec*  r2,    _prec*  r3,    _prec*  r4,    _prec*  r5,  _prec*  r6,
                      _prec*  u1, _prec*  v1,    _prec*  w1,    _prec*  lam,   _prec*  mu,  _prec*  qp,
                      _prec*  qs, _prec*  dcrjx, _prec*  dcrjy, _prec*  dcrjz, int nxt,    int nyt, 
                      cudaStream_t St, _prec*  coeff, _prec *d_vx1, _prec *d_vx2, int *d_ww, 
                      _prec *d_wwo, int s_i, int e_i, int s_j, int e_j, int d_i);
        
        void intp3d_H(_prec *u1l, _prec*  v1l, _prec *w1l, _prec *xxl, _prec *yyl, _prec *zzl, 
                _prec *xyl, _prec * xzl, _prec*  yzl,
                _prec *u1h, _prec *v1h, _prec*  w1h, _prec *xxh, _prec *yyh, _prec *zzh, 
                _prec *xyh, _prec *xzh, _prec*  yzh,
                int nxtl, int nytl, int rank, cudaStream_t St, int d_i);
        
        void swap_H(_prec * xxl, _prec*  yyl, _prec*  zzl, _prec*  xyl, _prec*  xzl, _prec*  yzl,_prec*  u1l, _prec*  v1l,_prec*  w1l,
        		  _prec * xxh, _prec*  yyh, _prec*  zzh, _prec*  xyh, _prec*  xzh, _prec*  yzh,_prec*  u1h, _prec*  v1h, _prec*  w1h,
        		  int nxtl, int nytl, _prec *buf_L, _prec *buf_R, _prec *buf_F, _prec *buf_B, int rank, 
                          cudaStream_t St, int d_i);
        
        void swaparea_update_corners(_prec *SL_swap, _prec *SR_swap, _prec *RF_swap, _prec *RB_swap, int nz, int off,
                 int nxt, int nyt);
        
        void addkinsrc_H(int i,   int dim,    int* psrc,  int npsrc,  cudaStream_t St, _prec*  mu,
                      _prec*  axx, _prec*  ayy,    _prec*  azz, _prec*  axz, _prec*  ayz, _prec*  axy,
                      _prec*  xx,  _prec*  yy,     _prec*  zz,  _prec*  xy,  _prec*  yz,  _prec*  xz, 
                      _prec *mom, double *d_srcfilt_d, int d_i);
        
        void addplanesrc_H(int i,  int dim,   int NST,  cudaStream_t St,
                      _prec *mu, _prec *lambda, int ND, int nxt, int nyt,
                      _prec*  axx, _prec*  ayy,    _prec*  azz,
                      _prec*  xx,  _prec*  yy,     _prec*  zz,  _prec*  xy,  _prec*  yz,  _prec*  xz, int d_i);

        void update_bound_y_H(_prec*  u1,   _prec*  v1, _prec*  w1, _prec*  f_u1,      _prec*  f_v1,      _prec*  f_w1, _prec*  b_u1, _prec*  b_v1,
                      _prec*  b_w1, int nxt,   int nzt,   cudaStream_t St1, cudaStream_t St2, int rank_f,  int rank_b, int d_i);

        void update_yldfac_buffer_x_H(_prec*  yldfac, _prec *buf_L, _prec *buf_R, int nyt, int nzt, cudaStream_t St1, cudaStream_t St2, 
             int rank_L, int rank_R, int d_i);
        
        void update_yldfac_data_x_H(_prec*  yldfac, _prec *buf_L, _prec *buf_R, int nyt, int nzt, cudaStream_t St1, cudaStream_t St2, 
             int rank_L, int rank_R, int d_i);
        
        void update_yldfac_buffer_y_H(_prec*  yldfac, _prec *buf_F, _prec *buf_B, int nxt, int nzt,
           cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, int d_i);
        
        void update_yldfac_data_y_H(_prec*  yldfac, _prec *buf_F, _prec *buf_B, int nxt, int nzt,
            cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, int d_i);
        
        void update_swapzone_buffer_x_H(_prec*  u1, _prec*  v1, _prec*  w1, _prec*  xx, _prec*  yy, _prec*  zz, _prec *xy, _prec *xz, _prec *yz, 
           _prec *buf_L, _prec *buf_R, int nyt, cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R, int zs, int ze, int meshtp);
        
        void update_swapzone_data_x_H(_prec*  u1, _prec*  v1, _prec*  w1, _prec*  xx, _prec*  yy, _prec*  zz, _prec *xy, _prec *xz, _prec *yz, 
           _prec *buf_L, _prec *buf_R, int nyt, cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R, int zs, int ze, int meshtp);

        void update_swapzone_buffer_y_H(_prec*  u1, _prec*  v1, _prec*  w1, _prec*  xx, _prec*  yy, _prec*  zz, _prec *xy, _prec *xz, _prec *yz, 
   _prec *buf_F, _prec *buf_B, int nxt, cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, int zs, int ze, int d_i);

        void update_swapzone_data_y_H(_prec*  u1, _prec*  v1, _prec*  w1, _prec*  xx, _prec*  yy, _prec*  zz, _prec *xy, _prec *xz, _prec *yz, 
   _prec *buf_F, _prec *buf_B, int nxt, cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, int zs, int ze, int d_i);
        
#ifdef __cplusplus
}
#endif
#endif// _KERNEL_HOST_H

