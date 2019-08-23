#ifndef _KERNEL_H
#define _KERNEL_H

#include "kernel_launch.h"

__global__ void dvelcx(_prec*  u1,    _prec*  v1,    _prec*  w1,    _prec*  xx,  _prec*  yy, _prec*  zz, _prec*  xy, _prec*  xz, _prec*  yz,
                       _prec*  dcrjx, _prec*  dcrjy, _prec*  dcrjz, _prec*  d_1, int s_i,   int e_i, int d_i);

__global__ void dvelcy(_prec*  u1,    _prec*  v1,    _prec*  w1,    _prec*  xx,  _prec*  yy,   _prec*  zz,   _prec*  xy, _prec*  xz, _prec*  yz,
                       _prec*  dcrjx, _prec*  dcrjy, _prec*  dcrjz, _prec*  d_1, _prec*  s_u1, _prec*  s_v1, _prec*  s_w1, int s_j, int e_j, int d_i);

__global__ void update_boundary_y(_prec*  u1, _prec*  v1, _prec*  w1, _prec*  s_u1, _prec*  s_v1, _prec*  s_w1, int rank, int flag, int d_i);

__global__ void fvelxyz(_prec*  u1, _prec*  v1, _prec*  w1, _prec*  lam_mu, int xls, int NX, int rankx, int d_i);

__global__ void dstrqc(_prec*  xx, _prec*  yy,    _prec*  zz,    _prec*  xy,    _prec*  xz,     _prec*  yz,
                       _prec*  r1, _prec*  r2,    _prec*  r3,    _prec*  r4,    _prec*  r5,     _prec*  r6,
                       _prec*  u1, _prec*  v1,    _prec*  w1,    _prec*  lam,   _prec*  mu,     _prec*  qp, _prec*  coeff, 
                       _prec*  qs, _prec*  dcrjx, _prec*  dcrjy, _prec*  dcrjz, _prec*  lam_mu, int NX,
                       int rankx, int ranky,    int s_i,      int e_i,      int s_j, int d_i);

__global__ void fstr (_prec*  zz, _prec*  xz, _prec*  yz, int s_i, int e_i, int s_j);

__global__ void drprecpc_calc(_prec *xx, _prec *yy, _prec *zz, 
      _prec *xy, _prec *xz, _prec *yz, _prec *mu, _prec *d1, 
      _prec *sigma2, _prec *yldfac,_prec *cohes, _prec *phi, _prec *neta,
      int s_i,      int e_i,      int s_j, int d_i);

__global__ void drprecpc_app(_prec *xx, _prec *yy, _prec *zz, 
      _prec *xy, _prec *xz, _prec *yz, _prec *mu, 
      _prec *sigma2, _prec *yldfac, 
      int s_i,      int e_i,      int s_j, int d_i);

__global__ void addsrc_cu(int i,      int READ_STEP, int dim,    int* psrc, int npsrc,
                          _prec*  axx, _prec*  ayy,    _prec*  azz, _prec*  axz, _prec*  ayz, _prec*  axy,
                          _prec*  xx,  _prec*  yy,     _prec*  zz,  _prec*  xy,  _prec*  yz,  _prec*  xz, int d_i);

__global__ void frcvel_cu(int i,      int READ_STEP, int dim,    int* psrc, int npsrc, int tskp, 
                          _prec*  axx, _prec*  ayy,    _prec*  azz, _prec*  axz, _prec*  ayz, _prec*  axy,
                          _prec*  u1,  _prec*  v1,     _prec*  w1, int xmin, int xmax, int d_i);

__global__ void fvel (_prec*  u1, _prec*  v1, _prec*  w1, _prec*  lam_mu, int NX, int rankx, int ranky, int s_i, int e_i, int s_j);

template <int BLOCK_Z, int BLOCK_Y>
__global__ void dvelcx_opt(_prec * __restrict__ u1,
                           _prec * __restrict__ v1, 
                           _prec * __restrict__ w1,
                           const _prec *xx,    const _prec *yy,    const _prec *zz,
                           const _prec *xy,    const _prec *xz,    const _prec *yz, 
                           const _prec *dcrjx, const _prec *dcrjy, const _prec *dcrjz,
                           const _prec *d_1,   
                           const int s_i,
                           const int e_i, const int d_i, const int ngrids);

template<int BLOCKX, int BLOCKY>
__global__ void 
__launch_bounds__(512,2)
dstrqc_new(_prec*  __restrict__ xx, _prec*  __restrict__ yy, _prec*  __restrict__ zz,
           _prec*  __restrict__ xy, _prec*  __restrict__ xz, _prec*  __restrict__ yz,
       _prec*  __restrict__ r1, _prec*  __restrict__ r2,  _prec*  __restrict__ r3, 
       _prec*  __restrict__ r4, _prec*  __restrict__ r5,  _prec*  __restrict__ r6,
       _prec*  __restrict__ u1, 
       _prec*  __restrict__ v1,    
       _prec*  __restrict__ w1,    
       _prec*  lam,   
       _prec*  mu,     
       _prec*  qp,
       _prec*  coeff, 
       _prec*  qs, 
       _prec*  dcrjx, _prec*  dcrjy, _prec*  dcrjz, _prec*  lam_mu, 
       //_prec *d_vx1, _prec *d_vx2, _prec *d_ww, _prec *d_wwo, //pengs version
       _prec *d_vx1, _prec *d_vx2, int *d_ww, _prec *d_wwo,
       int NX, int NPC, int rankx, int ranky, int nzt,   int s_i,      int e_i,      int s_j, int e_j, int d_i);

__global__ void 
__launch_bounds__(512,2)
drprecpc_calc_opt(_prec *xx, _prec *yy, _prec *zz, 
                  const _prec*  __restrict__ xy, 
                  const _prec*  __restrict__ xz, 
                  const _prec*  __restrict__ yz, 
                  _prec *mu, _prec *d1, 
                  _prec *sigma2, 
                  _prec *yldfac,_prec *cohes, _prec *phi,
                  _prec *neta,
                  int nzt, int s_i,      int e_i,      int s_j, int e_j, int d_i);

__global__ void update_yldfac_buffer_x(_prec*  yldfac, _prec *buf, int rank, int flag, int d_i);

__global__ void update_yldfac_data_x(_prec*  yldfac, _prec *buf, int rank, int flag, int d_i);

__global__ void update_yldfac_buffer_y(_prec*  yldfac, _prec *buf, int rank, int flag, int d_i);

__global__ void update_yldfac_data_y(_prec*  yldfac, _prec *buf, int rank, int flag, int d_i);

__global__ void update_swapzone_buffer_x(_prec*  u1, _prec*  v1, _prec*  w1, _prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
   _prec *buf, int rank, int flag, int zs, int ze, int meshtp);

__global__ void update_swapzone_data_x(_prec*  u1, _prec*  v1, _prec*  w1, _prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
   _prec *buf, int rank, int flag, int zs, int ze, int meshtp);

__global__ void update_swapzone_buffer_y(_prec*  u1, _prec*  v1, _prec*  w1, _prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
   _prec *buf, int rank, int flag, int zs, int ze, int meshtp);

__global__ void update_swapzone_data_y(_prec*  u1, _prec*  v1, _prec*  w1, _prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
   _prec *buf, int rank, int flag, int zs, int ze, int meshtp);

__global__ void addkinsrc_cu(int i, int dim,    int* psrc,  int npsrc, _prec*  mu,
                          _prec*  axx, _prec*  ayy,    _prec*  azz, _prec*  axz, _prec*  ayz, _prec*  axy,
                          _prec*  xx,  _prec*  yy,     _prec*  zz,  _prec*  xy,  _prec*  yz,  _prec*  xz, 
                          _prec *mom, double *d_srcfilt_d, int d_i);

__global__ void addplanesrc_cu(int i, int dim,  int NST, _prec*  mu, _prec*  lambda, int ND,
                          _prec*  axx, _prec*  ayy,    _prec*  azz,
                          _prec*  xx,  _prec*  yy,     _prec*  zz,  _prec*  xy,  _prec*  yz,  _prec*  xz,  
                          int d_i);
#endif

