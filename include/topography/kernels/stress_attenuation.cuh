#ifndef TOPO_STRESS_H
#define TOPO_STRESS_H
#include <awp/definitions.h>
#include <math.h>

void set_constants(const _prec dh, const _prec dt, const int nxt, const int
                nyt, const int nzt);

__global__ void dtopo_str_111(_prec*  __restrict__ xx, _prec*  __restrict__ yy, _prec*  __restrict__ zz,
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
       int NX, int ny, int nz, int rankx, int ranky, 
       int nzt, int s_i, int e_i, int s_j, int e_j);

#endif
