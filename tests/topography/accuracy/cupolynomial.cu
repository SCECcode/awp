#include "cupolynomial.cuh"

__global__ void poly_xy(_prec *out, 
                        const int wi0, const int win,
                        const int wj0, const int wjn,
                        const int wk0, const int wkn,
                        const int ri0, const int rin,
                        const int rj0, const int rjn,
                        const int rk0, const int rkn,
                        const int nx, const int ny, const int nz,
                        const int line, const int slice,
                        const int rx, const int ry,
                        const _prec a0, const _prec a1, const _prec a2,
                        const _prec p0, const _prec p1, const _prec p2, 
                        const _prec s0, const _prec s1, const _prec s2)
{
     // Indices used for output
     const int wk = threadIdx.x + blockIdx.x*blockDim.x + wk0;
     if ( wk >= wkn) return;
     const int wj = threadIdx.y + blockIdx.y*blockDim.y + wj0;
     if ( wj >= wjn) return;
     const int wi = threadIdx.z + blockIdx.z*blockDim.z + wi0;
     if ( wi >= win) return;

     // Indices used for input
     const int rk = threadIdx.x + blockIdx.x*blockDim.x + rk0;
     if ( rk >= rkn) return;
     const int rj = threadIdx.y + blockIdx.y*blockDim.y + rj0;
     if ( rj >= rjn) return;
     const int ri = threadIdx.z + blockIdx.z*blockDim.z + ri0;
     if ( ri >= rin) return;
     
     const int pos = wk + wj*line + wi*slice;
     out[pos] = a0*pow(ri + nx*rx - 0.5*s0, p0) 
              + a1*pow(rj + ny*ry - 0.5*s1, p1) 
              + a2*pow(rk         - 0.5*s2, p2);
}


__global__ void poly_z(_prec *out, 
                       const int wi0, const int win,
                       const int wj0, const int wjn,
                       const int wk0, const int wkn,
                       const int ri0, const int rin,
                       const int rj0, const int rjn,
                       const int rk0, const int rkn,
                       const int nx, const int ny, const int nz,
                       const int line, const int slice,
                       const int rx, const int ry,
                       const _prec a0, const _prec a1, const _prec a2,
                       const _prec p0, const _prec p1, const _prec p2, 
                       const _prec s0, const _prec s1, const _prec s2)
{
     // Indices used for output
     const int wk = threadIdx.x + blockIdx.x*blockDim.x + wk0;
     if ( wk >= wkn) return;
     const int wj = threadIdx.y + blockIdx.y*blockDim.y + wj0;
     if ( wj >= wjn) return;
     const int wi = threadIdx.z + blockIdx.z*blockDim.z + wi0;
     if ( wi >= win) return;

     // Indices used for input
     const int rk = threadIdx.x + blockIdx.x*blockDim.x + rk0;
     if ( rk >= rkn) return;
     const int rj = threadIdx.y + blockIdx.y*blockDim.y + rj0;
     if ( rj >= rjn) return;
     const int ri = threadIdx.z + blockIdx.z*blockDim.z + ri0;
     if ( ri >= rin) return;



/*
 *                                       n-4  n-3   n-2  n-1  
 *   z    ------o-----o-|---o-----o--|---o----o-----o---*
 *                      |            |     
 *                      |            |     
 *   zh   ---o-----o----|o-----o-----|^----o-----o--o
 *                      |            |n-4  n-3   n-2 n-1
 *
 *           Bottom           Interior           Top 
 */


     _prec zkp = 0.0; 
     if (rk == rkn - 1 && s2 == 1) {
           zkp = pow(rkn - 2, p2);
     } 
     else if (rk == rk0) {
        zkp = pow(rk, p2);
     }   
     else if (rk == rkn - 1 && s2 == 0) {
           zkp = 0;
     } 
     else {
        zkp = pow(rk- 0.5*s2, p2);
     }
     
     const int pos = wk + wj*line + wi*slice;
     out[pos] = a0*pow(ri + nx*rx - 0.5*s0, p0) 
              + a1*pow(rj + ny*ry - 0.5*s1, p1) 
              + a2*zkp;
}
