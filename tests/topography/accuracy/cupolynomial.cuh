#ifndef _POLYNOMIAL_H
#define _POLYNOMIAL_H
#endif

#include <awp/definitions.h>

#ifdef __cplusplus
extern "C" {
#endif
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
                                const _prec s0, const _prec s1, const _prec s2);

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
                       const _prec s0, const _prec s1, const _prec s2);
#ifdef __cplusplus
}
#endif
