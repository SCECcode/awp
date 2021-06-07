#ifndef _TOPOGRAPHY_TEST_H
#define _TOPOGRAPHY_TEST_H

#include <cuda.h>
#include <cuda_runtime.h>


#ifdef __cplusplus
extern "C" {
#endif
void topo_test_diffx_H(topo_t *T, _prec *out, const _prec *in);
//This function differs from the previous in that it calls an auto-generated
// compute kernel
void topo_test_cgdiffx_H(topo_t *T, _prec *out, const _prec *in);
void topo_test_diffy_H(topo_t *T, _prec *out, const _prec *in);
void topo_test_diffz_H(topo_t *T, _prec *out, const _prec *in);
// Construct polynomial on velocity grid
void topo_test_poly_H(topo_t *T, _prec *out, const _prec *coef,
                      const _prec *deg, const int *shift);
// Construct polynomial on stress grid
void topo_test_polystr_H(topo_t *T, _prec *out, const _prec *coef,
                      const _prec *deg, const int *shift);
void topo_test_polyzbnd_H(topo_t *T, _prec *out, const _prec *coef,
                       const _prec *deg, const int *shift);
void topo_test_polyf_H(topo_t *T, _prec *out, const _prec *coef,
                       const _prec *deg, const int *shift);
void topo_test_polyzbndf_H(topo_t *T, _prec *out, const _prec *coef,
                       const _prec *deg, const int *shift);
void topo_test_polyb_H(topo_t *T, _prec *out, const _prec *coef, 
                       const _prec *deg, const int *shift);
void topo_test_polyzbndb_H(topo_t *T, _prec *out, const _prec *coef, 
                       const _prec *deg, const int *shift);
void topo_test_polystrzbnd_H(topo_t *T, _prec *out, const _prec *coef,
                          const _prec *deg, const int *shift);
#ifdef __cplusplus
}
#endif

__global__ void dtopo_test_diffx(_prec *xx, const _prec *u1,
                                    const int wi0, const int win,
                                    const int wj0, const int wjn,
                                    const int wk0, const int wkn,
                                    const int ri0, const int rin,
                                    const int rj0, const int rjn,
                                    const int rk0, const int rkn,
                                    const int wline, const int wslice,
                                    const int rline, const int rslice);

__global__ void dtopo_test_diffx_111(_prec *xx, const _prec *u1, 
                                     const int nx, const int ny, const int nz);

__global__ void dtopo_test_diffy(_prec *xx, const _prec *u1,
                                    const int wi0, const int win,
                                    const int wj0, const int wjn,
                                    const int wk0, const int wkn,
                                    const int ri0, const int rin,
                                    const int rj0, const int rjn,
                                    const int rk0, const int rkn,
                                    const int wline, const int wslice,
                                    const int rline, const int rslice);

__global__ void dtopo_test_diffz_111(_prec *xz, const _prec *u1, const int nx, const int ny, const int nz);
__global__ void dtopo_test_diffz_112(_prec *xz, const _prec *u1, const int nx, const int ny, const int nz);

__global__ void dtopo_test_poly(_prec *out, 
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

__global__ void dtopo_test_polyzbnd(_prec *out, 
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

#endif
