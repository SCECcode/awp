#ifndef SOURCE_CUH
#define SOURCE_CUH

#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <interpolation/interpolation.cuh>

#ifdef __cplusplus
extern "C" {
#endif
void cusource_add_cartesian_H(const cu_interp_t *I, prec *out, const prec *in,
                              const prec h, const prec dt);
__global__ void cusource_add_cartesian(prec *out, const prec *in,
                                 const prec *lx, const prec *ly, const prec *lz,
                                 const int num_basis, const int *ix,
                                 const int *iy, const int *iz,
                                 const prec h, const prec dt,
                                 const int num_query, const grid3_t grid);
void cusource_add_curvilinear_H(const cu_interp_t *I, prec *out, const prec *in,
                                const prec h, const prec dt, const prec *f,
                                const int ny, const prec *dg);
__global__ void cusource_add_curvilinear(prec *out, const prec *in,
                                 const prec *lx, const prec *ly, const prec *lz,
                                 const int num_basis, const int *ix,
                                 const int *iy, const int *iz,
                                 const prec h, const prec dt,
                                 const int num_query, const grid3_t grid, 
                                 const prec *f, const int ny, const prec *dg);
#ifdef __cplusplus
}
#endif

#endif

