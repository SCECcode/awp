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
                                 const int *lidx,
                                 const prec h, const prec dt,
                                 const int num_query, const grid3_t grid);
void cusource_add_curvilinear_H(const cu_interp_t *I, prec *out, const prec *in,
                                const prec h, const prec dt, const prec *f,
                                const int ny, const prec *dg, const int zhat);
__global__ void cusource_add_curvilinear(prec *out, const prec *in,
                                 const prec *lx, const prec *ly, const prec *lz,
                                 const int num_basis, const int *ix,
                                 const int *iy, const int *iz,
                                 const int *lidx,
                                 const prec h, const prec dt,
                                 const int num_query, const grid3_t grid, 
                                 const prec *f, const int ny, const prec *dg, const int zhat);
void cusource_add_force_H(const cu_interp_t *I, prec *out, const prec *in,
                          const prec *d1, const prec h, const prec dt,
                          const prec quad_weight,
                          const prec *f, const int nx, const int ny,
                          const int nz, const prec *dg, const int sourcetype, const int dir);
__global__ void cusource_add_force(prec *out, const prec *in, const prec *d1,
                                   const prec *lx, const prec *ly,
                                   const prec *lz, const int num_basis,
                                   const int *ix, const int *iy, const int *iz,
                                   const int *lidx, const prec h, const prec dt,
                                   const prec quad_weight,
                                   const int num_query, const grid3_t grid,
                                   const prec *f, const int nx, const int ny,
                                   const int nz, const prec *dg);

__global__ void cusource_add_force_stress(prec *out, const prec *in, const prec *d1,
                                   const prec *lx, const prec *ly,
                                   const prec *lz, const int num_basis,
                                   const int *ix, const int *iy, const int *iz,
                                   const int *lidx, const prec h, const prec dt,
                                   const prec quad_weight,
                                   const int num_query, const grid3_t grid,
                                   const prec *f, const int nx, const int ny,
                                   const int nz, const prec *dg, const int dir);

__global__ void cusource_add_force_velocity(prec *out, const prec *in, const prec *d1,
                                   const prec *lx, const prec *ly,
                                   const prec *lz, const int num_basis,
                                   const int *ix, const int *iy, const int *iz,
                                   const int *lidx, const prec h, const prec dt,
                                   const prec quad_weight,
                                   const int num_query, const grid3_t grid,
                                   const prec *f, const int nx, const int ny,
                                   const int nz, const prec *dg, const int dir);
#ifdef __cplusplus
}
#endif

#endif

