#ifndef INTERPOLATION_CUH
#define INTERPOLATION_CUH

#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <awp/definitions.h>
#include <grid/grid_3d.h>

/* Interpolation data structure
 *
 * Members:
 * num_basis : Number of basis functions. Polynomial interpolation degree + 1.
 * num_query : Number of query points.
 * lx, ly, lz : Lagrange basis coefficients for each direction. 
 *       Size: num_basis x num_query
 * ix, iy, iz : Index to first grid point for each direction. Size: num_query
 * 
 */
typedef struct
{
        int num_basis;
        int num_query;
        size_t size_l;
        size_t size_i;
        prec *lx, *ly, *lz;
        int *ix, *iy, *iz;
        prec *d_lx, *d_ly, *d_lz;
        int *d_ix, *d_iy, *d_iz;
        grid3_t grid;
} cu_interp_t;

#ifndef INTERP_THREADS
#define INTERP_THREADS 256
#endif


#ifdef __cplusplus
extern "C" {
#endif

/*
 *  Initialize on host. 
 *  Compute Lagrange basis coefficients for each query point.
 *
 *  Arguments:
 *       out: interpolation data structure x: Array of grid points in the
 *              x-direction (size: grid_boundary_size(grid).x)
 *       y: Array of grid points in the y-direction (size:
 *              grid_boundary_size(grid).y)
 *       z: Array of grid points in the z-direction (size:
 *              grid_boundary_size(grid).z)
 *       grid: 3D grid data structure (see grid_3d.h)
 *       qx: Array of size: num_query that contains the x-coordinates of query
 *              points
 *       qy: Array of size: num_query that contains the y-coordinates of query
 *              points
 *       qz: Array of size: num_query that contains the z-coordinates of query
 *              points
 *       num_query: Number of query points
 *       degree: degree of interpolating polynomial
 *       num_buffer: Number of times to append interpolation results before
 *              flushing the data buffer.
 *       num_writes: Number of times to write to disk.  
 *
 *  Return value:
 *       Error code (SUCCESS, ...)
 */  
int cuinterp_init(cu_interp_t *out, const prec *x, const prec *y,
                    const prec *z, grid3_t grid, const prec *qx, const prec *qy,
                    const prec *qz, const int num_query, const int degree);
int cuinterp_lagrange_h(cu_interp_t *host, const prec *x, const prec *y,
                        const prec *z, const grid3_t grid, const prec *qx,
                        const prec *qy, const prec *qz);
int cuinterp_htod(cu_interp_t *interp);
int cuinterp_dtoh(cu_interp_t *interp);
int cuinterp_malloc(cu_interp_t *interp);
void cuinterp_finalize(cu_interp_t *interp);
void cuinterp_interp_H(const cu_interp_t *interp, prec *out, const prec *in);
 
__global__ void cuinterp_dinterp(prec *out, const prec *in,
                                 const prec *lx, const prec *ly, const prec *lz,
                                 const int num_basis, const int *ix,
                                 const int *iy, const int *iz,
                                 const int num_query, const grid3_t grid);

#ifdef __cplusplus
}
#endif
#endif

