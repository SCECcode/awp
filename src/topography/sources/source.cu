#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <assert.h>

#include <topography/sources/source.cuh>
#include <interpolation/interpolation.cuh>
#include <test/test.h>


// Enable or disable atomic operations. If the sources are overlapping, disabling atomics causes
// parallel synchronization issues. Only disable this macro if you know that the sources are
// non-overlapping.
#define USE_ATOMICS 1

void cusource_add_cartesian_H(const cu_interp_t *I, prec *out, const prec *in,
                              const prec h, const prec dt)
{
        dim3 block (INTERP_THREADS, 1, 1);
        dim3 grid((I->num_query + INTERP_THREADS - 1) / INTERP_THREADS,
                  1, 1);

        cusource_add_cartesian<<<grid, block>>>(
            out, in, I->d_lx, I->d_ly, I->d_lz, I->num_basis, I->d_ix, I->d_iy,
            I->d_iz, I->d_ridx, h, dt, I->num_query, I->grid);
        CUCHK(cudaGetLastError());
}

__global__ void cusource_add_cartesian(prec *out, const prec *in,
                                 const prec *lx, const prec *ly, const prec *lz,
                                 const int num_basis, const int *ix,
                                 const int *iy, const int *iz,
                                 const int *lidx,
                                 const prec h, const prec dt,
                                 const int num_query, const grid3_t grid)
{
        int q = threadIdx.x + blockDim.x * blockIdx.x;
        if (q >= num_query) {
                return;
        }

        prec dth = dt/(h * h * h);

        for (int i = 0; i < num_basis; ++i) {
        for (int j = 0; j < num_basis; ++j) {
        for (int k = 0; k < num_basis; ++k) {
                // Do not apply stencil at halo points
                if ( ix[q] + i >= 2 + nx + 3 * ngsl / 2 || ix[q] + i < 2 + ngsl / 2 ||
                     iy[q] + j >= 2 + ny + 3 * ngsl / 2 || iy[q] + j < 2 + ngsl / 2 ) continue;
                size_t pos = grid_index(grid, ix[q] + i, iy[q] + j, iz[q] + k);
                prec value = - dth * lx[q * num_basis + i] *
                            ly[q * num_basis + j] * lz[q * num_basis + k] *
                            in[lidx[q]];
#if USE_ATOMICS
                atomicAdd(&out[pos], value);
#else 
                out[pos] = value;
#endif


        }
        }
        }
}

void cusource_add_curvilinear_H(const cu_interp_t *I, prec *out, const prec *in,
                                const prec h, const prec dt, const prec *f,
                                const int ny, const prec *dg, const int zhat) 
{
        dim3 block (INTERP_THREADS, 1, 1);
        dim3 grid((I->num_query + INTERP_THREADS - 1) / INTERP_THREADS,
                  1, 1);

        cusource_add_curvilinear<<<grid, block>>>(
            out, in, I->d_lx, I->d_ly, I->d_lz, I->num_basis, I->d_ix, I->d_iy,
            I->d_iz, I->d_ridx, h, dt, I->num_query, I->grid, f, ny, dg, zhat);
        CUCHK(cudaGetLastError());
}

__global__ void cusource_add_curvilinear(prec *out, const prec *in,
                                 const prec *lx, const prec *ly, const prec *lz,
                                 const int num_basis, const int *ix,
                                 const int *iy, const int *iz,
                                 const int *lidx,
                                 const prec h, const prec dt,
                                 const int num_query, const grid3_t grid,
                                 const prec *f, const int ny, const prec *dg, const int zhat)
{
        int q = threadIdx.x + blockDim.x * blockIdx.x;
        if (q >= num_query) {
                return;
        }

#define _f(i, j) f[(j) + align + (i) * (2 * align + 2 * ngsl + ny + 4)]
#define _dg(k) dg[(k) + align]

        prec dth = dt / (h * h * h);

        // Reciprocal quadrature weights near the top boundary in the z-direction. First weight is
        // on the boundary
        // hweights: weights at the nodal grid points
        const prec hweights[4] = {3.55599789310935, 0.6905974224013051,
                                  1.4771520525102637, 0.914256470417062};
        // hhatweights: weights at the cell-centered grid points
        const prec hhatweights[4] = {2.9022824945274315, 2.28681149230364,
                                     0.7658753535345706, 1.0959408329892313};

        int nx = grid.size.x - 4 - 2 * ngsl;
        int nz = grid.size.z;
        for (int i = 0; i < num_basis; ++i) {
        for (int j = 0; j < num_basis; ++j) {
        for (int k = 0; k < num_basis; ++k) {
               prec Ji =
                   1.0 / (_f(i + ix[q], j + iy[q]) *
                          _dg(iz[q] + k));

                // Do not apply stencil at halo points
                if ( ix[q] + i >= 2 + nx + 3 * ngsl / 2 || ix[q] + i < 2 + ngsl / 2 ||
                     iy[q] + j >= 2 + ny + 3 * ngsl / 2 || iy[q] + j < 2 + ngsl / 2 ) continue;

                int pos =
                    (iz[q] + k) + align +
                    (2 * align + nz) * (ix[q] + i) * (2 * ngsl + ny + 4) +
                    (2 * align + nz) * (iy[q] + j);
                prec w = 1.0f;
                int offset_z = nz - (iz[q] + k + 2);
                int offset_zhat = nz - (iz[q] + k + 1);
                if (zhat == 0 &&  offset_z  < 4 && offset_z >= 0)
                        w = hweights[offset_z];
                if (zhat == 1 &&  offset_zhat < 4 && offset_zhat >= 0)
                        w = hhatweights[offset_zhat];
                prec value = - dth * lx[q * num_basis + i] *
                            ly[q * num_basis + j] * lz[q * num_basis + k] *
                            in[lidx[q]] * Ji * w;

#if USE_ATOMICS
                atomicAdd(&out[pos], value);
#else 
                out[pos] = value;
#endif
        }
        }
        }
}

void cusource_add_force_H(const cu_interp_t *I, prec *out, const prec *in,
                          const prec *d1, const prec h, const prec dt,
                          const prec quad_weight,
                          const prec *f, const int nx, const int ny,
                          const int nz, const prec *dg, const int sourcetype, const int dir) 
{
        dim3 block (INTERP_THREADS, 1, 1);
        dim3 grid((I->num_query + INTERP_THREADS - 1) / INTERP_THREADS,
                  1, 1);

        if (sourcetype == 0) {
        cusource_add_force<<<grid, block>>>(
            out, in, d1, I->d_lx, I->d_ly, I->d_lz, I->num_basis, I->d_ix,
            I->d_iy, I->d_iz, I->d_ridx, h, dt, quad_weight, I->num_query,
            I->grid, f, nx, ny, nz, dg);
        } 
        else if (sourcetype == 1) {
        cusource_add_force_stress<<<grid, block>>>(
            out, in, d1, I->d_lx, I->d_ly, I->d_lz, I->num_basis, I->d_ix,
            I->d_iy, I->d_iz, I->d_ridx, h, dt, quad_weight, I->num_query,
            I->grid, f, nx, ny, nz, dg, dir);

        }
        else {
            cusource_add_force_velocity<<<grid, block>>>(
            out, in, d1, I->d_lx, I->d_ly, I->d_lz, I->num_basis, I->d_ix,
            I->d_iy, I->d_iz, I->d_ridx, h, dt, quad_weight, I->num_query,
            I->grid, f, nx, ny, nz, dg, dir);
        }
        CUCHK(cudaGetLastError());
}

__global__ void cusource_add_force(prec *out, const prec *in, const prec *d1,
                                   const prec *lx, const prec *ly,
                                   const prec *lz, const int num_basis,
                                   const int *ix, const int *iy, const int *iz,
                                   const int *lidx, const prec h, const prec dt,
                                   const prec quad_weight,
                                   const int num_query, const grid3_t grid,
                                   const prec *f, const int nx, const int ny,
                                   const int nz, const prec *dg) 
{
        int q = threadIdx.x + blockDim.x * blockIdx.x;
        if (q >= num_query) {
                return;
        }

#define _f(i, j) f[(j) + align + (i) * (2 * align + 2 * ngsl + ny + 4)]
#define _dg(k) dg[(k) + align]

#define _rho(i, j, k)                                                   \
        d1[(k) + align + (2 * align + nz) * (i) * (2 * ngsl + ny + 4) + \
           (2 * align + nz) * (j)]

        prec dth = dt / (h * h * h);

        for (int i = 0; i < num_basis; ++i) {
        for (int j = 0; j < num_basis; ++j) {
        for (int k = 0; k < num_basis; ++k) {
                // Do not apply stencil at halo points
                if ( ix[q] + i >= 2 + nx + ngsl || ix[q] + i < 2 + ngsl ||
                     iy[q] + j >= 2 + ny + ngsl || iy[q] + j < 2 + ngsl ) continue;

                prec J =  _f(i + ix[q], j + iy[q]) * _dg(iz[q] + k);
                prec Ji = - quad_weight /(J * d1[q]);
                int pos =
                    (iz[q] + k) + align +
                    (2 * align + nz) * (ix[q] + i) * (2 * ngsl + ny + 4) +
                    (2 * align + nz) * (iy[q] + j);
                prec value = -dth * lx[q * num_basis + i] *
                            ly[q * num_basis + j] * lz[q * num_basis + k] * in[lidx[q]] * Ji;
#if USE_ATOMICS
                atomicAdd(&out[pos], value);
#else 
                out[pos] = value;
#endif
        }
        }
        }
}

__global__ void cusource_add_force_stress(prec *out, const prec *in, const prec *d1,
                                   const prec *lx, const prec *ly,
                                   const prec *lz, const int num_basis,
                                   const int *ix, const int *iy, const int *iz,
                                   const int *lidx, const prec h, const prec dt,
                                   const prec quad_weight,
                                   const int num_query, const grid3_t grid,
                                   const prec *f, const int nx, const int ny,
                                   const int nz, const prec *dg, const int dir) 
{
        int q = threadIdx.x + blockDim.x * blockIdx.x;
        if (q >= num_query) {
                return;
        }


        prec dth = 1.0 / (h * h);
        int k = nz - 1;

        for (int i = 0; i < num_basis; ++i) {
        for (int j = 0; j < num_basis; ++j) {
                // Do not apply stencil at halo points
                if ( ix[q] + i >= 2 + nx + ngsl || ix[q] + i < 2 + ngsl ||
                     iy[q] + j >= 2 + ny + ngsl || iy[q] + j < 2 + ngsl ) continue;

                int pos =
                    (k) + align +
                    (2 * align + nz) * (ix[q] + i) * (2 * ngsl + ny + 4) +
                    (2 * align + nz) * (iy[q] + j);
                prec value = dth * lx[q * num_basis + i] *
                            ly[q * num_basis + j] * in[lidx[q]];
                if (dir == 1 || dir == 2) {
                        out[pos] = value;
                        out[pos+1] = 2 * value - out[pos-1];
                        out[pos+2] = 2 * value - out[pos-2];
                }
                if (dir == 3) {
                        out[pos+1] = 2 * value - out[pos];
                        out[pos+2] = 2 * value - out[pos-1];
                }
        }
        }
}

__global__ void cusource_add_force_velocity(prec *out, const prec *in, const prec *d1,
                                   const prec *lx, const prec *ly,
                                   const prec *lz, const int num_basis,
                                   const int *ix, const int *iy, const int *iz,
                                   const int *lidx, const prec h, const prec dt,
                                   const prec quad_weight,
                                   const int num_query, const grid3_t grid,
                                   const prec *f, const int nx, const int ny,
                                   const int nz, const prec *dg, const int dir) 
{
        int q = threadIdx.x + blockDim.x * blockIdx.x;
        if (q >= num_query) {
                return;
        }


        prec dth = dt / (h * h * h);
        int k = nz - 1;

        for (int i = 0; i < num_basis; ++i) {
        for (int j = 0; j < num_basis; ++j) {
                // Do not apply stencil at halo points
                if ( ix[q] + i >= 2 + nx + ngsl || ix[q] + i < 2 + ngsl ||
                     iy[q] + j >= 2 + ny + ngsl || iy[q] + j < 2 + ngsl ) continue;

                int pos =
                    (k) + align +
                    (2 * align + nz) * (ix[q] + i) * (2 * ngsl + ny + 4) +
                    (2 * align + nz) * (iy[q] + j);
                prec value = dth * lx[q * num_basis + i] *
                            ly[q * num_basis + j] / d1[q] * in[lidx[q]];
#if USE_ATOMICS
                atomicAdd(&out[pos], 1.0 * value);
                atomicAdd(&out[pos - 1], -0.0 * value);
#else
                    out[pos] += value;
#endif
        }
        }
}
