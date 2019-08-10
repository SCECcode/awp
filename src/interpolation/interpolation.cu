#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <stdint.h>

#include <awp/error.h>
#include <awp/definitions.h>
#include <interpolation/interpolation.h>
#include <grid/grid_3d.h>
#include <interpolation/interpolation.cuh>
#include <test/test.h>

int cuinterp_init(cu_interp_t *out, const prec *x, const prec *y,
                    const prec *z, grid3_t grid, const prec *qx, const prec *qy,
                    const prec *qz, const int num_query, const int degree)
{
        out->num_basis = degree + 1;
        out->num_query = num_query;
        out->size_l = num_query * (degree + 1);
        out->size_i = num_query;
        out->grid = grid;

        if (num_query <= 0) {
                return ERR_NON_POSITIVE;
        }

        int err = cuinterp_malloc(out);

        if (err != SUCCESS) {
                return err;
        }

        cuinterp_lagrange_h(out, x, y, z, grid, qx, qy, qz);
        cuinterp_htod(out);

        return SUCCESS;
}

int cuinterp_lagrange_h(cu_interp_t *host, const prec *x, const prec *y,
                        const prec *z, const grid3_t grid, const prec *qx,
                        const prec *qy, const prec *qz)
{
        prec *xloc = (prec*)calloc(sizeof(xloc), host->num_basis);
        prec *yloc = (prec*)calloc(sizeof(yloc), host->num_basis);
        prec *zloc = (prec*)calloc(sizeof(zloc), host->num_basis);
        int err = SUCCESS;
        int deg = host->num_basis - 1;
        for (int q = 0; q < host->num_query; ++q) {
                err = interp_lagrange1_coef(
                    xloc, &host->lx[q * host->num_basis], &host->ix[q], x,
                    grid_boundary_size(grid).x, qx[q], deg);
                err = interp_lagrange1_coef(
                    yloc, &host->ly[q * host->num_basis], &host->iy[q], y,
                    grid_boundary_size(grid).y, qy[q], deg);
                err = interp_lagrange1_coef(
                    zloc, &host->lz[q * host->num_basis], &host->iz[q], z,
                    grid_boundary_size(grid).z, qz[q], deg);
        }
        free(xloc);
        free(yloc);
        free(zloc);
        return err;
}

int cuinterp_htod(cu_interp_t *T)
{
        int num_bytes =
            sizeof(prec) * T->num_basis * T->num_query;
        CUCHK(cudaMemcpy(T->d_lx, T->lx, num_bytes,
                                cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(T->d_ly, T->ly, num_bytes,
                                cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(T->d_lz, T->lz, num_bytes,
                                cudaMemcpyHostToDevice));
        num_bytes = sizeof(int) * T->num_query;
        CUCHK(cudaMemcpy(T->d_ix, T->ix, num_bytes,
                                cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(T->d_iy, T->iy, num_bytes,
                                cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(T->d_iz, T->iz, num_bytes,
                                cudaMemcpyHostToDevice));
        return SUCCESS;
}

int cuinterp_dtoh(cu_interp_t *T)
{
        int num_bytes =
            sizeof(prec) * T->num_basis * T->num_query;
        CUCHK(cudaMemcpy(T->lx, T->d_lx, num_bytes,
                              cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(T->ly, T->d_ly, num_bytes,
                              cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(T->lz, T->d_lz, num_bytes,
                                cudaMemcpyDeviceToHost));
        num_bytes = sizeof(int) * T->num_query;
        CUCHK(cudaMemcpy(T->ix, T->d_ix, num_bytes,
                              cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(T->iy, T->d_iy, num_bytes,
                              cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(T->iz, T->d_iz, num_bytes,
                                cudaMemcpyDeviceToHost));
        return SUCCESS;
}

int cuinterp_malloc(cu_interp_t *interp)
{
        interp->lx = (prec*)calloc(sizeof(prec), interp->size_l);
        interp->ly = (prec*)calloc(sizeof(prec), interp->size_l);
        interp->lz = (prec*)calloc(sizeof(prec), interp->size_l);
        if (!interp->lx || !interp->ly || !interp->lz ) {
                return ERR_INTERP_MALLOC;
        }

        interp->ix = (int*)calloc(sizeof(int), interp->size_i);
        interp->iy = (int*)calloc(sizeof(int), interp->size_i);
        interp->iz = (int*)calloc(sizeof(int), interp->size_i);

        if (!interp->ix || !interp->iy || !interp->iz ) {
                return ERR_INTERP_MALLOC;
        }

        size_t num_bytes =
            sizeof(prec) * interp->size_l;
        CUCHK(cudaMalloc(&interp->d_lx, num_bytes));
        CUCHK(cudaMalloc(&interp->d_ly, num_bytes));
        CUCHK(cudaMalloc(&interp->d_lz, num_bytes));

        num_bytes = sizeof(int) * interp->size_i;
        CUCHK(cudaMalloc(&interp->d_ix, num_bytes));
        CUCHK(cudaMalloc(&interp->d_iy, num_bytes));
        CUCHK(cudaMalloc(&interp->d_iz, num_bytes));

        return SUCCESS;
}

void cuinterp_finalize(cu_interp_t *interp)
{
        free(interp->lx);
        free(interp->ly);
        free(interp->lz);
        free(interp->ix);
        free(interp->iy);
        free(interp->iz);
        CUCHK(cudaFree(interp->d_lx));
        CUCHK(cudaFree(interp->d_ly));
        CUCHK(cudaFree(interp->d_lz));
        CUCHK(cudaFree(interp->d_ix));
        CUCHK(cudaFree(interp->d_iy));
        CUCHK(cudaFree(interp->d_iz));
}

void cuinterp_interp_H(const cu_interp_t *I, prec *out, const prec *in)
{
        dim3 block (INTERP_THREADS, 1, 1);
        dim3 grid((I->num_query + INTERP_THREADS - 1) / INTERP_THREADS,
                  1, 1);

        cuinterp_dinterp<<<grid, block>>>(out, in, I->d_lx, I->d_ly, I->d_lz,
                                          I->num_basis, I->d_ix, I->d_iy,
                                          I->d_iz, I->num_query, I->grid);
        CUCHK(cudaGetLastError());
}

__global__ void cuinterp_dinterp(prec *out, const prec *in,
                                 const prec *lx, const prec *ly, const prec *lz,
                                 const int num_basis, const int *ix,
                                 const int *iy, const int *iz,
                                 const int num_query, const grid3_t grid)
{
        int q = threadIdx.x + blockDim.x * blockIdx.x;
        if (q >= num_query) {
                return;
        }
        out[q] = 0.0;
        for (int i = 0; i < num_basis; ++i) {
        for (int j = 0; j < num_basis; ++j) {
        for (int k = 0; k < num_basis; ++k) {
                size_t pos = grid_index(grid, ix[q] + i, iy[q] + j, iz[q] + k);
                out[q] += lx[q * num_basis + i] * ly[q * num_basis + j] *
                          lz[q * num_basis + k] * in[pos];
        }
        }
        }
}

