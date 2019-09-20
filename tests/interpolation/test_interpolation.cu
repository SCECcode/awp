#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#include <awp/definitions.h>
#include <test/test.h>
#include <test/check.h>
#include <grid/shift.h>
#include <grid/grid_3d.h>
#include <interpolation/interpolation.h>
#include <interpolation/interpolation.cuh>

int test_interp(void);
int test_write(const int rank, const int size);

int main(int argc, char **argv)
{
        int err = 0;
        int rank, size;

        MPI_Init(&argc, &argv);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN); 
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (rank == 0) { 
                test_divider();
                printf("Testing cuinterpolation.cu\n");
                err |= test_interp();
        }

        if (rank == 0) {
                printf("Testing completed.\n");
                test_divider();
        }

        MPI_Finalize();

        return err;

}

int test_interp(void)
{
        int err = 0;

        prec *x1, *y1, *z1, *fcn3;
        int n = 10;
        int gsize[3] = {n, n, n};
        int num_query = 4;
        int deg = 3;
        int3_t shift = grid_yz();
        int3_t coord = {.x = 0, .y = 0, .z = 0};
        int3_t asize = {gsize[0], gsize[1], gsize[2]};
        int3_t bnd1 = {1, 1, 1};
        int3_t bnd2 = {0, 0, 0};
        prec h = 1.0/n;
        grid3_t grid = grid_init(asize, shift, coord, bnd1, bnd2, ngsl, h);

        x1 = (prec*)malloc(sizeof(x1) * grid.size.x);
        y1 = (prec*)malloc(sizeof(y1) * grid.size.y);
        z1 = (prec*)malloc(sizeof(z1) * grid.size.z);
        fcn3 = (prec*)malloc(grid.num_bytes);

        prec qx[4] = {0.0, 0.2, 0.4, 1.9};
        prec qy[4] = {0.0, 0.7, 0.4, 1.7};
        prec qz[4] = {0.0, 0.2, 0.3, 0.8};

        cu_interp_t I;

        grid_fill_x(x1, grid);
        grid_fill_y(y1, grid);
        grid_fill_z(z1, grid);

        grid_fill3_x(fcn3, x1, grid);

        cuinterp_init(&I, x1, y1, z1, grid, qx, qy, qz, num_query, deg);

        // Perform interpolation on both host and device and compare
        // Use tested host function (interp_lagrange3, see test_interpolation)
        {
                test_t test = test_init(" * interpolation", 0, 0);

                prec *d_fcn3;
                prec out[4];
                prec *d_res;
                prec *res;
                int res_bytes = num_query * sizeof(prec);

                err |= interp_lagrange3(out, fcn3, x1, y1, z1, grid, qx, qy, qz,
                                        num_query, deg);


                CUCHK(cudaMalloc((void**)&d_fcn3, grid.num_bytes));
                res = (prec*)malloc(res_bytes);
                CUCHK(cudaMalloc((void**)&d_res, res_bytes));
                CUCHK(cudaMemcpy(d_fcn3, fcn3, grid.num_bytes,
                                cudaMemcpyHostToDevice));


                cuinterp_interp_H(&I, d_res, d_fcn3);

                CUCHK(cudaMemcpy(res, d_res, res_bytes,
                                cudaMemcpyDeviceToHost));
                err |= s_assert(chk_inf(res, out, I.num_query) ==
                                0.0);

                err |= test_finalize(&test, err);

                cudaFree(d_res);
                cudaFree(d_fcn3);
                free(res);
        }

        cuinterp_finalize(&I);

        free(x1);
        free(y1);
        free(z1);
        free(fcn3);

        return err;
}

