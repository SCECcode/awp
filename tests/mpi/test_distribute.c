#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define ADDLINENUM 1
#define ADDRANK 1
#define RANK rank

#include <awp/definitions.h>
#include <mpi/distribute.h>
#include <test/test.h>
#include <test/check.h>
#include <grid/shift.h>

int test_indices(int rank, int size, enum eshift shift); 

int main(int argc, char **argv)
{
        int rank, size;
        MPI_Init(&argc, &argv);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN); 
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (size != 4)
        {
                if (rank == 0) {
                        printf("Test requires MPI size = 4.\n");
                        fflush(stdout);
                }
                        MPI_Abort(MPI_COMM_WORLD, -1);
                        return -1;
        }

        if (rank == 0) {
                test_divider();
                printf("Testing test_distribute.c\n");
        }

        test_indices(rank, size, GRID_U1);
        test_indices(rank, size, GRID_U2);
        test_indices(rank, size, GRID_U3);
        //TODO: Add tests for stress grids
        //err = test_indices(rank, size, GRID_XX);
        //err = test_indices(rank, size, GRID_YY);
        //err = test_indices(rank, size, GRID_ZZ);
        //err = test_indices(rank, size, GRID_XY);
        //err = test_indices(rank, size, GRID_XZ);
        //err = test_indices(rank, size, GRID_YZ);

        if (rank == 0) {
                printf("Testing completed.\n");
                test_divider();
        }

        MPI_Finalize();

        return test_last_error();
}

int test_indices(int rank, int size, enum eshift shifttype)
{
        char msg[90];
        sprintf(msg, " * indices: %s", grid_shift_label(shifttype)); 
        test_t test = test_init(msg, rank,  size);
        int err = 0;
        int n = 11;
        int blocks_x = 2;
        int gsize[3] = {n, n, n};
        prec h = 1.0/(n-1);
        
        prec *qx = malloc(sizeof qx * n);
        prec *qy = malloc(sizeof qy * n);
        prec *qz = malloc(sizeof qz * n);
        int  *grid_numbers = malloc(sizeof grid_numbers * n);
        for (int i = 0; i < n; ++i)
                grid_numbers[i] = 0.0f;

        int3_t shift = grid_shift(shifttype);
        
        int3_t coord = {.x = rank / blocks_x, .y = rank % blocks_x, .z = 0};
        int3_t asize = {gsize[0], gsize[1], gsize[2]};

        int3_t bnd1 = {0, 0, 1};
        int3_t bnd2 = {0, 0, 1};

        fcn_grid_t grid;
        fcn_grid_t ref_grid;

        if (shifttype == GRID_U1 || shifttype == GRID_U2 ||
            shifttype == GRID_U3) {
                // velocity grid
                grid = grid_init(asize, shift, coord, bnd1, bnd2, 0, h);
        } else {
                // stress grid
                grid = grid_init(asize, shift, coord, bnd1, bnd2, ngsl / 2, h);
        }


        // Reference grid
        ref_grid = grid;
        ref_grid.coordinate.x = 0;
        ref_grid.coordinate.y = 0;

        grid1_t grid_x = grid_grid1_x(ref_grid);
        grid1_t grid_y = grid_grid1_y(ref_grid);

        prec *x = malloc(sizeof(x) * grid_x.size);
        prec *y = malloc(sizeof(y) * grid_y.size);

        grid_fill1(x, grid_x);
        grid_fill1(y, grid_y);

        grid_x = grid_grid1_x(grid);
        grid_y = grid_grid1_y(grid);

        // local coordinates (not used)
        prec *xloc = malloc(sizeof(x) * grid_x.size);
        prec *yloc = malloc(sizeof(y) * grid_y.size);

        grid_fill1(xloc, grid_x);
        grid_fill1(yloc, grid_y);

        h = grid.gridspacing;

        n = 4;

        // Query points below are placed at, or near the boundary of the
        // partitions 

        // bottom left
        qx[0] = x[grid_x.size - 1];
        qy[0] = y[0];
        qz[0] = 0.0;

        // bottom right
        qx[1] = x[grid_x.size - 1] + h / 2 + 0.0001;
        qy[1] = y[0];
        qz[1] = 0.0;

        // top left
        qx[2] = x[0];
        qy[2] = y[grid_y.size - 1] + h;
        qz[2] = 0.0;

        // top right
        qx[3] = x[grid_x.size - 1] + h;
        qy[3] = y[grid_y.size - 1] + h;
        qz[3] = 0.0;

        size_t nidx = 0;
        int *indices;

        const int is_source = 0;
        dist_indices(&indices, &nidx, qx,  qy, n, grid, grid_numbers, 0, is_source);

        if (coord.x == 0 && coord.y == 0) {
                int ans[1] = {0};
                err |= s_assert(nidx == 1);
                err |= s_assert(chk_infi(ans, indices, nidx) == 0);
        }

        if (coord.x == 1 && coord.y == 0) {
                int ans[1] = {1};
                err |= s_assert(nidx == 1);
                err |= s_assert(chk_infi(ans, indices, nidx) == 0);
        }

        if (coord.x == 0 && coord.y == 1) {
                int ans[1] = {2};
                err |= s_assert(nidx == 1);
                err |= s_assert(chk_infi(ans, indices, nidx) == 0);
        }

        if (coord.x == 1 && coord.y == 1) {
                int ans[1] = {3};
                err |= s_assert(nidx == 1);
                err |= s_assert(chk_infi(ans, indices, nidx) == 0);
        }

        free(x);
        free(y);
        free(qx);
        free(qy);
        free(qz);
        free(indices);
        free(xloc);
        free(yloc);
        err |= test_finalize(&test, err);

        return test_last_error();

}

