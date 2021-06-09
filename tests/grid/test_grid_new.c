#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <awp/definitions.h>
#include <awp/error.h>
#include <test/test.h>
#include <test/array.h>
#include <grid/grid_new.h>
#include <grid/shift.h>

int test_global_to_local(int rank, int size);

int test_global_to_local(int rank, int size) {

    int err = 0;


    test_t test = test_init(" * global_to_local", rank, size);
    const int num_grids = 3;
    int nz[3] = {20, 10, 12};
    _prec h = 1.0;
    const prec H[3] = {grid_height(nz[0], h, 1), grid_height(nz[1], 3 * h, 0),
                       grid_height(nz[2], 9 * h, 0)};

    // Above free surface (in topo block)
    {
        _prec zglb = 0.2;
        _prec zloc = 0.0;
        int block_index = -1;
        int istopo = 1;
        global_to_local(&zloc, &block_index, zglb, h, nz, num_grids, istopo);
        err |= mpi_assert(block_index == 0, rank);
        err |= mpi_assert(fabs(zloc - (zglb + H[0])) < FLTOL, rank);
    }

    // Below free surface (in topo block)
    {
        _prec zglb = -0.2;
        _prec zloc = 0.0;
        int block_index = -1;
        int istopo = 1;
        global_to_local(&zloc, &block_index, zglb, h, nz, num_grids, istopo);
        err |= mpi_assert(block_index == 0, rank);
        err |= mpi_assert(fabs(zloc - (zglb + H[0]) ) < FLTOL, rank);
    }

    // In the overlap zone (belongs to the second block)
    {
        _prec zglb = -15.0;
        _prec zloc = 0.0;
        int block_index = -1;
        int istopo = 1;
        global_to_local(&zloc, &block_index, zglb, h, nz, num_grids, istopo);
        err |= mpi_assert(block_index == 1, rank);

        _prec zs = (zglb + H[0] + H[1] - grid_overlap(h) );
        err |= mpi_assert(fabs(zloc - zs) < FLTOL, rank);
    }


    // In the overlap zone (belongs to the third block)
    {
        _prec zglb = -19.0;
        _prec zloc = 0.0;
        int block_index = -1;
        int istopo = 1;
        global_to_local(&zloc, &block_index, zglb, h, nz, num_grids, istopo);
        err |= mpi_assert(block_index == 2, rank);

        _prec zs = (zglb + H[0] + H[1] + H[2] - grid_overlap(h) - grid_overlap(3 * h) );
        err |= mpi_assert(fabs(zloc - zs) < FLTOL, rank);
    }

    err |= test_finalize(&test, err);

    return err;

}

int main(int argc, char **argv)
{
        int err = 0;
        int rank, size;

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (rank == 0) {
                test_divider();
                printf("Testing grid_new.c\n");
                printf("\n");
                printf("Running tests:\n");
        }

        err |= test_global_to_local(rank, size);

        if (rank == 0) {
                printf("Testing completed.\n");
                test_divider();
        }

        MPI_Finalize();

        return err;
}

