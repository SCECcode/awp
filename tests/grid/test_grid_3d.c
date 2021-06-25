#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <awp/definitions.h>
#include <awp/error.h>
#include <test/test.h>
#include <test/array.h>
#include <grid/grid_3d.h>
#include <grid/shift.h>

int test_grid_fill(int rank, int size);
int test_grid_in_bounds(int rank, int size);
int test_grid_xyz(int rank, int size);
int test_grid3_xyz(int rank, int size);
int test_grid3_reduce(int rank, int size);
int test_shift(int rank, int size);
int test_global_to_local(int rank, int size);

   
int test_global_to_local(int rank, int size);
   

int main(int argc, char **argv)
{
        int err = 0;
        int rank, size;

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (rank == 0) {
                test_divider();
                printf("Testing grid_3d.c\n");
                printf("\n");
                printf("Running tests:\n");
        }

        err |= test_grid_fill(rank, size);
        err |= test_grid_in_bounds(rank, size);
        err |= test_grid_xyz(rank, size);
        err |= test_grid3_xyz(rank, size);
        err |= test_grid3_reduce(rank, size);
        err |= test_shift(rank, size);
        err |= test_global_to_local(rank, size);

        if (rank == 0) {
                printf("Testing completed.\n");
                test_divider();
        }

        MPI_Finalize();

        return err;
}

int test_grid_fill(int rank, int size)
{
        prec *x;
        int err = 0;
        int n = 101;
        prec h = 1.0;

        {
        test_t test = test_init(" * grid_fill::node", rank, size);
        grid1_t grid = {.id = rank, .shift = 0, .size = n, .gridspacing = h, 
                        .boundary1 = 0, .boundary2 = 0};

        x = malloc(sizeof(x) * grid.size);
        grid_fill1(x, grid, 1);
        err |= mpi_assert(!err, rank);
        err |= mpi_assert(fabs(x[0] - (0.0 +  rank * n)) < FLTOL, rank);
        err |= mpi_assert(fabs(x[1] - (1.0 +  rank * n)) < FLTOL, rank);
        err |= mpi_assert(fabs(x[n-1] - (n - 1 + rank * n) ) < FLTOL, rank);

        err |= test_finalize(&test, err);
        }
        
        {
        test_t test = test_init(" * grid_fill::node.left", rank, size);
        grid1_t grid = {.id = rank, .shift = 0, .size = n, .gridspacing = h, 
                        .boundary1 = 1, .boundary2 = 0};

        grid_fill1(x, grid, 1);
        err |= mpi_assert(!err, rank);
        err |= mpi_assert(fabs(x[0] - (0.0 +  rank * n)) < FLTOL, rank);
        err |= mpi_assert(fabs(x[1] - (1.0 +  rank * n)) < FLTOL, rank);
        err |= mpi_assert(fabs(x[n-1] - (n - 1 + rank * n) ) < FLTOL, rank);
        
        err |= test_finalize(&test, err);
        }

        {
        test_t test = test_init(" * grid_fill::node.right", rank, size);
        grid1_t grid = {.id = rank, .shift = 0, .size = n, .gridspacing = h, 
                        .boundary1 = 0, .boundary2 = 1};

        grid_fill1(x, grid, 1);
        err |= mpi_assert(!err, rank);
        err |= mpi_assert(fabs(x[0] - (0.0 +  rank * n)) < FLTOL, rank);
        err |= mpi_assert(fabs(x[1] - (1.0 +  rank * n)) < FLTOL, rank);
        err |= mpi_assert(fabs(x[n-1] - (0.0) ) < FLTOL, rank);
        
        err |= test_finalize(&test, err);
        }

        {
        test_t test = test_init(" * grid_fill::cell", rank, size);
        grid1_t grid = {.id = rank, .shift = 1, .size = n, .gridspacing = h, 
                        .boundary1 = 0, .boundary2 = 0};

        grid_fill1(x, grid, 1);
        err |= mpi_assert(!err, rank);
        err |= mpi_assert(fabs(x[0] - (0.5 +  rank * n)) < FLTOL, rank);
        err |= mpi_assert(fabs(x[1] - (1.5 +  rank * n)) < FLTOL, rank);
        err |= mpi_assert(fabs(x[n-1] - (n - 1  + 0.5 + rank * n) ) < FLTOL, 
                          rank);

        err |= test_finalize(&test, err);
        }

        {
        test_t test = test_init(" * grid_fill::cell.left", rank, size);
        grid1_t grid = {.id = rank, .shift = 1, .size = n, .gridspacing = h, 
                        .boundary1 = 1, .boundary2 = 0};

        grid_fill1(x, grid, 0);
        err |= mpi_assert(!err, rank);
        err |= mpi_assert(fabs(x[0] - (0.0 +  rank * n)) < FLTOL, rank);
        err |= mpi_assert(fabs(x[1] - (0.5 +  rank * n)) < FLTOL, rank);
        err |= mpi_assert(fabs(x[n-1] - (n - 1  - 0.5 + rank * n) ) < FLTOL, 
                          rank);

        err |= test_finalize(&test, err);
        }

        {
        test_t test = test_init(" * grid_fill::cell.right", rank, size);
        grid1_t grid = {.id = rank, .shift = 1, .size = n, .gridspacing = h, 
                        .boundary1 = 0, .boundary2 = 1};

        grid_fill1(x, grid, 0);
        err |= mpi_assert(!err, rank);
        err |= mpi_assert(fabs(x[0] - (-0.5 +  rank * n)) < FLTOL, rank);
        err |= mpi_assert(fabs(x[1] - (+0.5 +  rank * n)) < FLTOL, rank);
        err |= mpi_assert(fabs(x[n-1] - (n - 2  + rank * n) ) < FLTOL, 
                          rank);

        err |= test_finalize(&test, err);
        }

        free(x);
        return err;
}

int test_grid_in_bounds(int rank, int size)
{
        int err = 0;
        int n = 100;
        int gsize[3] = {n, n, n};
        prec *x;

        x = malloc(sizeof(x) * n);

        int3_t shift = {0, 0, 0};
        
        int3_t coord = {.x = 0, .y = 0, .z = 0};
        int3_t asize = {gsize[0], gsize[1], gsize[2]};

        int3_t bnd1 = {0, 0, 0};
        int3_t bnd2 = {1, 1, 1};

        {
        grid1_t grid1 = {.id = coord.x, .shift = shift.x, .size = asize.x,
                         .boundary1 = bnd1.x, .boundary2 = bnd2.x,
                         .alignment = 2 + ngsl,
                         .padding = 0,
                         .gridspacing = 1.0};
        grid_fill1(x, grid1, 0);

        test_t test = test_init(" * grid_in_bounds", rank, size);
        err |= mpi_assert(
            grid_in_bounds1(x, -0.2, grid1) == ERR_OUT_OF_BOUNDS_LOWER, rank);
        err |= mpi_assert(
            grid_in_bounds1(x, n+1, grid1) == ERR_OUT_OF_BOUNDS_UPPER, rank);
        err |= mpi_assert(
            grid_in_bounds1(x, n-2, grid1) == SUCCESS, rank);

        err |= test_finalize(&test, err);
        }

        // Check with padding
        {
        grid1_t grid1 = {.id = coord.x, .shift = shift.x, .size = asize.x,
                         .boundary1 = bnd1.x, .boundary2 = bnd2.x,
                         .alignment = 1 + ngsl,
                         .padding = 1,
                         .gridspacing = 1.0};
        grid_fill1(x, grid1, 0);

        test_t test = test_init(" * grid_in_bounds", rank, size);
        err |= mpi_assert(
            grid_in_bounds1(x, -1.2, grid1) == ERR_OUT_OF_BOUNDS_LOWER, rank);
        err |= mpi_assert(
            grid_in_bounds1(x, n+1, grid1) == ERR_OUT_OF_BOUNDS_UPPER, rank);
        err |= mpi_assert(
            grid_in_bounds1(x, n-4, grid1) == SUCCESS, rank);

        err |= test_finalize(&test, err);
        }

        free(x);

        return test_last_error();
}

int test_grid_xyz(int rank, int size)
{
        int err = 0;
        int n = 10;
        int gsize[3] = {n, n, n};
        prec h = 1.0;
        prec *x, *ans;

        x = malloc(sizeof(x) * n);
        ans = malloc(sizeof(ans) * n);

        int3_t shift = grid_yz();
        
        int3_t coord = {.x = rank, .y = rank, .z = rank};
        int3_t asize = {gsize[0], gsize[1], gsize[2]};

        int3_t bnd1 = {1, 1, 1};
        int3_t bnd2 = {1, 1, 1};

        fcn_grid_t grid = grid_init(asize, shift, coord, bnd1, bnd2, ngsl, h);
        x = malloc(sizeof(x) * grid.size.x);
        ans = malloc(sizeof(ans) * grid.size.x);
        {
        test_t test = test_init(" * grid_fill_x", rank, size);

        grid1_t grid1 = grid_grid1_x(grid);
        grid_fill1(ans, grid1, 1);
        grid_fill_x(x, grid);

        for (int i = 0; i < n; ++i) {
                err |= mpi_assert(fabs(x[i] - ans[i]) < FLTOL, rank);
        }

        err |= test_finalize(&test, err);
        }

        {
        test_t test = test_init(" * grid_fill_y", rank, size);

        grid1_t grid1 = grid_grid1_y(grid);
        grid_fill1(ans, grid1, 0);
        grid_fill_y(x, grid);

        for (int i = 0; i < n; ++i) {
                err |= mpi_assert(fabs(x[i] - ans[i]) < FLTOL, rank);
        }

        err |= test_finalize(&test, err);
        }

        {
        test_t test = test_init(" * grid_fill_z", rank, size);
        grid1_t grid1 = grid_grid1_z(grid);

        grid_fill1(ans, grid1, 0);
        grid_fill_z(x, grid);

        for (int i = 0; i < n; ++i) {
                err |= mpi_assert(fabs(x[i] - ans[i]) < FLTOL, rank);
        }

        err |= test_finalize(&test, err);
        }

        free(x);
        free(ans);

        return err;
}

int test_grid3_xyz(int rank, int size)
{
        int err = 0;
        int n = 10;
        int gsize[3] = {n, n, n};
        prec h = 1.0;
        prec *x1, *x3, *y1, *y3, *z1, *z3, *ans;

        ans = malloc(sizeof(ans) * n);
        

        int3_t shift = grid_yz();
        
        int3_t coord = {.x = rank, .y = rank, .z = rank};
        int3_t asize = {gsize[0], gsize[1], gsize[2]};

        int3_t bnd1 = {1, 1, 1};
        int3_t bnd2 = {1, 1, 1};

        fcn_grid_t grid = grid_init(asize, shift, coord, bnd1, bnd2, ngsl, h);

        x1 = malloc(sizeof(x1) * grid.size.x);
        y1 = malloc(sizeof(y1) * grid.size.y);
        z1 = malloc(sizeof(z1) * grid.size.z);

        x3 = malloc(sizeof(x3) * grid.mem.x * grid.mem.y * grid.mem.z);
        y3 = malloc(sizeof(y3) * grid.mem.x * grid.mem.y * grid.mem.z);
        z3 = malloc(sizeof(z3) * grid.mem.x * grid.mem.y * grid.mem.z);

        grid_fill_x(x1, grid); 
        grid_fill_y(y1, grid); 
        grid_fill_z(z1, grid); 
        grid_fill3_x(x3, x1, grid); 
        grid_fill3_y(y3, y1, grid); 
        grid_fill3_z(z3, z1, grid); 

        // Check a few values for the x-direction
        {
        test_t test = test_init(" * grid3_fill_x", rank, size);
        int i=0, j=0, k=0;
        err |= s_assert(fabs(x3[grid_index(grid, i, j, k)] - x1[i]) < FLTOL);
        i = n - 1;
        err |= s_assert(fabs(x3[grid_index(grid, i, j, k)] - x1[i]) < FLTOL);
        j = n - 1;
        err |= s_assert(fabs(x3[grid_index(grid, i, j, k)] - x1[i]) < FLTOL);
        k = n - 1;
        err |= s_assert(fabs(x3[grid_index(grid, i, j, k)] - x1[i]) < FLTOL);
        err |= test_finalize(&test, err);
        }

        // Check a few values for the y-direction
        {
        test_t test = test_init(" * grid3_fill_y", rank, size);
        int i=0, j=0, k=0;
        err |= s_assert(fabs(y3[grid_index(grid, i, j, k)] - y1[j]) < FLTOL);
        i = n - 1;           
        err |= s_assert(fabs(y3[grid_index(grid, i, j, k)] - y1[j]) < FLTOL);
        k = n - 1;           
        err |= s_assert(fabs(y3[grid_index(grid, i, j, k)] - y1[j]) < FLTOL);
        k = n - 1;           
        err |= s_assert(fabs(y3[grid_index(grid, i, j, k)] - y1[j]) < FLTOL);
        err |= test_finalize(&test, err);
        }

        // Check a few values for the z-direction
        {
        test_t test = test_init(" * grid3_fill_z", rank, size);
        int i=0, j=0, k=0;
        err |= s_assert(fabs(z3[grid_index(grid, i, j, k)] - z1[k]) < FLTOL);
        k = 2;
        i = n - 1;           
        err |= s_assert(fabs(z3[grid_index(grid, i, j, k)] - z1[k]) < FLTOL);
        j = n - 1;           
        err |= s_assert(fabs(z3[grid_index(grid, i, j, k)] - z1[k]) < FLTOL);
        k = n - 4;
        j = n - 2;           
        err |= s_assert(fabs(z3[grid_index(grid, i, j, k)] - z1[k]) < FLTOL);
        err |= test_finalize(&test, err);
        }

        free(x1);
        free(x3);
        free(y1);
        free(y3);
        free(z1);
        free(z3);
        free(ans);

        return err;
}

int test_grid3_reduce(int rank, int size)
{
        int err = 0;
        int n = 10;
        int gsize[3] = {n, n, n};
        prec h = 1.0;
        

        int3_t shift = grid_yz();
        
        int3_t coord = {.x = rank, .y = rank, .z = rank};
        int3_t asize = {gsize[0], gsize[1], gsize[2]};

        int3_t bnd1 = {1, 1, 0};
        int3_t bnd2 = {1, 1, 0};

        fcn_grid_t grid = grid_init(asize, shift, coord, bnd1, bnd2, 0, h);
        
        test_t test = test_init(" * reduce", rank, size);

        size_t num_elements = grid.mem.x * grid.mem.y * grid.mem.z;
        prec *x3 = malloc(sizeof(x3) * num_elements);
        array_fill(x3, 1.0, num_elements);
        double sum = grid_reduce3(x3, grid);
        double ans = n * n * n;
        err |= s_assert( fabs(sum  - ans) < FLTOL);
        err = test_finalize(&test, err);

        free(x3);

        return test_last_error();
}

int test_shift(int rank, int size)
{
        int err = 0;
        {
        test_t test = test_init(" * shift::u1 ", rank, size);
        int3_t shift = grid_shift(GRID_U1);
        int3_t ans = grid_u1();
        err |=
            s_assert(shift.x == ans.x && shift.y == ans.y && shift.z == ans.z);
        err = test_finalize(&test, err);
        }

        {
        test_t test = test_init(" * shift::u2 ", rank, size);
        int3_t shift = grid_shift(GRID_U2);
        int3_t ans = grid_u2();
        err |=
            s_assert(shift.x == ans.x && shift.y == ans.y && shift.z == ans.z);
        err = test_finalize(&test, err);
        }

        {
        test_t test = test_init(" * shift::u3 ", rank, size);
        int3_t shift = grid_shift(GRID_U3);
        int3_t ans = grid_u3();
        err |=
            s_assert(shift.x == ans.x && shift.y == ans.y && shift.z == ans.z);
        err = test_finalize(&test, err);
        }

        {
        test_t test = test_init(" * shift::xx ", rank, size);
        int3_t shift = grid_shift(GRID_XX);
        int3_t ans = grid_xx();
        err |=
            s_assert(shift.x == ans.x && shift.y == ans.y && shift.z == ans.z);
        err = test_finalize(&test, err);
        }

        {
        test_t test = test_init(" * shift::yy ", rank, size);
        int3_t shift = grid_shift(GRID_YY);
        int3_t ans = grid_yy();
        err |=
            s_assert(shift.x == ans.x && shift.y == ans.y && shift.z == ans.z);
        err = test_finalize(&test, err);
        }

        {
        test_t test = test_init(" * shift::zz ", rank, size);
        int3_t shift = grid_shift(GRID_ZZ);
        int3_t ans = grid_zz();
        err |=
            s_assert(shift.x == ans.x && shift.y == ans.y && shift.z == ans.z);
        err = test_finalize(&test, err);
        }

        {
        test_t test = test_init(" * shift::xy ", rank, size);
        int3_t shift = grid_shift(GRID_XY);
        int3_t ans = grid_xy();
        err |=
            s_assert(shift.x == ans.x && shift.y == ans.y && shift.z == ans.z);
        err = test_finalize(&test, err);
        }

        {
        test_t test = test_init(" * shift::xz ", rank, size);
        int3_t shift = grid_shift(GRID_XZ);
        int3_t ans = grid_xz();
        err |=
            s_assert(shift.x == ans.x && shift.y == ans.y && shift.z == ans.z);
        err = test_finalize(&test, err);
        }

        {
        test_t test = test_init(" * shift::yz ", rank, size);
        int3_t shift = grid_shift(GRID_YZ);
        int3_t ans = grid_yz();
        err |=
            s_assert(shift.x == ans.x && shift.y == ans.y && shift.z == ans.z);
        err = test_finalize(&test, err);
        }

        return test_last_error();
}

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

    return test_last_error();

}   
