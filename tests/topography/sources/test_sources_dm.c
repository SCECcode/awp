#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define STR_LEN 2048
#define ADDLINENUM 1
#define ADDRANK 1
#define RANK rank
#define STR_LEN 2048

#include <awp/definitions.h>
#include <test/test.h>
#include <awp/error.h>
#include <utils/array.h>
#include <grid/shift.h>
#include <topography/grids.h>
#include <topography/sources/source.h>
#include <topography/sources/sources.h>
#include <topography/receivers/receivers.h>
#include <readers/input.h>
int test_sources_dm(const char *inputfile, int rank, int size, const int px, const enum grid_types grid, const enum eshift grid2, const int3_t src1, const int3_t src2, const int3_t src3, int run_test);

int main(int argc, char **argv)
{
        int rank, size;
        MPI_Init(&argc, &argv);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN); 
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);


        int px = 2;


        int3_t src1 = {9, 9, 0};
        int3_t src2 = {3, 3, 0};
        int3_t src3 = {1, 1, 0};

        if (argc == 2) {
            printf("Source input file: %s \n", argv[1]);
            test_sources_dm(argv[1], rank, size, px, XY, GRID_XY, src1, src2, src3, 0);
            test_sources_dm(argv[1], rank, size, px, XZ, GRID_XZ, src1, src2, src3, 0);
            test_sources_dm(argv[1], rank, size, px, YZ, GRID_YZ, src1, src2, src3, 0);
            test_sources_dm(argv[1], rank, size, px, XX, GRID_XX, src1, src2, src3, 0);
            test_sources_dm(argv[1], rank, size, px, X, GRID_U1, src1, src2, src3, 0);
            test_sources_dm(argv[1], rank, size, px, Y, GRID_U2, src1, src2, src3, 0);
            test_sources_dm(argv[1], rank, size, px, Z, GRID_U3, src1, src2, src3, 0);
        } else {

            if (rank == 0) {
                    test_divider();
                    printf("Testing test_sources_dm.c\n");
            }

            test_sources_dm("source_xy.txt", rank, size, px, XY, GRID_XY, src1, src2, src3, 1);
            test_sources_dm("source_xz.txt", rank, size, px, XZ, GRID_XZ, src1, src2, src3, 1);
            test_sources_dm("source_yz.txt", rank, size, px, YZ, GRID_YZ, src1, src2, src3, 1);
            test_sources_dm("source_xx.txt", rank, size, px, XX, GRID_XX, src1, src2, src3, 1);
            test_sources_dm("source_x.txt", rank, size, px, X, GRID_U1, src1, src2, src3, 1);
            test_sources_dm("source_y.txt", rank, size, px, Y, GRID_U2, src1, src2, src3, 1);
            test_sources_dm("source_z.txt", rank, size, px, Z, GRID_U3, src1, src2, src3, 1);

            if (rank == 0) {
                    printf("Testing completed.\n");
                    test_divider();
            }

        }


        MPI_Finalize();

        return test_last_error();
}

int test_sources_dm(const char *sourcefile, int rank, int size, const int px, const enum grid_types grid, const enum eshift grid2, const int3_t src1, const int3_t src2, const int3_t src3, int run_test)
{
 
        char inputfile[STR_LEN];
        if (run_test)
            sprintf(inputfile, "../../../../tests/fixtures/%s", sourcefile);
        else 
            sprintf(inputfile, "%s", sourcefile);
                                       
        int coord_x = rank / px;
        int coord_y = rank % px;
        int coord_z = 0;
        // Grid points on the coarse grid
        int nx = 11;
        int ny = 11;
        int nz = 11;;
        prec h = 1.0;
        int ngrids = 3;
        int err = 0;
        int nzs[3] = {nz, nz, nz};
        grids_t grids[3] = {
            grids_init(9 * nx, 9 * ny, nz, coord_x, coord_y, coord_z, 0,
                       h),
            grids_init(3 * nx, 3 * ny, nz, coord_x, coord_y, coord_z, 0, 3 * h), 
            grids_init(nx, ny, nz, coord_x, coord_y, coord_z, 0, 9 * h)

        };

        test_t test;

        char testname[STR_LEN];
        sprintf(testname, " * sources_dm: %s", grid_shift_label(grid2));

        if (run_test) test = test_init(testname, rank, size);
        source_t M;
        if (grid2 == GRID_U1 || grid2 == GRID_U2 || grid2 == GRID_U3) {
            receivers_init(inputfile, grids, NULL, ngrids, NULL, MPI_COMM_WORLD, rank, size);
            M = receivers_get_receiver(grid);
        }
        else {
            sources_init(inputfile, grids, NULL, ngrids, NULL, NULL, MPI_COMM_WORLD, rank, size);
            M = sources_get_source(grid);
        }

        for (size_t i = 0; i < (size_t)ngrids; ++i) {
                grid3_t x = grids_select(grid, &grids[i]);

                if (!run_test) {
                    printf("   - Grid: %ld, grid spacing: %g \n", i, x.gridspacing); 
                }
                for (size_t j = 0; j < M.lengths[i]; ++j) {

                grid3_t vel_grid = grid_init_velocity_grid(
                                   x.inner_size, x.shift, x.coordinate,
                                   x.boundary1, x.boundary2, x.gridspacing);
                grid1_t x_grid = grid_grid1_x(vel_grid);
                grid1_t y_grid = grid_grid1_y(vel_grid);
                grid1_t z_grid = grid_grid1_z(vel_grid);

                prec *x1 = malloc(sizeof x1 * x_grid.size);
                prec *y1 = malloc(sizeof y1 * y_grid.size);
                prec *z1 = malloc(sizeof z1 * z_grid.size);

                grid_fill1(x1, x_grid, 1);
                grid_fill1(y1, y_grid, 0);
                grid_fill1(z1, z_grid, 0);
                // The user coordinate system (user) defines (0, 0, 0) at
                // material grid point and is a global coordinate system (a
                // single coordinate system defined for all blocks, irrespective
                // of MPI partition)
                //
                // The internal coordinate system (int) is local with respect to
                // each block and mpi partition. 
                //
                // However, Mxx.x, Mxx.y, Mxx.z contains the coordinates of the
                // source at a normal stress position in the internal
                // coordinates system that shifts by (0.5 h, -0.5 h, -0.5 h)
                // (see shift.c, xx = [1, 1, 1]), 
                //
                //
                int ix = M.interpolation[i].ix[j] - ngsl - 2;
                int iy = M.interpolation[i].iy[j] - ngsl - 2;
                int iz = M.interpolation[i].iz[j];
                
                // Once setup has been confirmed, we can add some test cases to
                // ensure that we don't break this configuration in the future.
                if (run_test) {
                    if (i == 0) err |= s_assert(ix == src1.x);
                    if (i == 1) err |= s_assert(ix == src2.x);
                    if (i == 2) err |= s_assert(ix == src3.x);

                    if (i == 0) err |= s_assert(iy == src1.y);
                    if (i == 1) err |= s_assert(iy == src2.y);
                    if (i == 2) err |= s_assert(iy == src3.y);
                }

                // Check that the global z-coordinate maps to the correct local z-coordinate
                _prec zloc;
                int block_index;
                global_to_local(&zloc, &block_index, M.zu[i][j],
                     1.0, nzs, 3, 0);
                err |= s_assert((size_t)block_index == i);
                err |= s_assert(fabs(zloc - M.z[i][j]) < FLTOL);


                if (err > 0 || !run_test)
                printf("     - %s(%ld), index         = [%d, %d, %d]\n"\
                       "               user(x, y, z) = [%g, %g, %g],\n"\
                       "               int(x, y, z)  = [%g, %g, %g]\n"\
                       "               int x = [%g %g %g ... ]\n"\
                       "               int y = [%g %g %g ... ]\n"\
                       "               int z = [%g %g %g ... ]\n", 
                                grid_shift_label(grid2), j, 
                                ix, iy, iz,
                                M.xu[i][j], M.yu[i][j], M.zu[i][j], 
                                M.x[i][j], M.y[i][j], M.z[i][j],
                                x1[0], x1[1], x1[2], 
                                y1[0], y1[1], y1[2],
				z1[0], z1[1], z1[2]);
                free(x1);
                free(y1);
                free(z1);

                }
        }
        if (run_test)
        err = test_finalize(&test, err);
        
        if (grid2 == GRID_U1 || grid2 == GRID_U2 || grid2 == GRID_U3) {
            receivers_finalize();
        } else {
            sources_finalize();
        }
        grids_finalize(grids);

        return test_last_error();
}
