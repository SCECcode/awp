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
#include <topography/sources/source.h>
#include <topography/sources/sources.h>
#include <readers/input.h>
int test_sources_dm(const char *inputfile, int rank, int size, const int px);

int main(int argc, char **argv)
{
        int rank, size;
        MPI_Init(&argc, &argv);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN); 
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        char inputfile[STR_LEN];
        int px = 2;

        if (argc == 2) {
                assert(strlen(argv[1]) < STR_LEN);
                sprintf(inputfile, "%s", argv[1]);
        }
        else {
                sprintf(inputfile, "../tests/fixtures/source_dm.txt");
        }

        if (rank == 0) {
                test_divider();
                printf("Testing test_sources_dm.c\n");
        }

        test_sources_dm(inputfile, rank, size, px);

        if (rank == 0) {
                printf("Testing completed.\n");
                test_divider();
        }

        MPI_Finalize();

        return test_last_error();
}

int test_sources_dm(const char *inputfile, int rank, int size, const int px) 
{
        int coord_x = rank / px;
        int coord_y = rank % px;
        int coord_z = 0;
        // Grid points on the coarse grid
        int nx = 11;
        int ny = 11;
        int nz = 11;
        prec h = 1.0;
        int ngrids = 3;
        int err = 0;
        grids_t grids[3] = {
            grids_init(9 * nx, 9 * ny, nz, coord_x, coord_y, coord_z, 0,
                       h),
            grids_init(3 * nx, 3 * ny, nz, coord_x, coord_y, coord_z, 0, 3 * h), 
            grids_init(nx, ny, nz, coord_x, coord_y, coord_z, 0, 9 * h)

        };

        test_t test;

        test = test_init(" * sources_dm", rank, size);
        sources_init(inputfile, grids, ngrids, NULL, NULL, MPI_COMM_WORLD, rank, size);
        source_t Mxx = sources_get_source(XX);

        for (size_t i = 0; i < (size_t)ngrids; ++i) {
                grid3_t xx = grids[i].xx;
                printf("   - Grid: %ld, grid spacing: %g \n", i, xx.gridspacing); 
                for (size_t j = 0; j < Mxx.lengths[i]; ++j) {

                grid3_t vel_grid = grid_init_velocity_grid(
                                   xx.inner_size, xx.shift, xx.coordinate,
                                   xx.boundary1, xx.boundary2, xx.gridspacing);
                grid1_t x_grid = grid_grid1_x(vel_grid);
                grid1_t y_grid = grid_grid1_y(vel_grid);
                grid1_t z_grid = grid_grid1_z(vel_grid);

                prec *x1 = malloc(sizeof x1 * x_grid.size);
                prec *y1 = malloc(sizeof y1 * y_grid.size);
                prec *z1 = malloc(sizeof z1 * z_grid.size);

                grid_fill1(x1, x_grid);
                grid_fill1(y1, y_grid);
                grid_fill1(z1, z_grid);
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
                // coordinates system that shifts by -0.5 * grid spacings in the 
                // y and z-directions (see shift.c, xx = [0, 1, 1]), 
                // but with adjustments to the x-direction,
                // and y-directions due to the DM and due having (0, 0, 0) at a
                // material point in the user coordinate system.
                //
                //
                int ix = Mxx.interpolation[i].ix[j] - ngsl;
                int iy = Mxx.interpolation[i].iy[j] - ngsl;
                int iz = Mxx.interpolation[i].iz[j];
                
                // Once setup has been confirmed, we can add some test cases to
                // ensure that we don't break this configuration in the future.
                if (i == 0) err |= s_assert(ix == 4);
                if (i == 1) err |= s_assert(ix == 1);
                if (i == 2) err |= s_assert(ix == 0);


                //FIXME: Resolve the y-direction
                //if (i == 0) err |= s_assert(iy == 2);

                printf("     - Mxx(%ld), index         = [%d, %d, %d]\n"\
                       "               user(x, y, z) = [%g, %g, %g],\n"\
                       "               int(x, y, z)  = [%g, %g, %g]\n"\
                       "               int x = [%g %g %g ... ]\n"\
                       "               int y = [%g %g %g ... ]\n"\
                       "               int z = [%g %g %g ... ]\n", 
                                j, 
                                ix, iy, iz,
                                Mxx.xu[i][j], Mxx.yu[i][j], Mxx.zu[i][j], 
                                Mxx.x[i][j], Mxx.y[i][j], Mxx.z[i][j],
                                x1[0], x1[1], x1[2], 
                                y1[0], y1[1], y1[2],
				z1[0], z1[1], z1[2]);
                free(x1);
                free(y1);
                free(z1);

                }
        }
        err = test_finalize(&test, err);
        
        sources_finalize();
        grids_finalize(grids);

        return test_last_error();
}
