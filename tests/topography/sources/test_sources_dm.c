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
        int ngrids = 2;
        int err = 0;
        grids_t grids[2] = {
            grids_init(3 * nx, 3 * ny, nz, coord_x, coord_y, coord_z, 0,
                       h),
            grids_init(nx, ny, nz, coord_x, coord_y, coord_z, 0, 3 * h)

        };

        test_t test;

        test = test_init(" * sources_dm", rank, size);
        sources_init(inputfile, grids, ngrids, NULL, MPI_COMM_WORLD, rank, size);
        err = test_finalize(&test, err);
        source_t Mxx = sources_get_source(XX);

        for (size_t i = 0; i < (size_t)ngrids; ++i) {
                printf("   - Grid: %ld \n", i); 
                for (size_t j = 0; j < Mxx.lengths[i]; ++j) {

                printf("     - Mxx(%ld), index: [%d, %d, %d], " \
                       "int(x, y, z) = [%f, %f, %f] \n", 
                                j, 
                                Mxx.interpolation[i].ix[j],
                                Mxx.interpolation[i].iy[j],
                                Mxx.interpolation[i].iz[j],
                                Mxx.x[i][j], Mxx.y[i][j], Mxx.z[i][j]);
                }
        }
        
        sources_finalize();
        grids_finalize(grids);

        return test_last_error();
}
