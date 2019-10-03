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

#include <awp/definitions.h>
#include <test/test.h>
#include <awp/error.h>
#include <utils/array.h>
#include <topography/sources/sources.h>
#include <readers/input.h>
int test_sources(const char *inputfile, int rank, int size, const int px);
void write_source(const char *filename, size_t num_sources, size_t num_steps);

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
                sprintf(inputfile, "../tests/fixtures/source.txt");
        }

        if (rank == 0) {
                test_divider();
                printf("Testing test_sources.c\n");
        }

        test_sources(inputfile, rank, size, px);

        if (rank == 0) {
                printf("Testing completed.\n");
                test_divider();
        }

        MPI_Finalize();

        return test_last_error();
}

int test_sources(const char *inputfile, int rank, int size, const int px) 
{
        int coord_x = rank / px;
        int coord_y = rank % px;
        int coord_z = 0;
        int nx = 64;
        int ny = 64;
        int nz = 128;
        prec h = 1.0/nx;
        prec dt = 0.1;
        int ngrids = 1;
        int err = 0;
        grids_t grids[] = {
            grids_init(nx, ny, nz, coord_x, coord_y, coord_z, 0, h)};

        test_t test;

        test = test_init(" * sources_init", rank, size);
        sources_init(inputfile, grids, ngrids, NULL, MPI_COMM_WORLD, rank, size);
        err = test_finalize(&test, err);

        input_t input;

        if (rank == 0) {
        AWPCHK(input_init(&input, inputfile));
        write_source("source_xx", input.length, input.steps);
        write_source("source_yy", input.length, input.steps);
        write_source("source_zz", input.length, input.steps);
        write_source("source_xy", input.length, input.steps);
        write_source("source_xz", input.length, input.steps);
        write_source("source_yz", input.length, input.steps);
        }

        AWPCHK(input_broadcast(&input, rank, 0, MPI_COMM_WORLD));
        
        test = test_init(" * sources_read", rank, size);

        for (size_t step = 0; step < input.steps; ++step) {
                sources_read(step);
        }

        err = test_finalize(&test, err);

        grids_t grid = grids[0];
        prec *d_xx, *d_yy, *d_zz, *d_xy, *d_xz, *d_yz;
        CUCHK(cudaMalloc((void**)&d_xx, sizeof d_xx * grid.xx.num_bytes));
        CUCHK(cudaMalloc((void**)&d_yy, sizeof d_yy * grid.yy.num_bytes));
        CUCHK(cudaMalloc((void**)&d_zz, sizeof d_zz * grid.zz.num_bytes));
        CUCHK(cudaMalloc((void**)&d_xy, sizeof d_xy * grid.xy.num_bytes));
        CUCHK(cudaMalloc((void**)&d_xz, sizeof d_xz * grid.xz.num_bytes));
        CUCHK(cudaMalloc((void**)&d_yz, sizeof d_yz * grid.yz.num_bytes));
        CUCHK(cudaMemset(d_xx, 0.0, grid.xx.num_bytes));
        CUCHK(cudaMemset(d_yy, 0.0, grid.yy.num_bytes));
        CUCHK(cudaMemset(d_zz, 0.0, grid.zz.num_bytes));
        CUCHK(cudaMemset(d_xy, 0.0, grid.xy.num_bytes));
        CUCHK(cudaMemset(d_xz, 0.0, grid.xz.num_bytes));
        CUCHK(cudaMemset(d_yz, 0.0, grid.yz.num_bytes));
        
        test = test_init(" * sources_add", rank, size);
        for (size_t step = 0; step < input.steps; ++step) {
                sources_read(step);
                sources_add_cartesian(d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, step,
                                      h, dt, 0);
        }

        err = test_finalize(&test, err);
        
        sources_finalize();
        if (rank == 0) {
                remove("source_xx");
                remove("source_yy");
                remove("source_zz");
                remove("source_xy");
                remove("source_xz");
                remove("source_yz");
        }
        grids_finalize(grids);

        CUCHK(cudaFree(d_xx));
        CUCHK(cudaFree(d_yy));
        CUCHK(cudaFree(d_zz));
        CUCHK(cudaFree(d_xy));
        CUCHK(cudaFree(d_xz));
        CUCHK(cudaFree(d_yz));

        return test_last_error();
}


void write_source(const char *filename, size_t num_sources, size_t num_steps)
{
        FILE *fp = fopen(filename, "wb");
        assert(fp != NULL);
        prec *src = malloc(sizeof(prec) * num_steps);

        array_range(src, num_steps);

        array_addc(src, num_steps, 1);
        for (size_t j = 0; j < num_sources; ++j) {
                fwrite(src, sizeof(prec), num_steps, fp);
        }
        fclose(fp);
        free(src);
}
