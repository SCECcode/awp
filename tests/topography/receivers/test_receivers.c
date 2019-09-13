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

#ifndef REMOVE_TEMPORARY_FILES
#define REMOVE_TEMPORARY_FILES 1
#endif

#include <awp/definitions.h>
#include <test/test.h>
#include <topography/receivers/receivers.h>
#include <topography/receivers/receiver.h>
#include <topography/grids.h>
#include <topography/input/input.h>

int test_receivers(const char *inputfile, int rank, int size, const int px);

int main(int argc, char **argv)
{
        int err = 0;
        int rank, size;
        MPI_Init(&argc, &argv);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN); 
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        char inputfile[] = "fixtures/receiver.txt";
        int px = 2;

        if (rank == 0) {
                test_divider();
                printf("Testing test_receivers.c\n");
        }

        err = test_receivers(inputfile, rank, size, px);

        if (rank == 0) {
                printf("Testing completed.\n");
                test_divider();
        }

        MPI_Finalize();

        return test_last_error();
}

int test_receivers(const char *inputfile, int rank, int size, const int px) 
{
        int coord_x = rank / px;
        int coord_y = rank % px;
        int coord_z = 0;
        int nx = 64;
        int ny = 64;
        int nz = 128;
        prec h = 1.0/nx;
        int ngrids = 1;
        size_t num_steps = 48;
        int err = 0;
        grids_t grids[] = {
            grids_init(nx, ny, nz, coord_x, coord_y, coord_z, 0, h)};

        grids_t grid = grids[0];

        prec *d_vx, *d_vy, *d_vz;

        CUCHK(cudaMalloc((void**)&d_vx, sizeof d_vx * grid.x.num_bytes));
        CUCHK(cudaMemset(d_vx, sizeof d_vx, grid.x.num_bytes));
        CUCHK(cudaMalloc((void**)&d_vy, sizeof d_vy * grid.y.num_bytes));
        CUCHK(cudaMemset(d_vy, sizeof d_vy, grid.y.num_bytes));
        CUCHK(cudaMalloc((void**)&d_vz, sizeof d_vz * grid.z.num_bytes));
        CUCHK(cudaMemset(d_vz, sizeof d_vz, grid.z.num_bytes));

        input_t input;

        if (rank == 0) {
                AWPCHK(input_init(&input, inputfile));
        }
        AWPCHK(input_broadcast(&input, rank, 0, MPI_COMM_WORLD));

        test_t test;

        test = test_init(" * receivers_init", rank, size);
        receivers_init(inputfile, grids, ngrids, NULL, MPI_COMM_WORLD, rank, size);
        err = test_finalize(&test, err);
        
        test = test_init(" * receivers_write", rank, size);

        size_t *steps = malloc(sizeof steps * num_steps);
        for (size_t i = 0; i < num_steps; ++i) {
                receivers_write(d_vx, d_vy, d_vz, i, num_steps);
                steps[i] = receivers_last_step();
        }
        
        receivers_finalize();
        
        // Remove all temporary files created
        if (rank == 0 && REMOVE_TEMPORARY_FILES) {
                char outfile[STR_LEN];
                for (size_t i = 0; i < num_steps; ++i) {
                        receivers_step_format(outfile, steps[i], "recv_x");
                        FILE *fp = fopen(outfile, "rb");
                        assert(fp != NULL);
                        fseek(fp, 0, SEEK_END); 
                        size_t file_size = ftell(fp); 
                        fclose(fp);
                        err |=
                            s_assert(file_size == sizeof(prec) * input.length *
                                                      input.gpu_buffer_size *
                                                      input.cpu_buffer_size *
                                                      input.num_writes);

                }

                for (size_t i = 0; i < num_steps; ++i) {
                        remove(outfile);
                        receivers_step_format(outfile, steps[i], "recv_x");
                        remove(outfile);
                        receivers_step_format(outfile, steps[i], "recv_y");
                        remove(outfile);
                        receivers_step_format(outfile, steps[i], "recv_z");
                        remove(outfile);
                }
        }

        err = test_finalize(&test, err);

        free(steps);
        grids_finalize(grids);

        CUCHK(cudaFree(d_vx));
        CUCHK(cudaFree(d_vy));
        CUCHK(cudaFree(d_vz));

        return test_last_error();
}

