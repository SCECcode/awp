#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <argparse/argparse.h>
#include <topography/topography.h>
#include <topography/topography.cuh>
#include <topography/examples/constant.h>
#include <mpi/partition.h>
#include <topography/kernels/unoptimized.cuh>
#include <topography/geometry.h>
 
static const char *const usages[] = {
    "topography_kernels [options] [[--] args]",
    "topography_kernels [options]",
    NULL,
};

static topo_t reference;
static int px = 0;
static int py = 0;
static int nx = 0;
static int ny = 0;
static int nz = 0;
static int nt = 0;
static prec h = 1.0;
static prec dt = 0.25;
static int coord[2] = {0, 0};
static int dim[2] = {0, 0};
static int rank, size;
static struct side_t side;
static cudaStream_t stream_1, stream_2, stream_i;

void init(topo_t *T);
void run(topo_t *T);

int main(int argc, char **argv)
{
        MPI_Init(&argc, &argv);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN); 
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        cudaStreamCreate(&stream_1);
        cudaStreamCreate(&stream_2);
        cudaStreamCreate(&stream_i);

        struct argparse_option options[] = {
            OPT_HELP(),
            OPT_GROUP("Options"),
            OPT_INTEGER('p', "px", &px,
                        "Number of processes in the X-direction", NULL, 0, 0),
            OPT_INTEGER('q', "py", &py,
                        "Number of processes in the Y-direction", NULL, 0, 0),
            OPT_INTEGER('x', "nx", &nx,
                        "Number of grid points in the X-direction", NULL, 0, 0),
            OPT_INTEGER('y', "ny", &ny,
                        "Number of grid points in the Y-direction", NULL, 0, 0),
            OPT_INTEGER('z', "nz", &nz,
                        "Number of grid points in the Z-direction", NULL, 0, 0),
            OPT_INTEGER('t', "nt", &nt,
                        "Number of iterations to perform", NULL, 0, 0),
            OPT_END(),
        };

        struct argparse argparse;
        argparse_init(&argparse, options, usages, 0);
        argparse_describe(
            &argparse,
            "\nPerformance analysis of CUDA compute kernels for AWP.", "\n");
        argc = argparse_parse(&argparse, argc, (const char**)argv);

        if (nx != 0) printf("nx: %d\n", nx);
        if (ny != 0) printf("ny: %d\n", ny);
        if (nz != 0) printf("nz: %d\n", nz);
        if (nt != 0) printf("nt: %d\n", nt);
        dim[0] = px;
        dim[1] = py;
        
        int period[2] = {0, 0};
        int err = 0;
        MPI_Comm comm;
        err = mpi_partition_2d(rank, dim, period, coord, &side, &comm);
        assert(err == 0);

        topo_t reference;


        init(&reference);
        run(&reference);
        topo_d_free(&reference);
        topo_free(&reference);

        return 0;
}

void init(topo_t *T)
{
        *T = topo_init(1, "", rank, side.left, side.right, side.front,
                              side.back, coord, px, py, nx, ny, nz, dt, h,
                              stream_1, stream_2, stream_i);
        topo_d_malloc(T);
        topo_d_zero_init(T);
        topo_init_metrics(T);
        topo_init_grid(T);

        _prec3_t hill_width = {.x = (_prec)nx / 2, .y = (_prec)ny / 2, .z = 0};
        _prec hill_height = 1000;
        _prec3_t hill_center = {.x = 0, .y = 0, .z = 0};
        // No canyon
        _prec3_t canyon_width = {.x = 0, .y = 0, .z = 0};
        _prec canyon_height = 0;
        _prec3_t canyon_center = {.x = 0, .y = 0, .z = 0};
        topo_init_gaussian_hill_and_canyon(T, hill_width, hill_height,
                                           hill_center, canyon_width,
                                           canyon_height, canyon_center);
}

void run(topo_t *T)
{
        for(int iter = 0; iter < nt; ++iter) {
                topo_velocity_interior_H(T);
                topo_velocity_front_H(T);
                topo_velocity_back_H(T);
        }
}
