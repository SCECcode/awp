#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <argparse/argparse.h>
#include <topography/topography.h>
#include <topography/initializations/constant.h>
#include <topography/initializations/random.h>
#include <topography/initializations/cerjan.h>
#include <test/check.h>
#include <mpi/partition.h>

#ifdef USE_OPTIMIZED_KERNELS
#include <topography/velocity.cuh>
#include <topography/stress.cuh>
#else
#include <topography/topography.cuh>
#endif
#include <topography/geometry.h>
#include <topography/host.h>
 
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
static int use_optimized_kernels = USE_OPTIMIZED_KERNELS;
static const char *outputdir;
static const char *inputdir;
static int run_velocity = 1;
static int run_stress = 1;

void init(topo_t *T);
void run(topo_t *T);
void write(topo_t *host, const char *outputdir);
void write_file(const char *path, const char *filename, const _prec *data, const
                int size);

int compare(topo_t *host, const char *inputdir);
void read_file(const char *path, const char *filename, _prec *data, const int
                size);

int main(int argc, char **argv)
{
        MPI_Init(&argc, &argv);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN); 
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        printf("Optimized kernels: %d.\n", use_optimized_kernels);
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
            OPT_STRING('o', "output", &outputdir,
                        "Write results to output directory", NULL, 0, 0),
            OPT_STRING('i', "input", &inputdir,
                        "Read results from input directory", NULL, 0, 0),
            OPT_INTEGER('s', "stress", &run_stress,
                        "Run stress kernels", NULL, 0, 0),
            OPT_INTEGER('v', "velocity", &run_velocity,
                        "Run velocity kernels", NULL, 0, 0),
            OPT_END(),
        };

        struct argparse argparse;
        argparse_init(&argparse, options, usages, 0);
        argparse_describe(
            &argparse,
            "\nPerformance analysis of CUDA compute kernels for AWP.", "\n");
        argc = argparse_parse(&argparse, argc, (const char**)argv);

        dim[0] = px;
        dim[1] = py;
        
        int period[2] = {0, 0};
        int err = 0;
        MPI_Comm comm;
        err = mpi_partition_2d(rank, dim, period, coord, &side, &comm);
        assert(err == 0);

        topo_t device;
        topo_t host;

        init(&device);
        host = device;
        topo_h_malloc(&host);
        cudaProfilerStart();
        run(&device);
        cudaDeviceSynchronize();
        cudaProfilerStop();

        topo_dtoh(&host, &device);
        write(&host, outputdir);
        err = compare(&host, inputdir);

        topo_h_free(&host);
        topo_d_free(&device);
        topo_free(&device);

        MPI_Finalize();

        return 0;
}

void init(topo_t *T)
{
        *T = topo_init(1, "", rank, side.left, side.right, side.front,
                              side.back, coord, px, py, nx, ny, nz, dt, h,
                              stream_1, stream_2, stream_i);
        topo_d_malloc(T);
        topo_d_zero_init(T);
        topo_d_cerjan_disable(T);
        topo_init_metrics(T);
        topo_init_grid(T);

        // Gaussian hill geometry
        _prec3_t hill_width = {.x = (_prec)nx / 2, .y = (_prec)ny / 2, .z = 0};
        _prec hill_height = nz;
        _prec3_t hill_center = {.x = 0, .y = 0, .z = 0};
        // No canyon
        _prec3_t canyon_width = {.x = 100, .y = 100, .z = 0};
        _prec canyon_height = 0;
        _prec3_t canyon_center = {.x = 0, .y = 0, .z = 0};
        topo_init_gaussian_hill_and_canyon(T, hill_width, hill_height,
                                           hill_center, canyon_width,
                                           canyon_height, canyon_center);

        // Set random initial conditions using fixed seed
        topo_d_random(T, 1, T->u1);
        topo_d_random(T, 2, T->v1);
        topo_d_random(T, 3, T->w1);

        topo_d_random(T, 4, T->xx);
        topo_d_random(T, 5, T->yy);
        topo_d_random(T, 6, T->zz);
        topo_d_random(T, 7, T->xy);
        topo_d_random(T, 8, T->xz);
        topo_d_random(T, 9, T->yz);

        topo_d_constant(T, 1, T->mui);
        topo_d_constant(T, 1, T->lami);

        topo_build(T);
}

void run(topo_t *T)
{
        for(int iter = 0; iter < nt; ++iter) {
                if (run_velocity) {
                        topo_velocity_interior_H(T);
                        topo_velocity_front_H(T);
                        topo_velocity_back_H(T);
                }

                CUCHK(cudaStreamSynchronize(T->stream_1));
                CUCHK(cudaStreamSynchronize(T->stream_2));
                CUCHK(cudaStreamSynchronize(T->stream_i));
                cudaDeviceSynchronize();

                if (run_stress) {
                        topo_stress_interior_H(T);
                        topo_stress_left_H(T);
                        topo_stress_right_H(T);
                }

                CUCHK(cudaStreamSynchronize(T->stream_1));
                CUCHK(cudaStreamSynchronize(T->stream_2));
                CUCHK(cudaStreamSynchronize(T->stream_i));
                cudaDeviceSynchronize();
        }
}

void write(topo_t *host, const char *outputdir)
{
        if (!outputdir) {
                return;
        }
        printf("writing to directory: %s \n", outputdir);

        mkdir(outputdir, 0700);

        int size = host->mx * host->my * host->mz;
        write_file(outputdir, "vx.bin", host->u1, size);
        write_file(outputdir, "vy.bin", host->v1, size);
        write_file(outputdir, "vz.bin", host->w1, size);
        write_file(outputdir, "xx.bin", host->xx, size);
}

void write_file(const char *path, const char *filename, const _prec *data,
                const int size) 
{
        char output[512];
        sprintf(output, "%s/%s", path, filename);
        FILE *fh = fopen(output, "wb");
        if (!fh) {
                printf("Unable to open: %s. \n", filename);
                MPI_Abort(MPI_COMM_WORLD, 1);
                exit(1);
        }
        fwrite(data, sizeof(prec), size, fh);
        fclose(fh);

}

int compare(topo_t *host, const char *inputdir)
{
        if (!inputdir) {
                return 0;
        }

        topo_t reference = *host;
        topo_h_malloc(&reference);

        int size = host->mx * host->my * host->mz;
        printf("reading from directory: %s \n", inputdir);
        read_file(inputdir, "vx.bin", reference.u1, size);
        read_file(inputdir, "vy.bin", reference.v1, size);
        read_file(inputdir, "vz.bin", reference.w1, size);
        read_file(inputdir, "xx.bin", reference.xx, size);

        prec *a[4] = {reference.u1, reference.v1, reference.w1, reference.xx};
        prec *b[4] = {host->u1, host->v1, host->w1, host->xx};
        const char *names[4] = {"vx", "vy", "vz", "xx"};
        double err[4];
        double total_error = 0;
        for (int i = 0; i < 4; ++i) {
                err[i] = chk_inf(a[i], b[i], size);
                printf("%s: %g ", names[i], err[i]);
                total_error += err[i];
        }
        printf("\n");

        topo_h_free(&reference);
        return total_error > 1e-6;
}

void read_file(const char *path, const char *filename, _prec *data, const int
                size)
{
        char input[512];
        sprintf(input, "%s/%s", path, filename);

        FILE *fh = fopen(input, "rb");
        if (!fh) {
                printf("Unable to open: %s. \n", filename);
                MPI_Abort(MPI_COMM_WORLD, 1);
                exit(1);
        }
        int count = fread(data, sizeof(prec), size, fh);
        fclose(fh);
}
