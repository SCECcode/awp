#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <argparse/argparse.h>
#include <topography/topography.h>
#include <topography/initializations/constant.h>
#include <topography/initializations/linear.h>
#include <topography/initializations/random.h>
#include <topography/initializations/cerjan.h>
#include <test/check.h>
#include <test/grid_check.h>
#include <mpi/partition.h>

#include <awp/kernel.h>

#include <topography/velocity.cuh>
#include <topography/stress_attenuation.cuh>
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
static int nx = 64;
static int ny = 64;
static int nz = 64;
static int nt = 10;
static prec h = 1.0;
static prec dt = 0.5/3;
static int coord[2] = {0, 0};
static int dim[2] = {0, 0};
static int rank, size;
static struct side_t side;
static cudaStream_t stream_1, stream_2, stream_i;

void init(topo_t *T);
void init_awp(topo_t *T);
void run(topo_t *T, topo_t *awp);
void write(topo_t *host, const char *outputdir);
void write_file(const char *path, const char *filename, const _prec *data, const
                int size);

int compare(topo_t *host, topo_t *awp);
void read_file(const char *path, const char *filename, _prec *data, const int
                size);

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
            "\nTest of Frequency dependent Q for AWP-TOPO.", "\n");
        argc = argparse_parse(&argparse, argc, (const char**)argv);

        dim[0] = px;
        dim[1] = py;
        
        int period[2] = {0, 0};
        int err = 0;
        MPI_Comm comm;
        err = mpi_partition_2d(rank, dim, period, coord, &side, &comm);
        assert(err == 0);

        topo_t topo, topo_h;
        topo_t awp, awp_h;

        init(&topo);
        init(&awp);
        init_awp(&awp);
        run(&topo, &awp);
        cudaDeviceSynchronize();

        topo_h = topo;
        awp_h = awp;
        topo_h_malloc(&topo_h);
        topo_h_malloc(&awp_h);

        topo_dtoh(&topo_h, &topo);
        topo_dtoh(&awp_h, &awp);

        err = compare(&topo_h, &awp_h);

        topo_d_free(&topo);
        topo_d_free(&awp);
        topo_h_free(&topo_h);
        topo_h_free(&awp_h);
        topo_free(&topo);
        topo_free(&awp);

        MPI_Finalize();

        return err;
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
        _prec hill_height = 0;
        _prec3_t hill_center = {.x = 0, .y = 0, .z = 0};
        // No canyon
        _prec3_t canyon_width = {.x = 100, .y = 100, .z = 0};
        _prec canyon_height = 0;
        _prec3_t canyon_center = {.x = 0, .y = 0, .z = 0};
        topo_init_gaussian_hill_and_canyon(T, hill_width, hill_height,
                                           hill_center, canyon_width,
                                           canyon_height, canyon_center);

        // Set random initial conditions using fixed seed
        topo_d_linear_i(T, T->u1);
        topo_d_constant(T, 0, T->v1);
        topo_d_constant(T, 0, T->w1);

        topo_d_constant(T, 1, T->xx);
        topo_d_random(T, 5, T->yy);
        topo_d_random(T, 6, T->zz);
        topo_d_random(T, 7, T->xy);
        topo_d_random(T, 8, T->xz);
        topo_d_random(T, 9, T->yz);

        topo_d_constant(T, 1, T->r1);
        topo_d_constant(T, 0, T->r2);
        topo_d_constant(T, 0, T->r3);
        topo_d_constant(T, 0, T->r4);
        topo_d_constant(T, 0, T->r5);
        topo_d_constant(T, 0, T->r6);

        topo_d_constant(T, 0, T->qpi);
        topo_d_constant(T, 0, T->qsi);
        
        topo_d_constant(T, 1.0, T->dcrjx);
        topo_d_constant(T, 1.0, T->dcrjy);
        topo_d_constant(T, 1.0, T->dcrjz);

        topo_d_constant(T, 1.0, T->wwo);
        topo_d_constanti(T, 1, T->ww);
        topo_d_constant(T, 1.0, T->vx1);
        topo_d_constant(T, 1.0, T->vx2);
        topo_d_constant(T, 1.0, T->coeff);

        topo_d_random(T, 1, T->mui);
        topo_d_random(T, 1, T->lami);
        topo_d_constant(T, 1, T->mui);
        topo_d_constant(T, 1, T->lami);
        topo_d_constant(T, 1, T->lam_mu);
        topo_build(T);

        topo_set_constants(T);
}

void init_awp(topo_t *T)
{
        _prec fmajor = 0, fminor = 0, Rz[9], RzT[9];
        printf("Initializing AWP: %d %d %d\n", nx, ny, nz);
        SetDeviceConstValue(&h, dt, &nx, &ny, &nz, 1, fmajor, fminor, Rz, RzT);
}

void run(topo_t *topo, topo_t *awp)
{
        for(int iter = 0; iter < nt; ++iter) {

               topo_stress_interior_H(topo);

	       dstrqc_H_new(awp->xx, awp->yy, awp->zz, awp->xy, awp->xz, awp->yz,
	        	awp->r1, awp->r2, awp->r3, awp->r4, awp->r5, awp->r6,
	        	awp->u1, awp->v1, awp->w1, awp->lami,
	        	awp->mui, awp->qpi, awp->coeff, awp->qsi, awp->dcrjx, awp->dcrjy, awp->dcrjz,
	        	ny,  nz,  awp->stream_1, awp->lam_mu,
	        	awp->vx1, awp->vx2, awp->ww, awp->wwo,
	        	nx, 0,  coord[0], coord[1],   ngsl + 2,  nx + ngsl2 - 1,
	        	 2 + ngsl,  ny + ngsl2 - 1, 0);

        }
}

int compare(topo_t *topo, topo_t *awp)
{

        prec *a[3] = {awp->xx, awp->r1, awp->w1};
        prec *b[3] = {topo->xx, topo->r1, topo->w1};
        const char *names[3] = {"sxx", "r1", "vz"};
        double err[3];
        int nxt = nx - ngsl;
        int nyt = ny - ngsl;
        int nzt = 50;
        double total_error = 0;
        int excl = 4;
        int i0 = excl + ngsl + 2;
        int in = i0 + nxt;
        int j0 = excl + ngsl + 2;
        int jn = j0 + nyt;
        int nbnd = 8;
        int k0 = align + excl + nbnd;
        int kn = k0 + nzt;
        int size = (in - i0) * (jn - j0) * (kn - k0);
        printf("Comparing in region [%d %d %d] [%d %d %d], size = %d \n", i0, j0, k0,
                        in, jn, kn,  size);
        for (int i = 0; i < 2; ++i) {
             err[i] = check_flinferr(a[i], b[i], 
                             i0, in, j0, jn, k0, kn,
                             topo->line, topo->slice);
                printf("%s: %g ", names[i], err[i]);
                total_error += err[i];
        }
        printf("\n");

        topo_h_free(&reference);
        return total_error > 1e-6;
}
