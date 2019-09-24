#include <stdio.h>
#include <time.h>
#include <math.h>
#include <errno.h>

#include <awp/definitions.h>
#include <test/test.h>
#include <topography/receivers/receivers.h>
#include <topography/receivers/receiver.h>
#include <readers/input.h>

static int use;

// Variables for defining last step output
static size_t last_step = 0;
static int leading_zeros;

static recv_t rx;
static recv_t ry;
static recv_t rz;

static input_t input;

void receivers_init(const char *filename, const grids_t *grids, int ngrids,
                    const f_grid_t *f,
                  const MPI_Comm comm, const int rank, const int size)
{
        use = strcmp(filename, "") != 0 ? 1 : 0;

        if (!use) return;

        if (rank == 0) {
                AWPCHK(input_init(&input, filename));
        }
        AWPCHK(input_broadcast(&input, rank, 0, comm));

        grids_t grid = grids[0];

       int *grid_number;

        rx = receiver_init("x", &input, grid.x, f, grid_number, rank, comm);
        ry = receiver_init("y", &input, grid.y, f, grid_number, rank, comm);
        rz = receiver_init("z", &input, grid.z, f, grid_number, rank, comm);
}

void receivers_finalize(void)
{
        if (!use) return;
        receiver_finalize(&rx);
        receiver_finalize(&ry);
        receiver_finalize(&rz);
}

void receivers_write(const prec *d_vx, const prec *d_vy, const prec *d_vz,
                     const size_t step, const size_t num_steps) {
        if (!use) return;
        char outputname[STR_LEN];
        leading_zeros = ceil(log10((double)num_steps)) + 1;
        last_step = receiver_step(input.gpu_buffer_size, input.cpu_buffer_size,
                                  input.num_writes, input.stride, step);

        receiver_filename(outputname, rx.filename, input.gpu_buffer_size,
                          input.cpu_buffer_size, input.num_writes, input.stride,
                          step, num_steps);
        receiver_write(&rx, step, outputname, d_vx);
        
        receiver_filename(outputname, ry.filename, input.gpu_buffer_size,
                          input.cpu_buffer_size, input.num_writes, input.stride,
                          step, num_steps);
        receiver_write(&ry, step, outputname, d_vy);
        
        receiver_filename(outputname, rz.filename, input.gpu_buffer_size,
                          input.cpu_buffer_size, input.num_writes, input.stride,
                          step, num_steps);
        receiver_write(&rz, step, outputname, d_vz);
}

size_t receivers_last_step(void)
{
        return last_step;
}

void receivers_step_format(char *out, size_t step, const char *base)
{
        sprintf(out, "%s_%0*ld", base, leading_zeros, step);
}

