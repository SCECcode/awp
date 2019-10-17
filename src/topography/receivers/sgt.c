#include <stdio.h>
#include <time.h>
#include <math.h>
#include <errno.h>

#include <awp/definitions.h>
#include <test/test.h>
#include <topography/receivers/receiver.h>
#include <topography/receivers/sgt.h>
#include <readers/input.h>

static int use;

static recv_t mat;

static recv_t Gxx;
static recv_t Gyy;
static recv_t Gzz;
static recv_t Gxy;
static recv_t Gxz;
static recv_t Gyz;

static input_t input;

// Variables for defining last step output
static size_t last_step = 0;
static int leading_zeros;

void sgt_init(const char *filename, const grids_t *grids, int ngrids,
                    const f_grid_t *f,
                  const MPI_Comm comm, const int rank, const int size)
{
        use = strcmp(filename, "") != 0 ? 1 : 0;

        if (!use) return;

        if (rank == 0) {
                AWPCHK(input_init(&input, filename));
        }
        AWPCHK(input_broadcast(&input, rank, 0, comm));


        Gxx = receiver_init("Gxx", XX, &input, grids, ngrids, f, rank, comm);
        Gyy = receiver_init("Gyy", YY, &input, grids, ngrids, f, rank, comm);
        Gzz = receiver_init("Gzz", ZZ, &input, grids, ngrids, f, rank, comm);
        Gxy = receiver_init("Gxy", XY, &input, grids, ngrids, f, rank, comm);
        Gxz = receiver_init("Gxz", XY, &input, grids, ngrids, f, rank, comm);
        Gyz = receiver_init("Gyz", YZ, &input, grids, ngrids, f, rank, comm);
        
        // Configure material input file so that it outputs without buffering
        input_t material_input = input;
        material_input.gpu_buffer_size = 1;
        material_input.cpu_buffer_size = 1;
        material_input.steps = 1;
        material_input.num_writes = 1;
        mat = receiver_init("", NODE, &material_input, grids, ngrids, f, rank,
                           comm);
}

void sgt_finalize(void)
{
        if (!use) return;
        receiver_finalize(&Gxx);
        receiver_finalize(&Gyy);
        receiver_finalize(&Gzz);
        receiver_finalize(&Gxy);
        receiver_finalize(&Gxz);
        receiver_finalize(&Gyz);
        receiver_finalize(&mat);
}

void sgt_write_material_properties(const prec *d_d1, const prec *d_lami,
                                   const prec *d_mui, const int grid_num) {
        if (!use) return;
        printf("Writing material properties\n");
        int len = strlen(mat.filename) + 4; 
        char *filename;
        filename = malloc(sizeof filename * len);
        sprintf(filename, "%sd1", mat.filename);
        receiver_write(&mat, 0, filename, d_lami, grid_num);
        sprintf(filename, "%slami", mat.filename);
        receiver_write(&mat, 0, filename, d_lami, grid_num);
        sprintf(filename, "%smui", mat.filename);
        receiver_write(&mat, 0, filename, d_mui, grid_num);
        free(filename);
}

void sgt_write(const prec *d_xx, const prec *d_yy, const prec *d_zz,
               const prec *d_xy, const prec *d_xz, const prec *d_yz,
                     const size_t step, const size_t num_steps,
                     const int grid_num) {
        if (!use) return;
        char outputname[STR_LEN];
        leading_zeros = ceil(log10((double)num_steps)) + 1;
        last_step = receiver_step(input.gpu_buffer_size, input.cpu_buffer_size,
                                  input.num_writes, input.stride, step);

        receiver_filename(outputname, Gxx.filename, input.gpu_buffer_size,
                          input.cpu_buffer_size, input.num_writes, input.stride,
                          step, num_steps);
        receiver_write(&Gxx, step, outputname, d_xx, grid_num);

        receiver_filename(outputname, Gyy.filename, input.gpu_buffer_size,
                          input.cpu_buffer_size, input.num_writes, input.stride,
                          step, num_steps);
        receiver_write(&Gyy, step, outputname, d_yy, grid_num);

        receiver_filename(outputname, Gzz.filename, input.gpu_buffer_size,
                          input.cpu_buffer_size, input.num_writes, input.stride,
                          step, num_steps);
        receiver_write(&Gzz, step, outputname, d_zz, grid_num);
        
        receiver_filename(outputname, Gxy.filename, input.gpu_buffer_size,
                          input.cpu_buffer_size, input.num_writes, input.stride,
                          step, num_steps);
        receiver_write(&Gxy, step, outputname, d_xy, grid_num);

        receiver_filename(outputname, Gxz.filename, input.gpu_buffer_size,
                          input.cpu_buffer_size, input.num_writes, input.stride,
                          step, num_steps);
        receiver_write(&Gxz, step, outputname, d_xz, grid_num);

        receiver_filename(outputname, Gyz.filename, input.gpu_buffer_size,
                          input.cpu_buffer_size, input.num_writes, input.stride,
                          step, num_steps);
        receiver_write(&Gyz, step, outputname, d_yz, grid_num);
}
