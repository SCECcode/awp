#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "topography.h"
#include "cutopography.cuh"
#include "cutopography_test.cuh"
#include "cutopography_test.cuh"
#include "check.h"
#include "grid_check.h"
#include "vtk.h"
#include "functions.h"
#include "metrics.h"
#include "geometry.h"
#include "shift.h"

typedef struct
{
        _prec tol;
        int verbose;
        int deg[3];
        int coef[3];
        int num_bytes;
        int size[3];
        int mem[3];
        int3_t shift;
        _prec *output;
        _prec *answer;
        _prec *error;
        fcn_grid_t velocity_grid;
        fcn_grid_t interior_grid;
        fcn_grid_t stress_grid;
        fcn_grid_t topography_grid;
        topo_t T;
        int velocity_offset_x[2];
        int velocity_offset_y[2];
        int stress_offset_x[2];
        int stress_offset_y[2];
        int offset_z[2];
} testdata_t;

void test_initialize(testdata_t *test);
void test_velocity(testdata_t *test);
void test_velocity_mod(testdata_t *test);
void test_stress(testdata_t *test);
void test_free(testdata_t *test);
double test_velocity_kernel(testdata_t *test, _prec *input, const _prec *input_coef,
                   const _prec *input_deg, const int *input_shift,
                   _prec *output, _prec *answer, const _prec *answer_coef,
                   _prec *answer_deg, const int *answer_shift);
double test_stress_kernel(testdata_t *test, _prec *input, const _prec *input_coef,
                   const _prec *input_deg, const int *input_shift,
                   _prec *output, _prec *answer, const _prec *answer_coef,
                   _prec *answer_deg, const int *answer_shift);
void copy_output_to_host(testdata_t *test, const _prec *input);
void copy_answer_to_host(testdata_t *test, const _prec *input);
double check_answer(const testdata_t *test, const int *shift, const int *offset_x,
                    const int *offset_y, const int *offset_z);
void write_vtk(const testdata_t *test);

int main(int argc, char **argv)
{
        printf("Testing topography.c, cutopography.c, cutopography_kernels.cu\n");
        testdata_t test;
        test.verbose = 0;
        //test_velocity(&test);
        test_velocity_mod(&test);
        test_stress(&test);
}

void test_initialize(testdata_t *test)
{
        int rank = 0;
        int x_rank_l = -1;
        int x_rank_r = -1;
        int y_rank_f = -1;
        int y_rank_b = -1;
        int coord[2] = {0, 0};
        int size[3] = {132, 132, 32};
        cudaStream_t stream_1, stream_2, stream_i;
        cudaStreamCreate(&stream_1);
        cudaStreamCreate(&stream_2);
        cudaStreamCreate(&stream_i);
        test->tol = 1e-6;
        _prec dt = 1.0;
        _prec h  = 1.0;
        test->T = topo_init(1, "topo", rank, x_rank_l, x_rank_r, y_rank_f,
                            y_rank_b, coord, size[0], size[1], size[2], dt, h,
                            stream_1, stream_2, stream_i);
        test->size[0] = test->T.nx;
        test->size[1] = test->T.ny;
        test->size[2] = test->T.nz;
        test->mem[0] = test->T.mx;
        test->mem[1] = test->T.my;
        test->mem[2] = test->T.mz;
        topo_d_malloc(&test->T);

        topo_init_metrics(&test->T);
        topo_init_grid(&test->T);
        topo_build(&test->T);

        test->velocity_offset_x[0] = test->T.off_x[1];
        test->velocity_offset_x[1] = test->T.off_x[2];
        test->velocity_offset_y[0] = test->T.off_y[1];
        test->velocity_offset_y[1] = test->T.off_y[2];
        test->stress_offset_x[0] = test->T.off_x[1] - ngsl/2;
        test->stress_offset_x[1] = test->T.off_x[3] + ngsl/2;
        test->stress_offset_y[0] = test->T.off_y[1] - ngsl/2;
        test->stress_offset_y[1] = test->T.off_y[3] + ngsl/2;
        test->offset_z[0] = test->T.off_z[1];
        test->offset_z[1] = test->T.off_z[2];

        int num_bytes = sizeof(_prec)*test->T.gridsize;
        test->num_bytes = num_bytes;
        test->output = malloc(num_bytes);
        test->answer = malloc(num_bytes);
        test->error = malloc(num_bytes);

        if (test->verbose) {
                metrics_print_info_f(&test->T.metrics_f);
                printf("offset x: %d %d \n", test->velocity_offset_x[0],
                                             test->velocity_offset_x[1]);
                printf("offset y: %d %d \n", test->velocity_offset_y[0],
                                             test->velocity_offset_y[1]);
                printf("offset z: %d %d \n", test->offset_z[0],
                                             test->offset_z[1]);
        }

        int3_t sizet = {.x = test->size[0],
                     .y = test->size[1],
                     .z = test->size[2]};
        int3_t coordt = {0, 0, 0};
        int3_t shift = {0, 0, 1};

        test->shift = shift;
        test->velocity_grid = fcn_init_grid(sizet, coordt, shift, 0, h);
        test->interior_grid = fcn_init_grid(sizet, coordt, shift, -ngsl/2, h);
        test->stress_grid = fcn_init_grid(sizet, coordt, shift, ngsl/2, h);
        test->topography_grid = fcn_init_grid(sizet, coordt, shift, ngsl, h);
}

void test_velocity(testdata_t *test)
{
        printf("Testing velocity update kernel... \n");
        printf(" * Testing u1 update equation. \n");
        {
        printf("    -- Testing quadratic function in x-direction. \n");
        _prec input_coef[3] = {1, 0, 0};
        _prec input_deg[3] = {2, 0, 0};
        int input_shift[3];
        shift_xx(input_shift);

        _prec answer_coef[3] = {2, 0, 0};
        _prec answer_deg[3] = {1, 0, 0};
        int answer_shift[3];
        shift_u1(answer_shift);
        test_initialize(test);
        test_velocity_kernel(test, test->T.xx, input_coef, input_deg,
                             input_shift, test->T.u1, test->T.yy, answer_coef,
                             answer_deg, answer_shift);
        test_free(test);
        }
}

void test_velocity_mod(testdata_t *test)
{
        printf("Testing velocity update kernel (kernels must be generated with debug=1, debug_ops=1... \n");
        printf(" * Testing u1 update equation. \n");
        {
        printf("    -- Testing DczPx*s11. \n");
        _prec input_coef[3] = {0, 0, 1};
        _prec input_deg[3] = {0, 0, 1};
        int input_shift[3];
        shift_xx(input_shift);

        _prec answer_coef[3] = {0, 0, 1};
        _prec answer_deg[3] = {0, 0, 0};
        int answer_shift[3];
        shift_u1(answer_shift);
        test_initialize(test);
        double err = test_velocity_kernel(
            test, test->T.xx, input_coef, input_deg, input_shift, test->T.u1,
            test->T.yy, answer_coef, answer_deg, answer_shift);

        printf("   Error: %g \n", err);
        write_vtk(test);
        test_free(test);
        }
        {
        printf("    -- Testing DczPy*s12. \n");
        _prec input_coef[3] = {0, 0, 1};
        _prec input_deg[3] = {0, 0, 1};
        int input_shift[3];
        shift_xy(input_shift);

        _prec answer_coef[3] = {0, 0, 1};
        _prec answer_deg[3] = {0, 0, 0};
        int answer_shift[3];
        shift_u1(answer_shift);
        test_initialize(test);
        double err = test_velocity_kernel(test, test->T.xy, input_coef, input_deg,
                             input_shift, test->T.u1, test->T.yy, answer_coef,
                             answer_deg, answer_shift);

        printf("   Error: %g \n", err);
        write_vtk(test);
        test_free(test);
        }



        return;

        printf(" * Testing u2 update equation. \n");
        {
        _prec input_coef[3] = {0, 0, 1};
        _prec input_deg[3] = {0, 0, 2};
        int input_shift[3];
        shift_xy(input_shift);

        _prec answer_coef[3] = {0, 0, 2};
        _prec answer_deg[3] = {0, 0, 1};
        int answer_shift[3];
        shift_u2(answer_shift);
        test_initialize(test);
        test_velocity_kernel(test, test->T.xy, input_coef, input_deg,
                             input_shift, test->T.v1, test->T.yy, answer_coef,
                             answer_deg, answer_shift);

        write_vtk(test);
        test_free(test);
        }

        printf(" * Testing u3 update equation. \n");
        {
        printf("    -- Testing quadratic function in x-direction. \n");
        // Only linear functions can be used in the test because interpolation
        // operators is only first order accurate near boundary
        _prec input_coef[3] = {0, 0, 1};
        _prec input_deg[3] = {0, 0, 2};
        int input_shift[3];
        shift_xz(input_shift);

        _prec answer_coef[3] = {0, 0, 2};
        _prec answer_deg[3] = {0, 0, 1};
        int answer_shift[3];
        shift_u3(answer_shift);
        test_initialize(test);
        double err = test_velocity_kernel(
            test, test->T.xz, input_coef, input_deg, input_shift, test->T.w1,
            test->T.yy, answer_coef, answer_deg, answer_shift);

        printf("       Error: %g \n", err);
        write_vtk(test);
        test_free(test);
        }
}

void test_stress(testdata_t *test)
{
        printf("Testing stress update kernel (kernels must be generated with debug=1, debug_ops=1... \n");
        printf(" * Testing s11 := PzDcx*u3. \n");
        {
        _prec input_coef[3] = {1, 0, 0};
        _prec input_deg[3] = {2, 0, 0};
        int input_shift[3];
        shift_u3(input_shift);

        _prec answer_coef[3] = {2, 0, 0};
        _prec answer_deg[3] = {1, 0, 0};
        int answer_shift[3];
        shift_xx(answer_shift);
        test_initialize(test);
        double err = test_stress_kernel(test, test->T.w1, input_coef, input_deg,
                             input_shift, test->T.xx, test->T.yy, answer_coef,
                             answer_deg, answer_shift);
        printf("   Error: %g \n", err);
        write_vtk(test);
        test_free(test);
        }

        printf(" * Testing s22 := PzDcy*u3. \n");
        {
        _prec input_coef[3] = {0, 1, 0};
        _prec input_deg[3] = {0, 2, 0};
        int input_shift[3];
        shift_u3(input_shift);

        _prec answer_coef[3] = {0, 2, 0};
        _prec answer_deg[3] = {0, 1, 0};
        int answer_shift[3];
        shift_yy(answer_shift);
        test_initialize(test);
        double err = test_stress_kernel(test, test->T.w1, input_coef, input_deg,
                             input_shift, test->T.yy, test->T.xz, answer_coef,
                             answer_deg, answer_shift);
        printf("   Error: %g \n", err);
        write_vtk(test);
        test_free(test);
        }

        printf(" * Testing s12 := PxDcz*u2. \n");
        {
        _prec input_coef[3] = {0, 0, 1};
        _prec input_deg[3] = {0, 0, 2};
        int input_shift[3];
        shift_u2(input_shift);

        _prec answer_coef[3] = {0, 0, 2};
        _prec answer_deg[3] = {0, 0, 1};
        int answer_shift[3];
        shift_xy(answer_shift);
        test_initialize(test);
        double err = test_stress_kernel(test, test->T.v1, input_coef, input_deg,
                             input_shift, test->T.xy, test->T.yy, answer_coef,
                             answer_deg, answer_shift);

        printf("   Error: %g \n", err);
        write_vtk(test);
        test_free(test);
        }

        printf(" * Testing s13 := PxDcz*u3. \n");
        {
        _prec input_coef[3] = {0, 0, 1};
        _prec input_deg[3] = {0, 0, 2};
        int input_shift[3];
        shift_u3(input_shift);

        _prec answer_coef[3] = {0, 0, 2};
        _prec answer_deg[3] = {0, 0, 1};
        int answer_shift[3];
        shift_xz(answer_shift);
        test_initialize(test);
        double err = test_stress_kernel(test, test->T.w1, input_coef, input_deg,
                             input_shift, test->T.xz, test->T.yy, answer_coef,
                             answer_deg, answer_shift);

        printf("   Error: %g \n", err);
        write_vtk(test);
        test_free(test);
        }


        printf(" * Testing s23 := PyDcz*u3. \n");
        {
        _prec input_coef[3] = {0, 1, 1};
        _prec input_deg[3] = {0, 1, 2};
        int input_shift[3];
        shift_u3(input_shift);

        _prec answer_coef[3] = {0, 0, 2};
        _prec answer_deg[3] = {0, 0, 1};
        int answer_shift[3];
        shift_yz(answer_shift);
        test_initialize(test);
        double err = test_stress_kernel(test, test->T.w1, input_coef, input_deg,
                             input_shift, test->T.yz, test->T.yy, answer_coef,
                             answer_deg, answer_shift);

        printf("   Error: %g \n", err);
        write_vtk(test);
        test_free(test);
        }

}

double test_velocity_kernel(testdata_t *test, _prec *input, const _prec *input_coef,
                          const _prec *input_deg, const int *input_shift,
                          _prec *output, _prec *answer,
                          const _prec *answer_coef, _prec *answer_deg,
                          const int *answer_shift) {
        topo_test_polystrzbnd_H(&test->T, input, input_coef, input_deg,
                            input_shift);
        topo_velocity_interior_H(&test->T);
        topo_test_polystrzbnd_H(&test->T, answer, answer_coef, answer_deg,
                            answer_shift);
        cudaDeviceSynchronize();

        copy_output_to_host(test, output);
        copy_answer_to_host(test, answer);

        int offset_x[2] = {test->velocity_offset_x[0],
                           test->velocity_offset_x[1]};
        int offset_y[2] = {test->velocity_offset_y[0],
                           test->velocity_offset_y[1]};
        int offset_z[2] = {test->offset_z[0]+8, test->offset_z[1]};

        double err = check_answer(test, answer_shift, offset_x, offset_y, offset_z);
        return err;
        
}

double test_stress_kernel(testdata_t *test, _prec *input, const _prec *input_coef,
                          const _prec *input_deg, const int *input_shift,
                          _prec *output, _prec *answer,
                          const _prec *answer_coef, _prec *answer_deg,
                          const int *answer_shift) {
        topo_test_polystrzbnd_H(&test->T, input, input_coef, input_deg,
                            input_shift);
        topo_stress_interior_H(&test->T);
        topo_test_polystrzbnd_H(&test->T, answer, answer_coef, answer_deg,
                            answer_shift);
        cudaDeviceSynchronize();

        copy_output_to_host(test, output);
        copy_answer_to_host(test, answer);

        //FIXME: should be stress_offset
        int offset_x[2] = {test->velocity_offset_x[0]+8,
                           test->velocity_offset_x[1]-8};
        int offset_y[2] = {test->velocity_offset_y[0]+8,
                           test->velocity_offset_y[1]-8};
        int offset_z[2] = {test->offset_z[0]+8, test->offset_z[1]};

        double err = check_answer(test, answer_shift, offset_x, offset_y, offset_z);
        return err;
        
}

void test_free(testdata_t *test)
{
        topo_free(&test->T);
        free(test->output);
        free(test->answer);
        cudaStreamDestroy(test->T.stream_1);
        cudaStreamDestroy(test->T.stream_2);
        cudaStreamDestroy(test->T.stream_i);
        topo_d_free(&test->T);
}

void copy_output_to_host(testdata_t *test, const _prec *input)
{
        cudaMemcpy(test->output, input, test->num_bytes,
                   cudaMemcpyDeviceToHost);
}

void copy_answer_to_host(testdata_t *test, const _prec *input)
{
        cudaMemcpy(test->answer, input, test->num_bytes,
                   cudaMemcpyDeviceToHost);
}

double check_answer(const testdata_t *test, const int *shift, const int *offset_x,
                    const int *offset_y, const int *offset_z) {
        // Do not check the ghost point on the nodal grid
        int skip = 0;
        if (shift[2] == 0) {
                skip = 1;
        }

        double err = check_flinferr(test->output, test->answer, 
                  offset_x[0], offset_x[1],
                  offset_y[0], offset_y[1], 
                  offset_z[0], offset_z[1] - skip, 
                  test->T.line,
                  test->T.slice);
        return err;
}

void write_vtk(const testdata_t *test)
{

        _prec *x = malloc(test->topography_grid.num_bytes);
        _prec *y = malloc(test->topography_grid.num_bytes);
        _prec *z = malloc(test->topography_grid.num_bytes);

        fcn_fill_grid(x, test->topography_grid, test->shift, 0);
        fcn_fill_grid(y, test->topography_grid, test->shift, 1);
        fcn_fill_grid(z, test->topography_grid, test->shift, 2);

        fcn_grid_t grid = test->interior_grid;
        const char *vtk_file = "output.vtk";
        vtk_write_grid(vtk_file, x, y, z, grid);
        size_t count = vtk_append_scalar(vtk_file, "output", test->output, grid);

        const char *vtk_file2 = "answer.vtk";
        vtk_write_grid(vtk_file2, x, y, z, grid);
        count = vtk_append_scalar(vtk_file2, "answer", test->answer, grid);

        const char *vtk_file3 = "error.vtk";
        fcn_difference(test->error, test->answer, test->output, grid); 
        fcn_abs(test->error, test->error, grid); 
        vtk_write_grid(vtk_file3, x, y, z, grid);
        count = vtk_append_scalar(vtk_file3, "error", test->error, grid);

}
