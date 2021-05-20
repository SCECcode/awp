#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>


#include <topography/topography.h>
#include <topography/metrics/metrics.h>
#include <topography/velocity.cuh>
#include <topography/stress.cuh>
//#include <topography/geometry/geometry.h>
#include <topography/geometry.h>
#include <grid/shift.h>
#include "functions.c"
#include "grid_check.c"
//#include "cutopography.cuh"
#include "mms.c"

void geom_mapping_z(_prec *out, const fcn_grid_t grid, const int3_t shift,
                    const f_grid_t *metrics_f,
                    const g_grid_t *metrics_g) {
        _prec *g;
        if (shift.z == 0) {
                g = metrics_g->g;
        }
        else {
                g = metrics_g->g_c;
        }

        int3_t nodes = grid_node();
        int3_t u1 = grid_u1();
        int3_t u2 = grid_u2();
        _prec *f;
        if (shift.x == nodes.x && shift.y == nodes.y) {
                f = metrics_f->f;
        } 
        else if(shift.x == u1.x && shift.y == u1.y) {
                f = metrics_f->f_1;
        }
        else if(shift.x == u2.x && shift.y == u2.y) {
                f = metrics_f->f_2;
        }
        else {
                f = metrics_f->f_c;
        }

        int f_offset_x = metrics_f->offset[0] + metrics_f->bounds_stress_x[0];
        int f_offset_y = metrics_f->offset[1] + metrics_f->bounds_stress_y[0];

        // Error: `grid` cannot be larger than the stress grid.
        //assert(f_offset_x + grid.size.x <= metrics_f->mem[0]);
        //assert(f_offset_y + grid.size.y <= metrics_f->mem[1]);

        for (int i = 0; i < grid.size.x; ++i) {
        for (int j = 0; j < grid.size.y; ++j) {
        for (int k = 0; k < grid.size.z; ++k) {
                int pos = grid.offset1.z + k +
                          (grid.offset1.y + j) * grid.line +
                          (grid.offset1.x + i) * grid.slice;
                int pos_g = k + metrics_g->offset;
                int pos_f = f_offset_y + j +
                            (i + f_offset_x) * metrics_f->slice;
                out[pos] = g[pos_g] * f[pos_f];
        }
        }
        }
}

using _prec=float;

typedef struct
{
        _prec *vx;
        _prec *vy;
        _prec *vz;
        _prec *sxx;
        _prec *syy;
        _prec *szz;
        _prec *sxy;
        _prec *sxz;
        _prec *syz;
        _prec *rho;
        int num_bytes;
} variables_t;

typedef struct
{
        _prec tol;
        _prec grid_spacing;
        int write_vtk;
        int verbose;
        int num_bytes;
        int3_t size;
        int3_t coord3;
        topo_t T;
        variables_t input;
        variables_t output;
        variables_t answer;
        _prec mms_wavenumber;
} testdata_t;

typedef struct
{
        // parameter coordinates
        _prec *x, *y, *z;
        // physical coordinate
        _prec *zp;
} grid_t;

typedef struct
{
        _prec interior;
        _prec boundary;
} err_t;

typedef struct
{
        err_t vx;
        err_t vy;
        err_t vz;
        err_t sxx;
        err_t syy;
        err_t szz;
        err_t sxy;
        err_t sxz;
        err_t syz;
} vars_err_t;


int3_t refine(const int3_t initial_size, const int grid);
void convergence_rates(vars_err_t *rates, const vars_err_t *err, const _prec *h,
                       const int num_refinements);
_prec convergence_rate(const _prec err1, const _prec err2, const _prec h1, const _prec h2);
void test_initialize(testdata_t *test, const int grid);
void test_velocity(testdata_t *test, vars_err_t *err);
void test_stress(testdata_t *test, vars_err_t *err);
void test_free(testdata_t *test);
void vars_init(variables_t *vars, const int num_bytes);
void vars_copy_to_device(topo_t *topo, const variables_t *vars);
void vars_copy_to_host(variables_t *vars, const topo_t *topo);
void vars_free(variables_t *vars);
void test_grid_data_init(grid_t *data, const testdata_t *test, const fcn_grid_t grid,
                    const int3_t shift);
void test_grid_data_free(grid_t *data);
err_t check_answer(const _prec *u, const _prec *v, const fcn_grid_t grid);
void init_sponge(topo_t *topo, const int num_bytes);


int main(int argc, char **argv)
{
        int num_refinements = 3;

        testdata_t test;
        int3_t initial_size = {32, 32, 32};

        vars_err_t err[num_refinements];
        int grid_sizes[num_refinements];
        _prec grid_spacings[num_refinements];


        printf("Convergence rate test\n");
        printf("-----------------------------------------------------\n");
        for (int grid = 0; grid < num_refinements; ++grid) {
                test.size = refine(initial_size, grid);
                grid_sizes[grid] = test.size.x;
                test_initialize(&test, grid);
                grid_spacings[grid] = test.grid_spacing;
                printf("Grid refinement: %d  grid size: {%d, %d, %d} \n", 
                        grid, test.size.x, test.size.y, test.size.z);
                test_velocity(&test, &err[grid]);
                printf("Testing stresses\n");
                test_free(&test);
                test_initialize(&test, grid);
                //test_stress(&test, &err[grid]);
                test_free(&test);
        }
        printf("-----------------------------------------------------\n");

        printf("\n");
        printf("\n");
        printf("Interior truncation error\n");
        printf("N \t vx        \t vy          \t vz \n");
        for (int i = 0; i < num_refinements; ++i) {
                printf("%d \t %e \t %e \t %e \n", grid_sizes[i],
                       err[i].vx.interior, err[i].vy.interior,
                       err[i].vz.interior);
        }

        printf("N \t sxx        \t syy          \t szz \n");
        for (int i = 0; i < num_refinements; ++i) {
                printf("%d \t %e \t %e \t %e \n", grid_sizes[i],
                       err[i].sxx.interior, err[i].syy.interior,
                       err[i].szz.interior);
        }

        printf("N \t sxy        \t sxz          \t syz \n");
        for (int i = 0; i < num_refinements; ++i) {
                printf("%d \t %e \t %e \t %e \n", grid_sizes[i],
                       err[i].sxy.interior, err[i].sxz.interior,
                       err[i].syz.interior);
        }
        
        printf("\n");
        printf("Boundary truncation error\n");
        printf("N \t vx        \t vy          \t vz \n");
        for (int i = 0; i < num_refinements; ++i) {
                printf("%d \t %e \t %e \t %e \n", grid_sizes[i],
                       err[i].vx.boundary, err[i].vy.boundary,
                       err[i].vz.boundary);
        }
        printf("N \t sxx        \t syy          \t szz \n");
        for (int i = 0; i < num_refinements; ++i) {
                printf("%d \t %e \t %e \t %e \n", grid_sizes[i],
                       err[i].sxx.boundary, err[i].syy.boundary,
                       err[i].szz.boundary);
        }

        printf("N \t sxy        \t sxz          \t syz \n");
        for (int i = 0; i < num_refinements; ++i) {
                printf("%d \t %e \t %e \t %e \n", grid_sizes[i],
                       err[i].sxy.boundary, err[i].sxz.boundary,
                       err[i].syz.boundary);
        }

        vars_err_t rates[num_refinements - 1]; 
        for (int i = 0; i < num_refinements - 1; ++i) {
                convergence_rates(rates, err, grid_spacings, num_refinements); 
        }

        printf("\n");
        printf("\n");
        printf("Interior convergence  rates\n");
        printf("N \t vx        \t vy          \t vz \n");
        for (int i = 0; i < num_refinements - 1; ++i) {
                printf("%d \t %e \t %e \t %e \n", grid_sizes[i+1],
                       rates[i].vx.interior, rates[i].vy.interior,
                       rates[i].vz.interior);
        }

        printf("N \t sxx        \t syy          \t szz \n");
        for (int i = 0; i < num_refinements - 1; ++i) {
                printf("%d \t %e \t %e \t %e \n", grid_sizes[i+1],
                       rates[i].sxx.interior, rates[i].syy.interior,
                       rates[i].szz.interior);
        }

        printf("N \t sxy        \t sxz          \t syz \n");
        for (int i = 0; i < num_refinements - 1; ++i) {
                printf("%d \t %e \t %e \t %e \n", grid_sizes[i+1],
                       rates[i].sxy.interior, rates[i].sxz.interior,
                       rates[i].syz.interior);
        }
        
        printf("\n");
        printf("Boundary convergence  rates\n");
        printf("N \t vx        \t vy          \t vz \n");
        for (int i = 0; i < num_refinements - 1; ++i) {
                printf("%d \t %e \t %e \t %e \n", grid_sizes[i+1],
                       rates[i].vx.boundary, rates[i].vy.boundary,
                       rates[i].vz.boundary);
        }

        printf("N \t sxx        \t syy          \t szz \n");
        for (int i = 0; i < num_refinements - 1; ++i) {
                printf("%d \t %e \t %e \t %e \n", grid_sizes[i+1],
                       rates[i].sxx.boundary, rates[i].syy.boundary,
                       rates[i].szz.boundary);
        }

        printf("N \t sxy        \t sxz          \t syz \n");
        for (int i = 0; i < num_refinements - 1; ++i) {
                printf("%d \t %e \t %e \t %e \n", grid_sizes[i+1],
                       rates[i].sxy.boundary, rates[i].sxz.boundary,
                       rates[i].syz.boundary);
        }
        return 0;
}

int3_t refine(const int3_t initial_size, const int grid) 
{
        int3_t out;
        out.x = initial_size.x*pow(2, grid);
        out.y = initial_size.y*pow(2, grid);
        out.z = initial_size.z*pow(2, grid);
        return out;
}

void convergence_rates(vars_err_t *rates, const vars_err_t *err, const _prec *h,
                       const int num_refinements) {
        for (int i = 0; i < num_refinements - 1; ++i) {
                rates[i].vx.interior = convergence_rate(
                    err[i].vx.interior, err[i + 1].vx.interior, h[i], h[i + 1]);
                rates[i].vy.interior = convergence_rate(
                    err[i].vy.interior, err[i + 1].vy.interior, h[i], h[i + 1]);
                rates[i].vz.interior = convergence_rate(
                    err[i].vz.interior, err[i + 1].vz.interior, h[i], h[i + 1]);
                rates[i].sxx.interior =
                    convergence_rate(err[i].sxx.interior,
                                     err[i + 1].sxx.interior, h[i], h[i + 1]);
                rates[i].syy.interior =
                    convergence_rate(err[i].syy.interior,
                                     err[i + 1].syy.interior, h[i], h[i + 1]);
                rates[i].szz.interior =
                    convergence_rate(err[i].szz.interior,
                                     err[i + 1].szz.interior, h[i], h[i + 1]);
                rates[i].sxy.interior =
                    convergence_rate(err[i].sxy.interior,
                                     err[i + 1].sxy.interior, h[i], h[i + 1]);
                rates[i].sxz.interior =
                    convergence_rate(err[i].sxz.interior,
                                     err[i + 1].sxz.interior, h[i], h[i + 1]);
                rates[i].syz.interior =
                    convergence_rate(err[i].syz.interior,
                                     err[i + 1].syz.interior, h[i], h[i + 1]);

                rates[i].vx.boundary = convergence_rate(
                    err[i].vx.boundary, err[i + 1].vx.boundary, h[i], h[i + 1]);
                rates[i].vy.boundary = convergence_rate(
                    err[i].vy.boundary, err[i + 1].vy.boundary, h[i], h[i + 1]);
                rates[i].vz.boundary = convergence_rate(
                    err[i].vz.boundary, err[i + 1].vz.boundary, h[i], h[i + 1]);
                rates[i].sxx.boundary =
                    convergence_rate(err[i].sxx.boundary,
                                     err[i + 1].sxx.boundary, h[i], h[i + 1]);
                rates[i].syy.boundary =
                    convergence_rate(err[i].syy.boundary,
                                     err[i + 1].syy.boundary, h[i], h[i + 1]);
                rates[i].szz.boundary =
                    convergence_rate(err[i].szz.boundary,
                                     err[i + 1].szz.boundary, h[i], h[i + 1]);
                rates[i].sxy.boundary =
                    convergence_rate(err[i].sxy.boundary,
                                     err[i + 1].sxy.boundary, h[i], h[i + 1]);
                rates[i].sxz.boundary =
                    convergence_rate(err[i].sxz.boundary,
                                     err[i + 1].sxz.boundary, h[i], h[i + 1]);
                rates[i].syz.boundary =
                    convergence_rate(err[i].syz.boundary,
                                     err[i + 1].syz.boundary, h[i], h[i + 1]);
        }
}

_prec convergence_rate(const _prec err1, const _prec err2, const _prec h1,
                       const _prec h2) {
        return log(err1/err2)/log(h1/h2);
}

void test_initialize(testdata_t *test, const int grid)
{
        int rank = 0;
        int x_rank_l = -1;
        int x_rank_r = -1;
        int y_rank_f = -1;
        int y_rank_b = -1;
        int coord[2] = {0, 0};
        int px = 1;
        int py = 1;
        cudaStream_t stream_1, stream_2, stream_i;
        cudaStreamCreate(&stream_1);
        cudaStreamCreate(&stream_2);
        cudaStreamCreate(&stream_i);
        test->tol = 1e-6;
        _prec dt = 1.0;
        _prec h  = 1.0/(test->size.x - 1);
        printf("Test size: %d %d %d \n", test->size.x, test->size.y, test->size.z);
        char gridname[2048];
        sprintf(gridname, "topography_%d.bin", grid);
        test->T = topo_init(1, gridname, rank, x_rank_l, x_rank_r, y_rank_f,
                            y_rank_b, coord, px, py, test->size.x, test->size.y,
                            test->size.z, dt, h, stream_1, stream_2, stream_i);
        topo_d_malloc(&test->T);
        test->coord3.x = coord[0];
        test->coord3.y = coord[1];
        test->grid_spacing = h;
        test->write_vtk = 0;
        test->mms_wavenumber = 32;

        _prec amplitude = 0.0;
        _prec3_t width = {.x = 0.1, .y = 0.1, .z = 0};
        _prec3_t center = {.x = 0.5, .y = 0.5, .z = 0};

        topo_init_metrics(&test->T);
        topo_init_geometry(&test->T);
        //topo_init_gaussian_geometry(&test->T, amplitude, width, center);
        topo_build(&test->T);

        int num_items = test->T.mx*test->T.my*test->T.mz;
        vars_init(&test->input, num_items);
        vars_init(&test->output,num_items);
        vars_init(&test->answer,num_items);

        init_sponge(&test->T, sizeof(_prec)*num_items);
}

void test_velocity(testdata_t *test, vars_err_t *err)
{

        int3_t shift = {0, 0, 0};
        fcn_grid_t velocity_grid = fcn_init_grid(
            test->size, shift, test->coord3, 0, test->grid_spacing);
        fcn_grid_t stress_grid = fcn_init_grid(test->size, shift, test->coord3,
                                               ngsl / 2, test->grid_spacing);
        grid_t gvx;
        grid_t gvy;
        grid_t gvz;
        grid_t gsxx;
        grid_t gsyy;
        grid_t gszz;
        grid_t gsxy;
        grid_t gsxz;
        grid_t gsyz;
        test_grid_data_init(&gvx,  test, stress_grid, grid_u1());  
        test_grid_data_init(&gvy,  test, stress_grid, grid_u2());  
        test_grid_data_init(&gvz,  test, stress_grid, grid_u3());  
        test_grid_data_init(&gsxx, test, stress_grid, grid_xx());  
        test_grid_data_init(&gsyy, test, stress_grid, grid_yy());  
        test_grid_data_init(&gszz, test, stress_grid, grid_zz());  
        test_grid_data_init(&gsxy, test, stress_grid, grid_xy());  
        test_grid_data_init(&gsxz, test, stress_grid, grid_xz());  
        test_grid_data_init(&gsyz, test, stress_grid, grid_yz());  


        // Input
        _prec properties[2] = {test->mms_wavenumber, 0};
        fcn_apply(test->input.sxx, mms_init_sxx, gsxx.x, gsxx.y, gsxx.zp,
                  properties, stress_grid);
        fcn_apply(test->input.syy, mms_init_syy, gsyy.x, gsyy.y, gsyy.zp,
                  properties, stress_grid);
        fcn_apply(test->input.szz, mms_init_szz, gszz.x, gszz.y, gszz.zp,
                  properties, stress_grid);
        fcn_apply(test->input.sxy, mms_init_sxy, gsxy.x, gsxy.y, gsxy.zp,
                  properties, stress_grid);
        fcn_apply(test->input.sxz, mms_init_sxz, gsxz.x, gsxz.y, gsxz.zp,
                  properties, stress_grid);
        fcn_apply(test->input.syz, mms_init_syz, gsyz.x, gsyz.y, gsyz.zp,
                  properties, stress_grid);

        vars_copy_to_device(&test->T, &test->input);

        topo_velocity_interior_H(&test->T);

        // Output
        vars_copy_to_host(&test->output, &test->T);
        
        //Check answer
        fcn_apply(test->answer.vx, mms_final_vx, gvx.x, gvx.y, gvx.zp,
                  properties, velocity_grid);
        fcn_apply(test->answer.vy, mms_final_vy, gvy.x, gvy.y, gvy.zp,
                  properties, velocity_grid);
        fcn_apply(test->answer.vz, mms_final_vz, gvz.x, gvz.y, gvz.zp,
                  properties, velocity_grid);

        err_t tmp = check_answer(test->output.vx, test->answer.vx, velocity_grid);
        err->vx = tmp;

        tmp = check_answer(test->output.vy, test->answer.vy, velocity_grid);
        err->vy = tmp;

        tmp = check_answer(test->output.vz, test->answer.vz, velocity_grid);
        err->vz = tmp;

        char vtk_file[128];
        if (test->write_vtk) {
                sprintf(vtk_file, "input_sxx.vtk");
                vtk_write_grid(vtk_file, gsxx.x, gsxx.y, gsxx.zp,
                               velocity_grid);
                vtk_append_scalar(vtk_file, "z", test->input.sxx,
                                  velocity_grid);

                sprintf(vtk_file, "output_vx.vtk");
                vtk_write_grid(vtk_file, gvx.x, gvx.y, gvx.zp, velocity_grid);
                vtk_append_scalar(vtk_file, "z", test->output.vx,
                                  velocity_grid);

                sprintf(vtk_file, "answer_vx.vtk");
                vtk_write_grid(vtk_file, gvx.x, gvx.y, gvx.zp, velocity_grid);
                vtk_append_scalar(vtk_file, "z", test->answer.vx,
                                  velocity_grid);
        }
}

void test_stress(testdata_t *test, vars_err_t *err)
{

        int3_t shift = {0, 0, 0};
        fcn_grid_t velocity_grid = fcn_init_grid(
            test->size, shift, test->coord3, 0, test->grid_spacing);
        fcn_grid_t stress_grid = fcn_init_grid(test->size, shift, test->coord3,
                                               ngsl / 2, test->grid_spacing);
        grid_t gvx;
        grid_t gvy;
        grid_t gvz;
        grid_t gsxx;
        grid_t gsyy;
        grid_t gszz;
        grid_t gsxy;
        grid_t gsxz;
        grid_t gsyz;
        test_grid_data_init(&gvx,  test, stress_grid, grid_u1());  
        test_grid_data_init(&gvy,  test, stress_grid, grid_u2());  
        test_grid_data_init(&gvz,  test, stress_grid, grid_u3());  
        test_grid_data_init(&gsxx, test, stress_grid, grid_xx());  
        test_grid_data_init(&gsyy, test, stress_grid, grid_yy());  
        test_grid_data_init(&gszz, test, stress_grid, grid_zz());  
        test_grid_data_init(&gsxy, test, stress_grid, grid_xy());  
        test_grid_data_init(&gsxz, test, stress_grid, grid_xz());  
        test_grid_data_init(&gsyz, test, stress_grid, grid_yz());  

        // Input
        _prec properties[2] = {test->mms_wavenumber, 0};
        fcn_apply(test->input.vx, mms_init_vx, gvx.x, gvx.y, gvx.zp,
                  properties, stress_grid);
        fcn_apply(test->input.vy, mms_init_vy, gvy.x, gvy.y, gvy.zp,
                  properties, stress_grid);
        fcn_apply(test->input.vz, mms_init_vz, gvz.x, gvz.y, gvz.zp,
                  properties, stress_grid);
        
        vars_copy_to_device(&test->T, &test->input);

        topo_stress_interior_H(&test->T);

        // Output
        vars_copy_to_host(&test->output, &test->T);

        // Answer
        fcn_apply(test->answer.sxx, mms_final_sxx, gsxx.x, gsxx.y, gsxx.zp,
                  properties, velocity_grid);
        fcn_apply(test->answer.syy, mms_final_syy, gsyy.x, gsyy.y, gsyy.zp,
                  properties, velocity_grid);
        fcn_apply(test->answer.szz, mms_final_szz, gszz.x, gszz.y, gszz.zp,
                  properties, velocity_grid);
        fcn_apply(test->answer.sxy, mms_final_sxy, gsxy.x, gsxy.y, gsxy.zp,
                  properties, velocity_grid);
        fcn_apply(test->answer.sxz, mms_final_sxz, gsxz.x, gsxz.y, gsxz.zp,
                  properties, velocity_grid);
        fcn_apply(test->answer.syz, mms_final_syz, gsyz.x, gsyz.y, gsyz.zp,
                  properties, velocity_grid);

        err_t tmp;
        tmp = check_answer(test->output.sxx, test->answer.sxx, velocity_grid);
        err->sxx = tmp;
        tmp = check_answer(test->output.syy, test->answer.syy, velocity_grid);
        err->syy = tmp;
        tmp = check_answer(test->output.szz, test->answer.szz, velocity_grid);
        err->szz = tmp;
        tmp = check_answer(test->output.sxy, test->answer.sxy, velocity_grid);
        err->sxy = tmp;
        tmp = check_answer(test->output.sxz, test->answer.sxz, velocity_grid);
        err->sxz = tmp;
        tmp = check_answer(test->output.syz, test->answer.syz, velocity_grid);
        err->syz = tmp;
}

void test_free(testdata_t *test)
{
        topo_free(&test->T);
        cudaStreamDestroy(test->T.stream_1);
        cudaStreamDestroy(test->T.stream_2);
        cudaStreamDestroy(test->T.stream_i);
        vars_free(&test->input);
        vars_free(&test->output);
        vars_free(&test->answer);
}

void vars_init(variables_t *vars, const int num_items)
{
        int item_size = sizeof(_prec);
        vars->vx = (_prec*)calloc(num_items, item_size);
        vars->vy = (_prec*)calloc(num_items, item_size);
        vars->vz = (_prec*)calloc(num_items, item_size);
        vars->sxx =(_prec*) calloc(num_items, item_size);
        vars->syy =(_prec*) calloc(num_items, item_size);
        vars->szz =(_prec*) calloc(num_items, item_size);
        vars->sxy =(_prec*) calloc(num_items, item_size);
        vars->sxz =(_prec*) calloc(num_items, item_size);
        vars->syz =(_prec*) calloc(num_items, item_size);
        vars->rho =(_prec*) calloc(num_items, item_size);
        vars->num_bytes = num_items*item_size;
        for (int i = 0; i < num_items; ++i)
            vars->rho[i] = 1.0;
}

void vars_copy_to_device(topo_t *topo, const variables_t *vars)
{
        cudaMemcpy(topo->u1, vars->vx, vars->num_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(topo->v1, vars->vy, vars->num_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(topo->w1, vars->vz, vars->num_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(topo->xx, vars->sxx, vars->num_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(topo->yy, vars->syy, vars->num_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(topo->zz, vars->szz, vars->num_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(topo->xy, vars->sxy, vars->num_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(topo->xz, vars->sxz, vars->num_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(topo->yz, vars->syz, vars->num_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(topo->rho, vars->rho, vars->num_bytes,
                   cudaMemcpyHostToDevice);
}

void vars_copy_to_host(variables_t *vars, const topo_t *topo)
{
        cudaMemcpy(vars->vx, topo->u1, vars->num_bytes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(vars->vy, topo->v1, vars->num_bytes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(vars->vz, topo->w1, vars->num_bytes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(vars->sxx, topo->xx, vars->num_bytes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(vars->syy, topo->yy, vars->num_bytes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(vars->szz, topo->zz, vars->num_bytes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(vars->sxy, topo->xy, vars->num_bytes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(vars->sxz, topo->xz, vars->num_bytes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(vars->syz, topo->yz, vars->num_bytes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(vars->rho, topo->rho, vars->num_bytes,
                   cudaMemcpyDeviceToHost);
}
void vars_free(variables_t *vars)
{
        free(vars->vx);
        free(vars->vy);
        free(vars->vz);
        free(vars->sxx);
        free(vars->syy);
        free(vars->szz);
        free(vars->sxy);
        free(vars->sxz);
        free(vars->syz);
        free(vars->rho);
}

void test_grid_data_init(grid_t *data, const testdata_t *test, const fcn_grid_t grid,
                    const int3_t shift) {
        data->x = (_prec*)malloc(grid.num_bytes);
        data->y = (_prec*)malloc(grid.num_bytes);
        data->z = (_prec*)malloc(grid.num_bytes);
        data->zp =(_prec*) malloc(grid.num_bytes);

        fcn_fill_grid(data->x, grid, shift, 0);
        fcn_fill_grid(data->y, grid, shift, 1);
        fcn_fill_grid(data->z, grid, shift, 2);

        fcn_shift(data->x, data->x, grid, -ngsl*grid.gridspacing);
        fcn_shift(data->y, data->y, grid, -ngsl*grid.gridspacing);

        geom_mapping_z(data->zp, grid, shift, &test->T.metrics_f,
                       &test->T.metrics_g);
}

void test_grid_data_free(grid_t *data)
{
        free(data->x);
        free(data->y);
        free(data->z);
        free(data->zp);
}

err_t check_answer(const _prec *u, const _prec *v, const fcn_grid_t grid)
{
        // Maximum truncation error of the entire domain (boundary truncation
        // error)
        int nb = TOP_BOUNDARY_SIZE;
        _prec boundary = check_flinferr(u, v, 
                  grid.offset1.x + nb, grid.offset2.x - nb,
                  grid.offset1.y + nb, grid.offset2.y - nb,
                  grid.offset1.z + nb, grid.offset2.z - grid.exclude_top_row,
                  grid.line,
                  grid.slice);
        
        // Maximum truncation error in the interior of the domain
        _prec interior = check_flinferr(u, v, 
                  grid.offset1.x + nb, grid.offset2.x - nb,
                  grid.offset1.y + nb, grid.offset2.y - nb,
                  grid.offset1.z + nb, 
                  grid.offset2.z - nb - grid.exclude_top_row,
                  grid.line,
                  grid.slice);
        err_t out = {interior, boundary};
        return out;
}

void init_sponge(topo_t *topo, const int num_bytes)
{
        _prec *ones = (_prec*)malloc(num_bytes);
        for (size_t i = 0; i < num_bytes/(sizeof(_prec)); ++i) {
                ones[i] = 1.0;
        }

        cudaMemcpy(topo->dcrjx, ones, num_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(topo->dcrjy, ones, num_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(topo->dcrjz, ones, num_bytes, cudaMemcpyHostToDevice);
        free(ones);
}

