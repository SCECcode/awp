#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <awp/definitions.h>
#include <grid/grid_3d.h>
#include <grid/shift.h>
#include <topography/metrics/metrics.h>
#include <topography/metrics/kernel.h>
#include <topography/metrics/shift.h>
#include <test/test.h>
#include "interpolation/interpolation.h"


// This parameter pads the compute region. Its needed for the computation of
// derivative and interpolation stencils. Do not change its value.

f_grid_t metrics_init_f(const int *size, const _prec gridspacing,
                            const int pad) {
        f_grid_t out = {.size = {size[0], size[1], 1},
                        .mem = {size[0] + 4 + 2 * pad,
                                size[1] + 4 + 2 * pad + 2 * align, 1},
                        .bounds_x = {-pad, size[0] + pad},
                        .bounds_y = {-pad, size[1] + pad},
                        .bounds_stress_x = {-pad / 2, size[0] + pad / 2},
                        .bounds_stress_y = {-pad / 2, size[1] + pad / 2},
                        .offset = {2 + pad, 2 + pad + align, 0},
                        .hi = 1.0 / gridspacing};
        out.line = out.mem[2];
        out.slice = out.mem[1] * out.mem[2];

        metrics_h_malloc_f(&out);
        metrics_d_malloc_f(&out);

        return out;
}

void metrics_build_f(f_grid_t *f)
{
        metrics_interpolate_f(f);
        metrics_differentiate_f(f);
        metrics_d_copy_f(f);
}

void metrics_free_f(f_grid_t *f)
{
        metrics_h_free_f(f);
        metrics_d_free_f(f);
}

void metrics_print_info_f(const f_grid_t *f)
{
        printf("Topography function\n");
        printf("   size = %d %d %d \n", f->size[0], f->size[1], f->size[2]);
        printf("   mem = %d %d %d \n", f->mem[0], f->mem[1], f->mem[2]);
        printf("   bounds_x = %d %d \n", f->bounds_x[0], f->bounds_x[1]);
        printf("   bounds_y = %d %d \n", f->bounds_y[0], f->bounds_y[1]);
        printf("   bounds_stress_x = %d %d \n", f->bounds_stress_x[0],
               f->bounds_stress_x[1]);
        printf("   bounds_stress_y = %d %d \n", f->bounds_stress_y[0],
               f->bounds_stress_y[1]);
        printf("   offset = %d %d %d \n", f->offset[0], f->offset[1],
               f->offset[2]);
}

size_t metrics_sizeof_f(const f_grid_t *f)
{
        return sizeof(_prec) * f->mem[0] * f->mem[1];
}

void metrics_h_malloc_f(f_grid_t *f)
{
        size_t size = metrics_sizeof_f(f);
        f->f = malloc(size);
        f->f_1 = malloc(size);
        f->f_2 = malloc(size);
        f->f_c = malloc(size);
        f->f1_1 = malloc(size);
        f->f1_2 = malloc(size);
        f->f1_c = malloc(size);
        f->f2_1 = malloc(size);
        f->f2_2 = malloc(size);
        f->f2_c = malloc(size);
}

void metrics_d_malloc_f(f_grid_t *f)
{
        size_t size = metrics_sizeof_f(f);
        CUCHK(cudaMalloc((void **)&f->d_f, size));
        CUCHK(cudaMalloc((void **)&f->d_f_1, size));
        CUCHK(cudaMalloc((void **)&f->d_f_2, size));
        CUCHK(cudaMalloc((void **)&f->d_f_c, size));
        CUCHK(cudaMalloc((void **)&f->d_f1_1, size));
        CUCHK(cudaMalloc((void **)&f->d_f1_2, size));
        CUCHK(cudaMalloc((void **)&f->d_f1_c, size));
        CUCHK(cudaMalloc((void **)&f->d_f2_1, size));
        CUCHK(cudaMalloc((void **)&f->d_f2_2, size));
        CUCHK(cudaMalloc((void **)&f->d_f2_c, size));
}

void metrics_d_copy_f(f_grid_t *f)
{
        size_t size = metrics_sizeof_f(f);
	CUCHK(cudaMemcpy(f->d_f, f->f, size, cudaMemcpyHostToDevice));
	CUCHK(cudaMemcpy(f->d_f_1, f->f_1, size, cudaMemcpyHostToDevice));
	CUCHK(cudaMemcpy(f->d_f_2, f->f_2, size, cudaMemcpyHostToDevice));
	CUCHK(cudaMemcpy(f->d_f_c, f->f_c, size, cudaMemcpyHostToDevice));
	CUCHK(cudaMemcpy(f->d_f1_1, f->f1_1, size, cudaMemcpyHostToDevice));
	CUCHK(cudaMemcpy(f->d_f1_2, f->f1_2, size, cudaMemcpyHostToDevice));
	CUCHK(cudaMemcpy(f->d_f1_c, f->f1_c, size, cudaMemcpyHostToDevice));
	CUCHK(cudaMemcpy(f->d_f2_1, f->f2_1, size, cudaMemcpyHostToDevice));
	CUCHK(cudaMemcpy(f->d_f2_2, f->f2_2, size, cudaMemcpyHostToDevice));
	CUCHK(cudaMemcpy(f->d_f2_c, f->f2_c, size, cudaMemcpyHostToDevice));
}

void metrics_h_free_f(f_grid_t *f)
{
        free(f->f);
        free(f->f_1); 
        free(f->f_2); 
        free(f->f_c); 
        free(f->f1_1);
        free(f->f1_2);
        free(f->f1_c);
        free(f->f2_1);
        free(f->f2_2);
        free(f->f2_c);
}

void metrics_d_free_f(f_grid_t *f)
{
        CUCHK(cudaFree(f->d_f));
        CUCHK(cudaFree(f->d_f_1));
        CUCHK(cudaFree(f->d_f_2));
        CUCHK(cudaFree(f->d_f_c));
        CUCHK(cudaFree(f->d_f1_1));
        CUCHK(cudaFree(f->d_f1_2));
        CUCHK(cudaFree(f->d_f1_c));
        CUCHK(cudaFree(f->d_f2_1));
        CUCHK(cudaFree(f->d_f2_2));
        CUCHK(cudaFree(f->d_f2_c));
}

void metrics_interpolate_f(f_grid_t *f)
{
        metrics_f_interp_1_111(f->f_1, f->f, f->size[0], f->size[1],
                               f->size[2]);
        metrics_f_interp_2_111(f->f_2, f->f, f->size[0], f->size[1],
                               f->size[2]);
        metrics_f_interp_c_111(f->f_c, f->f, f->size[0], f->size[1],
                               f->size[2]);
}

void metrics_differentiate_f(f_grid_t *f)
{
        // Differentiate in the r1-direction to edge-1, edge-2, cell-center
        // positions
        metrics_f_diff_1_1_111(f->f1_1, f->f_c, f->hi, f->size[0], f->size[1],
                             f->size[2]);
        metrics_f_diff_1_2_111(f->f1_2, f->f, f->hi, f->size[0], f->size[1],
                             f->size[2]);
        metrics_f_diff_1_2_111(f->f1_c, f->f_1, f->hi, f->size[0], f->size[1],
                             f->size[2]);
        // Differentiate in the r2-direction to edge-1, edge-2, cell-center
        // positions
        metrics_f_diff_2_1_111(f->f2_1, f->f, f->hi, f->size[0], f->size[1],
                             f->size[2]);
        metrics_f_diff_2_2_111(f->f2_2, f->f_c, f->hi, f->size[0], f->size[1],
                             f->size[2]);
        metrics_f_diff_2_1_111(f->f2_c, f->f_2, f->hi, f->size[0], f->size[1],
                             f->size[2]);
}

void metrics_shift_f(f_grid_t *fout, const f_grid_t *fin)
{
        int nx = fout->size[0];
        int ny = fout->size[1];
        metrics_shift_f_apply(fout->f, fin->f, nx, ny);
        metrics_shift_f_apply(fout->f_1, fin->f_1, nx, ny);
        metrics_shift_f_apply(fout->f_2, fin->f_2, nx, ny);
        metrics_shift_f_apply(fout->f_c, fin->f_c, nx, ny);

        metrics_shift_f_apply(fout->f1_1, fin->f1_1, nx, ny);
        metrics_shift_f_apply(fout->f1_2, fin->f1_2, nx, ny);
        metrics_shift_f_apply(fout->f1_c, fin->f1_c, nx, ny);

        metrics_shift_f_apply(fout->f2_1, fin->f2_1, nx, ny);
        metrics_shift_f_apply(fout->f2_2, fin->f2_2, nx, ny);
        metrics_shift_f_apply(fout->f2_c, fin->f2_c, nx, ny);
}

int metrics_interpolate_f_point(const f_grid_t *f, prec *out, const prec *in,
                                const prec *x, const prec *y,
                                grid3_t grid, const prec *qx,
                                const prec *qy, const int m, const int deg) 
{
        int err = 0;
        prec *lx, *ly, *xloc, *yloc;
        lx = calloc(sizeof(lx), (deg + 1));
        ly = calloc(sizeof(ly), (deg + 1));
        xloc = calloc(sizeof(xloc), (deg + 1));
        yloc = calloc(sizeof(yloc), (deg + 1));

        for (int q = 0; q < m; ++q) { 
                int ix = 0; int iy = 0;
                err = interp_lagrange1_coef(
                    xloc, lx, &ix, x, grid_boundary_size(grid).x, qx[q], deg);
                err = interp_lagrange1_coef(
                    yloc, ly, &iy, y, grid_boundary_size(grid).y, qy[q], deg);
                out[q] = 0.0;
                for (int i = 0; i < deg + 1; ++i) {
                for (int j = 0; j < deg + 1; ++j) {
                        int pos = metrics_f_index(f, ix + i, iy + j);
                        out[q] += lx[i] * ly[j] * in[pos];
                }
                }
        }

        free(lx);
        free(ly);
        free(xloc);
        free(yloc);

        return err;
}

g_grid_t metrics_init_g(const int *size, const _prec gridspacing)
{
        g_grid_t out = {
            .size = size[2],
            .mem = size[2] + 2*align,
            .bounds_z = {0, size[2]},
            .offset = align,
            .hi = 1.0/gridspacing
        };
        out.line = out.mem;

        metrics_h_malloc_g(&out);
        metrics_d_malloc_g(&out);

        return out;
}

void metrics_build_g(g_grid_t *g)
{
        metrics_interpolate_g(g);
        metrics_differentiate_g(g);
        metrics_d_copy_g(g);
}

void metrics_free_g(g_grid_t *g)
{
        metrics_h_free_g(g);
        metrics_d_free_g(g);
}

size_t metrics_sizeof_g(const g_grid_t *g)
{
        return sizeof(_prec) * g->mem;
}

void metrics_print_info_g(const g_grid_t *g)
{
        printf("Grid stretching function \n");
        printf("    size = %d \n", g->size);
        printf("    mem = %d \n", g->mem);
        printf("    bounds_z = %d %d \n", g->bounds_z[0], g->bounds_z[1]);
        printf("    offset = %d \n", g->offset);
}

void metrics_h_malloc_g(g_grid_t *g)
{
        size_t size = metrics_sizeof_g(g);
        g->g = (_prec*)malloc(size);
        g->g_c = (_prec*)malloc(size);
        g->g3 = (_prec*)malloc(size);
        g->g3_c = (_prec*)malloc(size);
}

void metrics_d_malloc_g(g_grid_t *g)
{
        size_t size = 100*metrics_sizeof_g(g);
        CUCHK(cudaMalloc((void **)&g->d_g, size));
        CUCHK(cudaMalloc((void **)&g->d_g_c, size));
        CUCHK(cudaMalloc((void **)&g->d_g3, size));
        CUCHK(cudaMalloc((void **)&g->d_g3_c, size));
}

void metrics_d_copy_g(g_grid_t *g)
{
        size_t size = metrics_sizeof_g(g);
        CUCHK(cudaMemcpy(g->d_g, g->g, size, cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(g->d_g_c, g->g_c, size, cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(g->d_g3, g->g3, size, cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(g->d_g3_c, g->g3_c, size, cudaMemcpyHostToDevice));
}

void metrics_h_free_g(g_grid_t *g)
{
        free(g->g);
        free(g->g_c);
        free(g->g3);
        free(g->g3_c);
}

void metrics_d_free_g(g_grid_t *g)
{
        CUCHK(cudaFree(g->d_g));
        CUCHK(cudaFree(g->d_g_c));
        CUCHK(cudaFree(g->d_g3));
        CUCHK(cudaFree(g->d_g3_c));
}

void metrics_interpolate_g(g_grid_t *g)
{
        metrics_g_interp_110(g->g_c, g->g, 1, 1, g->size); 
        metrics_g_interp_111(g->g_c, g->g, 1, 1, g->size); 
        metrics_g_interp_112(g->g_c, g->g, 1, 1, g->size); 
}

void metrics_differentiate_g(g_grid_t *g)
{
        // Differentiate in the r3-direction to the nodes
        metrics_g_diff_3_110(g->g3, g->g_c, g->hi, 1, 1, g->size); 
        metrics_g_diff_3_111(g->g3, g->g_c, g->hi, 1, 1, g->size); 
        metrics_g_diff_3_112(g->g3, g->g_c, g->hi, 1, 1, g->size); 
        // Set ghost point value to anything else than zero
        // If this value is zero and if it is inverted it will become infinity
        // and break the computation.
        g->g3[g->offset + g->size - 1] = 1.0;

        // Differentiate in the r3-direction to the cell-centers
        metrics_g_diff_c_110(g->g3_c, g->g, g->hi, 1, 1, g->size); 
        metrics_g_diff_c_111(g->g3_c, g->g, g->hi, 1, 1, g->size); 
        metrics_g_diff_c_112(g->g3_c, g->g, g->hi, 1, 1, g->size); 
}

/* 
 * Convert metrics_g grid type to fcn_grid_t grid type. 
 */ 
fcn_grid_t metrics_grid_g(const g_grid_t *g)
{
        int3_t shift = grid_node();
        int3_t size = {.x = 1, .y = 1, .z = g->size};
        // MPI coordinate is not important for grid that only lives in the
        // vertical direction
        int3_t coord = {0, 0, 0};

        int3_t bnd1 = {0, 0 ,0}, bnd2 = {0, 0, 1};
        fcn_grid_t grid =
            grid_init(size, shift, coord, bnd1, bnd2, 2 + ngsl, 1.0 / g->hi);
        grid.offset1.x = 0;
        grid.offset2.x = 1;
        grid.offset1.y = 0;
        grid.offset2.y = 1;
        grid.line = 1;
        grid.slice = 2*align + size.z;
        return grid;
}

