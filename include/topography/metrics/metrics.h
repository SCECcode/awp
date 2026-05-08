/* Metrics
 * This module constructs the metric coefficients used by the topography kernel
 * functions. Example use cases and tests are defined in `test_metrics.c`.
 *
 * There are only two parts to constructing the metrics coefficients: 
 * 1. Define a topography function `f(x1, x2)`
 * 2. Define a grid stretching function `g(x3)`.
 *
 * It is up to the user to choose the topography function and grid stretching
 * function. The topography function describes the shape of the free surface
 * boundary. The grid stretching function determines the distribution of grid
 * points in the vertical direction. This function must satisfy:
 *  g(1) = 1 (Condition at the free surface)
 *  g(0) = 0 (Condition at depth)
 *
 *  More details about these functions are documented below.
 *
 */ 
#ifndef METRICS_H
#define METRICS_H

#include <stddef.h>
#include <awp/definitions.h>
#include <grid/grid_3d.h>

// This parameter pads the compute region. Its needed for the computation of
// derivative and interpolation stencils. Do not change its value.

#define metrics_f_index(g,i,j) ((g)->offset[1] + (g)->bounds_y[0] + j) + \
                               ((g)->offset[0] + (g)->bounds_x[0] + i) * \
                               (g)->slice

static const int metrics_padding = 8;

/*
 * Topography function `f(x1, x2)`. 
 * The topography function is a 2D scalar field that defines the elevation at
 * each grid point at the sea level (or some other chosen datum). 
 *
 * All functions and variables associated with this function have `_f` in their
 * name.
 *
 * For the topography kernel functions to perform correctly, the topography
 * function must be defined and in both the compute regions and ghost regions.
 * Hence, the domain size of the topography function is (nx + 2 * ghost cells )
 * x (ny + 2 * ghost cells). The topography function needs this extra padding to
 * ensure that when it is interpolated or differentiated, those values are valid
 * at all of the stress grids points (nx + ghost cells ) x (ny + ghost cells).
 * Each time the function is interpolated in a direction, its domain in which it
 * is valid decreases by 4 points (2 points for each side). This decrease comes
 * from using a fourth order centered finite difference approximation for either
 * interpolation or differentiation.
 *
 */ 
typedef struct
{
        int size[3];
        int mem[3];
        int offset[3];
        // Bounds covering (nx + 2 * ghost cells ) x (ny + 2 * ghost cells)
        int bounds_x[2];
        int bounds_y[2];
        // Bounds covering (nx + ghost cells) x (ny + ghost cells)
        // (stress-grid)
        int bounds_stress_x[2];
        int bounds_stress_y[2];
        // 1/grid spacing
        _prec hi;
        int line;
        int slice;
        // valid at :  (nx + 2 * ghost cells ) x (ny + 2 * ghost cells) 
        _prec *f;
        // valid at :  (nx + ghost cells ) x (ny + ghost cells) 
        _prec *f_1;
        _prec *f_2;
        _prec *f_c;
        // Derivatives
        // Naming convention
        // f1_c : Derivative of `f` with respect to `x1` evaluated at grid `c`
        // (cell-center), etc.
        // valid at :  (nx + ghost cells ) x (ny + ghost cells) 
        _prec *f1_1;
        _prec *f1_2;
        _prec *f1_c;
        _prec *f2_1;
        _prec *f2_2;
        _prec *f2_c;

        // Device variables
        _prec* __restrict__ d_f;
        _prec* __restrict__ d_f_1;
        _prec* __restrict__ d_f_2;
        _prec* __restrict__ d_f_c;
        _prec* __restrict__ d_f1_1;
        _prec* __restrict__ d_f1_2;
        _prec* __restrict__ d_f1_c;
        _prec* __restrict__ d_f2_1;
        _prec* __restrict__ d_f2_2;
        _prec* __restrict__ d_f2_c;
} f_grid_t;
/*
 * Grid stretching function
 *
 * The grid function uses one-sided difference operators to make sure that it is
 * always well-behaved through-out the entire vertical section of the grid
 * block. Therefore, there is no need to define the grid stretching function
 * outside its domain (like it is for the topography function).
 */ 
typedef struct
{
        int size;
        int mem;
        int line;
        int offset;
        int bounds_z[2];
        // 1/grid spacing
        _prec hi;
        // valid at :  align + (0, nz)
        _prec *g;
        // valid at :  align + (0, nz)
        _prec *g_c;
        // Derivatives
        // valid at :  align + (0, nz)
        _prec *g3;
        _prec *g3_c;
        
        _prec* __restrict__ d_g;
        _prec* __restrict__ d_g_c;
        _prec* __restrict__ d_g3;
        _prec* __restrict__ d_g3_c;

} g_grid_t;

f_grid_t metrics_init_f(const int *size, const _prec gridspacing,
                            const int pad);
void metrics_build_f(f_grid_t *f);
void metrics_free_f(f_grid_t *f);
void metrics_print_info_f(const f_grid_t *f);
size_t metrics_sizeof_f(const f_grid_t *f);

void metrics_h_malloc_f(f_grid_t *f);
void metrics_d_malloc_f(f_grid_t *f);
void metrics_h_copy_f(f_grid_t *f);
void metrics_d_copy_f(f_grid_t *f);
void metrics_h_free_f(f_grid_t *f);
void metrics_d_free_f(f_grid_t *f);
void metrics_interpolate_f(f_grid_t *f);
void metrics_differentiate_f(f_grid_t *f);
void metrics_shift_f(f_grid_t *fout, const f_grid_t *fin);
int metrics_interpolate_f_point(const f_grid_t *f, prec *out, const prec *in,
                                const prec *x, const prec *y,
                                const grid3_t grid, const prec *qx,
                                const prec *qy, const int m, const int deg);

int metrics_interpolate_jacobian(const f_grid_t *fgrid, float *out, const float *f, const float *g,
                        const float *x, const float *y, const float *z,
                        grid3_t grid, const float *qx,
                        const float *qy, const float *qz, const int m, const int deg);

g_grid_t metrics_init_g(const int *size, const _prec gridspacing);
void metrics_build_g(g_grid_t *g);
void metrics_free_g(g_grid_t *g);
void metrics_print_info_g(const g_grid_t *g);
size_t metrics_sizeof_g(const g_grid_t *g);

void metrics_h_malloc_g(g_grid_t *g);
void metrics_d_malloc_g(g_grid_t *g);
void metrics_d_copy_g(g_grid_t *g);
void metrics_h_free_g(g_grid_t *g);
void metrics_d_free_g(g_grid_t *g);
void metrics_interpolate_g(g_grid_t *g);
void metrics_differentiate_g(g_grid_t *g);


fcn_grid_t metrics_grid_g(const g_grid_t *g);

#endif

