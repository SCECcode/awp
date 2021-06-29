#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <grid/shift.h>
#include <topography/geometry/geometry.h>
#include <topography/metrics/metrics.h>
#include <topography/mapping.h>
#include <grid/grid_3d.h>
#include <functions/functions.h>
#include <test/test.h>

void geom_cartesian_topography(f_grid_t *metrics_f)
{
        int3_t shift = grid_node();
        _prec coef[3] = {1.0, 0.0, 0.0};
        _prec deg[3] = {0.0, 0.0, 0.0};
        _prec args[13] = {coef[0], coef[1], coef[2],
                          deg[0], deg[1], deg[2],
                          shift.x, shift.y, shift.z,
                          metrics_f->size[0], metrics_f->size[1],
                          0, 0};

        fcn_poly(metrics_f->f,
                 metrics_f->offset[0] + metrics_f->bounds_x[0], 
                 metrics_f->offset[0] + metrics_f->bounds_x[1],
                 metrics_f->offset[1] + metrics_f->bounds_y[0], 
                 metrics_f->offset[1] + metrics_f->bounds_y[1],
                 0, 1, 
                 metrics_f->line, metrics_f->slice, args);
}

void geom_no_grid_stretching(g_grid_t *metrics_g)
{
        fcn_grid_t grid = metrics_grid_g(metrics_g);
        grid1_t grid1 = grid_grid1_z(grid);
        grid1.shift = grid_node().z;
        grid1.boundary1 = 0;
        grid1.boundary2 = 1;
        grid_fill1(&metrics_g->g[grid1.alignment], grid1, 0);
        // Shift grid vector so that the internal coordinate system places z = 0 at the first grid
        // point immediately above the DM overlap zone
        for (int i = 0; i < grid1.size; ++i) {
                metrics_g->g[i + grid1.alignment] -= MAPPING_START_POINT * grid1.gridspacing;
        }
}

void geom_gaussian(f_grid_t *metrics_f, const _prec *x, const _prec *y,
                   const fcn_grid_t grid,
                   const _prec amplitude,
                   const _prec3_t width, const _prec3_t center) {
        int len_x = metrics_f->bounds_x[1] - metrics_f->bounds_x[0];
        int len_y = metrics_f->bounds_y[1] - metrics_f->bounds_y[0];

        int off_x = metrics_f->offset[0] + metrics_f->bounds_x[0];
        int off_y = metrics_f->offset[1] + metrics_f->bounds_y[0];

        size_t last = grid.offset1.z + (grid.offset1.y + grid.size.y / 2) * grid.line +
                      (grid.offset1.x + grid.size.x / 2 ) * grid.slice;
        prec xm = x[last];
        prec ym = y[last];

        // Grid spacing in vertical direction for a grid satsifying:
        // 0 <= z' < =1
        // This normalization constant is used so that the user can specify
        // block dimension using physical units.
        _prec normalize = 1.0 / grid.gridspacing / (grid.size.z - 2);

        _prec  max = 0.0;
        for (int i = 0; i < len_x; ++i) {
        for (int j = 0; j < len_y; ++j) {
                int pos_f = (off_y + j) + (off_x + i) * metrics_f->slice;
                int pos = grid.offset1.z + (grid.offset1.y + j) * grid.line +
                            (grid.offset1.x + i) * grid.slice;
                metrics_f->f[pos_f] = 1.0 + 
                        normalize * amplitude * exp(-0.5 / pow(width.x, 2) *
                                        pow(x[pos] - center.x - xm, 2) -
                                    -0.0 / pow(width.y, 2) *
                                        pow(y[pos] - center.y - ym, 2));
                if (metrics_f->f[pos_f] > max) {
                        max = metrics_f->f[pos_f];
                }
        }
        }
}

void geom_incline_plane(f_grid_t *metrics_f, const _prec *x, const _prec *y,
                   const grid3_t grid, const _prec phi_x, const _prec phi_y,
                   const int px, const int py)
{
        int len_x = metrics_f->bounds_x[1] - metrics_f->bounds_x[0];
        int len_y = metrics_f->bounds_y[1] - metrics_f->bounds_y[0];

        int off_x = metrics_f->offset[0] + metrics_f->bounds_x[0];
        int off_y = metrics_f->offset[1] + metrics_f->bounds_y[0];

        // Introduce a normalization constant so that the z-coordinate maps to
        // the unit interval:
        // 0 <= z' < =1
        // This normalization constant is used so that the user can specify
        // block dimension using physical units.
        _prec normalize = 1.0 / grid.gridspacing / (grid.size.z - 2);
        prec xm = 0.5 * grid.gridspacing * (px * grid.inner_size.x - 1);
        prec ym = 0.5 * grid.gridspacing * (py * grid.inner_size.y - 1);

        for (int i = 0; i < len_x; ++i) {
        for (int j = 0; j < len_y; ++j) {
                int pos_f = (off_y + j) + (off_x + i) * metrics_f->slice;
                metrics_f->f[pos_f] = 1.0 + (x[i] - xm) * tan(phi_x) * normalize
                                          + (y[j] - ym) * tan(phi_y) * normalize; 
        }
        }

}

void geom_gaussian_hill_and_canyon_xz(
    f_grid_t *metrics_f, const _prec *x, const _prec *y, const grid3_t grid,
    const _prec3_t hill_width, const _prec hill_height,
    const _prec3_t hill_center, const _prec3_t canyon_width,
    const _prec canyon_height, const _prec3_t canyon_center, const int px,
    const int py) 
{
        int len_x = metrics_f->bounds_x[1] - metrics_f->bounds_x[0];
        int len_y = metrics_f->bounds_y[1] - metrics_f->bounds_y[0];

        int off_x = metrics_f->offset[0] + metrics_f->bounds_x[0];
        int off_y = metrics_f->offset[1] + metrics_f->bounds_y[0];

        // Introduce a normalization constant so that the z-coordinate maps to
        // the unit interval:
        // 0 <= z' < =1
        // This normalization constant is used so that the user can specify
        // block dimension using physical units.
        _prec normalize = 1.0 / grid.gridspacing / (grid.size.z - 2);
        prec xm = 0.5 * grid.gridspacing * (px * grid.inner_size.x - 1);

        for (int i = 0; i < len_x; ++i) {
        for (int j = 0; j < len_y; ++j) {
                int pos_f = (off_y + j) + (off_x + i) * metrics_f->slice;
                prec hill =
                    hill_height * exp(- powf(x[i] - xm - hill_center.x, 2) /
                                        powf(hill_width.x, 2));
                prec canyon =
                    canyon_height * exp(-powf(x[i] - xm - canyon_center.x, 2) /
                                       powf(canyon_width.x, 2));
                metrics_f->f[pos_f] = 1.0 + (hill - canyon) * normalize;
        }
        }

}

void geom_gaussian_hill_and_canyon(
    f_grid_t *metrics_f, const _prec *x, const _prec *y, const grid3_t grid,
    const _prec3_t hill_width, const _prec hill_height,
    const _prec3_t hill_center, const _prec3_t canyon_width,
    const _prec canyon_height, const _prec3_t canyon_center, const int px,
    const int py) 
{
        int len_x = metrics_f->bounds_x[1] - metrics_f->bounds_x[0];
        int len_y = metrics_f->bounds_y[1] - metrics_f->bounds_y[0];

        int off_x = metrics_f->offset[0] + metrics_f->bounds_x[0];
        int off_y = metrics_f->offset[1] + metrics_f->bounds_y[0];

        // Introduce a normalization constant so that the z-coordinate maps to
        // the unit interval:
        // 0 <= z' < =1
        // This normalization constant is used so that the user can specify
        // block dimension using physical units.
        _prec normalize = 1.0 / grid.gridspacing / (grid.size.z - 2);
        prec xm = 0.5 * grid.gridspacing * (px * grid.inner_size.x - 1);
        prec ym = 0.5 * grid.gridspacing * (py * grid.inner_size.y - 1);

        for (int i = 0; i < len_x; ++i) {
        for (int j = 0; j < len_y; ++j) {
                int pos_f = (off_y + j) + (off_x + i) * metrics_f->slice;
                prec hill =
                    hill_height * exp(-powf(x[i] - xm - hill_center.x, 2) /
                                       powf(hill_width.x, 2)
                                      -powf(y[j] - ym - hill_center.y, 2) /
                                       powf(hill_width.y, 2));
                prec canyon =
                    canyon_height * exp(-powf(x[i] - xm - canyon_center.x, 2) /
                                         powf(canyon_width.x, 2)
                                        -powf(y[j] - ym - canyon_center.y, 2) /
                                         powf(canyon_width.y, 2));
                metrics_f->f[pos_f] = 1.0 + (hill - canyon) * normalize;
        }
        }
}

void geom_custom(const f_grid_t *metrics_f, const grid3_t grid, const int px,
                 const int py, prec *f) 
{
        int len_x = metrics_f->bounds_x[1] - metrics_f->bounds_x[0];
        int len_y = metrics_f->bounds_y[1] - metrics_f->bounds_y[0];

        int off_x = metrics_f->offset[0] + metrics_f->bounds_x[0];
        int off_y = metrics_f->offset[1] + metrics_f->bounds_y[0];

        // Introduce a normalization constant so that the z-coordinate maps to
        // the unit interval:
        // 0 <= z' < =1
        // This normalization constant is used so that the user can specify
        // block dimension using physical units.
        _prec normalize = 1.0 / grid.gridspacing / (grid.size.z - 2 - MAPPING_START_POINT);

        for (int i = 0; i < len_x; ++i) {
        for (int j = 0; j < len_y; ++j) {
                int pos_f = (off_y + j) + (off_x + i) * metrics_f->slice;
                f[pos_f] = 1.0 + f[pos_f] * normalize;
        }
        }
}

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

