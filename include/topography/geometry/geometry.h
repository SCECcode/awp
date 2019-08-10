/* This module contains pre-defined surface topography and grid
 * stretching functions. 
 *
 */ 
#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <topography/metrics/metrics.h>
#include <grid/grid_3d.h>

void geom_cartesian_topography(f_grid_t *metrics_f);
void geom_no_grid_stretching(g_grid_t *metrics_g);

void geom_gaussian(f_grid_t *metrics_f, const _prec *x, const _prec *y,
                   const fcn_grid_t grid, const _prec amplitude,
                   const _prec3_t width, const _prec3_t center);

void geom_gaussian_hill_and_canyon_xz(
    f_grid_t *metrics_f, const _prec *x, const _prec *y, const grid3_t grid,
    const _prec3_t hill_width, const _prec hill_height,
    const _prec3_t hill_center, const _prec3_t canyon_width,
    const _prec canyon_height, const _prec3_t canyon_center, const int px,
    const int py);
void geom_gaussian_hill_and_canyon(
    f_grid_t *metrics_f, const _prec *x, const _prec *y, const grid3_t grid,
    const _prec3_t hill_width, const _prec hill_height,
    const _prec3_t hill_center, const _prec3_t canyon_width,
    const _prec canyon_height, const _prec3_t canyon_center, const int px,
    const int py);
void geom_custom(const f_grid_t *metrics_f, const grid3_t grid, const int px,
                 const int py, prec *f);

/*
 * Generate built-in incline plane geometry
 *
 * Angle of incline is defined by `phi` and is measured in radians. An angle of
 * `phi = 0` gives a horizontal plane. The variable `phi_x` is the angle of
 * incline for the x-direction, and `phi_y` is the angle of incline for the
 * y-direction.
 *
 */ 
void geom_incline_plane(f_grid_t *metrics_f, const _prec *x, const _prec *y,
                   const grid3_t grid, const _prec phi_x, const _prec phi_y,
                   const int px, const int py);

void geom_ramp(_prec *out, const fcn_grid_t grid, const f_grid_t *metrics_f,
               const _prec *x, const _prec *y, const _prec3_t ramp);

void geom_mapping_z(_prec *out, const fcn_grid_t grid, const int3_t shift,
                    const f_grid_t *metrics_f,
                    const g_grid_t *metrics_g);

#endif

