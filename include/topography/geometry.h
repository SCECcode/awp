#pragma once
#ifndef TOPO_GEOMETRY_H
#define TOPO_GEOMETRY_H

#include <awp/definitions.h>
#include <topography/topography.h>

#ifdef __cplusplus
extern "C" {
#endif

void topo_init_grid(topo_t *T);
void topo_init_gaussian_geometry(topo_t* T, const _prec amplitude,
                                 const _prec3_t width, const _prec3_t center);
void topo_init_incline_plane(topo_t* T, const _prec phi_x, const _prec phi_y);
void topo_init_gaussian_hill_and_canyon_xz(topo_t *T, const _prec3_t hill_width,
                                        const _prec hill_height, 
                                        const _prec3_t hill_center,
                                        const _prec3_t canyon_width,
                                        const _prec canyon_height,
                                        const _prec3_t canyon_center);
void topo_init_gaussian_hill_and_canyon(topo_t *T, const _prec3_t hill_width,
                                        const _prec hill_height,
                                        const _prec3_t hill_center,
                                        const _prec3_t canyon_width,
                                        const _prec canyon_height,
                                        const _prec3_t canyon_center);

#ifdef __cplusplus
}
#endif
#endif
