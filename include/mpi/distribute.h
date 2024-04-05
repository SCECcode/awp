#ifndef DISTRIBUTE_H
#define DISTRIBUTE_H

#include <awp/definitions.h>
#include <grid/grid_3d.h>
#include <topography/grids.h>
#include <topography/sources/source.h>

#ifdef __cplusplus
extern "C" {
#endif
#
enum dist_options {DIST_COUNT, DIST_INSERT_INDICES};

int dist_indices(int **indices, size_t *nidx, const prec *qx, const prec *qy,
                 const size_t n, const grid3_t grid, const int *grid_numbers,
                 const int grid_number, const enum source_type st, const enum dist_options mode);

int dist_indices_in_bounds(const prec qx, const prec qy,
                           const prec *x, const size_t mx, 
                           const prec *y, const size_t my,
                           const prec hx, const prec hy,
                           const enum source_type st);
#ifdef __cplusplus
}
#endif
#endif
