#ifndef DISTRIBUTE_H
#define DISTRIBUTE_H

#include <awp/definitions.h>
#include <grid/grid_3d.h>

#ifdef __cplusplus
extern "C" {
#endif

int dist_indices(int **indices, size_t *nidx, const prec *qx, const prec *qy,
                 const size_t n, grid3_t grid, const int *grid_numbers, const int grid_number, const int is_source);

#ifdef __cplusplus
}
#endif
#endif
