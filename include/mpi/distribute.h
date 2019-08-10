#ifndef DISTRIBUTE_H
#define DISTRIBUTE_H

#include <awp/definitions.h>
#include <grid/grid_3d.h>

#ifdef __cplusplus
extern "C" {
#endif

int dist_indices(int **indices, size_t *nidx, const prec *qx, const prec *qy,
                 const size_t n, grid3_t grid);

#ifdef __cplusplus
}
#endif
#endif
