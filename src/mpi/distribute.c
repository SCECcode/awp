#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <mpi/distribute.h>
#include <awp/error.h>
#include <grid/grid_3d.h>
#include <topography/grids.h>
#include <topography/sources/source.h>
#include <test/test.h>

/* Distributes indices based on which part of space they belong to. 

        indices: (output) indices for a particular query point (qx[i], qy[i]) that lies in `grid`. 
        nidx: Number of indices written
        qx: Array containing query points  (x-coordinate)
        qy: Array containing query points  (y-coordinate)
        n: Number of query points (length of qx, qx)
        grids: The grids to conduct the search for
        grid_numbers: Array that contains the grid number that each query point belongs to (in the
                      z-direction) 
        is_source: Set grid bounds based on source partitioning 
                  (disable to set grid bounds for receiver partitioning)
        mode: Choose between counting indices in current partition (DIST_COUNT), or populate
              index array (DIST_INSERT_INDICES)

*/

__inline__ int dist_indices_in_bounds(const prec qx, const prec qy,
                                      const prec *x, const prec *y,
                                      grid1_t grid_x, grid1_t grid_y,
                                      const enum source_type st) {
        int inbounds_x = 0;
        int inbounds_y = 0;
        switch (st) {
                case MOMENT_TENSOR:
                        inbounds_x = grid_in_bounds_moment_tensor(x, qx, grid_x);
                        inbounds_y = grid_in_bounds_moment_tensor(y, qy, grid_y);
                        break;
                case FORCE:
                        inbounds_x = grid_in_bounds_force(x, qx, grid_x);
                        inbounds_y = grid_in_bounds_force(y, qy, grid_y);
                        break;
                case RECEIVER:
                        inbounds_x = grid_in_bounds_receiver(x, qx, grid_x);
                        inbounds_y = grid_in_bounds_receiver(y, qy, grid_y);
                        break;
                case SGT:
                        inbounds_x = grid_in_bounds_sgt(x, qx, grid_x);
                        inbounds_y = grid_in_bounds_sgt(y, qy, grid_y);
                        break;
        }
        if (inbounds_x == SUCCESS && inbounds_y == SUCCESS)
                return 1;
        return 0;
}

int dist_indices(int **indices, size_t *nidx, const prec *qx, const prec *qy,
                 const size_t n, const grid3_t grid, const int *grid_numbers,
                 const int grid_number, const enum source_type st, const enum dist_options mode)
{

        size_t nlocal = 0;

        grid1_t grid_x = grid_grid1_x(grid);
        grid1_t grid_y = grid_grid1_y(grid);

        prec *x = malloc(sizeof(x) * grid_x.size);
        prec *y = malloc(sizeof(y) * grid_y.size);

        grid_fill1(x, grid_x, 1);
        grid_fill_y_dm(y, grid_y, grid_number);

        size_t j = *nidx;
        for (size_t i = 0; i < n; ++i)
        {
                if (dist_indices_in_bounds(qx[i], qy[i], x, y, grid_x, grid_y, st) &&
                    grid_numbers[i] == grid_number)
                {
                        switch (mode)
                        {
                        case DIST_COUNT:
                                nlocal++;
                                break;
                        case DIST_INSERT_INDICES:
                                (*indices)[j] = i;
                                j++;
                        }
                }
        }

        free(x);
        free(y);

        switch (mode)
        {
        case DIST_COUNT:
                *nidx = nlocal;
                break;
        case DIST_INSERT_INDICES:
                break;
        }

        return SUCCESS;
}
