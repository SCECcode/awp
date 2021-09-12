#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <mpi/distribute.h>
#include <awp/error.h>
#include <grid/grid_3d.h>
#include <topography/grids.h>
#include <topography/sources/source.h>
#include <interpolation/interpolation.h>
#include <test/test.h>


int is_in_bounds(const _prec *x, const int mx, const _prec q, const int overlap)
{
        int nearest;
        int bounds_err = interp_argnearest(&nearest, x, mx, q);
        if (bounds_err != SUCCESS) return bounds_err;
        
        if (nearest < 2 + ngsl - overlap) return ERR_OUT_OF_BOUNDS_LOWER;
        if (nearest >= mx - 2 - ngsl + overlap) return ERR_OUT_OF_BOUNDS_UPPER;
        return SUCCESS;
}

__inline__ int dist_indices_in_bounds(const prec qx, const prec qy,
                                      const prec *x, const int mx, 
                                      const prec *y, const int my,
                                      const enum source_type st) {
        int inbounds_x = 0;
        int inbounds_y = 0;
        switch (st) {
            case MOMENT_TENSOR:
            case FORCE:
                        inbounds_x = is_in_bounds(x, mx, qx, ngsl);
                        inbounds_y = is_in_bounds(y, my, qy, ngsl);
                        break;
            case RECEIVER:
            case SGT:
                        inbounds_x = is_in_bounds(x, mx, qx, 0);
                        inbounds_y = is_in_bounds(y, my, qy, 0);
                        break;
        }
        if (inbounds_x == SUCCESS && inbounds_y == SUCCESS)
                return 1;
        return 0;
}

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
int dist_indices(int **indices, size_t *nidx, const prec *qx, const prec *qy,
                 const size_t n, 
                 const grid3_t grid, const int *grid_numbers,
                 const int grid_number, const enum source_type st, const enum dist_options mode)
{

        size_t nlocal = 0;

        grid1_t grid_x = grid_grid1_x(grid);
        grid1_t grid_y = grid_grid1_y(grid);
        int mx = grid_x.size;
        int my = grid_y.size;

        prec *x = malloc(sizeof(x) * grid_x.size);
        prec *y = malloc(sizeof(y) * grid_y.size);

        grid_fill1(x, grid_x, 1);
        grid_fill_y_dm(y, grid_y, grid_number);

        size_t j = *nidx;
        for (size_t i = 0; i < n; ++i)
        {
                if (dist_indices_in_bounds(qx[i], qy[i], x, mx, y, my, st) &&
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
