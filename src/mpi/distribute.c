#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include <mpi/distribute.h>
#include <awp/error.h>
#include <grid/grid_3d.h>
#include <topography/grids.h>
#include <topography/sources/source.h>
#include <interpolation/interpolation.h>
#include <test/test.h>

int grid_in_bounds_output(const prec *x, const size_t mx, const prec q, const prec h)
{
        prec x_left = x[2 + ngsl] - h / 2.0;
        prec x_right = x[mx - 3 - ngsl] + h / 2.0;

        if ( q - x_left < 0 ) {
                return ERR_OUT_OF_BOUNDS_LOWER;
        }
        if ( q - x_right >= 0) {
                return ERR_OUT_OF_BOUNDS_UPPER;
        }

        return SUCCESS;

}

int grid_in_bounds_input(const prec *x, const int mx, const prec q, const prec h)
{
    // Split the input (moment tensor / force) based on the subdomain it belongs to. Inputs that
    // fall in the overlap zone belongs to both processes. The source/force kernels have guard
    // statements that make sure that no sources/forces are applied outside the actual compute
    // region.
        if ( q - x[0] < h / 2 ) {
                return ERR_OUT_OF_BOUNDS_LOWER;
        }
        if ( q - x[mx - 1] >=  h / 2) {
                return ERR_OUT_OF_BOUNDS_UPPER;
        }
        return SUCCESS;
}

int dist_indices_in_bounds(const prec qx, const prec qy,
                           const prec *x, const size_t mx, 
                           const prec *y, const size_t my,
                           const prec hx, const prec hy,
                           const enum source_type st) {
        int inbounds_x = 0;
        int inbounds_y = 0;
        switch (st) {
            case MOMENT_TENSOR:
            case FORCE:
                        inbounds_x = grid_in_bounds_input(x, mx, qx, hx);
                        inbounds_y = grid_in_bounds_input(y, my, qy, hy);
                        break;
            case RECEIVER:
            case SGT:
                        inbounds_x = grid_in_bounds_output(x, mx, qx, hx);
                        inbounds_y = grid_in_bounds_output(y, my, qy, hy);
                        break;
            default:
                fprintf(stderr, "Unknown source type passed to %s:%s!\n",
                        __FILE__, __func__);
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
        size_t mx = grid_x.size;
        size_t my = grid_y.size;
        prec hx = grid_x.gridspacing;
        prec hy = grid_y.gridspacing;

        prec *x = malloc(sizeof(x) * grid_x.size);
        prec *y = malloc(sizeof(y) * grid_y.size);

        grid_fill1(x, grid_x, 1);
        grid_fill_y_dm(y, grid_y, grid_number);

        size_t j = *nidx;
        for (size_t i = 0; i < n; ++i)
        {
                if (dist_indices_in_bounds(qx[i], qy[i], x, mx, y, my, hx, hy, st) &&
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
