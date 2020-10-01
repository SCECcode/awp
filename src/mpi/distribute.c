#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <mpi/distribute.h>
#include <awp/error.h>
#include <test/test.h>

/* Distributes indices based on which part of space they belong to. 

        indices: (output) indices for a particular query point (qx[i], qy[i]) that lies in `grid`. 
        nidx: Number of indices written
        qx: Array containing query points  (x-coordinate)
        qy: Array containing query points  (y-coordinate)
        n: Number of query points (length of qx, qx)
        grid: The grid to conduct the search for
        grid_numbers: Array that contains the grid number that each query point belongs to (in the
                      z-direction) 
        grid_number: The grid number for `grid`
        is_source: Set grid bounds based on source partitioning 
                  (disable to set grid bounds for receiver partitioning)

*/
int dist_indices(int **indices, size_t *nidx, const prec *qx, const prec *qy,
                 const size_t n, grid3_t grid, const int *grid_numbers,
                 const int grid_number, const int is_source) {
        grid1_t grid_x = grid_grid1_x(grid);
        grid1_t grid_y = grid_grid1_y(grid);

        prec *x = malloc(sizeof(x) * grid_x.size);
        prec *y = malloc(sizeof(y) * grid_y.size);

        grid_fill1(x, grid_x);
        grid_fill1(y, grid_y);


        size_t nlocal = 0; 

        for (size_t i = 0; i < n; ++i) {
                int inbounds_x = is_source ? grid_in_bounds_part_x(x, qx[i], grid_x) : grid_in_bounds_ext1(x, qx[i], grid_x);
                int inbounds_y = is_source ? grid_in_bounds_part_y(y, qy[i], grid_y) : grid_in_bounds_ext1(y, qy[i], grid_y);
                if (inbounds_x == SUCCESS && inbounds_y == SUCCESS &&
                    grid_numbers[i] == grid_number) {
                        nlocal++;
                }
        }
        
        *nidx = nlocal;
        *indices = malloc(sizeof(indices) * nlocal);
        int *idx = *indices;

        int j = 0;
        for (size_t i = 0; i < n; ++i) {
                int inbounds_x = is_source ? grid_in_bounds_part_x(x, qx[i], grid_x) : grid_in_bounds_ext1(x, qx[i], grid_x);
                int inbounds_y = is_source ? grid_in_bounds_part_y(y, qy[i], grid_y) : grid_in_bounds_ext1(y, qy[i], grid_y);
                if (inbounds_x == SUCCESS && inbounds_y == SUCCESS &&
                    grid_numbers[i] == grid_number) {
                        idx[j] = i;
                        j++;
                }
        }

        free(x);
        free(y);
        
        return SUCCESS;
}


