#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <mpi/distribute.h>
#include <awp/error.h>
#include <test/test.h>

int dist_indices(int **indices, size_t *nidx, const prec *qx, const prec *qy,
                 const size_t n, grid3_t grid) 
{
        grid1_t grid_x = grid_grid1_x(grid);
        grid1_t grid_y = grid_grid1_y(grid);

        prec *x = malloc(sizeof(x) * grid_x.size);
        prec *y = malloc(sizeof(y) * grid_y.size);

        grid_fill1(x, grid_x);
        grid_fill1(y, grid_y);


        size_t nlocal = 0; 

        //FIXME: Add checks for PML region, and boundary regions

        for (size_t i = 0; i < n; ++i) {
                int inbounds_x = grid_in_bounds_ext1(x, qx[i], grid_x);
                int inbounds_y = grid_in_bounds_ext1(y, qy[i], grid_y);
                if (inbounds_x == SUCCESS && inbounds_y == SUCCESS) {
                        nlocal++;
                }
        }
        
        *nidx = nlocal;
        *indices = malloc(sizeof(indices) * nlocal);
        int *idx = *indices;

        int j = 0;
        for (size_t i = 0; i < n; ++i) {
                int inbounds_x = grid_in_bounds_ext1(x, qx[i], grid_x);
                int inbounds_y = grid_in_bounds_ext1(y, qy[i], grid_y);
                if (inbounds_x == SUCCESS && inbounds_y == SUCCESS) {
                        idx[j] = i;
                        j++;
                }
        }

        free(x);
        free(y);
        
        return SUCCESS;
}


