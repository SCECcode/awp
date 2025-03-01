#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define STR_LEN 2048
#define ADDLINENUM 1
#define ADDRANK 1
#define RANK rank
#define STR_LEN 2048

#include <awp/definitions.h>
#include <test/test.h>
#include <awp/error.h>
#include <utils/array.h>
#include <grid/shift.h>
#include <topography/grids.h>
#include <grid/grid_3d.h>
#include <topography/sources/source.h>
#include <topography/sources/sources.h>
#include <topography/receivers/receivers.h>
#include <readers/input.h>

void init(float **x, float **y, int nx, int ny, int px, int py, float h);
int inbounds(int nx, int ny, int blocknum, int px, int py, float h, int degree, const enum source_type st);

int main(int argc, char **argv)
{
        test_divider();
        printf("Testing test_source_distribution.c\n");

        int nx = 32;
        int ny = 32;
        float h = 1.0f;

        int degree = 3;

        int err = 0;
        err |= inbounds(nx, ny, 0, 0, 0, h, 3, RECEIVER);
        err |= inbounds(nx, ny, 0, 1, 0, h, 3, RECEIVER);




}


int inbounds(int nx, int ny, int blocknum, int px, int py, float h, int degree, const enum source_type st) {

        int err = 0;
        float *x, *y;
        init(&x, &y, nx, ny, px, py, h);

        int overlap = 0;
        switch(st) {
            case MOMENT_TENSOR:
                overlap = 2;
                break;
            case FORCE: 
                overlap = 2;
                break;
            case RECEIVER:
                overlap = 0;
                break;
            case SGT:
                overlap = 0;
                break;
        }

        float qx, qy;

        int half_width = (degree + 1) / 2;


        // inbounds
        {
            qx = (nx - half_width - 1 - overlap) * h;
            qy = (ny - half_width - 1 - overlap) * h;
            printf("Query point: (%g, %g), \n", qx, qy);
            printf("x = %g %g %g, y = %g %g %g \n", x[0], x[1], x[2], y[0], y[1], y[2]);
            printf("Velocity bounds. In bounds if %g <= qx <= %g, %g <= qy <= %g \n", 
                    x[2 + ngsl], x[2 + ngsl + nx - 1],
                    y[2 + ngsl], y[2 + ngsl + ny - 1]);
            printf("Stress bounds. In bounds if %g <= qx <= %g, %g <= qy <= %g \n", 
                    x[2 + ngsl / 2], x[2 + 3 / 2 * ngsl + nx - 1],
                    y[2 + ngsl / 2], y[2 + 3 / 2 * ngsl + ny - 1]);
        }

        free(x);
        free(y);

        return err;
}


void init(float **x, float **y, int nx, int ny, int px, int py, float h) {
        int3_t gsize = {nx, ny, 1};
        int3_t shift = {0, 0, 0};
        int3_t coordinate = {px, py, 0};
        int3_t boundary1 = {0, 0, 0};
        int3_t boundary2 = {0, 0, 0};

        grid3_t grid = grid_init(gsize, shift, coordinate, boundary1, boundary2, ngsl + 2, h); 
        grid1_t xgrid = grid_grid1_x(grid); 
        grid1_t ygrid = grid_grid1_y(grid); 
        *x = malloc(sizeof(float) * xgrid.size);
        *y = malloc(sizeof(float) * ygrid.size);

        grid_fill1(*x, xgrid, 1);
        grid_fill_y_dm(*y, ygrid, 0);
}

