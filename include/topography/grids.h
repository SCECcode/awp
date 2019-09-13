#ifndef GRIDS_H
#define GRIDS_H

#include <grid/grid_3d.h>

typedef struct
{
        prec *x;
        prec *y;
        prec *z;
        int3_t length;
} grid_data_t;

typedef struct
{
        grid3_t x;
        grid3_t y;
        grid3_t z;

        grid3_t xx;
        grid3_t yy;
        grid3_t zz;
        grid3_t xy;
        grid3_t xz;
        grid3_t yz;
} grids_t;

grids_t grids_init(const int nx, const int ny, const int nz, const int coord_x,
                   const int coord_y, const int coord_z,
                   const int topography,
                   const prec gridspacing);

void grids_finalize(grids_t *grids);

void grid_data_init(grid_data_t *grid_data, const grid3_t grid);
void grid_data_free(grid_data_t *grid_data);



#endif


