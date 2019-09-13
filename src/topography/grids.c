#include <stdio.h>
#include <stdlib.h>

#include <awp/definitions.h>
#include <topography/grids.h>
#include <test/test.h>
#include <grid/grid_3d.h>
#include <grid/shift.h>

grids_t grids_init(const int nx, const int ny, const int nz, const int coord_x,
                   const int coord_y, const int coord_z,
                   const int topography, 
                   const prec gridspacing) 
{
        int3_t size = {.x = nx, .y = ny, .z = nz};
        int3_t coord = {.x = coord_x, .y = coord_y, .z = 0};

        //FIXME: Adjust depending on grid type: DM, topography, free surface.
        int3_t bnd1 = {0, 0, 0};
        int3_t bnd2 = {0, 0, topography};

        grids_t grids;

        prec h = gridspacing;

        // velocity grids
        grids.x = grid_init(size, grid_x(), coord, bnd1, bnd2, 0, h);
        grids.y = grid_init(size, grid_y(), coord, bnd1, bnd2, 0, h);
        grids.z = grid_init(size, grid_z(), coord, bnd1, bnd2, 0, h);

        // stress grids
        grids.xx = grid_init(size, grid_xx(), coord, bnd1, bnd2, ngsl / 2, h);
        grids.yy = grid_init(size, grid_yy(), coord, bnd1, bnd2, ngsl / 2, h);
        grids.zz = grid_init(size, grid_zz(), coord, bnd1, bnd2, ngsl / 2, h);
        grids.xy = grid_init(size, grid_xy(), coord, bnd1, bnd2, ngsl / 2, h);
        grids.xz = grid_init(size, grid_xz(), coord, bnd1, bnd2, ngsl / 2, h);
        grids.yz = grid_init(size, grid_yz(), coord, bnd1, bnd2, ngsl / 2, h);

        return grids;
}

void grids_finalize(grids_t *grids)
{
}

void grid_data_init(grid_data_t *grid_data, const grid3_t grid)
{
        grid1_t xgrid = grid_grid1_x(grid);
        grid1_t ygrid = grid_grid1_y(grid);
        grid1_t zgrid = grid_grid1_z(grid);
        grid_data->x = malloc(sizeof grid_data->x * xgrid.size);
        grid_data->y = malloc(sizeof grid_data->y * ygrid.size);
        grid_data->z = malloc(sizeof grid_data->z * zgrid.size);
        grid_fill1(grid_data->x, xgrid);
        grid_fill1(grid_data->y, ygrid);
        grid_fill1(grid_data->z, zgrid);
}

void grid_data_free(grid_data_t *grid_data)
{
        free(grid_data->x);
        free(grid_data->y);
        free(grid_data->z);
}

