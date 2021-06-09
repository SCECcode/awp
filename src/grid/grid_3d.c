#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <test/test.h>
#include <grid/grid_3d.h>
#include <awp/error.h>

/* Initialize grid
 * 
 * Input arguments:
 *      size : Size of velocity grid (nxt, nyt, nzt)
 *      shift : Grid that the grid function belongs to (see shift.h)
 *      coordinate : MPI coordinate (x, y, 0)
 *      boundary1 : Define boundary grid points, lower (see boundary_types in
 *             grid_3d.h)
 *      boundary2 : Define boundary grid points, upper (see boundary_types in
 *             grid_3d.h)
 *      padding : Adjust grid size and alignments depending on what type of
 *              field is stored on the grid.
 *       padding = 0 (velocity grid)
 *       padding = ngsl/2 (stress grid)
 *       padding = ngsl (topography grid)
 *
 */
grid3_t grid_init(const int3_t size, const int3_t shift,
                         const int3_t coordinate, 
                         const int3_t boundary1, const int3_t boundary2,
                         const int padding,
                         const _prec gridspacing) {
        int3_t out_size = size;
        fcn_grid_t out;
        out.size = out_size;
        out.inner_size = out_size;
        out.mem = fcn_get_memory_size(size);
        out.num_bytes = sizeof(_prec)*(out.mem.x*out.mem.y*out.mem.z);
        out.alignment = fcn_get_alignment();
        out.offset1 = fcn_get_offset1(out.alignment, size);
        out.offset2 = fcn_get_offset2(out.alignment, size);
        out.line = fcn_get_line(out.mem);
        out.slice = fcn_get_slice(out.mem);
        out.boundary1 = boundary1;
        out.boundary2 = boundary2;
        out.gridspacing = gridspacing;
        out.coordinate = coordinate;
        out.shift = shift;
        out.padding.x = padding;
        out.padding.y = padding;
        out.padding.z = 0;

        out.size.x += padding*2;
        out.size.y += padding*2;
        out.alignment.x -= padding;
        out.alignment.y -= padding;
        out.offset1.x -= padding;
        out.offset2.x += padding;
        out.offset1.y -= padding;
        out.offset2.y += padding;

        // Alignment must be non-negative. Padding size is too large.
        assert(out.alignment.x >= 0);
        assert(out.alignment.y >= 0);

        // Remove top row of grid points for regular grids in z-direction
        if (shift.z == 0) {
                out.exclude_top_row = 1;
        }

        assert(out.offset1.x >= 0);
        assert(out.offset1.y >= 0);
        assert(out.offset2.x <= out.mem.x);
        assert(out.offset2.y <= out.mem.y);
        return out;
}

grid3_t grid_init_velocity_grid(const int3_t size, const int3_t shift,
                         const int3_t coordinate, const int3_t boundary1,
                         const int3_t boundary2,
                         const _prec gridspacing)
{
        return grid_init(size, shift, coordinate, boundary1, boundary2, 0,
                         gridspacing);
}

grid3_t grid_init_stress_grid(const int3_t size, const int3_t shift,
                         const int3_t coordinate, const int3_t boundary1,
                         const int3_t boundary2,
                         const _prec gridspacing)
{
        return grid_init(size, shift, coordinate, boundary1, boundary2,
                         ngsl / 2, gridspacing);
}

grid3_t grid_init_metric_grid(const int3_t size, const int3_t shift,
                         const int3_t coordinate, const int3_t boundary1,
                         const int3_t boundary2,
                         const _prec gridspacing)
{
        return grid_init(size, shift, coordinate, boundary1, boundary2, ngsl,
                         gridspacing);
}

grid3_t grid_init_full_grid(const int3_t size, const int3_t shift,
                         const int3_t coordinate, const int3_t boundary1,
                         const int3_t boundary2,
                         const _prec gridspacing)
{
        return grid_init(size, shift, coordinate, boundary1, boundary2, ngsl + 2,
                         gridspacing);
}

//FIXME: remove this function. It should be replaced by "grid_init"
fcn_grid_t fcn_init_grid(const int3_t size, const int3_t shift,
                         const int3_t coordinate, const int padding,
                         const _prec gridspacing) {
        int3_t out_size = size;
        fcn_grid_t out;
        out.size = out_size;
        out.mem = fcn_get_memory_size(size);
        out.num_bytes = sizeof(_prec)*(out.mem.x*out.mem.y*out.mem.z);
        out.alignment = fcn_get_alignment();
        out.offset1 = fcn_get_offset1(out.alignment, size);
        out.offset2 = fcn_get_offset2(out.alignment, size);
        out.line = fcn_get_line(out.mem);
        out.slice = fcn_get_slice(out.mem);
        out.gridspacing = gridspacing;
        out.coordinate = coordinate;
        out.inner_size = size;

        out.size.x += padding*2;
        out.size.y += padding*2;
        out.alignment.x -= padding;
        out.alignment.y -= padding;
        out.offset1.x -= padding;
        out.offset2.x += padding;
        out.offset1.y -= padding;
        out.offset2.y += padding;

        // Remove top row of grid points for regular grids in z-direction
        if (shift.z == 0) {
                out.exclude_top_row = 1;
        }

        assert(out.offset1.x >= 0);
        assert(out.offset1.y >= 0);
        assert(out.offset2.x <= out.mem.x);
        assert(out.offset2.y <= out.mem.y);
        return out;
}

int3_t grid_boundary_size(const grid3_t grid)
{
        int3_t out = {.x = grid.size.x,
                      .y = grid.size.y,
                      .z = grid.size.z};

        if (grid.shift.x == 0 && grid.boundary2.x == CLOSED_BOUNDARY) {
                out.x -= 1; 
        }
        if (grid.shift.y == 0 && grid.boundary2.y == CLOSED_BOUNDARY) {
                out.y -= 1; 
        }
        if (grid.shift.z == 0 && grid.boundary2.z == CLOSED_BOUNDARY) {
                out.z -= 1; 
        }

        return out;
}

int3_t fcn_get_memory_size(const int3_t size)
{
        int3_t mem;
        mem.x = size.x + 4 + ngsl2;
        mem.y = size.y + 4 + ngsl2;
        mem.z = size.z + 2*align;
        return mem;
}

int3_t fcn_get_alignment(void)
{
        int3_t out;
        out.x = 2 + ngsl;
        out.y = 2 + ngsl;
        out.z = align;
        return out;
}

int fcn_get_line(const int3_t mem)
{
        return mem.z;
}

int fcn_get_slice(const int3_t mem)
{
        return mem.z*mem.y;
}

int3_t fcn_get_offset1(const int3_t alignment, const int3_t size)
{
        int3_t offset;
        offset.x = alignment.x;
        offset.y = alignment.y;
        offset.z = alignment.z;
        return offset;
}

int3_t fcn_get_offset2(const int3_t alignment, const int3_t size)
{
        int3_t offset;
        offset.x = alignment.x + size.x;
        offset.y = alignment.y + size.y;
        offset.z = alignment.z + size.z;
        return offset;
}

void fcn_print_info(const fcn_grid_t grid)
{
        printf("size = {%d, %d, %d} \n", grid.size.x, grid.size.y, grid.size.z);
        printf("size = {%d, %d, %d} \n", grid.size.x, grid.size.y, grid.size.z);
        printf("mem = {%d, %d, %d} \n", grid.mem.x, grid.mem.y, grid.mem.z);
        printf("coordinate = {%d, %d, %d} \n", grid.coordinate.x, grid.coordinate.y, grid.coordinate.z);
        printf("offset1 = {%d, %d, %d} \n", grid.offset1.x, grid.offset1.y, grid.offset1.z);
        printf("offset2 = {%d, %d, %d} \n", grid.offset2.x, grid.offset2.y, grid.offset2.z);
        printf("alignment = {%d, %d, %d} \n", grid.alignment.x, grid.alignment.y, grid.alignment.z);
        printf("line = %d, slice = %d \n", grid.line, grid.slice);
        printf("numbytes = %d \n", grid.num_bytes);
        printf("exclude_top_row = %d \n", grid.exclude_top_row);
}

grid1_t grid_grid1_x(const grid3_t grid)
{
        grid1_t grid1 = {.id = grid.coordinate.x, .size = grid.size.x,
                         .shift = grid.shift.x,
                         .alignment = grid.alignment.x,
                         .boundary1 = grid.boundary1.x,
                         .boundary2 = grid.boundary2.x,
                         .padding = grid.padding.x,
                         .gridspacing = grid.gridspacing};
        return grid1;
}
grid1_t grid_grid1_y(const grid3_t grid)
{
        grid1_t grid1 = {.id = grid.coordinate.y, .size = grid.size.y,
                         .shift = grid.shift.y,
                         .alignment = grid.alignment.y,
                         .boundary1 = grid.boundary1.y,
                         .boundary2 = grid.boundary2.y,
                         .padding = grid.padding.y,
                         .gridspacing = grid.gridspacing};
        return grid1;
}

grid1_t grid_grid1_z(const grid3_t grid)
{
        grid1_t grid1 = {.id = grid.coordinate.z, .size = grid.size.z,
                         .shift = grid.shift.z,
                         .alignment = grid.alignment.z,
                         .boundary1 = grid.boundary1.z,
                         .boundary2 = grid.boundary2.z,
                         .padding = grid.padding.z,
                         .end = grid.size.z - 1 - grid.boundary2.z,
                         .gridspacing = grid.gridspacing};
        return grid1;
}

int grid_fill1(prec *out, const grid1_t grid, const int isxdir)
{
        _prec h = grid.gridspacing;
        for (int i = 0; i < grid.size; ++i) {
                out[i] = h * (i + grid.id * (grid.size - 2 * grid.padding) - 0.5 * grid.shift + isxdir * grid.shift - grid.padding);
        }

        if (grid.shift && grid.boundary1) {
                out[0] = h * grid.id * grid.size;
        }

        if (grid.shift && grid.boundary2) {
                out[grid.size - 1] =
                    h * ((grid.id + 1) * (grid.size - 2 * grid.padding) - 2 - grid.padding);
        }

        if (!grid.shift && grid.boundary2) {
                out[grid.size - 1] = 0.0;
        }
        return grid.size;
}


int grid_in_bounds1(const _prec *x, const _prec q, const grid1_t grid)
{
        size_t idx = 0;

        if (q - x[idx] < -FLTOL ) {
                return ERR_OUT_OF_BOUNDS_LOWER;
        }
        if (q - x[grid.size - 1 - idx] > FLTOL &&
            grid.boundary2 != CLOSED_BOUNDARY) {
                return ERR_OUT_OF_BOUNDS_UPPER;
        }
        // Grid contains a boundary ghost point, check second last point in
        // grid point array.
        if (q > x[grid.size - 2 - idx] && grid.shift == 0 &&
            grid.boundary2 == CLOSED_BOUNDARY) {
                return ERR_OUT_OF_BOUNDS_UPPER;
        }
        return SUCCESS;
}

int grid_in_bounds_receiver(const _prec *x, const _prec q, const grid1_t grid)
{
        _prec h = grid.gridspacing;
        if ( q - (x[0] - h / 2) < 0 ) {
                return ERR_OUT_OF_BOUNDS_LOWER;
        }
        if ( q - (x[grid.size - 1] + h / 2) >= 0) {
                return ERR_OUT_OF_BOUNDS_UPPER;
        }
        return SUCCESS;

}

int grid_in_bounds_sgt(const _prec *x, const _prec q, const grid1_t grid)
{
        _prec h = grid.gridspacing;
        if ( q - (x[0] - h / 2 + h * ngsl / 2) < 0 ) {
                return ERR_OUT_OF_BOUNDS_LOWER;
        }
        if ( q - (x[grid.size - 1] + h / 2 - h * ngsl / 2) >= 0) {
                return ERR_OUT_OF_BOUNDS_UPPER;
        }

        return SUCCESS;

}

int grid_in_bounds_force(const _prec *x, const _prec q, const grid1_t grid)
{
        _prec h = grid.gridspacing;
        if ( q - (x[0] - 2 * h) < 0 ) {
                return ERR_OUT_OF_BOUNDS_LOWER;
        }
        if ( q - (x[grid.size - 1] + 2 * h) >= 0) {
                return ERR_OUT_OF_BOUNDS_UPPER;
        }
        return SUCCESS;
}

int grid_in_bounds_moment_tensor(const _prec *x, const _prec q, const grid1_t grid)
{
        _prec h = grid.gridspacing;
        if ( q - (x[0] - ngsl / 2 * h) < 0 ) {
                return ERR_OUT_OF_BOUNDS_LOWER;
        }
        if ( q - (x[grid.size - 1] + ngsl / 2 * h) >= 0) {
                return ERR_OUT_OF_BOUNDS_UPPER;
        }
        return SUCCESS;
}

int grid_fill_x(prec *out, const fcn_grid_t grid)
{
        grid1_t grid1 = grid_grid1_x(grid);
        return grid_fill1(out, grid1, 1);
}

int grid_fill_y(prec *out, const fcn_grid_t grid)
{
        grid1_t grid1 = grid_grid1_y(grid);
        return grid_fill1(out, grid1, 0);
}

int grid_fill_z(prec *out, const fcn_grid_t grid)
{
        grid1_t grid1 = grid_grid1_z(grid);
        return grid_fill1(out, grid1, 0);
}

int grid_fill3_x(_prec *out, const _prec *x, const grid3_t grid)
{
        for (int i = 0; i < grid.size.x; ++i) {
        for (int j = 0; j < grid.size.y; ++j) {
        for (int k = 0; k < grid.size.z; ++k) {
                int pos = grid_index(grid, i, j, k);
                out[pos] = x[i]; 
        }
        }
        }
        return grid.size.x * grid.size.y * grid.size.z;
}

int grid_fill3_y(_prec *out, const _prec *y, const grid3_t grid)
{
        for (int i = 0; i < grid.size.x; ++i) {
        for (int j = 0; j < grid.size.y; ++j) {
        for (int k = 0; k < grid.size.z; ++k) {
                int pos = grid_index(grid, i, j, k);
                out[pos] = y[j]; 
        }
        }
        }
        return grid.size.x * grid.size.y * grid.size.z;
}

int grid_fill3_z(_prec *out, const _prec *z, const grid3_t grid)
{
        for (int i = 0; i < grid.size.x; ++i) {
        for (int j = 0; j < grid.size.y; ++j) {
        for (int k = 0; k < grid.size.z; ++k) {
                int pos = grid_index(grid, i, j, k);
                out[pos] = z[k]; 
        }
        }
        }
        return grid.size.x * grid.size.y * grid.size.z;
}

int grid_pow3(_prec *out, const _prec p, const grid3_t grid)
{
        for (int i = 0; i < grid.size.x; ++i) {
        for (int j = 0; j < grid.size.y; ++j) {
        for (int k = 0; k < grid.size.z; ++k) {
                int pos = grid_index(grid, i, j, k);
                out[pos] = pow(out[pos], p); 
        }
        }
        }
        return grid.size.x * grid.size.y * grid.size.z;
}

double grid_reduce3(const _prec *in, const grid3_t grid)
{
        double out = 0.0;
        for (int i = 0; i < grid.size.x; ++i) {
        for (int j = 0; j < grid.size.y; ++j) {
        for (int k = 0; k < grid.size.z; ++k) {
                int pos = grid_index(grid, i, j, k);
                out += in[pos];
        }
        }
        }
        return out;
}

_prec grid_overlap(const _prec h) {
    return 7.0 * h;
}
_prec grid_height(const int nz, const _prec h, const int istopo) {
    return istopo == 1 ? (nz - 2) * h : (nz - 1) * h;
}
void global_to_local(_prec *zloc, int *block_index, const _prec z,
                     const _prec h, const int *nz, const int num_grids,
                     const int istopo) {
    _prec z0 = z;
    _prec bi = -1;

    _prec hloc = h;
    _prec H = 0.0;
    // Go from top grid to bottom grid
    for (int i = 0; i < num_grids; ++i ) {

        if (i > 0) 
            z0 -= grid_overlap(hloc / 3);

        // Check minimum number of grid points per block
        assert(nz[i] >= 7);

        _prec overlap = grid_overlap(hloc);
        
        H = i == 0 ? grid_height(nz[i], hloc, istopo) : grid_height(nz[i], hloc, 0);

        z0 += H;
        hloc *= 3;
        bi = i;

        //printf("z0 + H = %g i = %d \n", z0, i);

        // Check if the coordinate is in the overlap zone, if so, push it to the next grid
        if (z0 > 0 && z0 < grid_overlap(hloc / 3) ) {
            //printf("in overlap zone, z0 = %g i = %d overlap = %g \n", z0, i, overlap);
            continue;
        }

        if (z0 > 0) break;
        
        //printf("next,  z0 = %g i = %d \n", z0, i);

    }

    // Check if the mapping succeeded or not
    if (z0 < 0) {
        printf("WARNING: Failed to map z=%g to a block.\n", z);
    }

    *zloc = z0;
    *block_index = bi;
}
