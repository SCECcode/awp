#ifndef GRID_3D_H
#define GRID_3D_H
#ifdef __cplusplus
extern "C" {
#endif

#include <awp/definitions.h>

/* Type of grid points distribution at boundaries
 * Open boundary: Cell-centered grid points can extend outside the boundary.
 * Closed boundary: Cell-centered grid points align on the boundary.
 */ 
enum grid_boundary_types {OPEN_BOUNDARY = 0, CLOSED_BOUNDARY = 1};


/*
 * 1D grid data structure
 *
 * A grid is split up into three regions, left alignment, inner grid size, and
 * right alignment. The left and right alignment regions are determined by ghost
 * the number of ghost points in the x, y directions, or memory data alignment
 * in the z-direction. In AWP, these alignments are set to:
 * alignment.x :  2 + ngsl
 * alignment.y :  2 + ngsl
 * alignment.z :  align
 *
 * However, for say the stress grids, the alignments in the x, y directions are
 * equal to 
 * alignment.x :  2 + ngsl/2
 * alignment.y :  2 + ngsl/2
 * alignment.z :  align
 *
 * This change is controlled by the parameter `padding` that is passed to the
 * grid initialization function. The padding parameter increases the width of
 * the inner grid size region while decreasing the size of each alignment
 * region by the same amount.
 *
 * padding = 0 
 *
 * |o--o--o--o | o--o--o--o--o--o | o--o--o--o | 
 *
 *      gl             size             gl     
 *
 *      gl = ngsl + 2 
 *
 * padding = 2
 * |o--o | o--o--o--o--o--o--o--o--o--o | o--o | 
 *
 *   gl             size                   gl     
 *
 *   gl = ngsl
 *
 *
 */ 
typedef struct
{
        int id;
        int size;
        int shift;
        int alignment;
        int boundary1;
        int boundary2;
        int padding;
        // index to last grid point that physically exists in the grid
        int end;
        _prec gridspacing;

} grid1_t;

/* Grid 3D data structure
 * size: Grid size including padding in the x, y-directions
 * inner_size: Grid size excluding padding in the x, y-directions
 * outer_size: Same as size
 * mem: Size of allocated memory (takes padding in x,y-directions, alignment in
 *      z-direction, and extra x,y boundary padding into account)
 * coordinate: MPI coordinate 
 * offset1: Offset to first inner element 
 * offset2: Offset to last inner element + 1
 * boundary1: Type of boundary grid points, lower 
 * boundary2: Type of boundary grid points, upper
 * alignment: Helper data used to determine offsets
 * gridspacing: Grid spacing (same in all three directions)
 * num_bytes: Number of bytes to allocate for storing a field on the grid.
 * line: Helper data that specifies the line stride.
 * slice: Helper data that specifies the slice stride.
 * exclude_top_row: Helper data for determining if the top row of grid points
 *      should be excluded or not. Used for visualization purposes. FIXME:
 *      Consider removing this primitive since it is probably no longer needed.
 */ 
typedef struct
{
        int3_t size;
        int3_t inner_size;
        int3_t outer_size;
        int3_t mem;
        int3_t coordinate;
        int3_t shift;
        int3_t offset1;
        int3_t offset2;
        int3_t boundary1;
        int3_t boundary2;
        int3_t alignment;
        _prec gridspacing;
        int3_t padding;
        int num_bytes;
        int line;
        int slice;
        int exclude_top_row;
} fcn_grid_t;

typedef fcn_grid_t grid3_t;

#define grid_index(g,i,j,k)  (g).offset1.z + (k) +                        \
                            ((g).offset1.y + (j)) * (g).line +           \
                            ((g).offset1.x + (i)) * (g).slice           

#define grid_index2(g,i,j) ((g).offset1.y + j) + ((g).offset1.x + i) * (g).slice

grid3_t grid_init_velocity_grid(const int3_t size, const int3_t shift,
                         const int3_t coordinate, const int3_t boundary1,
                         const int3_t boundary2,
                         const _prec gridspacing);

grid3_t grid_init_stress_grid(const int3_t size, const int3_t shift,
                         const int3_t coordinate, const int3_t boundary1,
                         const int3_t boundary2,
                         const _prec gridspacing);

grid3_t grid_init_metric_grid(const int3_t size, const int3_t shift,
                         const int3_t coordinate, const int3_t boundary1,
                         const int3_t boundary2,
                         const _prec gridspacing);

grid3_t grid_init_full_grid(const int3_t size, const int3_t shift,
                         const int3_t coordinate, const int3_t boundary1,
                         const int3_t boundary2,
                         const _prec gridspacing);

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
                         const int3_t coordinate, const int3_t boundary1,
                         const int3_t boundary2, const int padding,
                         const _prec gridspacing);

fcn_grid_t fcn_init_grid(const int3_t size, const int3_t shift,
                         const int3_t coordinate, const int padding,
                         const _prec gridspacing);

/*
 * Obtain the boundary grid size that takes change in the number of grid points
 * for the different grids into account.
 */ 
int3_t grid_boundary_size(const grid3_t grid);

int3_t fcn_get_memory_size(const int3_t size);
int3_t fcn_get_alignment(void);
int fcn_get_line(const int3_t mem);
int fcn_get_slice(const int3_t mem);
int3_t fcn_get_offset1(const int3_t alignment, const int3_t size);
int3_t fcn_get_offset2(const int3_t alignment, const int3_t size);
void fcn_print_info(const fcn_grid_t grid);

// Return 1D grid data structure for each direction
grid1_t grid_grid1_x(const grid3_t grid);
grid1_t grid_grid1_y(const grid3_t grid);
grid1_t grid_grid1_z(const grid3_t grid);

/*
 * Fill the array `out` with the grid point values that are defined by a grid
 * in one dimension.
 *
 * Arguments:
 *      out: Array to fill
 *      n: Array size. Must be greater than the grid size.
 *      grid: 1D grid data structure.
 *
 * Return value:
 *      Number of elements written.
 */ 
int grid_fill1(prec *out, const grid1_t grid);

/*
 * Check if a query point is in bounds or not. The query point is in bounds if
 * `x(0) <= q < x(end)`.
 *
 * Arguments:
 *      x: Array of size grid.size that contains grid points
 *      q: Query point.
 *
 * Return value:
 *      Error code (SUCCESS, ERR_OUT_OF_BOUNDS_LOWER, ERR_OUT_OF_BOUNDS_UPPER)
 *
 */
int grid_in_bounds1(const _prec *x, const _prec q, const grid1_t grid);

int grid_in_bounds_ext1(const _prec *x, const _prec q, const grid1_t grid);

int grid_in_bounds_force(const _prec *x, const _prec q, const grid1_t grid);
int grid_in_bounds_receiver(const _prec *x, const _prec q, const grid1_t grid);
int grid_in_bounds_moment_tensor(const _prec *x, const _prec q, const grid1_t grid);

/*
 * Fill the array `out` with the grid point values in the x-direction of a grid
 * in three dimensions.
 *
 * Arguments:
 *      out: Array to fill (size: grid.size.x)
 *      grid: 3D grid data structure.
 *
 * Return value:
 *      Number of elements written.
 */ 
int grid_fill_x(_prec *out, const fcn_grid_t grid);

/*
 * Fill the 1D array `out` with the grid point values in the y-direction of a
 * grid in three dimensions.
 *
 * Arguments:
 *      out: Array to fill (size: grid.size.y)
 *      grid: 3D grid data structure.
 *
 * Return value:
 *      Number of elements written.
 */ 
int grid_fill_y(_prec *out, const fcn_grid_t grid);

/*
 * Fill the 1D array `out` with the grid point values in the z-direction of a
 * grid in three dimensions.
 *
 * Arguments:
 *      out: Array of size ( grid.size.y ) to assign values to (output).
 *      grid: 3D grid data structure.
 *
 * Return value:
 *      Number of elements written.
 */ 
int grid_fill_z(_prec *out, const fcn_grid_t grid);

/*
 * Fill a 3D grid with grid point values in the x-direction.
 *
 * Arguments:
 *      out: Array of size (grid.mem.x * grid.mem.y * grid.mem.z ) to assign
 *              values to (output). 
 *      x: Array of size grid.size.x that contains values in the x-direction.
 *      grid: 3D grid data structure.
 *
 * Return value:
 *      Number of elements written.
 */
int grid_fill3_x(_prec *out, const _prec *x, const grid3_t grid);

/*
 * Fill a 3D grid with grid point values in the y-direction.
 *
 * Arguments:
 *      out: Array of size (grid.mem.x * grid.mem.y * grid.mem.z ) to assign
 *              values to (output). 
 *      y: Array of size grid.size.y that contains values in the y-direction.
 *      grid: 3D grid data structure.
 *
 * Return value:
 *      Number of elements written.
 */
int grid_fill3_y(_prec *out, const _prec *y, const grid3_t grid);

/*
 * Fill a 3D grid with grid point values in the z-direction.
 *
 * Arguments:
 *      out: Array of size (grid.mem.x * grid.mem.y * grid.mem.z ) to assign
 *              values to (output). 
 *      y: Array of size grid.size.z that contains values in the z-direction.
 *      grid: 3D grid data structure.
 *
 * Return value:
 *      Number of elements written.
 */
int grid_fill3_z(_prec *out, const _prec *y, const grid3_t grid);

/*
 * Raise each value defined on a 3D grid to some power.
 *
 * Arguments:
 *      out: Array of size (grid.mem.x * grid.mem.y * grid.mem.z ) to assign
 *              values to (output). 
 *      p: Power.
 *      grid: 3D grid data structure.
 *
 * Return value:
 *      Number of elements written.
 */
int grid_pow3(_prec *out, const _prec p, const grid3_t grid);


/* Reduce a grid function to a single value.
 *
 */
double grid_reduce3(const _prec *in, const grid3_t grid);


#ifdef __cplusplus
}
#endif
#endif

