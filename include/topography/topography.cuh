#ifndef _TOPOGRAPHY_H
#define _TOPOGRAPHY_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <topography/topography.h>
#include <awp/definitions.h>
/*
 * This is the header file for the topography GPU compute kernels.
 * To understand what these kernels do, it is important to have a good sense
 * of how the velocity and stress components are arranged with respect to each
 * other.
 *
 * Shifts
 *
 *  If the field has a `1` in any of its given components, then it is shifted by
 *  -h/2 for that direction. Shifted grids are called "hat" grids and are
 *  denoted by "xh", for a shift in the x-direction, and so forth. Note that the
 *  boundary point is included in the shifted grid. The free surface boundary is
 *  placed at the highest grid index in the z-direction.
 *  
 *        vx : (1, 1, 1)
 *        vy : (0, 0, 1)
 *        vz : (0, 1, 0)
 *        xx : (0, 1, 1)
 *        yy : (0, 1, 1)
 *        zz : (0, 1, 1)
 *        xy : (1, 0, 1)
 *        xz : (1, 1, 0)
 *        yz : (0, 0, 0)
 *  
 * Boundary kernel updates
 *  
 *  Below follows some examples for various directions. Pay particular attention
 *  to the z-direction because this is where the free surface boundary condition
 *  needs to be enforced.
 *  
 *  Velocity update
 *  
 *  dvx/dt = Dx*xx
 *  
 *        xx   i-2   i-1    i    i+1   i+2
 *                      |                 |
 *   x    o-----*-----*-|---*-----*-----o-|---o-----o
 *                      |                 |
 *                      |                 |
 *   xh   ---o-----o----|o-----o-----o----|o-----o---
 *                      |                 |
 *        vx             i    i+1   i+2
 *           
 *           Left           Interior           Right 
 *  
 *  Stress update
 *  
 *  dzz/dt = Dz*vz
 *  
 *        w1  k-2    k-1    k    k+1    k+2
 *                      |                 |
 *   z    o-----*-----*-|---*-----*-----o-|---o-----o
 *                      |                 |
 *                      |                 |
 *   zh   o--o-----o----|o-----o-----o----|o-----o--o
 *                      |                 |
 *        zz             k    k+1    k+2
 *  
 *           Left           Interior           Right 
 *  
 *  Velocity update
 *  
 *  dvx/dt = Dz*xz
 *  dvy/dt = Dz*yz
 *  
 *  
 *        xz, yz  k-2    k-1    k    k+1    k+2
 *                      |                 |
 *   z    ------*-----*-|---*-----*-----o-|---o-----o-----o
 *                      |                 |
 *                      |                 |
 *   zh   ---o-----o----|o-----o-----o----|o-----o-----o--o
 *                      |                 |
 *        vx, vy         k    k+1    k+2
 *  
 *           Bottom           Interior          Top
 *  
 *  Due to the different number of grid points in the two grids, it is difficult
 *  to compute on them both in the same compute kernel. To remedy the situation,
 *  we make a slight modification that introduces an additional grid point in
 *  the first grid. No computation will take place for this point.
 *  
 *  The value placed in the last position does not matter because it will not be
 *  used in practice. The presence of the extra point requires us to introduce
 *  an extra derivative (or what whatever the operator is) stencil associated
 *  with this point. Since we do not care about the solution here, we simply
 *  introduce a stencil that has all of its coefficients set to zero. 
 *  
 *  We then make the following partitioning of the two grids:
 *  
 *   z    o-----o-----o-|---o-----o--|---o----o-----o---*
 *                      |            |     
 *                      |            |     
 *   zh   o--o-----o----|o-----o-----|^----o-----o--o
 *                      |            |     
 *           Bottom           Interior           Top 
 *
 *  
 *  
 *      * : Extra point in which no computation is performed
 *      ^ : Interior point moved to boundary computation
 *  
 *  As we can see in the Figure, the same number of points in each of the three
 *  regions are updated for both grids. For example, in the right region, we
 *  update four points belonging to the `x` grid. Although, the last point is
 *  the extra point that is never used. In the same region, four points are also
 *  updated for the `xh` grid. Since the number of points is the same, the
 *  computation can take place in the same compute kernel, meaning that the it
 *  can take place within the same loop body.
 *  
 *  Perhaps it seems a little strange that on the left side there are only three
 *  points being updated per grid, whereas on the right side there are four
 *  points being updated per grid. The reason for this discrepancy has to do
 *  with the fact that the initial operator defined on the `x` grid has three
 *  boundary points per side. Therefore, we must pack four points into the right
 *  boundary region (accounting for the extra point). To ensure that the same
 *  number of points are updated for both grids, we simply move one of the
 *  interior points originally belonging to the interior of the `xh` grid into
 *  the right boundary region.
 *
 *  Maybe this problem with introducing extra points can be avoided by a grid
 *  that looks like this:
 *
 *   z    o-----o-----o-|---o-----o-----o-|---o-----o--o
 *                      |                 |
 *                      |                 |
 *   zh   o--o-----o----|o-----o-----o----|o-----o-----o
 *                      |                 |
 *           Bottom           Interior           Top 
 *
 */

#ifdef __cplusplus
extern "C" {
#endif
void topo_init_material_H(topo_t *T);
void topo_velocity_interior_H(topo_t *T);
void topo_velocity_front_H(topo_t *T);
void topo_velocity_back_H(topo_t *T);
void topo_stress_interior_H(topo_t *T);
void topo_stress_left_H(topo_t *T);
void topo_stress_right_H(topo_t *T);
#ifdef __cplusplus
}
#endif
#

// Number of threads per block to use
#ifndef TBX
#define TBX 1
#endif
#ifndef TBY
#define TBY 1
#endif
#ifndef TBZ
#define TBZ 16
#endif

// Number of boundary points. Must match the number of boundary stencils used in
// the interpolation and differentiation stencils
#define BOTTOM_BOUNDARY_SIZE 7
#define TOP_BOUNDARY_SIZE 7



#endif

