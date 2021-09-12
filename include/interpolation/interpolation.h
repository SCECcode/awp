#ifndef INTERPOLATION_H
#define INTERPOLATION_H
#ifdef __cplusplus
extern "C" {
#endif

#include <awp/pmcl3d_cons.h>
#include <grid/grid_3d.h>

/* 
 * Find the index of the point in a grid nearest to a query point.
 *
 *  Input arguments:
 *      nearest: Index of nearest grid point (output)
 *      grid: Array of grid points 
 *      n : Number of grid points
 *      q : Query point
 *
 *  Return value:
 *      Error code (SUCCESS, ERR_OUT_OF_BOUNDS_LOWER, ERR_OUT_OF_BOUNDS_UPPER)
 *
 */ 
int interp_argnearest(int *nearest, const prec *grid, const int n, const prec
                      query);

/*
 * Find the index of the point in a grid nearest to a query point (wraps
 * interp_argnearest).
 *
 *  Input arguments:
 *      nearest: Index of nearest grid point (output)
 *      x: Array of grid points 
 *      q: Query point
 *      grid: Grid data structure.
 *
 *  Return value:
 *      Error code (SUCCESS, ERR_OUT_OF_BOUNDS_LOWER, ERR_OUT_OF_BOUNDS_UPPER)
 */
int interp_grid_argnearest(int *nearest, const prec *x, const prec q,
                           grid1_t grid);

/*
 *   Find the range of indices nearest the query point qp. 
 *
 *   --------[---*----)---------
 *
 *   * index of grid point nearest to query point
 *   [ left index (`left` steps away from qp) 
 *   ) right index (`right` steps away + 1 from qp) 
 *
 *   If the stencil hits the boundary, then the stencil length is preserved.
 *
 *   Input arguments:
 *       first: First index of grid points in stencil (output)
 *       last: and last index + 1 of grid points in stencil. (output)
 *       lower: Number of points to the left.
 *       upper: Number of points to the right.
 *       nearest: Index of grid point nearest to query point. 
 *       n: Number of grid points.
 *
 *   Returns:
 *       Error code (SUCCESS, ERR_OUT_OF_BOUNDS_LOWER, ERR_OUT_OF_BOUNDS_UPPER)
 *
 */
int interp_argnearest_range(int *first, int *last,
                            const int lower, const int upper,
                            const int nearest,
                            const int n);

/* Perform 1D Lagrange interpolation by interpolating in the neighborhood of
 * each query point. 
 *
 * Arguments:
 *      out : Array of interpolated values, one per query point (output)
 *      x : Grid points
 *      in : Values of function to interpolate at each grid point
 *      n : Number of grid points
 *      query : Array of query points
 *      deg : Degree of interpolating polynomial
 *
 * Returns:
 *       Error code (SUCCESS, ERR_OUT_OF_BOUNDS_LOWER, ERR_OUT_OF_BOUNDS_UPPER)
 */ 
int interp_lagrange1(prec *out, const prec *x, const prec *in, const int n,
                     const prec *query, const int m, const int deg);

/* Obtain lagrange interpolation coefficients and grid points for building an
 * interpolant in the neighborhood of a query point.
 *
 * Arguments:
 *      xloc: Array of size deg + 1 used for assigning grid points in the
 *              neighborhood of the query point (output).
 *      l: Array of size deg + 1 used for evaluating lagrange basis functions at
 *              the query point. (output). The first coefficient l[0] is the
 *              Lagrange basis function belonging to grid point xloc[0] and
 *              evaluated at the query point.
 *      first: Index that gives x[first] = xloc[0], i.e, the index of the grid
 *      point that references the first grid point in the local neighborhood
 *              surrounding the query point (output).
 *          x: Array of size n that contains the grid points.
 *          q: Query point that should satsify x[0] <= q <= x[n-1].
 *
 * Returns:
 *       Error code (SUCCESS, ERR_OUT_OF_BOUNDS_LOWER, ERR_OUT_OF_BOUNDS_UPPER)
 */       
int interp_lagrange1_coef(prec *xloc, prec *l, int *first, const prec *x,
                          const int n, const prec query, const int deg);


int interp_lagrange3(prec *out, const prec *in, const prec *x, const prec *y,
                     const prec *z, const grid3_t grid, const prec *qx,
                     const prec *qy, const prec *qz, const int m,
                     const int deg);


#ifdef __cplusplus
}
#endif
#endif

