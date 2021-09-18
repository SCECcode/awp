#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>


#include <grid/grid_3d.h>
#include <interpolation/lagrange.h>
#include <interpolation/interpolation.h>
#include <awp/error.h>
#include <test/test.h>

int interp_argnearest(int *nearest, const prec *points, const int n, const prec
                      query)
{
        if (query < points[0]) {
                *nearest = 0;
                return ERR_OUT_OF_BOUNDS_LOWER;
        }
        
        if (query > points[n - 1]) {
                *nearest = n - 1;
                return ERR_OUT_OF_BOUNDS_UPPER;
        }

        int argmin = n + 1;
        _prec dist = INFINITY;
        for (int i = 0; i < n; ++i) {
                _prec current_dist = fabs(query - points[i]);
                if (current_dist  < dist) {
                        dist = current_dist;
                        argmin = i;
                }
        }
        *nearest = argmin;
        return SUCCESS;
}

int interp_grid_argnearest(int *nearest, const prec *x, const prec q, grid1_t
                grid)
{
        if (grid.boundary2 == CLOSED_BOUNDARY && grid.shift == 0) {
                return interp_argnearest(nearest, x, grid.size - 1,  q);
        } 
        return interp_argnearest(nearest, x, grid.size, q);
}

int interp_argnearest_range(int *first, int *last,
                            const int lower, const int upper,
                            const int nearest,
                            const int n)
{
        int err = 0;
        int inearest = nearest;
        int diffleft = inearest - lower;
        int diffright = n - inearest - upper - 1;
        int ileft = max(diffleft, 0) + min(diffright, 0);
        int iright = min(inearest + upper + 1, n) - min(diffleft, 0);
        *first = ileft;
        *last = iright;
        return err;
}

int interp_lagrange1(prec *out, const prec *x, const prec *in, const int n,
                     const prec *query, const int m, const int deg)
{
        int err = 0;
        prec *l, *xloc;
        l = calloc(sizeof(l), (deg + 1));
        xloc = calloc(sizeof(xloc), (deg + 1));

        for (int i = 0; i < m; ++i) { 
                int lower = 0;
                interp_lagrange1_coef(xloc, l, &lower, x, n, query[i], deg);
                out[i] = 0.0;
                for (int j = 0; j < deg + 1; ++j) {
                        out[i] += l[j]*in[lower + j];
                }

        }

        free(l);
        free(xloc);

        return err;

}

int interp_lagrange1_coef(prec *xloc, prec *l, int *first, const prec *x,
                          const int n, const prec query, const int deg)
{
        int err = 0;
        int nearest = 0;
        int lower = (int)ceil((double)deg * 0.5);
        int upper = (int)ceil((double)deg * 0.5);

        prec q = query;

        err = interp_argnearest(&nearest, x, n, query);

        if (err == ERR_OUT_OF_BOUNDS_LOWER) {
                q = x[0];
        }
        if (err == ERR_OUT_OF_BOUNDS_UPPER) {
                q = x[n-1];
        }

        lower = interp_get_lower(x[nearest], query, deg);
        upper = interp_get_upper(x[nearest], query, deg);

        err |=
            interp_argnearest_range(&lower, &upper, lower, upper, nearest, n);
        for (int j = 0; j < deg + 1; ++j) {
                xloc[j] = x[lower + j];
        }
        *first = lower;
        err |= lagrange_basis(l, deg + 1, xloc, q);
        return err;
}

int interp_lagrange3(prec *out, const prec *in, const prec *x, const prec *y,
                     const prec *z, const grid3_t grid, const prec *qx,
                     const prec *qy, const prec *qz, const int m,
                     const int deg) {
        int err = 0;
        prec *lx, *ly, *lz, *xloc, *yloc, *zloc;
        lx = calloc(sizeof(lx), (deg + 1));
        ly = calloc(sizeof(ly), (deg + 1));
        lz = calloc(sizeof(lz), (deg + 1));
        xloc = calloc(sizeof(xloc), (deg + 1));
        yloc = calloc(sizeof(yloc), (deg + 1));
        zloc = calloc(sizeof(zloc), (deg + 1));

        for (int q = 0; q < m; ++q) { 
                int ix = 0; int iy = 0; int iz = 0;

                err = interp_lagrange1_coef(
                    xloc, lx, &ix, x, grid_boundary_size(grid).x, qx[q], deg);
                err = interp_lagrange1_coef(
                    yloc, ly, &iy, y, grid_boundary_size(grid).y, qy[q], deg);
                err = interp_lagrange1_coef(
                    zloc, lz, &iz, z, grid_boundary_size(grid).z, qz[q], deg);
                out[q] = 0.0;
                for (int i = 0; i < deg + 1; ++i) {
                for (int j = 0; j < deg + 1; ++j) {
                for (int k = 0; k < deg + 1; ++k) {
                        int pos = grid_index(grid, ix + i, iy + j, iz + k);
                        out[q] += lx[i] * ly[j] * lz[k] * in[pos];
                }
                }
                }
        }

        free(lx);
        free(ly);
        free(lz);
        free(xloc);
        free(yloc);
        free(zloc);

        return err;
}

int interp_get_lower(const prec xnearest, const prec query, const int deg) {
        int lower = (int)ceil((double)deg * 0.5);
        if (deg % 2 == 1) {
                if (xnearest - query > 0) {
                        lower = (int)ceil((double)deg * 0.5);
                } else {
                        lower = (int)floor((double)deg * 0.5);
                }
        }
        return lower;
}

int interp_get_upper(const prec xnearest, const prec query, const int deg) {
        int upper = (int)ceil((double)deg * 0.5);
        if (deg % 2 == 1) {
                if (xnearest - query > 0) {
                        upper = (int)floor((double)deg * 0.5);
                } else {
                        upper = (int)ceil((double)deg * 0.5);
                }


        }
        return upper;
}


