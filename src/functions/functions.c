#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <functions/functions.h>
#include <grid/grid_3d.h>

/*
 * Fill a grid with grid point values for a given axis.
 *
 * Example:
 * `fcn_fill_grid(grid, 0)` constructs the x-axis.
 */
void fcn_fill_grid(_prec *out, const fcn_grid_t grid, const int3_t shift,
                   const int axis) {
        int i1 = grid.offset1.x;
        int j1 = grid.offset1.y;
        int k1 = grid.offset1.z;
        int i2 = grid.offset2.x;
        int j2 = grid.offset2.y;
        int k2 = grid.offset2.z;

        _prec a[3];
        for (int i = 0; i < 3; ++i) {
                a[i] = 0.0;
        }
        a[axis] = 1;
        _prec h = grid.gridspacing;

        for (int i = i1; i < i2; ++i) {
        for (int j = j1; j < j2; ++j) {
        for (int k = k1; k < k2; ++k) {
                _prec zkp = 0.0; 
                if (k == k2 - 1 && shift.z == 1) {
                      zkp = k2 - 2;
                } 
                else if ( k == grid.offset1.z) {
                      zkp = k1;
                }
                else if ( k == k2 - 1 && shift.z == 0) {
                      zkp = k1;
                } 
                else {
                   zkp = k - 0.5*shift.z;
                }

                int pos = k + j*grid.line + i*grid.slice;
                out[pos] =
                    h * a[0] * (i - i1 + grid.coordinate.x * grid.inner_size.x -
                                0.5 * shift.x) +
                    h * a[1] * (j - j1 + grid.coordinate.y * grid.inner_size.y -
                                0.5 * shift.y) +
                    h * a[2] * (zkp - k1);
        }
        }
        }
}


void fcn_shift(_prec *out, _prec *in, const fcn_grid_t grid, const _prec shift)
{
        for (int i = 0; i < grid.size.x; ++i) {
        for (int j = 0; j < grid.size.y; ++j) {
        for (int k = 0; k < grid.size.z; ++k) {
                int pos = grid.offset1.z + k +
                          (grid.offset1.y + j) * grid.line +
                          (grid.offset1.x + i) * grid.slice;
                out[pos] = in[pos] + shift;
        }
        }
        }
}

void fcn_power(_prec *out, _prec *in, const fcn_grid_t grid,
               const _prec exponent) {
        for (int i = 0; i < grid.size.x; ++i) {
        for (int j = 0; j < grid.size.y; ++j) {
        for (int k = 0; k < grid.size.z; ++k) {
                int pos = grid.offset1.z + k +
                          (grid.offset1.y + j) * grid.line +
                          (grid.offset1.x + i) * grid.slice;
                out[pos] = pow(in[pos], exponent);
        }
        }
        }
}

void fcn_normalize(_prec *out, _prec *in, const fcn_grid_t grid)
{

        int i1 = grid.offset1.x;
        int j1 = grid.offset1.y;
        int k1 = grid.offset1.z;
        int i2 = grid.offset2.x;
        int j2 = grid.offset2.y;
        int k2 = grid.offset2.z - grid.exclude_top_row;
        int pos1 = k1 + j1 * grid.line + i1 * grid.slice;
        int pos2 = (k2 - 1) + (j2 - 1) * grid.line + (i2 - 1) * grid.slice;

        _prec normalization = 1.0/(in[pos2] - in[pos1]);
        for (int i = i1; i < i2; ++i) {
        for (int j = j1; j < j2; ++j) {
        for (int k = k1; k < k2; ++k) {
                int pos = k + j * grid.line + i * grid.slice;
                out[pos] = (in[pos] - in[pos1])*normalization;
        }
        }
        }
}


void fcn_difference(_prec *out, _prec *in1, _prec *in2, const fcn_grid_t grid)
{

        int i1 = grid.offset1.x;
        int j1 = grid.offset1.y;
        int k1 = grid.offset1.z;
        int i2 = grid.offset2.x;
        int j2 = grid.offset2.y;
        int k2 = grid.offset2.z - grid.exclude_top_row;

        for (int i = i1; i < i2; ++i) {
        for (int j = j1; j < j2; ++j) {
        for (int k = k1; k < k2; ++k) {
                int pos = k + j * grid.line + i * grid.slice;
                out[pos] = in1[pos] - in2[pos];
        }
        }
        }
}

void fcn_apply(_prec *out, fcn_gridp fcn, const _prec *x, const _prec *y,
               const _prec *z, const _prec *properties, const fcn_grid_t grid) {
        for (int i = 0; i < grid.size.x; ++i) {
        for (int j = 0; j < grid.size.y; ++j) {
        for (int k = 0; k < grid.size.z; ++k) {
                int pos = grid.offset1.z + k +
                          (grid.offset1.y + j) * grid.line +
                          (grid.offset1.x + i) * grid.slice;
                out[pos] = fcn(x[pos], y[pos], z[pos], properties); 
        }
        }
        }
}

void fcn_abs(_prec *out, _prec *in, const fcn_grid_t grid)
{

        int i1 = grid.offset1.x;
        int j1 = grid.offset1.y;
        int k1 = grid.offset1.z;
        int i2 = grid.offset2.x;
        int j2 = grid.offset2.y;
        int k2 = grid.offset2.z - grid.exclude_top_row;

        for (int i = i1; i < i2; ++i) {
        for (int j = j1; j < j2; ++j) {
        for (int k = k1; k < k2; ++k) {
                int pos = k + j * grid.line + i * grid.slice;
                out[pos] = fabs(in[pos]);
        }
        }
        }
}

void fcn_constant(_prec *out, 
                  const int i0, const int in, 
                  const int j0, const int jn, 
                  const int k0, const int kn, 
                  const int line, const int slice,
                  const _prec *args)
{
        for (int i = i0; i < in; ++i) {
        for (int j = j0; j < jn; ++j) {
        for (int k = k0; k < kn; ++k) {
                int pos = k + j*line + i*slice; 
                out[pos] = args[0];
        }
        }
        }

}
void fcn_poly(_prec *out, 
              const int i0, const int in, 
              const int j0, const int jn, 
              const int k0, const int kn, 
              const int line, const int slice,
              const _prec *args)
{
        const _prec a0 = args[0];
        const _prec a1 = args[1];
        const _prec a2 = args[2];
        const _prec p0 = args[3];
        const _prec p1 = args[4];
        const _prec p2 = args[5];
        const _prec s0 = args[6];
        const _prec s1 = args[7];
        const _prec s2 = args[8];
        const int   nx = (int)args[9];
        const int   ny = (int)args[10];
        const int   rx = (int)args[11];
        const int   ry = (int)args[12];
        for (int i = i0; i < in; ++i) {
        for (int j = j0; j < jn; ++j) {
        for (int k = k0; k < kn; ++k) {
                int pos = k + j*line + i*slice; 
                out[pos] =  a0*pow(i + rx*nx - 0.5*s0, p0)
                          + a1*pow(j + ry*ny - 0.5*s1, p1)
                          + a2*pow(k         - 0.5*s2, p2);
        }
        }
        }
}

//TODO: Deprecate this function. Use `fcn_polynomial` instead.
void fcn_polybndz(_prec *out, 
                  const int i0, const int in, 
                  const int j0, const int jn, 
                  const int k0, const int kn, 
                  const int line, const int slice,
                  const _prec *args)
{
        const _prec a0 = args[0];
        const _prec a1 = args[1];
        const _prec a2 = args[2];
        const _prec p0 = args[3];
        const _prec p1 = args[4];
        const _prec p2 = args[5];
        const _prec s0 = args[6];
        const _prec s1 = args[7];
        const _prec s2 = args[8];
        const int   nx = (int)args[9];
        const int   ny = (int)args[10];
        const int   rx = (int)args[11];
        const int   ry = (int)args[12];
        for (int i = i0; i < in; ++i) {
        for (int j = j0; j < jn; ++j) {
        for (int k = k0; k < kn; ++k) {
                _prec zkp = 0.0; 
                if (k == kn - 1 && s2 == 1) {
                      zkp = pow(kn - 2, p2);
                } 
                else if ( k == k0) {
                      zkp = pow(k0, p2);
                }
                else if ( k == kn - 1 && s2 == 0) {
                      zkp = 0;
                } 
                else {
                   zkp = pow(k - 0.5*s2, p2);
                }
                int pos = k + j*line + i*slice; 
                out[pos] =  a0*pow(i + rx*nx + 0.5*s0, p0)
                          + a1*pow(j + ry*ny + 0.5*s1, p1)
                          + a2*zkp;
        }
        }
        }
}

