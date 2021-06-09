#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <awp/definitions.h>
#include <awp/error.h>
#include <grid/shift.h>
#include <grid/grid_3d.h>
#include <interpolation/interpolation.h>
#include <test/test.h>
#include <test/array.h>

int test_argnearest(void);
int test_argnearest_range(void);
int test_lagrange1(int p);
int test_lagrange1_coef(int p);
int test_lagrange3(void);

int main(int argc, char **argv)
{
        int err = 0;
        test_divider();
        printf("Testing interpolation.c\n");
        err |= test_argnearest();
        err |= test_argnearest_range();
        err |= test_lagrange1(0);
        err |= test_lagrange1(1);
        err |= test_lagrange1(2);
        err |= test_lagrange1(3);
        err |= test_lagrange1_coef(0);
        err |= test_lagrange1_coef(1);
        err |= test_lagrange1_coef(2);
        err |= test_lagrange1_coef(3);
        err |= test_lagrange3();
        printf("Testing completed.\n");
        test_divider();

        return test_last_error();
}

int test_argnearest(void)
{
        int err = 0;
        int n = 4;
        _prec *x;
        x = malloc(sizeof(x) * n);

        for (int i = 0; i < n; ++i) {
                x[i] = i;
        }
        
        {
        test_t test = test_init(" * argnearest:interior", 0, 0);
        int nearest = -1;
        err |= interp_argnearest(&nearest, x, n, 2.1);
        err |= s_assert(nearest == 2);
        err |= test_finalize(&test, err);
        }
        
        {
        int nearest = -1;
        test_t test = test_init(" * argnearest:boundary", 0, 0);
        err |= interp_argnearest(&nearest, x, n, 0.5);
        err |= s_assert(!err);
        err |= s_assert(nearest == 0);
        err |= test_finalize(&test, err);
        }

        {
        int nearest = -1;
        test_t test = test_init(" * argnearest:bounds check", 0, 0);
        err |= s_no_except(interp_argnearest(&nearest, x, n, -1) == 
                           ERR_OUT_OF_BOUNDS_LOWER);
        err |= s_assert(nearest == 0);
        err |= s_no_except(interp_argnearest(&nearest, x, n, n + 1) == 
                           ERR_OUT_OF_BOUNDS_UPPER);

        err |= test_finalize(&test, err);
        }


        {
        // Grid with closed right boundary 
        grid1_t grid = {.id = 0, .shift = 0, .size = n, .gridspacing = 1, 
                        .boundary1 = 0, .boundary2 = 1};
        grid_fill1(x, grid, 1);
        test_t test = test_init(" * grid_argnearest:bounds_except", 0, 0);
        int nearest = -1;
        err |= s_no_except(interp_grid_argnearest(&nearest, x, -1, grid) == 
                           ERR_OUT_OF_BOUNDS_LOWER);
        err |= s_assert(nearest == 0);

        // Last grid point x[n-1] is a ghost point and therefore n - 1 is out of
        // bounds
        err |= s_no_except(interp_grid_argnearest(&nearest, x, n - 1, grid) == 
                           ERR_OUT_OF_BOUNDS_UPPER);
        err |= s_assert(nearest == 2);
        err |= test_finalize(&test, err);

        // In bounds
        err |= s_no_except(interp_grid_argnearest(&nearest, x, n - 2, grid) == 
                           SUCCESS);
        err |= s_assert(nearest == 2);
        }

        free(x);

        return err;
}

int test_argnearest_range(void)
{
        int err = 0;
        int n = 10;
        _prec *x;
        _prec xs;
        x = malloc(sizeof(x) * n);

        for (int i = 0; i < n; ++i) {
                x[i] = i;
        }

        {
        int nearest = -1, lower = 0, upper = 0;
        test_t test = test_init(" * argnearest_range:interior", 0, 0);

        // Query point in the interior of the domain, away from boundaries, no
        // left, right bounds
        xs = 4.3;

        err |= interp_argnearest(&nearest, x, n, xs);
        err |= interp_argnearest_range(&lower, &upper, lower, upper, nearest, n,
                                       xs);
        err |= s_assert(nearest == 4);
        err |= s_assert(lower == 4);
        err |= s_assert(upper == 5);

        // Query point in the interior of the domain, away from boundaries, 
        // three point wide stencil
        lower = 1;
        upper = 1;
        err |= interp_argnearest(&nearest, x, n, xs);
        err |= interp_argnearest_range(&lower, &upper, lower, upper, nearest, n,
                                       xs);

        err |= test_finalize(&test, err);
        err |= s_assert(nearest == 4);
        err |= s_assert(lower == 3);
        err |= s_assert(upper == 6);
        }

        {
        err = 0;
        test_t test = test_init(" * argnearest_range:left", 0, 0);

        // Query point close to the left boundary, 5 point wide stencil
        int nearest = -1, lower = 2, upper = 2;
        xs = 0.1;
        err |= interp_argnearest(&nearest, x, n, xs);
        err |= interp_argnearest_range(&lower, &upper, lower, upper, nearest, n,
                                       xs);
        err |= s_assert(nearest == 0);
        err |= s_assert(lower == 0);
        err |= s_assert(upper == 5);

        err |= test_finalize(&test, err);
        }

        {
        err = 0;
        test_t test = test_init(" * argnearest_range:right", 0, 0);

        // Query point close to the right boundary, 5 point wide stencil
        int nearest = -1, lower = 2, upper = 2;
        xs = 8.9;
        err |= interp_argnearest(&nearest, x, n, xs);
        err |= interp_argnearest_range(&lower, &upper, lower, upper, nearest, n,
                                       xs);
        err |= s_assert(nearest == 9);
        err |= s_assert(lower == 5);
        err |= s_assert(upper == 10);

        err |= test_finalize(&test, err);
        }

        free(x);
        return err;
}

int test_lagrange1(int p)
{
        int err = 0;
        int n = 10;
        int m = p + 1;
        _prec *x;
        _prec *val;
        _prec *queries;
        _prec *fcn;
        x = malloc(sizeof(x) * n);
        fcn = malloc(sizeof(x) * n);
        val = malloc(sizeof(val) * m);
        queries = malloc(sizeof(queries) * 4);
        queries[0] = 0.25;
        queries[1] = 0.625;
        queries[2] = 0.75;
        queries[3] = 0.5;

        for (int i = 0; i < n; ++i) {
                x[i] = i;
                fcn[i] = pow(i, p);
        }

        char testname[STR_LEN];
        sprintf(testname, " * lagrange1: x^%d", p);

        test_t test = test_init(testname, 0, 0);
        err |= interp_lagrange1(val, x, fcn, n, queries, m, p);

        error_print(err);
        for (int i = 0; i < m; ++i) {
                err |= s_assert(fabs(val[i] - pow(queries[i], p)) < FLTOL);
        }
        err |= test_finalize(&test, err);

        free(x);
        free(fcn);
        free(val);
        free(queries);
        return err;
}

int test_lagrange1_coef(int p)
{

//int interp_lagrange1_coef(float *xloc, float *l, int *first, const prec *x,
//                          const int n, const prec query, const int deg)
        char testname[STR_LEN];
        sprintf(testname, " * lagrange1_coef: x^%d", p);
        test_t test = test_init(testname, 0, 0);

        int err = 0;
        prec *x, *xloc, *l;
        size_t n = 10;
        size_t m = p + 1;
        int first;
        x = malloc(sizeof(x) * n);
        xloc = malloc(sizeof(x) * m);
        l = malloc(sizeof(x) * m);
        for (size_t i = 0; i < n; ++i) {
                x[i] = i;
        }

        // Nearest point is x = 1 < 1.05, add an extra point in the upper
        // stencil
        {
                prec query = 1.05;
                err |= interp_lagrange1_coef(xloc, l, &first, x, n, query, p);
                if (p > 0) {
                        err |= s_assert(xloc[0] <= query);
                        err |= s_assert(xloc[p] >= query);
                }
                double sum = 0.0;
                for (size_t i = 0; i < m; ++i) {
                        sum += l[i];
                }
                err |= s_assert(fabs(sum - 1.0) < FLTOL);

        }
        // Nearest point is x = 1 > 0.95, add an extra point in the lower
        // stencil
        {
                prec query = 0.95;
                err |= interp_lagrange1_coef(xloc, l, &first, x, n, query, p);
                if (p > 0) {
                        err |= s_assert(xloc[0] <= query);
                        err |= s_assert(xloc[p] >= query);
                }
                double sum = 0.0;
                for (size_t i = 0; i < m; ++i) {
                        sum += l[i];
                }
                err |= s_assert(fabs(sum - 1.0) < FLTOL);
        }


        
        err |= test_finalize(&test, err);

        free(x);
        free(xloc);
        free(l);

        return test_last_error();
}

int test_lagrange3(void)
{
        int err = 0;

        int n = 10;
        prec *x1, *y1, *z1, *fcn1;
        prec *x3, *y3, *z3, *fcn3;

        int gsize[3] = {n, n, n};
        int3_t shift = grid_yz();
        int3_t coord = {.x = 0, .y = 0, .z = 0};
        int3_t asize = {gsize[0], gsize[1], gsize[2]};
        int3_t bnd1 = {1, 1, 1};
        int3_t bnd2 = {1, 0, 0};
        prec h = 1.0/n;
        int deg = 3;

        grid3_t grid = grid_init(asize, shift, coord, bnd1, bnd2, ngsl, h);
        
        x1 = malloc(sizeof(x1) * grid.size.x);
        y1 = malloc(sizeof(y1) * grid.size.y);
        z1 = malloc(sizeof(z1) * grid.size.z);
        fcn1 = malloc(sizeof(fcn1) * grid.size.x);

        x3 = malloc(grid.num_bytes);
        y3 = malloc(grid.num_bytes);
        z3 = malloc(grid.num_bytes);
        fcn3 = malloc(grid.num_bytes);

        grid_fill_x(x1, grid);
        grid_fill_y(y1, grid);
        grid_fill_z(z1, grid);

        int m = 4;
        prec qx[4] = {0.0, 0.2, 0.4, 0.9};
        prec qy[4] = {0.0, 0.7, 0.4, 0.7};
        prec qz[4] = {0.0, 0.2, 0.3, 0.8};

        prec ax[4] = {0.0, 0.2, 0.4, 0.9};
        prec ay[4] = {0.0, 0.7, 0.4, 0.7};
        prec az[4] = {0.0, 0.2, 0.3, 0.8};
        prec out[4];


        // X-direction
        {
        grid_fill3_x(fcn3, x1, grid);
        grid_pow3(fcn3, deg, grid);
        test_t test = test_init(" * lagrange3: x^3", 0, 0);
        err |=
            interp_lagrange3(out, fcn3, x1, y1, z1, grid, qx, qy, qz, m, deg);
        for(int i = 0; i < m; ++i) {
                err|= s_assert(fabs(out[i] - pow(ax[i], deg)) < FLTOL);
        }
        error_print(err);
        err |= test_finalize(&test, err);
        }

        // Y-direction
        {
        grid_fill3_y(fcn3, y1, grid);
        grid_pow3(fcn3, deg, grid);
        test_t test = test_init(" * lagrange3: y^3", 0, 0);
        err |=
            interp_lagrange3(out, fcn3, x1, y1, z1, grid, qx, qy, qz, m, deg);
        for(int i = 0; i < m; ++i) {
                err|= s_assert(fabs(out[i] - pow(ay[i], deg)) < FLTOL);
        }
        error_print(err);
        err |= test_finalize(&test, err);
        }

        // Z-direction
        {
        grid_fill3_z(fcn3, z1, grid);
        grid_pow3(fcn3, deg, grid);
        test_t test = test_init(" * lagrange3: z^3", 0, 0);
        err |=
            interp_lagrange3(out, fcn3, x1, y1, z1, grid, qx, qy, qz, m, deg);
        for(int i = 0; i < m; ++i) {
                err|= s_assert(fabs(out[i] - pow(az[i], deg)) < FLTOL);
        }
        error_print(err);
        err |= test_finalize(&test, err);
        }

        free(x1);
        free(y1);
        free(z1);
        free(fcn1);
        free(x3);
        free(y3);
        free(z3);
        free(fcn3);
        return err;
}

