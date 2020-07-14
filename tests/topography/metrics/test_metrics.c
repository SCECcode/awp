#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <test/check.h>
#include <test/grid_check.h>
#include <functions/functions.h>
#include <topography/metrics/metrics.h>
#include <grid/shift.h>

typedef struct
{
        _prec tol;
        _prec *f_tmp;
        _prec *g_tmp;
        int verbose;
        int offset_x[2];
        int offset_y[2];
        int offset_z[2];
        int offset_zh[4];
        int offset_stress_x[2];
        int offset_stress_y[2];

} test_t;

f_grid_t test_f_init(test_t *test);
void set_offsets(test_t *test, const f_grid_t *f, const g_grid_t *g);
void test_f_d_malloc(f_grid_t *f, const test_t *test);
void test_f_assign(f_grid_t *f, const test_t *test);
void test_f_interpolate_f_1_constant(f_grid_t *f, const test_t *test);
void test_f_interpolate_f_1_linear(f_grid_t *f, const test_t *test);
void test_f_interpolate_f_2_linear(f_grid_t *f, const test_t *test);
void test_f_interpolate_f_c_linear(f_grid_t *f, const test_t *test);
void test_f_interpolate(f_grid_t *f, const test_t *test);
void test_f_differentiate_f1_quadratic(f_grid_t *f, const test_t *test);
void test_f_differentiate_f2_quadratic(f_grid_t *f, const test_t *test);

g_grid_t test_g_init(test_t *test);
void test_g_interpolate_linear(g_grid_t *g, const test_t *test);
void test_g_differentiate_quadratic(g_grid_t *g, const test_t *test);
void test_free(test_t *test);

int main(int argc, char **argv)
{
        printf("Testing metrics.c\n");
        test_t test = {.tol = 1e-3, .verbose = 0};

        f_grid_t f = test_f_init(&test);
        g_grid_t g = test_g_init(&test);
        set_offsets(&test, &f, &g);

        printf("Testing topography function ...\n");
        test_f_assign(&f, &test);
        test_f_interpolate_f_1_constant(&f, &test);
        test_f_interpolate_f_1_linear(&f, &test);
        test_f_interpolate_f_2_linear(&f, &test);
        test_f_interpolate_f_c_linear(&f, &test);
        test_f_differentiate_f1_quadratic(&f, &test);
        test_f_differentiate_f2_quadratic(&f, &test);
        metrics_free_f(&f);

        printf("Testing grid stretching function ...\n");
        test_g_interpolate_linear(&g, &test);
        test_g_differentiate_quadratic(&g, &test);
        metrics_free_g(&g);

        test_free(&test);
        printf("Testing completed.\n");

        return 0;
}

f_grid_t test_f_init(test_t *test)
{
        int size[3] = {32, 16, 1};
        _prec gridspacing = 1.0;
        f_grid_t out = metrics_init_f(size, gridspacing, 8);
        assert(out.size[0] == size[0]);
        assert(out.size[1] == size[1]);
        assert(out.size[2] == size[2]);
        if (test->verbose)
                metrics_print_info_f(&out);

        int num_bytes = metrics_sizeof_f(&out);
        test->f_tmp = (_prec*)malloc(num_bytes);
        for (int i = 0; i < out.mem[0]*out.mem[1]; ++i) {
                test->f_tmp[i] = 0.0;
        }
        return out;
}

void set_offsets(test_t *test, const f_grid_t *f, const g_grid_t *g)
{
        test->offset_x[0] = f->bounds_x[0] + f->offset[0];
        test->offset_x[1] = f->bounds_x[1] + f->offset[0];
        test->offset_y[0] = f->bounds_y[0] + f->offset[1];
        test->offset_y[1] = f->bounds_y[1] + f->offset[1];

        test->offset_stress_x[0] = f->bounds_stress_x[0] + f->offset[0];
        test->offset_stress_x[1] = f->bounds_stress_x[1] + f->offset[0];
        test->offset_stress_y[0] = f->bounds_stress_y[0] + f->offset[1];
        test->offset_stress_y[1] = f->bounds_stress_y[1] + f->offset[1];

        test->offset_z[0] = g->offset;
        test->offset_z[1] = g->offset + g->size;

        test->offset_zh[0] = 0;
        test->offset_zh[1] = 0;
        test->offset_zh[2] = 1;
        test->offset_zh[3] = 1;
}

void test_f_assign(f_grid_t *f, const test_t *test)
{
        printf(" -- Testing that constant function is applied everywhere\n");
        _prec coef[3] = {0.0, 0.0, 0.0};
        _prec deg[3] = {0.0, 0.0, 0.0};
        int shift[3];
        shift_node(shift);

        _prec args[13] = {coef[0], coef[1], coef[2],
                          deg[0], deg[1], deg[2],
                          shift[0], shift[1], shift[2],
                          f->size[0], f->size[1],
                          0, 0};

        // Set f_1 to zero
        fcn_poly(f->f_1,
                 test->offset_x[0], test->offset_x[1],
                 test->offset_y[0], test->offset_y[1],
                 0, 1, 
                 f->line, f->slice, args);

        // Set f to one
        args[0] = 1.0;
        fcn_poly(f->f,
                 test->offset_x[0], test->offset_x[1],
                 test->offset_y[0], test->offset_y[1],
                 0, 1, 
                 f->line, f->slice, args);

        _prec expected_size = (_prec)(f->bounds_x[1] - f->bounds_x[0]) *
                            (f->bounds_y[1] - f->bounds_y[0]);

        int regions[1] = {1};
        _prec ferr[1] = {0.0};

        check_all(check_fl1err, f->f, f->f_1, test->offset_x, test->offset_y,
                  test->offset_zh, 1, 1, f->line, f->slice, test->tol, regions,
                  ferr);

        assert(fabs(ferr[0] - expected_size) < test->tol);
}

void test_f_interpolate_f_1_constant(f_grid_t *f, const test_t *test)
{
        printf(" -- Testing that constant function is interpolated"            \
               " to f_1 grid at stress points\n");

        int regions[1] = {1};
        int shift[3];
        shift_node(shift);
        _prec coef[3] = {1.0, 0.0, 0.0};
        _prec deg[3] = {0.0, 0.0, 0.0};
        _prec args[13] = {coef[0], coef[1], coef[2],
                          deg[0], deg[1], deg[2],
                          shift[0], shift[1], shift[2],
                          f->size[0], f->size[1],
                          0, 0};

        fcn_poly(f->f,
                 test->offset_x[0], test->offset_x[1],
                 test->offset_y[0], test->offset_y[1],
                 0, 1, 
                 f->line, f->slice, args);
        

        metrics_build_f(f);
        _prec ferr[1] = {0.0};

        check_all(check_fl1err, test->f_tmp, f->f_1, test->offset_stress_x,
                  test->offset_stress_y, test->offset_zh, 1, 1, f->line,
                  f->slice, test->tol, regions, ferr);

        _prec expected_size = (f->bounds_stress_x[1] - f->bounds_stress_x[0]) *
                              (f->bounds_stress_y[1] - f->bounds_stress_y[0]);
        assert(fabs(ferr[0] - expected_size) < test->tol);
}

void test_f_interpolate_f_1_linear(f_grid_t *f, const test_t *test)
{
        printf(" -- Testing that linear function is interpolated"              \
               " to f_1 grid at stress points\n");

        // Linear function in y
        {
        int shift[3];
        shift_node(shift);
        _prec coef[3] = {0.0, 1.0, 0.0};
        _prec deg[3] = {0.0, 1.0, 0.0};
        _prec args[13] = {coef[0], coef[1], coef[2],
                          deg[0], deg[1], deg[2],
                          shift[0], shift[1], shift[2],
                          f->size[0], f->size[1],
                          0, 0};
        fcn_poly(f->f,
                 test->offset_x[0], test->offset_x[1],
                 test->offset_y[0], test->offset_y[1],
                 0, 1, 
                 f->line, f->slice, args);
        
        }

        // Expected result for f_1
        {
        int shift[3];
        shift_u1(shift);
        _prec coef[3] = {0.0, 1.0, 0.0};
        _prec deg[3] = {0.0, 1.0, 0.0};
        _prec args[13] = {coef[0], coef[1], coef[2],
                          deg[0], deg[1], deg[2],
                          shift[0], shift[1], shift[2],
                          f->size[0], f->size[1],
                          0, 0};
        fcn_poly(test->f_tmp, 
                 f->bounds_stress_x[0] + f->offset[0],
                 f->bounds_stress_x[1] + f->offset[0],
                 f->bounds_stress_y[0] + f->offset[1],
                 f->bounds_stress_y[1] + f->offset[1], 0, 1, f->line, f->slice,
                 args);
        _prec ferr[1] = {0.0};
        int regions[1] = {1};
        metrics_build_f(f);
        check_all(check_fl1err, test->f_tmp, f->f_1, test->offset_stress_x,
                  test->offset_stress_y, test->offset_zh, 1, 1, f->line,
                  f->slice, test->tol, regions, ferr);
        assert(fabs(ferr[0]) < test->tol);
        }
}

void test_f_interpolate_f_2_linear(f_grid_t *f, const test_t *test)
{
        printf(" -- Testing that linear function is interpolated"              \
               " to f_2 grid at stress points\n");

        // Linear function in x
        {
        int shift[3];
        shift_node(shift);
        _prec coef[3] = {1.0, 0.0, 0.0};
        _prec deg[3] = {1.0, 0.0, 0.0};
        _prec args[13] = {coef[0], coef[1], coef[2],
                          deg[0], deg[1], deg[2],
                          shift[0], shift[1], shift[2],
                          f->size[0], f->size[1],
                          0, 0};
        fcn_poly(f->f,
                 test->offset_x[0], test->offset_x[1],
                 test->offset_y[0], test->offset_y[1],
                 0, 1, 
                 f->line, f->slice, args);
        
        }
        // Expected result for f_2
        {
        int shift[3];
        shift_u2(shift);
        _prec coef[3] = {1.0, 0.0, 0.0};
        _prec deg[3] = {1.0, 0.0, 0.0};
        _prec args[13] = {coef[0], coef[1], coef[2],
                          deg[0], deg[1], deg[2],
                          shift[0], shift[1], shift[2],
                          f->size[0], f->size[1],
                          0, 0};
        fcn_poly(test->f_tmp, 
                 test->offset_stress_x[0], test->offset_stress_x[1],
                 test->offset_stress_y[0], test->offset_stress_y[1],
                 0, 1, f->line, f->slice,
                 args);
        _prec ferr[1] = {0.0};
        int regions[1] = {1};
        metrics_build_f(f);
        check_all(check_fl1err, test->f_tmp, f->f_2, test->offset_stress_x,
                  test->offset_stress_y, test->offset_zh, 1, 1, f->line,
                  f->slice, test->tol, regions, ferr);
        assert(fabs(ferr[0]) < test->tol);
        }
}

void test_f_interpolate_f_c_linear(f_grid_t *f, const test_t *test)
{
        printf(" -- Testing that linear function is interpolated"              \
               " to f_c grid at stress points\n");
        // Linear function in x and y
        {
        int shift[3];
        shift_node(shift);
        _prec coef[3] = {1.0, 1.0, 0.0};
        _prec deg[3] = {1.0, 1.0, 0.0};
        _prec args[13] = {coef[0], coef[1], coef[2],
                          deg[0], deg[1], deg[2],
                          shift[0], shift[1], shift[2],
                          f->size[0], f->size[1],
                          0, 0};
        fcn_poly(f->f,
                 test->offset_x[0], test->offset_x[1],
                 test->offset_y[0], test->offset_y[1],
                 0, 1, 
                 f->line, f->slice, args);
        
        }
        // Expected result for f_c
        {
        int shift[3];
        shift_xx(shift);
        _prec coef[3] = {1.0, 1.0, 0.0};
        _prec deg[3] = {1.0, 1.0, 0.0};
        _prec args[13] = {coef[0], coef[1], coef[2],
                          deg[0], deg[1], deg[2],
                          shift[0], shift[1], shift[2],
                          f->size[0], f->size[1],
                          0, 0};
        fcn_poly(test->f_tmp, 
                 test->offset_stress_x[0], test->offset_stress_x[1],
                 test->offset_stress_y[0], test->offset_stress_y[1],
                 0, 1, f->line, f->slice,
                 args);
        _prec ferr[1] = {0.0};
        int regions[1] = {1};
        metrics_build_f(f);
        check_all(check_fl1err, test->f_tmp, f->f_c, test->offset_stress_x,
                  test->offset_stress_y, test->offset_zh, 1, 1, f->line,
                  f->slice, test->tol, regions, ferr);
        assert(fabs(ferr[0]) < test->tol);
        }
}


void test_f_differentiate_f1_quadratic(f_grid_t *f, const test_t *test)
{

        {
        int shift[3];
        shift_node(shift);
        _prec coef[3] = {1.0, 0.0, 0.0};
        _prec deg[3] = {2.0, 0.0, 0.0};
        _prec args[13] = {coef[0], coef[1], coef[2],
                          deg[0], deg[1], deg[2],
                          shift[0], shift[1], shift[2],
                          f->size[0], f->size[1],
                          0, 0};
        fcn_poly(f->f,
                 test->offset_x[0], test->offset_x[1],
                 test->offset_y[0], test->offset_y[1],
                 0, 1, 
                 f->line, f->slice, args);
        
        }

        char *grids[3] = {"f_1", "f_2", "f_c"};
        int su1[3];
        shift_u1(su1);
        int su2[3];
        shift_u2(su2);
        int sxx[3];
        shift_xx(sxx);
        int *shifts[3] = {su1, su2, sxx};
        _prec *out[3] = {f->f1_1, f->f1_2, f->f1_c};

        for (int i = 0; i < 3; ++i) {
                printf(" -- Testing that x^2 is differentiated"             \
                       " to %s grid at stress points\n", grids[i]);
                int shift[3] = {shifts[i][0], shifts[i][1], shifts[i][2]};
                _prec coef[3] = {2.0, 0.0, 0.0};
                _prec deg[3] = {1.0, 0.0, 0.0};
                _prec args[13] = {coef[0], coef[1], coef[2],
                                  deg[0], deg[1], deg[2],
                                  shift[0], shift[1], shift[2],
                                  f->size[0], f->size[1],
                                  0, 0};
                fcn_poly(test->f_tmp, 
                         test->offset_stress_x[0], test->offset_stress_x[1],
                         test->offset_stress_y[0], test->offset_stress_y[1],
                         0, 1, f->line, f->slice,
                         args);
                _prec ferr[1] = {0.0};
                int regions[1] = {1};
                metrics_build_f(f);
                check_all(check_flinferr, test->f_tmp, out[i], 
                          test->offset_stress_x,
                          test->offset_stress_y, 
                          test->offset_zh, 1, 1, f->line,
                          f->slice, test->tol, regions, ferr);
                assert(fabs(ferr[0]) < test->tol);
        }
}

void test_f_differentiate_f2_quadratic(f_grid_t *f, const test_t *test)
{

        {
        int shift[3];
        shift_node(shift);
        _prec coef[3] = {0.0, 1.0, 0.0};
        _prec deg[3] = {0.0, 2.0, 0.0};
        _prec args[13] = {coef[0], coef[1], coef[2],
                          deg[0], deg[1], deg[2],
                          shift[0], shift[1], shift[2],
                          f->size[0], f->size[1],
                          0, 0};
        fcn_poly(f->f,
                 test->offset_x[0], test->offset_x[1],
                 test->offset_y[0], test->offset_y[1],
                 0, 1, 
                 f->line, f->slice, args);
        
        }

        char *grids[3] = {"f_1", "f_2", "f_c"};
        int su1[3];
        shift_u1(su1);
        int su2[3];
        shift_u2(su2);
        int sxx[3];
        shift_xx(sxx);
        int *shifts[3] = {su1, su2, sxx};
        _prec *out[3] = {f->f2_1, f->f2_2, f->f2_c};

        for (int i = 0; i < 3; ++i) {
                printf(" -- Testing that y^2 is differentiated"             \
                       " to %s grid at stress points\n", grids[i]);
                int shift[3] = {shifts[i][0], shifts[i][1], shifts[i][2]};
                _prec coef[3] = {0.0, 2.0, 0.0};
                _prec deg[3] = {0.0, 1.0, 0.0};
                _prec args[13] = {coef[0], coef[1], coef[2],
                                  deg[0], deg[1], deg[2],
                                  shift[0], shift[1], shift[2],
                                  f->size[0], f->size[1],
                                  0, 0};
                fcn_poly(test->f_tmp, 
                         test->offset_stress_x[0], test->offset_stress_x[1],
                         test->offset_stress_y[0], test->offset_stress_y[1],
                         0, 1, f->line, f->slice,
                         args);
                _prec ferr[1] = {0.0};
                int regions[1] = {1};
                metrics_build_f(f);
                check_all(check_flinferr, test->f_tmp, out[i], 
                          test->offset_stress_x,
                          test->offset_stress_y, 
                          test->offset_zh, 1, 1, f->line,
                          f->slice, test->tol, regions, ferr);
                assert(fabs(ferr[0]) < test->tol);
        }
}

g_grid_t test_g_init(test_t *test)
{
        int size[3] = {32, 16, 32};
        _prec gridspacing = 1.0;
        g_grid_t out = metrics_init_g(size, gridspacing);
        assert(out.size == size[2]);
        if (test->verbose)
                metrics_print_info_g(&out);
        int num_bytes = metrics_sizeof_g(&out);
        test->g_tmp = (_prec*)malloc(num_bytes);
        for (int i = 0; i < out.mem; ++i) {
                test->g_tmp[i] = 0.0;
        }
        return out;
}

void test_g_interpolate_linear(g_grid_t *g, const test_t *test)
{
        printf(" -- Testing that linear function is interpolated"            \
               " to cell-centers\n");

        {
        int shift[3];
        shift_node(shift);
        _prec coef[3] = {0.0, 0.0, 1.0};
        _prec deg[3] = {0.0, 0.0, 1.0};
        _prec args[13] = {coef[0], coef[1], coef[2],
                          deg[0], deg[1], deg[2],
                          shift[0], shift[1], shift[2],
                          0, 0,
                          0, 0};
        fcn_polybndz(g->g,
                 0, 1,
                 0, 1,
                 test->offset_z[0], test->offset_z[1], 
                 1, 1, args);
        
        }

        metrics_build_g(g);

        {
        int shift[3];
        shift_xx(shift);
        _prec coef[3] = {0.0, 0.0, 1.0};
        _prec deg[3] = {0.0, 0.0, 1.0};
        _prec args[13] = {coef[0], coef[1], coef[2],
                          deg[0], deg[1], deg[2],
                          shift[0], shift[1], shift[2],
                          0, 0,
                          0, 0};
        fcn_polybndz(test->g_tmp,
                 0, 1,
                 0, 1,
                 test->offset_z[0], test->offset_z[1], 
                 1, 1, args);
        _prec err =
            check_fl1err(test->g_tmp, g->g_c, 0, 1, 0, 1, test->offset_z[0],
                           test->offset_z[1], 1, 1);
        assert(fabs(err) < test->tol);
        
        }
}

void test_g_differentiate_quadratic(g_grid_t *g, const test_t *test)
{
        {
                int shift[3];
                shift_node(shift);
                _prec coef[3] = {0.0, 0.0, 1.0};
                _prec deg[3] = {0.0, 0.0, 2.0};
                _prec args[13] = {coef[0], coef[1], coef[2],
                                  deg[0], deg[1], deg[2],
                                  shift[0], shift[1], shift[2],
                                  0, 0,
                                  0, 0};
                fcn_polybndz(g->g,
                         0, 1,
                         0, 1,
                         test->offset_z[0], test->offset_z[1], 
                         1, 1, args);
                
                metrics_build_g(g);
        }

        int su3[3];
        int sxx[3];
        shift_u3(su3);
        shift_xx(sxx);
        int *shifts[2] = {su3, sxx};
        _prec *out[2] = {g->g3, g->g3_c};
        char *grids[2] = {"nodes", "cell-centers"};


        for (int i = 0; i < 2; ++i) {
                printf(" -- Testing that y^2 is differentiated"             \
                       " to %s\n", grids[i]);


                _prec coef[3] = {0.0, 0.0, 2.0};
                _prec deg[3] = {0.0, 0.0, 1.0};
                _prec args[13] = {coef[0], coef[1], coef[2],
                                  deg[0], deg[1], deg[2],
                                  shifts[i][0], shifts[i][1], shifts[i][2],
                                  0, 0,
                                  0, 0};
                fcn_polybndz(test->g_tmp,
                         0, 1,
                         0, 1,
                         test->offset_z[0], test->offset_z[1], 
                         1, 1, args);
                // Skip testing boundary points because the interpolation is
                // only exact for first order polynomials
                int bnd = 6;
                _prec err =
                    check_fl1err(test->g_tmp, out[i], 0, 1, 0, 1, 
                                  test->offset_z[0]+bnd,
                                   test->offset_z[1]-bnd, 1, 1);

                assert(fabs(err) < test->tol);
        }
}

void test_free(test_t *test)
{
        free(test->f_tmp);
        free(test->g_tmp);
}

