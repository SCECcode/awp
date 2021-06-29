#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <topography/mapping.h>

#ifdef NDEBUG
#undef NDEBUG
#endif

void test_convergence(const double dzb, const double dzt, const int n, const double eps);
void test_convergence(const double dzb, const double dzt, const int n, const double eps) {

    printf("Testing dzb=%f, dzt=%f n=%d eps=%f \n", dzb, dzt, n, eps);
    double h = 1.0 / (n - 1);
    struct mapping map = init_mapping(dzb, dzt, h);
    for (int i = 0; i < n; ++i) {
        double z = i * h;
        double r = invert(z, &map, 0.5 * eps, 10000);
        double zeval = eval(r, &map);
        assert(fabs(zeval - z) < eps);
    }
}

int main(int argc, char **argv) {

    double eps = 1e-4;

    // Check that if the mapping is linear then, z = r
    double n = 4;
    double h = 1.0 / (n - 1);
    double dzb = h;
    double dzt = h;
    struct mapping map = init_mapping(dzb, dzt, h);

    assert(find_cell_r(0.2 * h, &map) == 0);
    assert(find_cell_r(1.1 * h, &map) == 1);
    assert(find_cell_r(1.0 - 0.5 * h, &map) == 2);

    for (int i = 0; i <  n; ++i) {
        double r = i * h;
        assert(fabs(r - eval(r, &map)) < eps * h);
    }
    
    for (int i = 0; i < n; ++i) {
        double r = i * h;
        assert(fabs(r - invert(r, &map, 0.5 * eps, 1000)) < eps);
    }

    test_convergence(0.1, 0.01, 11, eps);
    test_convergence(0.1, 0.1, 10, eps);
    test_convergence(1e-2, 0.1, 10, eps);
    test_convergence(5e-3, 0.1, 10, eps);
    test_convergence(1e-2, 0.1, 100, eps);
    test_convergence(5e-3, 0.1, 100, eps);
    test_convergence(0.01, 0.1, 1000, eps);
    test_convergence(1e-2, 1e-2, 1000, eps);
    test_convergence(1e-4, 1e-5, 10000, eps);
    test_convergence(1e-3, 1e-6, 10000, eps);
}

