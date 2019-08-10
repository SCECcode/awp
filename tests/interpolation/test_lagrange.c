#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <awp/definitions.h>
#include <test/test.h>
#include <interpolation/lagrange.h>
#include <interpolation/error.h>


int test_basis(void);

int main(int argc, char **argv)
{
        int err = 0;
        test_divider();
        printf("Testing lagrange.c\n");
        printf("\n");
        printf("Running tests:\n");
        err |= test_basis();
        printf("Testing completed.\n");
        test_divider();
        return err;
}

int test_basis(void)
{
        int err = 0;
        int n = 4;
        _prec *x, *l;
        _prec x0 = 0.0;

        x = malloc(sizeof(x) * n);
        l = malloc(sizeof(l) * n);

        for (int i = 0; i < n; ++i) {
                x[i] = i;
        }

        {
        test_t test = test_init(" * lagrange_basis:init", 0, 0);
        x0 = 0.0;
        err |= lagrange_basis(l, n, x, x0);
        err |= s_assert(!err);
        err |= test_finalize(&test, err);
        }

        {
        test_t test = test_init(" * lagrange_basis:bounds check", 0, 0);
        x0 = -0.1;
        err |= s_no_except(lagrange_basis(l, n, x, x0) == 
                           ERR_LAGRANGE_LOWER_EXTRAPOLATION);
        x0 = x[3] + 0.1;
        err |= s_no_except(lagrange_basis(l, n, x, x0) == 
                           ERR_LAGRANGE_UPPER_EXTRAPOLATION);
        err |= test_finalize(&test, err);
        }
        
        {
        test_t test = test_init(" * lagrange_basis:interpolate constant", 0, 0);
        x0 = 0.5;
        err |= lagrange_basis(l, n, x, x0);
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
                sum += l[i];
        }
        err |= s_assert(fabs(sum - 1.0) < FLTOL);
        err |= test_finalize(&test, err);
        }

        {
        test_t test = test_init(" * lagrange_basis:interpolate x^(n-1)", 0, 0);
        x0 = 0.5;
        err |= lagrange_basis(l, n, x, x0);
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
                sum += l[i]*pow(x[i], n - 1);
        }
        err |= s_assert(fabs(sum - pow(x0, n - 1)) < FLTOL);
        err |= test_finalize(&test, err);
        }

        free(x);
        free(l);

        return err;
}

