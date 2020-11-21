#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <test/check.h>

double chk_inf(const prec *a, const prec *b, const int n)
{
        double err = 0;
        for (int i = 0; i < n; ++i) {
                double diff = fabs(a[i] - b[i]);
                err = diff > err ? diff : err;
        }
        return err;
}

int chk_infi(const int *a, const int *b, const int n)
{
        int err = 0;
        for (int i = 0; i < n; ++i) {
                int diff = abs(a[i] - b[i]);
                err = diff > err ? diff : err;
        }
        return err;
}

int chk_infl(const size_t *a, const size_t *b, const size_t n)
{
        int err = 0;
        for (size_t i = 0; i < n; ++i) {
                int diff = abs((int)a[i] - (int)b[i]);
                err = diff > err ? diff : err;
        }
        return err;
}

double chk_2(const prec *a, const prec *b, const int n)
{
        double err = 0;
        for (int i = 0; i < n; ++i) {
                double diff = pow(a[i] - b[i], 2);
                err += diff;
        }
        return sqrt(err);
}
