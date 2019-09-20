#include <math.h>

#include <interpolation/lagrange.h>
#include <awp/error.h>

/*
 * Compute Lagrange polynomial basis and evaluate it at x0:
 *
 *      l_j(x0) = product_{i != j} (x0 - x_i)/(x_j - x_i)
 *
 * Input arguments:
 *      l  : Array to hold Lagrange coefficients l_j(x0)
 *      n  : Number of nodes
 *      x0 : Point to evaluate lagrange polynomial at
 *      x : Array of nodes
 *
 * Return value:
 *      Error code. 
 */ 
int lagrange_basis(_prec *l, const int n, const _prec *x, const _prec x0)
{
        if (n == 1) {
                l[0] = 1.0;
                return SUCCESS;
        }
        if (x0 - x[0] < -FLTOL) {
                return ERR_LAGRANGE_LOWER_EXTRAPOLATION;
        }

        if (x0 - x[n-1] > FLTOL) {
                return ERR_LAGRANGE_UPPER_EXTRAPOLATION;
        }

        _prec L = lagrange_node_poly(x0, x, n);
        for (int j = 0; j < n; ++j) {
                l[j] = 1.0;

                if (fabs(x0 - x[j]) < FLTOL) {
                        continue;
                }

                double denom = 1.0;
                for (int k = 0; k < n; ++k) {
                        if (j == k) {
                                continue;
                        }
                        denom *= x[j] - x[k];
                }
                double lambda = 1.0/denom;
                l[j] = (_prec)L*lambda/(x0 - x[j]); 
        }

        return SUCCESS;
}

/*
 * Compute Lagrange node polynomial at x0:
 *
 *      l(x0) = product_{i=1}^n (x0 - x)
 *
 * Input arguments:
 *      x0 : Point to evaluate lagrange polynomial at
 *      x : Array of node positions
 *      n : Number of node positions
 *
 * Return value:
 *      Value of the node polynomial evaluated at `x0`
 *
 */ 
double lagrange_node_poly(const _prec x0, const _prec *x, const size_t n)
{
        double l = 1.0;
        for (size_t i = 0; i < n; ++i) {
                l *= (double)(x0 - x[i]);
        }
        return l;
}

