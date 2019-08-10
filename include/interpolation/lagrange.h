#ifndef LAGRANGE_H
#define LAGRANGE_H
#ifdef __cplusplus
extern "C" {
#endif
#include <stdlib.h>
#include <awp/definitions.h>

int lagrange_basis(_prec *l, const int n, const _prec *x, const _prec x0);
double lagrange_node_poly(const _prec x0, const _prec *x, const size_t n);
#ifdef __cplusplus
}
#endif
#endif 


