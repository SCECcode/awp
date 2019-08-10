#ifndef CHECK_H
#define CHECK_H
#ifdef __cplusplus
extern "C" {
#endif
#include <awp/definitions.h>

/*
 * Compare two arrays in the maximum norm.
 *
 * Arguments:
 *      a: Array of size n     
 *      b: Array of size n     
 *      n: Number of elements to compare
 *
 * Return value:
 *      Maximum absolute difference 
 */      
double chk_inf(const prec *a, const prec *b, const int n);

int chk_infi(const int *a, const int *b, const int n);
int chk_infl(const size_t *a, const size_t *b, const size_t n);

#ifdef __cplusplus
}
#endif
#endif

