#ifndef GRID_CHECK_H
#define GRID_CHECK_H
/*
 * This module is used to compare two arrays that are allocated on the host that
 * have a memory layout that matches the 2D grid decomposition of a grid. In
 * map view, this layout can be represented by:
 *
 *            back 
 *
 *        | 0 | 1 | 2 |
 *        |   |---|   |
 *  left  | 3 | 4 | 5 |  right
 *        |   |---|   |
 *        | 6 | 7 | 8 |
 *
 *            front
 *
 * All sections except for section `4` are ghost regions. 
 *
 * To use this module, first prepare two arrays that you want to compare. Then
 * call any of the comparison functions to compute the error between the arrays
 * and specify where you want the error to be computed.
 * The comparison functions are:
 *  - check_fl1err  : L1-error (sum of absolute value of all terms)
 *  - check_fl2err  : L2-error
 *  - check_finferr : L-infinity-error (maximum absolute value)
 *
 * Where you want the error to be computed is specified by the offsets `off_x`,
 * `off_y`, and `off_z`, one for each direction. Each offset contains two
 * values: the starting index and the exclusive ending index. 
 *
 * Use `check_all` to compute the error for each region in the above figure. If
 * there is an error in a particular region, it will be flagged as `1`. The
 * index of the error array maps to indices in the figure above. Pass
 * `check_fl2err` as the `fcn` argument if you want to use this function for the
 * comparison.
 *
 * The function `check_printerr` can be used to print a figure, like the one
 * shown above, that shows what region contain errors and what the errors are.
 *
 * The function `check_printerr53` includes 2 additional regions in the
 * x-direction. 
 *
 * The function `check_printerr55` includes 2 additional regions in both the
 * x and y-directions. 
 *
 *
 */

typedef double (*check_fun)(const _prec *, const _prec *, 
                            const int, const int, 
                            const int, const int, 
                            const int, const int, 
                            const int, const int);

double check_fl1err(const _prec *u, const _prec *v, 
                    const int i0, const int in, 
                    const int j0, const int jn, 
                    const int k0, const int kn, 
                    const int line, const int slice);

double check_fl2err(const _prec *u, const _prec *v, 
                    const int i0, const int in, 
                    const int j0, const int jn, 
                    const int k0, const int kn, 
                    const int line, const int slice);

double check_flinferr(const _prec *u, const _prec *v, 
                      const int i0, const int in, 
                      const int j0, const int jn, 
                      const int k0, const int kn, 
                      const int line, const int slice);

int check_all(check_fun fp, 
              const _prec *field, const _prec *result, 
              const int *off_x, const int *off_y, const int *off_z, 
              const int nx, const int ny,
              const int line, const int slice, 
              const _prec tol,
              const int *regions,
              _prec *regions_out);

void check_printerr(const char *fcn, const int rank, const char *field_str, 
                     const _prec *err);
void check_printerr53(const char *fcn, const int rank, const char *field_str, 
                      const _prec *err);
void check_printerr55(const char *fcn, const int rank, const char *field_str, 
                      const _prec *err);


#endif
