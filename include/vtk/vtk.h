#ifndef VTK_H
#define VTK_H
#ifdef __cplusplus
extern "C" {
#endif
/* The VTK module provides function for writing and reading .vtk files stored in
 * the vtk legacy file format.
 *
 * TODO: Implement functions for reading vtk files.
 */
#include <grid/grid_3d.h>

size_t vtk_write_grid(const char *fname, 
                      const _prec *x, 
                      const _prec *y, 
                      const _prec *z, 
                      const fcn_grid_t grid
                      );

size_t vtk_write_grid_xz(const char *fname, 
                      const _prec *x,
                      const _prec *z,
                      const fcn_grid_t grid);

size_t vtk_append_scalar(const char *fname,
                         const char *label, 
                         const _prec *data,
                         const fcn_grid_t grid
                         );

size_t vtk_append_scalar_xz(const char *fname,
                         const char *label, 
                         const _prec *data,
                         const fcn_grid_t grid
                         );

#ifdef __cplusplus
}
#endif
#endif

