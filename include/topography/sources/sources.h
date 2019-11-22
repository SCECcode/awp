#ifndef SOURCES_H
#define SOURCES_H
/* This module is a container interface for handling moment tensor sources.
 */

#include <mpi.h>

#include <topography/grids.h>
#include <topography/metrics/metrics.h>
#include <topography/sources/source.h>

void sources_init(const char *filename, const grids_t *grids, int ngrids,
                  const f_grid_t *f, const MPI_Comm comm, const int rank,
                  const int size);
void sources_read(const size_t step);
void sources_add_cartesian(prec *d_xx, prec *d_yy, prec *d_zz, prec *d_xy,
                           prec *d_xz, prec *d_yz, const size_t step,
                           const prec h, const prec dt, const int grid_num);
void sources_add_curvilinear(prec *d_xx, prec *d_yy, prec *d_zz, prec *d_xy,
                           prec *d_xz, prec *d_yz, 
                           const size_t step,
                           const prec h, const prec dt, 
                           const f_grid_t *f, const g_grid_t *g,
                           const int grid_num);

source_t sources_get_source(enum grid_types grid_type); 
void sources_finalize(void);

#endif

