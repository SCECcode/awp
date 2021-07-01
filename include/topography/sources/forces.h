#ifndef FORCES_H
#define FORCES_H
/* This module is an interface for handling boundary point forces.
 */

#include <mpi.h>

#include <topography/grids.h>
#include <topography/mapping.h>
#include <topography/sources/source.h>
#include <topography/metrics/metrics.h>

void forces_init(const char *filename, const grids_t *grids, const struct mapping *map, int ngrids,
                  const f_grid_t *f, const g_grid_t *g, const MPI_Comm comm, const int rank,
                  const int size, const float *d_rho, const int istopo);
int forces_boundary_check(const source_t *Fx);
void forces_read(const size_t step);
void forces_add(prec *d_u1, prec *d_v1, prec *d_w1, const prec *d_d1,
                const size_t step, const prec h, const prec dt,
                const f_grid_t *f, const g_grid_t *g, const int grid_num);
void forces_add_cartesian(prec *d_xz, prec *d_yz, prec *d_zz, const size_t step,
                const int nx, const int ny, const int nz, const prec h, const prec dt, const int grid_num);
void forces_add_cartesian_velocity(prec *d_vx, prec *d_vy, prec *d_vz, const size_t step,
                const int nx, const int ny, const int nz, const prec h, const prec dt, const int grid_num);
void forces_finalize(void);

#endif


