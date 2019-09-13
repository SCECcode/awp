#ifndef RECEIVERS_H
#define RECEIVERS_H
/* This module is a container interface for handling receivers.
 */

#include <mpi.h>

#include <topography/grids.h>
#include <topography/metrics/metrics.h>

void receivers_init(const char *filename, const grids_t *grids, int ngrids,
                    const f_grid_t *f, const MPI_Comm comm, const int rank,
                    const int size);
void receivers_finalize(void);
void receivers_write(const prec *d_vx, const prec *d_vy, const prec *d_vz,
                     const size_t step, const size_t num_steps);
size_t receivers_last_step(void);
void receivers_step_format(char *out, size_t step, const char *base);

#endif

