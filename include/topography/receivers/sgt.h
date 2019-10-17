#ifndef SGT_H
#define SGT_H
/* This module is an interface for handling SGT outputs.
 */

#include <mpi.h>

#include <topography/grids.h>
#include <topography/metrics/metrics.h>

void sgt_init(const char *filename, const grids_t *grids, int ngrids,
                    const f_grid_t *f, const MPI_Comm comm, const int rank,
                    const int size);
void sgt_finalize(void);
void sgt_write_material_properties(const prec *d_d1, const prec *d_lami,
                                   const prec *d_mui, const int grid_num);
void sgt_write(const prec *d_xx, const prec *d_yy, const prec *d_zz,
               const prec *d_xy, const prec *d_xz, const prec *d_yz,
                     const size_t step, const size_t num_steps,
                     const int grid_num);

#endif

