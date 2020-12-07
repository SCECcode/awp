#ifndef SOURCE_H
#define SOURCE_H

#include <mpi.h>

#include <awp/definitions.h>
#include <awp/pmcl3d_cons.h>
#include <buffers/buffer.h>
#include <test/test.h>
#include <awp/error.h>
#include <readers/input.h>
#include <grid/grid_3d.h>
#include <topography/grids.h>
#include <topography/metrics/metrics.h>
#include <mpi/io.h>
#include <interpolation/interpolation.cuh>

// Offsets in grid spacings factor with respect to the previous grid
#define SOURCE_DM_OFFSET_X 0
#define SOURCE_DM_OFFSET_Y -1

// Shift due to inconsistency with the user coordinate (0, 0, 0) defined at a
// material grid point, but (0, 0, 0) defined at the shear stress xz in the
// internal coordinate system (see shift.c)
//#define SOURCE_OFFSET_X -0.5
#define SOURCE_OFFSET_X -0.5

typedef struct {
        int *indices;
        int *offsets;
        int *blocklen;
        size_t length;
        // parameter space coordinates
        int *global_indices[MAXGRIDS];
        // Adjusted coordinates for placing sources/receivers consistent with
        // DM with respect to grids that use a local coordinate system (internal
        // coordinate system)
        prec *x[MAXGRIDS];
        prec *y[MAXGRIDS];
        prec *z[MAXGRIDS];
        // User coordinates that are specified in the input file
        prec *xu[MAXGRIDS];
        prec *yu[MAXGRIDS];
        prec *zu[MAXGRIDS];
        int *type[MAXGRIDS];
        size_t lengths[MAXGRIDS];
        size_t num_elements;
        cu_interp_t interpolation[MAXGRIDS];
        mpi_io_idx_t io;
        buffer_t buffer;
        prec *host_buffer_extra;
        MPI_Comm comm;
        int use;
        char filename[STR_LEN*2];
        int ngrids;
        size_t steps;

} source_t;


source_t source_init(const char *file_end, 
                     const enum grid_types grid_type,
                     const input_t *input,
                     const grids_t *grids, 
                     const int ngrids,
                     const f_grid_t *f, 
                     const int rank,
                     const MPI_Comm comm);

void source_finalize(source_t *src);

void source_find_grid_number(const input_t *input, const
                             grids_t *grids, int *grid_number, 
                             const int *indices,
                             const int length,
                             const int num_grids);
void source_init_common(source_t *src, const char *filename,
                        const enum grid_types grid_type, 
                        const input_t *input, 
                        const grids_t *grids, 
                        const int ngrids,
                        const f_grid_t *f,
                        const int rank, 
                        const MPI_Comm comm,
                        const int is_source);
MPI_Comm source_communicator(source_t *src, const int rank,
                             const MPI_Comm comm);
void source_read(source_t *src, size_t step);
void source_add_cartesian(prec *out, source_t *src, const size_t step,
                          const prec h, const prec dt, const int grid_num);

// zhat: indicates if the source should be applied on the cell-centered grid in
// the z-direction or not
void source_add_curvilinear(prec *out, source_t *src, const size_t step,
                            const prec h, const prec dt, const prec *f,
                            const int ny, const prec *dg, const int grid_num, const int zhat);

void source_add_force(prec *out, const prec *d1, source_t *src,
                      const size_t step, const prec h, const prec dt,
                      const prec quad_weight,
                      const prec *f, const int nx, const int ny, const int nz, 
                      const prec *dg,
                      const int grid_num);

#endif

