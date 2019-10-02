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

typedef struct {
        int *indices;
        int *offsets;
        int *blocklen;
        size_t length;
        // parameter space coordinates
        int *global_indices[MAXGRIDS];
        prec *x[MAXGRIDS];
        prec *y[MAXGRIDS];
        prec *z[MAXGRIDS];
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
                             const grids_t *grids, int *grid_number, 
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
                        const MPI_Comm comm);
MPI_Comm source_communicator(source_t *src, const int rank,
                             const MPI_Comm comm);
void source_read(source_t *src, size_t step);
void source_add_cartesian(prec *out, source_t *src, const size_t step,
                          const prec h, const prec dt, const int grid_num);

void source_add_curvilinear(prec *out, source_t *src, const size_t step,
                            const prec h, const prec dt, const prec *f,
                            const int ny, const prec *dg, const int grid_num);

#endif

