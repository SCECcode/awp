#ifndef SOURCE_H
#define SOURCE_H

#include <mpi.h>

#include <awp/definitions.h>
#include <buffers/buffer.h>
#include <test/test.h>
#include <awp/error.h>
#include <readers/input.h>
#include <grid/grid_3d.h>
#include <topography/metrics/metrics.h>
#include <mpi/io.h>
#include <interpolation/interpolation.cuh>

typedef struct {
        int *indices;
        int *offsets;
        int *blocklen;
        // parameter space coordinates
        prec *x;
        prec *y;
        prec *z;
        int *type;
        size_t num_elements;
        size_t length;
        cu_interp_t interpolation;
        mpi_io_idx_t io;
        buffer_t buffer;
        prec *host_buffer_extra;
        MPI_Comm comm;
        int use;
        char filename[STR_LEN*2];

} source_t;


source_t source_init(const char *file_end, const input_t *input,
                     const grid3_t grid, const f_grid_t *f, 
                     const int *grid_number,
                     const int rank,
                     const MPI_Comm comm);

void source_finalize(source_t *src);
void source_init_common(source_t *src, const char *filename,
                        const input_t *input, const grid3_t grid, 
                        const f_grid_t *f,
                        const int *grid_number,
                        const int rank, 
                        const MPI_Comm comm);
MPI_Comm source_communicator(source_t *src, const int rank,
                             const MPI_Comm comm);
void source_read(source_t *src, size_t step);
void source_add_cartesian(prec *out, source_t *src, const size_t step,
                          const prec h, const prec dt);

void source_add_curvilinear(prec *out, source_t *src, const size_t step,
                            const prec h, const prec dt, const prec *f,
                            const int ny, const prec *dg);

#endif

