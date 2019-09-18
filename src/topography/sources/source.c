#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include <buffers/buffer.h>
#include <grid/shift.h>
#include <utils/array.h>
#include <grid/grid_3d.h>
#include <mpi/distribute.h>
#include <interpolation/interpolation.h>
#include <interpolation/interpolation.cuh>
#include <topography/sources/source.h>
#include <topography/sources/source.cuh>
#include <topography/grids.h>

void source_init_indexed(source_t *src, const input_t *input, size_t num_reads);

source_t source_init(const char *file_end, const input_t *input,
                     const grid3_t grid, 
                     const f_grid_t *f, const int rank,
                     const MPI_Comm comm)
{
        source_t src;

        source_init_common(&src, file_end, input, grid, f, rank, comm);

        if (!src.use) {
                return src;
        }


        size_t num_reads =
            input->steps / (input->cpu_buffer_size * input->gpu_buffer_size);
        source_init_indexed(&src, input, num_reads);
        src.io = mpi_io_idx_init(src.comm, rank, src.offsets, src.blocklen,
                                     src.length, num_reads);

        return src;
}


void source_finalize(source_t *src)
{
        if (!src->use) return;
        free(src->indices);
        buffer_finalize(&src->buffer);
        free(src->blocklen);
        free(src->offsets);
        free(src->host_buffer_extra);
        free(src->x);
        free(src->y);
        free(src->z);
}

void source_init_common(source_t *src, const char *filename,
                        const input_t *input, const grid3_t grid, 
                        const f_grid_t *f,
                        const int rank, const MPI_Comm comm)
{
        sprintf(src->filename, "%s_%s", input->file, filename);

        AWPCHK(dist_indices(&src->indices, &src->length, input->x,
                            input->y, input->length, grid));


        src->use = src->length > 0 ? 1 : 0;

        MPI_Comm_split(comm, src->use, rank, &src->comm);

        if (!src->use) {
                return;
        }

        // Init grid that covers interior and halo regions
        grid3_t full_grid = grid_init_metric_grid(
                    grid.inner_size, grid.shift, grid.coordinate,
                    grid.boundary1, grid.boundary2, grid.gridspacing);
        grid_data_t xyz;
        grid_data_init(&xyz, full_grid);

        // Init arrays that contains local coordinates
        src->x = malloc(sizeof src->x *  src->length);
        src->y = malloc(sizeof src->y *  src->length);
        src->z = malloc(sizeof src->z *  src->length);

        for (size_t i = 0; i < src->length; ++i) {
                src->x[i] = input->x[src->indices[i]];
                src->y[i] = input->y[src->indices[i]];
                src->z[i] = input->z[src->indices[i]];
        }

        // Map input coordinates to parameter space
        if (f != NULL) {
                grid3_t metric_grid = grid_init_metric_grid(
                    grid.inner_size, grid_node(), grid.coordinate,
                    grid.boundary1, grid.boundary2, grid.gridspacing);

                // x, y, z grid vectors compatible with topography grid
                grid1_t x_grid = grid_grid1_x(metric_grid);
                grid1_t y_grid = grid_grid1_y(metric_grid);
                grid1_t z_grid = grid_grid1_z(metric_grid);

                prec *x1 = malloc(sizeof x1 * x_grid.size);
                prec *y1 = malloc(sizeof y1 * y_grid.size);
                prec *z1 = malloc(sizeof z1 * z_grid.size);

                grid_fill1(x1, x_grid);
                grid_fill1(y1, y_grid);
                grid_fill1(z1, z_grid);

                // Interpolate topography data to source location in
                // (x,y) space
                prec *f_interp = malloc(sizeof f_interp * src->length);

                metrics_interpolate_f_point(f, f_interp, f->f, x1, y1,
                                            metric_grid, src->x, src->y,
                                            src->length, input->degree);

                if (input->dimension == 3) { 
                        // Map to parameter space
                        for (size_t k = 0; k < src->length; ++k) {
                                src->z[k] = src->z[k] / f_interp[k];
                        }
                } else {
                        // Automatically map source to free surface
                        for (size_t k = 0; k < src->length; ++k) {
                                src->z[k] = z1[z_grid.size - 2];
                        }
                }

                
                // TODO: Add inversion step if grid stretching function is used

                free(f_interp);
                free(x1);
                free(y1);
                free(z1);

        } 
        // Regular AWP
        //else {
        //        if (input->dimension != 3) { 
        //                // Automatically map source to free surface
        //                for (size_t k = 0; k < src->length; ++k) {
        //                        src->z[k] = z1[z_grid.size - 1];
        //                }
        //        }
        //}

        // Compute interpolation coefficients on the full grid
        AWPCHK(cuinterp_init(&src->interpolation, xyz.x, xyz.y, xyz.z, full_grid,
                                     src->x, src->y, src->z, src->length,
                                     input->degree));
        grid_data_free(&xyz);

        src->buffer = buffer_init(src->length,
                                 input->gpu_buffer_size,
                                 input->cpu_buffer_size, input->stride);

        // Extra space for host buffer
        src->host_buffer_extra = malloc(src->buffer.h_buffer_bytes);
}

void source_init_indexed(source_t *src, const input_t *input, size_t num_reads)
{
        if (!src->use) return;
        src->blocklen = malloc(sizeof(src->blocklen) * input->length);
        src->offsets = malloc(sizeof(src->offsets) * input->length);
        size_t num_elements = input->steps / num_reads;
        src->num_elements = num_elements;
        for (size_t i = 0; i < src->length; ++i) {
                src->blocklen[i] = num_elements;
        }
        for (size_t i = 0; i < src->length; ++i) {
                src->offsets[i] = src->indices[i] * num_elements;
        }
}

void source_read(source_t *src, size_t step)
{
        if (!src->use)
                return;
        if (buffer_is_host_empty(&src->buffer, step)) {
             prec *host_ptr = buffer_get_host_ptr(&src->buffer, step);
             mpi_io_idx_read(&src->io, host_ptr, src->filename);

             // Transpose data from (index, time) to (time, index)
             // (last index is contiguous)
             size_t rows = src->length;
             size_t cols = src->buffer.num_host * src->buffer.num_device;
             array_transpose(src->host_buffer_extra, host_ptr, rows, cols);
             SWAP(src->host_buffer_extra, src->buffer.h_buffer, prec*);
        }
        
        if (buffer_is_device_empty(&src->buffer, step)) {
                buffer_copy_to_device(&src->buffer, step);
        }
        
}

void source_add_cartesian(prec *out, source_t *src, const size_t step,
                          const prec h, const prec dt)
{
        if (!src->use || !buffer_is_device_ready(&src->buffer, step)) 
                return;


        prec *source_data = buffer_get_device_ptr(&src->buffer, step);
        cusource_add_cartesian_H(&src->interpolation, out, source_data, h, dt);
}

void source_add_curvilinear(prec *out, source_t *src, const size_t step,
                            const prec h, const prec dt, const prec *f,
                            const int ny, 
                            const prec *dg) 
{
        if (!src->use || !buffer_is_device_ready(&src->buffer, step)) 
                return;


        prec *source_data = buffer_get_device_ptr(&src->buffer, step);
        cusource_add_curvilinear_H(&src->interpolation, out, source_data, h, dt,
                                   f, ny, dg);
}

