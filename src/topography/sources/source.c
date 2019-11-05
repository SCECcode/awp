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

#define OVERLAP 7.0

void source_init_indexed(source_t *src, const input_t *input, size_t num_reads);

source_t source_init(const char *file_end, 
                     const enum grid_types grid_type, 
                     const input_t *input,
                     const grids_t *grids, 
                     const int ngrids,
                     const f_grid_t *f, 
                     const int rank,
                     const MPI_Comm comm)
{
        source_t src;

        source_init_common(&src, file_end, grid_type, input, grids, ngrids, f,
                           rank, comm);

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
        for (int i = 0; i < src->ngrids; ++i) {
                if (src->x[i] != NULL) free(src->x[i]);
                if (src->y[i] != NULL) free(src->y[i]);
                if (src->z[i] != NULL) free(src->z[i]);
                if (src->xu[i] != NULL) free(src->xu[i]);
                if (src->yu[i] != NULL) free(src->yu[i]);
                if (src->zu[i] != NULL) free(src->zu[i]);
                if (src->type[i] != NULL) free(src->type[i]);
        }
}

void source_find_grid_number(const input_t *input, const
                             grids_t *grids, int *grid_number, 
                             const int *indices,
                             const int length,
                             const int num_grids)
{

        prec *z1 = malloc(sizeof z1 * grids[0].z.size.z);

        for (int j = 0; j < length; ++j) {
                grid_number[j] = -1;
        }

        grid1_t z_grid = grid_grid1_z(grids[0].z);
        grid_fill1(z1, z_grid);

        _prec lower = 0;
        _prec upper = 0;
        _prec overlap = 0;
        for (int i = 0; i < num_grids; ++i) {
                z_grid = grid_grid1_z(grids[i].z);
                grid_fill1(z1, z_grid);
                upper  = lower;
                lower  = lower - z1[z_grid.end];
                // Shift lower position by "overlap" for all grids except the
                // last grid. No shift is applied if there is only one grid.
                if (i + 1 != num_grids) {
                        overlap = z_grid.gridspacing * OVERLAP;
                } else {
                        overlap = 0.0f;
                }
                lower = lower + overlap;
                for (int j = 0; j < length; ++j) {
                        _prec z = input->z[indices[j]];
                        // Take into account that topography can yield positive
                        // z-values
                        if (input->type[indices[j]] == INPUT_SURFACE_COORD) {
                                grid_number[j] = 0;
                                continue;
                        }
                        else if (z > 0) {
                                grid_number[j] = 0;
                        }
                        else if (lower < z && z <= upper) {
                                grid_number[j] = i;
                        }
                        else if (lower - overlap <= z && z <= lower) {
                                fprintf(stderr, 
                                "Source/receiver id=%d is in overlap zone.\n",
                                j);
                        }

                }

        }

        free(z1);

        for (int j = 0; j < length; ++j) {
                if (grid_number[j] == -1) {
                        fprintf(stderr, 
                                "Failed to assign source/receiver id=%d "\
                                " to a grid.\n", j);
                        exit(1);
                }

        }

}

void source_init_common(source_t *src, const char *filename,
                        const enum grid_types grid_type,
                        const input_t *input, 
                        const grids_t *grids, 
                        const int ngrids,
                        const f_grid_t *f, 
                        const int rank, const MPI_Comm comm)
{
        sprintf(src->filename, "%s_%s", input->file, filename);



        // Shift by 0.5 such that x = 0, y = 0 is
        // located at a material or topography grid
        // point.

        _prec *x = malloc(sizeof x * input->length);
        _prec *y = malloc(sizeof y * input->length);

        {
                int *grid_number = malloc(sizeof grid_number * input->length);
                int *indices = malloc(sizeof indices * input->length);

                for (size_t i = 0; i < input->length; ++i) {
                        indices[i] = i;
                }

                source_find_grid_number(input, grids, grid_number, indices,
                                        input->length, ngrids);

                grid3_t grid_prev;
                for (int j = 0; j < ngrids; ++j) {
                        grid3_t grid = grids_select(grid_type, &grids[j]);
                        if (j > 0) {
                                grid_prev =
                                    grids_select(grid_type, &grids[j - 1]);
                        }
                        for (size_t i = 0; i < input->length; ++i) {
                                if (grid_number[i] != j) continue;
                                
                                x[i] = input->x[i];
                                y[i] = input->y[i];

                                // Skip DM-specific shift for the top grid block
                                if (grid_number[i] == 0) 
                                        continue;

                                // Apply DM-specific shift for all other blocks
                                x[i] =
                                    x[i] + SOURCE_OFFSET_X * grid.gridspacing +
                                    SOURCE_DM_OFFSET_X * grid_prev.gridspacing;
                                y[i] = 
                                    y[i] + 
                                    SOURCE_DM_OFFSET_Y * grid_prev.gridspacing;
                        }
                }

                free(indices);
                free(grid_number);
        }


        grid3_t grid = grids_select(grid_type, &grids[0]);


        AWPCHK(dist_indices(&src->indices, &src->length, x,
                            y, input->length, grid));



        src->ngrids = ngrids;
        src->use = src->length > 0 ? 1 : 0;

        MPI_Comm_split(comm, src->use, rank, &src->comm);

        if (!src->use) {
                return;
        }

        for (int j = 0; j < ngrids; ++j) {
                src->lengths[j] = 0;
        }

        int *grid_number = malloc(sizeof grid_number * src->length);
        source_find_grid_number(input, grids, grid_number, src->indices,
                                src->length, ngrids);

        for (size_t i = 0; i < src->length; ++i) {
                for (int j = 0; j < ngrids; ++j) {
                        if (grid_number[i] == j) src->lengths[j] += 1;
                }
        }

        // Init arrays that contains local coordinates
        for (int j = 0; j < ngrids; ++j) {
                src->global_indices[j] =
                    calloc(sizeof src->global_indices[j], src->lengths[j]);
                src->x[j] = malloc(sizeof src->x * src->lengths[j]);
                src->y[j] = malloc(sizeof src->y * src->lengths[j]);
                src->z[j] = malloc(sizeof src->z * src->lengths[j]);
                src->xu[j] = malloc(sizeof src->x * src->lengths[j]);
                src->yu[j] = malloc(sizeof src->y * src->lengths[j]);
                src->zu[j] = malloc(sizeof src->z * src->lengths[j]);
                src->type[j] = malloc(sizeof src->type *  src->lengths[j]);
        }

        for (int j = 0; j < ngrids; ++j) {
                int local_idx = 0;
                for (size_t i = 0; i < src->length; ++i) {
                        if (grid_number[i] != j) continue;
                        src->global_indices[j][local_idx] = i;
                        src->x[j][local_idx] = x[src->indices[i]];
                        src->y[j][local_idx] = y[src->indices[i]];
                        src->z[j][local_idx] = input->z[src->indices[i]];
                        src->xu[j][local_idx] = input->x[src->indices[i]];
                        src->yu[j][local_idx] = input->y[src->indices[i]];
                        src->zu[j][local_idx] = input->z[src->indices[i]];
                        src->type[j][local_idx] = input->type[src->indices[i]];
                        local_idx++;
                }
        }

        _prec overlap = 0.0;
        _prec lower = 0.0;
        _prec block_height = 0.0;
        for (int j = 0; j < ngrids; ++j) {
                grid = grids_select(grid_type, &grids[j]);
        
                grid3_t metric_grid = grid_init_metric_grid( grid.inner_size,
                                grid_node(), grid.coordinate, grid.boundary1,
                                grid.boundary2, grid.gridspacing);

                if (f!= NULL && j == 0) {
                        block_height = grid.gridspacing * (grid.size.z - 2);
                }
                else {
                        block_height = grid.gridspacing * (grid.size.z - 1);
                }

                lower  = lower - block_height + overlap;

                if (src->lengths[j] == 0) {
                        src->x[j] = NULL;
                        src->y[j] = NULL;
                        src->z[j] = NULL;
                        src->type[j] = NULL;
                }

                if (src->lengths[j] != 0 && f != NULL && j == 0) {
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
                        prec *f_interp =
                            malloc(sizeof f_interp * src->lengths[j]);

                        metrics_interpolate_f_point(f, f_interp, f->f, x1, y1,
                                                    metric_grid, src->x[j], src->y[j],
                                                    src->lengths[j], input->degree);


                        for (size_t k = 0; k < src->lengths[j]; ++k) {
                                switch (src->type[j][k]) {
                                        // Map to parameter space
                                        case INPUT_VOLUME_COORD:
                                                src->z[j][k] =
                                                    (block_height + src->z[j][k]) /
                                                    f_interp[k];
                                                break;
                                        case INPUT_SURFACE_COORD:
                                                src->z[j][k] = z1[z_grid.size - 2];
                                                break;
                                                // FIXME: INPUT_BATHYMETRY_COORD
                                                // Implement treatment for ocean
                                                // bathymetry.
                                                // Recommendation: Add a
                                                // function to "receivers.c" and
                                                // a function to to "receiver.c"
                                                // Place the implementation in
                                                // "receiver.c" but call this
                                                // function for each receiver
                                                // component in "receivers.c"
                                }
                        }

                        // TODO: Add inversion step if grid stretching function
                        // is used

                        free(f_interp);
                        free(x1);
                        free(y1);
                        free(z1);
                } 
                // Regular AWP
                else {
                      for (size_t k = 0; k < src->lengths[j]; ++k) {
                              switch (src->type[j][k]) {
                                      case INPUT_VOLUME_COORD:
                                              src->z[j][k] = (src->z[j][k] - lower);
                                              break;
                                      // Map to parameter space
                                      case INPUT_SURFACE_COORD:
                                              // Only coordinates in the top
                                              // block can be surface
                                              // coordinates
                                              assert(j == 0);
                                              src->z[j][k] = block_height;
                                              break;
                              }
                      }
                }

                overlap = grid.gridspacing * OVERLAP;

                if (src->lengths[j] == 0) continue;

                // Init grid that covers interior and halo regions
                grid3_t full_grid = grid_init_metric_grid(
                            grid.inner_size, grid.shift, grid.coordinate,
                            grid.boundary1, grid.boundary2, grid.gridspacing);
                grid_data_t xyz;
                grid_data_init(&xyz, full_grid);

                // Compute interpolation coefficients on the full grid
                AWPCHK(cuinterp_init(&src->interpolation[j], xyz.x, xyz.y, xyz.z,
                                        full_grid, src->x[j], src->y[j], src->z[j],
                                        src->global_indices[j],
                                        src->lengths[j], input->degree));
 


                                        
                grid_data_free(&xyz);
        } // end loop j

        free(grid_number);
        free(x);
        free(y);
        

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
                          const prec h, const prec dt, const int grid_num)
{
        if (!src->use || !buffer_is_device_ready(&src->buffer, step) ||
             src->lengths[grid_num] == 0) 
                return;


        prec *source_data = buffer_get_device_ptr(&src->buffer, step);
        cusource_add_cartesian_H(&src->interpolation[grid_num], 
                                 out, source_data, h, dt);
}

void source_add_curvilinear(prec *out, source_t *src, const size_t step,
                            const prec h, const prec dt, const prec *f,
                            const int ny, 
                            const prec *dg, const int grid_num) 
{
        if (!src->use || !buffer_is_device_ready(&src->buffer, step) ||
             src->lengths[grid_num] == 0) 
                return;

        prec *source_data = buffer_get_device_ptr(&src->buffer, step);
        cusource_add_curvilinear_H(&src->interpolation[grid_num], out,
                                   source_data, h, dt, f, ny, dg);
}

void source_add_force(prec *out, const prec *d1, source_t *src,
                      const size_t step, const prec h, const prec dt,
                      const prec quad_weight,
                      const prec *f, const int nx, const int ny, const int nz, 
                      const prec *dg,
                      const int grid_num) 
{
        if (!src->use || !buffer_is_device_ready(&src->buffer, step) ||
             src->lengths[grid_num] == 0) 
                return;

        prec *source_data = buffer_get_device_ptr(&src->buffer, step);
        cusource_add_force_H(&src->interpolation[grid_num], out,
                                   source_data, d1, h, dt, quad_weight, f, nx, ny, nz, dg);
}

