#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#include <topography/receivers/receiver.h>
#include <buffers/buffer.h>
#include <grid/shift.h>
#include <utils/array.h>
#include <mpi/distribute.h>
#include <interpolation/interpolation.cuh>
#include <topography/sources/source.cuh>

void receiver_init_indexed(recv_t *recv, const input_t *input,
                           size_t num_reads);

recv_t receiver_init(const char *filename, 
                     const enum grid_types grid_type,
                     const input_t *input,
                     const grids_t *grids, 
                     const int ngrids,
                     const f_grid_t *f, 
                     const int rank,
                     const MPI_Comm comm) 
{
        recv_t recv;

        strcpy(recv.filename, filename);

        source_init_common(&recv, filename, grid_type, input, grids, ngrids, f,
                           rank, comm, RECEIVER);

        if (!recv.use) {
                return recv;
        }

        receiver_init_indexed(&recv, input, input->num_writes);
        recv.io = mpi_io_idx_init(recv.comm, rank, recv.offsets, recv.blocklen,
                                     recv.length, input->num_writes);
        return recv;
}

void receiver_finalize(recv_t *recv)
{
        source_finalize(recv);
}

void receiver_write(recv_t *recv, size_t step, const char *filename,
                const prec *in, const int grid_num)
{
        if (!recv->use)
                return;

        if (recv->lengths[grid_num] != 0 &&
            buffer_is_device_ready(&recv->buffer, step)) {
                prec *d_ptr = buffer_get_device_ptr(&recv->buffer, step);
                cuinterp_interp_H(&recv->interpolation[grid_num], d_ptr, in);
        }

        if (grid_num + 1 == recv->ngrids &&
            buffer_is_device_full(&recv->buffer, step)) {
                buffer_copy_to_host(&recv->buffer, step);
        }

        if (grid_num + 1 == recv->ngrids && buffer_is_host_full(&recv->buffer, step)) {
                prec *host_ptr = recv->buffer.h_buffer;
                // Transpose data from (time, index) to (index, time)
                // (last index is contiguous)
                size_t cols = recv->length;
                size_t rows = recv->buffer.num_host * recv->buffer.num_device;
                array_transpose(recv->host_buffer_extra, host_ptr, rows, cols);
                SWAP(recv->host_buffer_extra, recv->buffer.h_buffer, prec *);
                mpi_io_idx_write(&recv->io, recv->buffer.h_buffer, filename);
        }
}

void receiver_init_indexed(recv_t *recv, const input_t *input,
                           size_t num_writes) 
{
        if (!recv->use) return;
        recv->blocklen = malloc(sizeof(recv->blocklen) * input->length);
        recv->offsets = malloc(sizeof(recv->offsets) * input->length);
        size_t num_elements = input->cpu_buffer_size * input->gpu_buffer_size;
        recv->num_elements = num_elements;
        for (size_t i = 0; i < recv->length; ++i) {
                recv->blocklen[i] = num_elements;
        }
        for (size_t i = 0; i < recv->length; ++i) {
                recv->offsets[i] = recv->indices[i] * num_elements;
        }
}

void receiver_filename(char *out, const char *base,
                       const size_t gpu_buffer_size,
                       const size_t cpu_buffer_size, const size_t num_writes,
                       const int stride, const size_t step,
                       const size_t maxsteps)
{
        size_t filestep = receiver_step(gpu_buffer_size, cpu_buffer_size,
                                        num_writes, stride, step);
        int leading_zeros = ceil(log10((double)maxsteps)) + 1;
        sprintf(out, "%s_%0*ld", base, leading_zeros, filestep);
}

size_t receiver_step(const size_t gpu_buffer_size, const size_t cpu_buffer_size,
                     const size_t num_writes, const int stride,
                     const size_t step)
{
        size_t unit = gpu_buffer_size * cpu_buffer_size * num_writes * stride;
        size_t filestep = (1 + step / unit) * unit;
        return filestep;
}

