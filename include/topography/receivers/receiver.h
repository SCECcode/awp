#ifndef RECEIVER_H
#define RECEIVER_H

#include <topography/sources/source.h>

typedef source_t recv_t;
recv_t receiver_init(const char *filename, 
                     const enum grid_types grid_type,
                     const input_t *input,
                     const grids_t *grids, 
                     const int ngrids,
                     const f_grid_t *f, 
                     const int rank,
                     const MPI_Comm comm);
void receiver_finalize(recv_t *recv);
void receiver_write(recv_t *recv, size_t step, const char *filename,
                const prec *in);
/* Determine filename of output file depending on which time step to write to
 * file and buffer settings etc. The closest time step not included in the file
 * is appended to the filename.
 *
 * Arguments:
 *      out: String to write filename to.
 *      base: Basename to append time step info to.
 *      gpu_buffer_size: Size of the GPU buffer.
 *      cpu_buffer_size: Size of the CPU buffer.
 *      num_writes: Number of times to write to the file.
 *      step: Current time step.
 *      stride: Skip every x steps.
 *      max: Maximum number of steps to take (used for determining number of
 *              leading zeros).
 *
 * Example:
 *      If `gpu_buffer_size=cpu_buffer_size=10`, `stride = 2`, `num_writes = 3`,
 *      `maxsteps = 2000` then the steps `0, 2, 4, 6, .. , 598` should be
 *      written to the same file. The name of this file as determined by this
 *      function is `base_0600`. The reason is that there are in total `10 * 10
 *      * 3` time steps to write, but only the every second step should be
 *      written. If `step = 600`, then the filename is `base_1200`.
 */
void receiver_filename(char *out, const char *base,
                       const size_t gpu_buffer_size,
                       const size_t cpu_buffer_size, const size_t num_writes,
                       const int stride, const size_t step,
                       const size_t maxsteps);

// Return step value for the function `receiver_filename` 
size_t receiver_step(const size_t gpu_buffer_size, const size_t cpu_buffer_size,
                     const size_t num_writes, const int stride,
                     const size_t step);

#endif

