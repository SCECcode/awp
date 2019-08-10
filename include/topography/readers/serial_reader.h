#ifndef TOPOGRAPY_SERIAL_READER_H
#define TOPOGRAPY_SERIAL_READER_H
#include <awp/definitions.h>
/*
 * This module reads topography data from file (elevation map data).
 *
 */

/* Read topography data in serial. Each process will open the same file.
 *
 * Arguments:
 *      filename: File to open
 *      rank: MPI rank
 *      px: Number of processes in the x-direction
 *      py: Number of processes in the y-direction
 *      coord: MPI coordinate.
 *      nx: Number of grid points in the x-direction (local to each process)
 *      ny: Number of grid points in the y-direction (local to each process)
 *      alloc: Allocate memory for output data
 *      out: Topography data read from file.
 *
 * Return value:
 *      Error code.
 *
 */
int topo_read_serial(const char *filename, const int rank, const int px,
                     const int py, const int *coord, const int nx,
                     const int ny, const int alloc, prec **out);
#endif

