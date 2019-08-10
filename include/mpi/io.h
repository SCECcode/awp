#ifndef MPI_IO_H
#define MPI_IO_H
#ifdef __cplusplus
extern "C" {
#endif
/* This module provides functions for reading and writing files using MPI IO.
 *
 * Usage:
 *
 *
 * Contiguous input and output
 * The functions mpi_io_.. are used for reading and writing data in which all
 * processes write or read data in a contiguous fashion.
 *
 *
 * File view:
 *
 *            ------------------------------------
 * Blocks     |  0   |      1      |  2  |   3   |
 *            ------------------------------------ 
 *
 *            Numbers indicate process ranks 
 *
 *  In the example above, each process writes one block of data to the file. The
 *  blocks are ordered by the process ranks. The order is not preserved when
 *  writing multiple times to the same file. Writing twice results in:
 *
 *  ------------------------------------------------------------------------
 *  |  0   |      1      |  2  |   3   |  0   |      1      |  2  |   3   |
 *  ------------------------------------------------------------------------
 *
 *  Each process is responsible for writing `data_size` number of floating point
 *  elements and writes them to the file with forming any gaps. The sections
 *  which each process writes to is ordered based on their rank.
 *
 *  Indexed input and output
 *  The functions mpi_io_idx_... are used for reading writing data in a strided
 *  fashion using MPI_Indexed. Each process is responsible for writing a certain
 *  number of blocks that can be of different lengths. The blocks are written to
 *  different locations in the file by specifying an index of each block.
 *
 * File view:
 *
 *           --------------------------------------
 * Blocks    | 0 | 1   | 2  | 3 |  2 |  2  |   0   |
 *           -------------------------------------- 
 *
 *           Numbers indicate process ranks 
 *
 * In the example above, process 0 is responsible for writing the first and last
 * block to the file. Indexed IO supports preserving block order between
 * consecutive writes. Specify the number of times to write to file upon
 * initialization (or read from file). 
 *
 *
 * First write:
 *           -----------------------------------------------
 *           |  0  |           |  1   |       |      |1|   |
 *           -----------------------------------------------
 * Second write:
 *           -----------------------------------------------
 *           |  0  |  0  |     |  1   |   1   |      |1|1| |
 *           -----------------------------------------------
 * Third write:
 *           -----------------------------------------------
 *           |  0  |  0  |  0  |  1   |   1   |   1  |1|1|1|
 *           -----------------------------------------------
 *
 * The example above shows a file that has been configured for writing three
 * times to it.
 *
 */ 

#include <mpi.h>

#include <awp/definitions.h>


/* Data structure for contiguous IO
 *
 * Members:
 *      num_elements: Number of elements to read/write for the given rank.
 *      total_num_bytes: Total number of bytes to read or write for all
 *              ranks.
 *      offset: Offset for the given rank in terms of the number of bytes.
 *      comm: MPI Communicator group
 *      rank: MPI rank
 */ 
typedef struct {
        MPI_Aint num_elements;
        MPI_Aint total_num_bytes;
        MPI_Offset offset;
        MPI_Comm comm;
        int rank;
} mpi_io_t;

/* Data structure for indexed IO
 *
 * Members:
 *      dtype: MPI Indexed data description.
 *      num_elements: Number of elements to read or write for the given rank.
 *      num_bytes: Number of bytes to read or write for the given rank.
 *      comm: MPI Communicator group
 *      rank: MPI rank
 */ 
typedef struct {
        MPI_Datatype dtype;
        MPI_Offset offset;
        MPI_Aint num_elements;
        MPI_Aint num_bytes;
        MPI_Comm comm;
        size_t num_writes;
        size_t current_write;
        int rank;
} mpi_io_idx_t;

/* Initialize contiguous IO.
 *
 * Arguments:
 *      comm: MPI communicator group
 *      rank: MPI process rank
 *      num_elements: Number of elements (floating-point numbers) to read/write
 *              for the given rank.
 */      
mpi_io_t mpi_io_init(MPI_Comm comm, int rank, MPI_Aint num_elements);

/* Write data to disk. 
 *
 * Arguments:
 * m: Contiguous IO data structure 
 * data: Data to write for the given process. Must contain at least `data_size`
 *      number of elements.
 * filename: Filename to write to including file extension. Data will be
 *      appended to the end of the file if the file already exists, and create
 *      if it does not exist.
 */
void mpi_io_write(mpi_io_t *m, prec *data, const char *filename);

/* Read data from disk. 
 *
 * Arguments:
 * m: Contiguous IO data structure 
 * data: Data buffer to read to for the given process. Must contain at least
 *      `m->num_elements` number of elements.
 * filename: Filename to read, including file extension. 
 */
void mpi_io_read(mpi_io_t *m, prec *data, const char *filename);

/* Initialize indexed IO.
 *
 * Arguments:
 *      comm: MPI communicator group
 *      rank: MPI process rank
 * indices: Index positions of each block for the given rank. Indices are
 *      specified in terms of element count.
 * blocklen: Length of each block for the given rank in terms of the number of
 *      elements. 
 * num_blocks: Number of blocks.     
 * num_writes: Number of times to write or read from file. 
 */      
mpi_io_idx_t mpi_io_idx_init(MPI_Comm comm, int rank, int *indices,
                int *blocklen, size_t num_blocks, size_t num_writes);

/* Write data to disk. 
 *
 * Arguments:
 * m: Indexed IO data structure 
 * data: Data to write for the given process. Must contain at least
 *      `m->num_elements` number of elements.
 * filename: Filename to write to including file extension. Data will be
 *      appended to the end of the file if the file already exists, and create
 *      if it does not exist.
 */
void mpi_io_idx_write(mpi_io_idx_t *m, prec *data, const char *filename);

/* Read data from disk. 
 *
 * Arguments:
 * m: Indexed IO data structure 
 * data: Data buffer to read to for the given process. Must contain at least
 *      `m->num_elements` number of elements.
 * filename: Filename to read, including file extension. 
 */
void mpi_io_idx_read(mpi_io_idx_t *m, prec *data, const char *filename);


// Free allocated memory
void mpi_io_idx_finalize(mpi_io_idx_t *m);

#ifdef __cplusplus
}
#endif
#endif

