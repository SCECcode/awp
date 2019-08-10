#ifndef INPUT_H
#define INPUT_H
#ifdef __cplusplus
extern "C" {
#endif
#define INPUT_MAJOR 1
#define INPUT_MINOR 0
#define INPUT_PATCH 0
#define INPUT_DATA_STRING_LENGTH 2048

#include <mpi.h>

#include <awp/definitions.h>
#include <readers/version.h>
/* Input files are specified in the ASCII file format. Each input file contains
 * two sections, the header section and the body section. The header section
 * begins with the version number, followed by key, value argument pairs, one
 * per row. The body section starts with the keyword "coordinates".
 *
 * Example:
 * 1.0.0
 * file=output
 * length=3
 *
 * coordinates
 * 1.0 0.0 0.0
 * 0.0 1.0 0.0
 * 0.0 0.0 1.0
 *
 *
 * Header:
 *  version: Version number.
 *  file: Reference to binary data file to write, or read.
 *  length: Number of elements in the body section (rows)
 *  gpu_buffer_size: Size of the gpu buffer in [length]. In other words, number
 *      of times to write to buffer before it is full. 
 * cpu_buffer_size: Size of the cpu buffer in [gpu_buffer_size]. In other words,
 *      number of times of times to copy the gpu buffer into the cpu buffer 
 *      before the cpu buffer becomes full.
 * steps: Number of time steps to read from binary file. 
 * stride: Number of steps to skip each time before writing to file. If
 *      stride=3, then output is written at steps = 1, 3, 6, 9, ... .
 * degree: Degree of interpolating polynomial. Use degree = 0 to use the nearest
 *      grid point.
 * system: Type of coordinate system to use. 
 * dimension: Dimensionality of input/output. This value is equal to the number
 *      of columns in the data section. dimension = 2 is for surface
 *      input/output.
 * num_components: Number of components of input/output data. Velocity output is
 * three components, and stress is six components.
 *
 * Body:
 *  x, y, z : Coordinate (float)
 */

typedef struct
{
        // Header section
        version_t version;
        char file[INPUT_DATA_STRING_LENGTH];
        size_t length;
        size_t gpu_buffer_size;
        size_t cpu_buffer_size;
        size_t steps;
        size_t num_writes;
        int stride;
        int degree;
        int system;
        int dimension;
        int num_components;
        // Body section
        prec *x, *y, *z;

} input_t;

/*
 * Default initialization of input data structure
 */ 
int input_init_default(input_t *out);

/*
 * Initialize input by reading the ASCII configuration file
 * and populate the input_t data structure.
 */ 
int input_init(input_t *out, const char *filename);

// Write input data structure to disk.
int input_write(input_t *out, const char *filename);

/*
 * Check if a file is readable (if it is exists).
 *
 * Input arguments:
 *        filename: Filename to file to test.
 *
 * Return value: 
 *        0 on SUCCESS. Otherwise an error code.
 */
int input_file_readable(const char *filename);

/*
 * Check if a file is writeable (can be created).
 * It is not possible to write to a file if its is located in a directory that
 * does not exist.
 *
 * Input arguments:
 *       filename: Filename to file to test.
 *
 * Return value: 
 *      0 on SUCCESS. Otherwise an error code.
 */
int input_file_writeable(const char *filename);

int input_parse(input_t *out, const char *line);

/*
 * Parse input arguments of the form "variable=name"
 
 * Input arguments:
 *       variable: pointer to write parsed variable name to.
 *       value: pointer to write parsed variable value to.
 *       line: string to parse.
 *
 * Return value: 
 *      0 on SUCCESS. Otherwise an error code.
 */ 
int input_parse_arg(char *variable, char *name, const char *line);

// Compare if two header data descriptions are the same
// Return 1 they are equal, 0 otherwise.
int input_equals(input_t *a, input_t *b);

int input_broadcast(input_t *out, int rank, int root, MPI_Comm communicator);

/* Check that all header arguments are valid.
 *
 * Return value:
 *      SUCCESS if all arguments are OK. Otherwise returns an error code for the
 *      first invalid argument encountered.
 */ 
int input_check_header(const input_t *out);

// Finalize. Call this function to perform cleanup when done.
void input_finalize(input_t *out);
#ifdef __cplusplus
}
#endif
#endif 

