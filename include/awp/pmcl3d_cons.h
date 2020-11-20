#ifndef DEFINITIONS_H
#define BLOCK_SIZE_X 2
#define BLOCK_SIZE_Y 2
#define BLOCK_SIZE_Z 32
#endif
// Set floating-point precision. Make sure to configure both `_prec` and
// `_mpi_prec`.
// FIXME: The mesh reader can currently only run in single precision mode.
#if DOUBLE_PRECISION
#define _prec double
#define _mpi_prec MPI_DOUBLE
#else
#define _prec float
#define _mpi_prec MPI_FLOAT
#endif
#define align 32
#define loop  1 
// Do not change the number of ghost cells.
#define ngsl 4     /* number of ghost cells x loop */
#define ngsl2 8  /* ngsl * 2 */

#define Both  0
#define Left  1
#define Right 2
#define Front 3
#define Back  4

#define NEDZ_EP 160 /*max k to save final plastic strain*/

/*
 * Intercept CUDA errors. Usage: pass the CUDA
 * library function call to this macro. 
 * For example, CUCHK(cudaMalloc(...));
 * This check will be disabled if the preprocessor macro NDEBUG is defined (same
 * macro that disables assert() )
 */
#ifndef TEST_H
#ifndef CUCHK
#ifndef NDEBUG
#define CUCHK(call) {                                                         \
  cudaError_t err = call;                                                     \
  if( cudaSuccess != err) {                                                   \
  fprintf(stderr, "CUDA error in %s:%i %s(): %s.\n",                          \
          __FILE__, __LINE__, __func__, cudaGetErrorString(err) );            \
  fflush(stderr);                                                             \
  exit(EXIT_FAILURE);                                                         \
  }                                                                           \
}                                                                             
#else
#define CUCHK(call) {}
#endif
#endif
#endif

// intercept MPI errors. Same usage as for CUCHK
#ifndef NDEBUG
#define MPICHK(err) {                                                         \
 if (err != MPI_SUCCESS) {                                                    \
 char error_string[2048];                                                     \
 int length_of_error_string;                                                  \
 MPI_Error_string((err), error_string, &length_of_error_string);              \
 fprintf(stderr, "MPI error: %s:%i %s(): %s\n",                               \
         __FILE__, __LINE__, __func__, error_string);                         \
 MPI_Abort(MPI_COMM_WORLD, err);                                              \
 fflush(stderr);                                                              \
 exit(EXIT_FAILURE);                                                          \
}                                                                             \
}
#else
#define MPICHK(err) {}
#endif

//precompiles variables for DM
#define MAXGRIDS 10

/*order in which variables are stored in swap buffers for transition zone
DO NOT CHANGE */
#define sbvpos_u1 0
#define sbvpos_v1 1
#define sbvpos_w1 2
#define sbvpos_xx 3
#define sbvpos_yy 4
#define sbvpos_zz 5
#define sbvpos_xy 6
#define sbvpos_xz 7
#define sbvpos_yz 8

//WEDMI window length = 2*WWL + 1
#define WWL 2

//HIGHEST ORDER OF FILTER is MAXFILT-1
#define MAXFILT 20

#define MPIRANKIO 400000

// IN_FILE_LEN : Maximum number of characters in the input filenames 
#define IN_FILE_LEN 50

