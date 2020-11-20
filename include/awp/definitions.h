#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define BLOCK_SIZE_X 2
#define BLOCK_SIZE_Y 2
#define BLOCK_SIZE_Z 32

#include <mpi.h>

#ifndef _prec
#if DOUBLE_PRECISION
typedef double _prec;
#else
typedef float _prec;
#endif
#endif

#if DOUBLE_PRECISION
#define MPI_PREC MPI_DOUBLE
#else
#define MPI_PREC MPI_FLOAT
#endif


#if DOUBLE_PRECISION
typedef double prec;
#else
typedef float prec;
#endif

#ifndef ngsl
#define ngsl 4
#endif

#ifndef ngsl2
#define ngsl2 8
#endif

#ifndef align
#define align 32
#endif

#define FLTOL 1e-5
#define max(x, y) ((x) > (y) ? (x) : (y))
#define min(x, y) ((x) < (y) ? (x) : (y))

#define STR_LEN 2048



typedef struct
{
        _prec x, y, z;
} _prec3_t;

typedef struct
{
        int x, y, z;
} int3_t;

#define SWAP(x, y, T) do { T SWAP = x; x = y; y = SWAP; } while (0)


#endif

