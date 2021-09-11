#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define STR_LEN 2048
#define ADDLINENUM 1
#define ADDRANK 1
#define RANK rank
#define STR_LEN 2048

#include <awp/definitions.h>
#include <test/test.h>
#include <awp/error.h>
#include <utils/array.h>
#include <grid/shift.h>
#include <topography/grids.h>
#include <topography/sources/source.h>
#include <topography/sources/sources.h>
#include <topography/receivers/receivers.h>
#include <readers/input.h>

int main(int argc, char **argv)
{
        int rank, size;
        MPI_Init(&argc, &argv);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN); 
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        test_divider();
        printf("Testing test_source_distribution.c\n");

}
