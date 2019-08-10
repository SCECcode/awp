#pragma once
#ifndef MPI_PARTITION_H
#define MPI_PARTITION_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

struct side_t {
        int left;
        int right;
        int front;
        int back;
};

int mpi_partition_2d(const int rank, const int *dim, const int *period,
                     int *coord, struct side_t *side, MPI_Comm *comm);


#ifdef __cplusplus
}
#endif
#endif
