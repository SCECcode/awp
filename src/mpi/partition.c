#include <mpi/partition.h>

int mpi_partition_2d(const int rank, const int *dim, const int *period,
                     int *coord, struct side_t *side, MPI_Comm *comm) 
{
        int err;
        int reorder = 1;
        err = MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, comm);
        err = MPI_Cart_shift(MPI_COMM_WORLD, 0, 1, &side->left, &side->right);
        err = MPI_Cart_shift(MPI_COMM_WORLD, 1, 1, &side->front, &side->back);
        err = MPI_Cart_coords(MPI_COMM_WORLD, rank, 2, coord);
        err = MPI_Barrier(MPI_COMM_WORLD);
        return err;
}

