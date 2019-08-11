#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#include <awp/definitions.h>
#include <test/array.h>
#include <test/test.h>
#include <test/check.h>

int main(int argc, char **argv)
{
        int rank, size;
        MPI_Init(&argc, &argv);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN); 
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int err = 0;
        int *data, *owner, *updated_data;
        int num_elements = 2;
        int n = 10;
        
        data  = malloc(sizeof(int) * n);
        owner = malloc(sizeof(int) * n);

        test_t test = test_init("MPI Indexed test", rank, size);

        for (int i = 0; i < n; ++i) {
                data[i] = i;
        }

        for (int i = 0; i < n; ++i) {
                owner[i] = ((i+3) / size) % size ;
        }

        if (rank == 0) {
                print_master("Initial data array:\n");
                inspect_da(data, 10);
                print_master("Data owner array:\n");
                inspect_da(owner, 10);
        }

        // Determine number of blocks this process owns
        int blocks_per_rank = 0;
        for (int i = 0; i < n; ++i) {
                if (owner[i] == rank) {
                        blocks_per_rank++;
                }
        }
        
        // Determine the number of blocks per process. This information is known
        // to all processes
        int* block_count = calloc(sizeof(int), size);
        for (int i = 0; i < size; ++i) {
                for (int j = 0; j < n; ++j) { 
                        if (owner[j] == i) {
                                block_count[i]++;
                        }
                }
        }

        inspect_d(blocks_per_rank);
        inspect_da(block_count, size);

        int m = blocks_per_rank * num_elements;
        updated_data = malloc(sizeof(int) * blocks_per_rank * num_elements);
        int *displacements = malloc(sizeof(int) * blocks_per_rank);
        int *block_length = malloc(sizeof(int) * blocks_per_rank);

        // Fill array with data owned by each process,
        // and setup displacements and block lengths per process
        int k = 0;
        for (int i = 0; i < n; ++i) {
                if (owner[i] == rank) {
                        for (int j = 0; j < num_elements; ++j) {
                                updated_data[j + k * num_elements] = data[i];
                        }
                        displacements[k] = i * num_elements;
                        block_length[k] = num_elements;
                        k++;
                }
        }
        
        inspect_da(displacements, blocks_per_rank);
        inspect_da(block_length, blocks_per_rank);

        print_master("Updated data array\n");
        inspect_da(updated_data, blocks_per_rank * num_elements);

        MPI_Datatype indextype;
        MPICHK(MPI_Type_indexed(blocks_per_rank, block_length, displacements,
                                MPI_INT, &indextype));
        MPI_Type_commit(&indextype);

        MPI_File fh;
        MPI_Status filestatus;

        MPICHK(MPI_File_open(MPI_COMM_WORLD, "temp.bin",
                             MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL,
                             &fh));
        MPICHK(MPI_File_set_view(fh, 0, MPI_INT, indextype, "native",
                                 MPI_INFO_NULL));
        MPICHK(MPI_File_write_all(fh, updated_data, m, MPI_INT, &filestatus));
        MPICHK(MPI_File_close(&fh));

        if (rank == 0) {
                FILE *fh;
                fh = fopen("temp.bin", "rb");
                int *buf;
                buf = calloc(sizeof(int), n * num_elements);
                CCHK(fread(buf, sizeof(int), n * num_elements, fh) == 0);
                fclose(fh);
                remove("temp.bin");
                print("Data written to file:\n");
                inspect_da(buf, n * num_elements);


                int *ans = malloc(sizeof(int) * n * num_elements);
                for (int i = 0; i < n; ++i) {
                        for (int j = 0; j < num_elements; ++j) {
                                ans[j + i * num_elements] = i;
                        }
                }
                print("Expected output:\n");
                inspect_da(ans, n * num_elements);

                err |= chk_infi(ans, buf, n * num_elements);

                free(buf);
                free(ans);
        }

        err |= test_finalize(&test, err);


        MPICHK(MPI_Type_free(&indextype));

        free(data);
        free(owner);
        free(updated_data);
        free(block_count);
        free(displacements);
        free(block_length);

        print_master("Testing completed.\n");
        test_divider();
        
        MPI_Finalize();

        return err;
}

