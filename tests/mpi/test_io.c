#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define ADDLINENUM 1
#define ADDRANK 1
#define RANK rank

#include <awp/definitions.h>
#include <buffers/buffer.h>
#include <test/test.h>
#include <test/check.h>
#include <test/array.h>
#include <mpi/io.h>

#define PRINT 0

int test_mpi_write(int rank, int size);
size_t init_indexed(int **indices, int **block_length, prec **data,
                    prec **global_data, size_t num_elements, size_t data_size,
                    int rank, int size);
int test_mpi_indexed_write(int rank, int size);
int test_mpi_read(int rank, int size);
int test_mpi_indexed_read(int rank, int size);

int main(int argc, char **argv)
{
        int err = 0;
        int rank, size;
        MPI_Init(&argc, &argv);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN); 
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (rank == 0) {
                test_divider();
                printf("Testing mpi/io.c\n");
        }

        err |= test_mpi_write(rank, size);
        err |= test_mpi_read(rank, size);
        err |= test_mpi_indexed_write(rank, size);
        err |= test_mpi_indexed_read(rank, size);

        if (rank == 0) {
                printf("Testing completed.\n");
                test_divider();
        }

        MPI_Finalize();

        return err;
}

int test_mpi_write(int rank, int size)
{
        int err = 0;
        MPI_Aint data_size = 4;
        size_t num_writes = 8;
        prec *data = malloc(sizeof data * data_size);

        array_range(data, data_size);

        test_t test = test_init("mpi_write", rank, size);

        mpi_io_t io =
            mpi_io_init(MPI_COMM_WORLD, rank, data_size);

        for (size_t i = 0; i < num_writes; ++i) {
                mpi_io_write(&io, data, "temp.bin");
        }

        if (rank == 0) {
                FILE *fh;
                size_t total_size = data_size * size * num_writes;
                prec *filedata = malloc(sizeof filedata * total_size);
                fh = fopen("temp.bin", "rb");
                CCHK(fread(filedata, sizeof(prec), total_size, fh) == 0);
                fclose(fh);
                remove("temp.bin");

                prec *ans = malloc(sizeof ans * total_size);

                for (size_t i = 0; i < (size_t)num_writes; ++i) {
                for (size_t j = 0; j < (size_t)size; ++j) {
                for (size_t k = 0; k < (size_t)data_size; ++k) {
                        ans[k + j * data_size + i * size * data_size ] = k;
                }
                }
                }

                err = (int)chk_inf(filedata, ans, total_size);

        }
        
        err |= test_finalize(&test, err);

        return err;
}

int test_mpi_read(int rank, int size)
{
        int err = 0;
        MPI_Aint data_size = 4;
        size_t num_writes = 8;
        prec *data = malloc(sizeof data * data_size);
        prec *read_data = malloc(sizeof read_data * data_size);

        array_range(data, data_size);

        test_t test = test_init("mpi_read", rank, size);

        mpi_io_t write =
            mpi_io_init(MPI_COMM_WORLD, rank, data_size);

        for (size_t i = 0; i < num_writes; ++i) {
                array_addc(data, data_size, i);
                mpi_io_write(&write, data, "temp.bin");
        }

        mpi_io_t read =
            mpi_io_init(MPI_COMM_WORLD, rank, data_size);

        array_range(data, data_size);
        for (size_t i = 0; i < num_writes; ++i) {
                array_addc(data, data_size, i);
                mpi_io_read(&read, read_data, "temp.bin");
                err |= (int)chk_inf(data, read_data, data_size);
        }

        if (rank == 0) {
                remove("temp.bin");
        }
        
        err |= test_finalize(&test, err);

        return test_last_error();
}

size_t init_indexed(int **indices, int **block_length, prec **data,
                    prec **global_data, size_t num_elements, size_t data_size,
                    int rank, int size) 
{
        *global_data = malloc(sizeof global_data * data_size);
        prec *owner = malloc(sizeof owner * data_size);

        for (size_t i = 0; i < data_size; ++i) {
                (*global_data)[i] = (prec)i;
                owner[i] = ((i+3) / size) % size ;
        }

        size_t block_count = 0;
        for (size_t i = 0; i < data_size; ++i) {
                if (owner[i] == rank) {
                        block_count++;
                }
        }

        *data = malloc(sizeof(prec) * block_count * num_elements);
        *indices = malloc(sizeof(int) * block_count);
        *block_length = malloc(sizeof(int) * block_count);

        int k = 0;
        for (size_t i = 0; i < data_size; ++i) {
                if (owner[i] == rank) {
                        for (size_t j = 0; j < num_elements; ++j) {
                                (*data)[j + k * num_elements] =
                                    (*global_data)[i];
                        }
                        (*indices)[k] = i * num_elements;
                        (*block_length)[k] = num_elements;
                        k++;
                }
        }

        free(owner);

        return block_count;

}

int test_mpi_indexed_write(int rank, int size)
{
        int err = 0;
        size_t data_size = 4;
        size_t num_elements = 2;
        size_t num_writes = 2;

        test_t test = test_init("mpi_indexed_write", rank, size);

        int *indices, *block_length;
        prec *data, *global_data;
        size_t block_count =
            init_indexed(&indices, &block_length, &data, &global_data,
                         num_elements, data_size, rank, size);

        if (PRINT) {
                inspect_ga(global_data, data_size);
                inspect_da(indices, block_count);
                inspect_da(block_length, block_count);
                inspect_ga(data, block_count * num_elements);
        }

        mpi_io_idx_t io =
            mpi_io_idx_init(MPI_COMM_WORLD, rank, indices, block_length,
                            block_count, num_writes);

        for (size_t i = 0; i < num_writes; ++i) {
                mpi_io_idx_write(&io, data, "temp.bin");
        }

        if (rank == 0) {
                FILE *fh;
                fh = fopen("temp.bin", "rb");
                prec *buf;
                size_t total_size = num_writes * num_elements * data_size;
                buf = calloc(sizeof(prec), total_size );
                CCHK(fread(buf, sizeof(prec), total_size, fh) == 0);
                fclose(fh);
                remove("temp.bin");
                if (PRINT) {
                        print("Data written to file:\n");
                        inspect_ga(buf, total_size);
                }

                prec *ans = malloc(sizeof(prec) * total_size);
                for (size_t i = 0; i < data_size; ++i) {
                for (size_t j = 0; j < num_elements; ++j) {
                for (size_t k = 0; k < num_writes; ++k) {
                        ans[k + j * num_writes +
                            i * num_elements * num_writes] = global_data[i];
                }
                }
                }

                if (PRINT) {
                        print("Expected output:\n");
                        inspect_ga(ans, total_size);
                }

                err |= (int)chk_inf(ans, buf, total_size);

                free(buf);
                free(ans);
        }

        mpi_io_idx_finalize(&io);


        err |= test_finalize(&test, err);

        free(global_data);
        free(data);
        free(indices);
        free(block_length);

        return 0;
}

int test_mpi_indexed_read(int rank, int size)
{
        int err = 0;
        size_t data_size = 4;
        size_t num_elements = 2;
        size_t num_writes = 2;

        test_t test = test_init("mpi_indexed_read", rank, size);

        int *indices, *block_length;
        prec *data, *global_data;
        size_t block_count =
            init_indexed(&indices, &block_length, &data, &global_data,
                         num_elements, data_size, rank, size);

        mpi_io_idx_t write =
            mpi_io_idx_init(MPI_COMM_WORLD, rank, indices, block_length,
                            block_count, num_writes);

        for (size_t i = 0; i < num_writes; ++i) {
                mpi_io_idx_write(&write, data, "temp.bin");
        }

        mpi_io_idx_finalize(&write);

        mpi_io_idx_t read =
            mpi_io_idx_init(MPI_COMM_WORLD, rank, indices, block_length,
                            block_count, num_writes);

        size_t read_data_size = block_count * num_elements;
        prec *read_data = malloc(sizeof read_data * read_data_size);
        for (size_t i = 0; i < num_writes; ++i) {
                mpi_io_idx_read(&read, read_data, "temp.bin");
                err |= (int)chk_inf(read_data, data, read_data_size);
        }

        mpi_io_idx_finalize(&read);

        err |= test_finalize(&test, err);

        if (rank == 0) {
                remove("temp.bin");
        }

        free(global_data);
        free(data);
        free(read_data);
        free(indices);
        free(block_length);

        return 0;
}

