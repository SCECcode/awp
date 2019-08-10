#include <stdio.h>
#include <stdlib.h>

#include <mpi/io.h>
#include <test/test.h>

mpi_io_t mpi_io_init(MPI_Comm comm, int rank, MPI_Aint num_elements)
{
        mpi_io_t out = {
            .comm = comm, .rank = rank, .num_elements = num_elements};
        MPICHK2(MPI_Allreduce(&out.num_elements, &out.total_num_bytes, 1,
               MPI_AINT, MPI_SUM, out.comm), rank);
        MPICHK2(MPI_Exscan(&out.num_elements, &out.offset, 1, MPI_OFFSET,
                          MPI_SUM, out.comm), rank);
        out.offset *= sizeof(prec);
        out.total_num_bytes *= sizeof(prec);
        out.rank = rank;

        return out;
}

void mpi_io_write(mpi_io_t *m, prec *data, const char *filename)
{
        MPI_File fh;
        MPI_Status filestatus;
        MPICHK2(MPI_File_open(m->comm, filename,
                             MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL,
                             &fh),
               m->rank);
        MPICHK2(MPI_File_write_at_all(fh,
                                     m->offset,
                                     data, m->num_elements, MPI_PREC,
                                     &filestatus),
               m->rank);
        m->offset += m->total_num_bytes;
        MPICHK2(MPI_File_close(&fh), m->rank);
}

void mpi_io_read(mpi_io_t *m, prec *data, const char *filename)
{
        MPI_File fh;
        MPI_Status filestatus;
        MPICHK2(MPI_File_open(m->comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL,
                             &fh),
               m->rank);
        MPICHK2(MPI_File_read_at_all(fh,
                                     m->offset,
                                     data, m->num_elements, MPI_PREC,
                                     &filestatus),
               m->rank);
        m->offset += m->total_num_bytes;
        MPICHK2(MPI_File_close(&fh), m->rank);
}

mpi_io_idx_t mpi_io_idx_init(MPI_Comm comm, int rank, int *indices,
                int *blocklen, size_t num_blocks, size_t num_writes)
{
        mpi_io_idx_t out = {.comm = comm, .rank = rank};
        int *offsets = malloc(sizeof(offsets) * num_blocks);
        out.num_bytes = 0;
        out.num_elements = 0;
        out.offset = 0;
        for (size_t i = 0; i < num_blocks; ++i) {
                offsets[i] = indices[i] * num_writes;
                out.num_elements += blocklen[i];
        }
        out.num_writes = num_writes;
        out.current_write = 0;
        out.num_bytes = blocklen[0] * sizeof(prec);

        MPICHK2(MPI_Type_indexed(num_blocks, blocklen, offsets, MPI_PREC,
                                &out.dtype),
               rank);
        MPI_Type_commit(&out.dtype);
        free(offsets);
        return out;
}

void mpi_io_idx_write(mpi_io_idx_t *m, prec *data, const char *filename)
{
        MPI_File fh;
        MPI_Status filestatus;
        MPICHK2(MPI_File_open(m->comm, filename,
                             MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL,
                             &fh),
               m->rank);
        MPICHK2(MPI_File_set_view(fh, m->offset, MPI_PREC, m->dtype, "native",
                                 MPI_INFO_NULL),
               m->rank);
        MPICHK2(MPI_File_write_all(fh, data, m->num_elements, MPI_PREC,
                                  &filestatus),
               m->rank);
        m->offset += m->num_bytes;
        m->current_write++;
        if (m->current_write == m->num_writes) {
                m->current_write = 0;
                m->offset = 0;
        }
        MPICHK2(MPI_File_close(&fh), m->rank);
}

void mpi_io_idx_read(mpi_io_idx_t *m, prec *data, const char *filename)
{
        MPI_File fh;
        MPI_Status filestatus;
        MPICHK2(MPI_File_open(m->comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL,
                             &fh),
               m->rank);
        MPICHK2(MPI_File_set_view(fh, m->offset, MPI_PREC, m->dtype, "native",
                                 MPI_INFO_NULL),
               m->rank);
        MPICHK2(MPI_File_read_all(fh, data, m->num_elements, MPI_PREC,
                                  &filestatus),
               m->rank);
        m->offset += m->num_bytes;
        if (m->current_write == m->num_writes) {
                m->current_write = 0;
                m->offset = 0;
        }
        MPICHK2(MPI_File_close(&fh), m->rank);
}

void mpi_io_idx_finalize(mpi_io_idx_t *m)
{
        MPICHK2(MPI_Type_free(&m->dtype), m->rank);
}

