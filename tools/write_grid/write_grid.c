/*
 * This program reads in a topography binary file and produces a binary file
 * that contains the grid coordinates (x_i, y_j, z_k) for each grid point in the
 * curvilinear grid.
 *
 */ 
#define VERSION_MAJOR 1
#define VERSION_MINOR 0
#define VERSION_PATCH 0

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#include <test/test.h>
#include <topography/readers/serial_reader.h>
#include <awp/definitions.h>

// Command line arguments
static int nx;
static int ny;
static int nz;
static prec h;
static int px;
static int py;
const char *input;
const char *output;

static int nvars = 3;

struct Mpi
{
  // MPI vars
  int rank;
  int size;
  int err;
  int dim[2];
  int period[2];
  int reorder;
  int nxt, nyt, nzt;
  int coord[2];
  MPI_Comm MCW, MC1;
};

void mpi_init(struct Mpi *m, int argc, char **argv);
void mpi_cart(struct Mpi *m, const int *size, const int *part);
MPI_Datatype data_type(const struct Mpi *mpi);

int main(int argc, char **argv)
{
        struct Mpi m;
        mpi_init(&m, argc, argv);

        if (argc <= 7 && m.rank == 0) {
                printf(
                    "usage: %s <input> <output> <nx> <ny> <nz> <h> <px> <py> "
                    "\n",
                    argv[0]);
                printf("AWP curvilinear grid writer, v%d.%d.%d\n",
                       VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
                printf("\n");
                printf("Args:\n");
                printf(" input          Topography binary file\n");
                printf(" output         Binary file to write\n");
                printf(" nx int         Number of grid points in the "
                    "x-direction\n");
                printf(" ny int         Number of grid points in the "
                    "y-direction\n");
                printf(" nz int         Number of grid points in the "
                    "z-direction\n");
                printf(" h float        Grid spacing\n");
                printf(" px int         Number of MPI partitions in "
                    "the x-direction\n");
                printf(" py int         Number of MPI partitions in "
                    "the y-direction\n");
                MPI_Finalize();
                return -1;
        }

        input = argv[1];
        output = argv[2];
        nx = atoi(argv[3]);
        ny = atoi(argv[4]);
        nz = atoi(argv[5]);
        h = atof(argv[6]);
        px = atoi(argv[7]);
        py = atoi(argv[8]);

        if (m.rank == 0) {
                printf("AWP curvilinear grid writer, v%d.%d.%d\n",
                       VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
                printf(
                    "input = %s output= %s nx = %d ny = %d nz = %d h = %g px = "
                    "%d py = "
                    "%d\n",
                    input, output, nx, ny, nz, h, px, py);
                int size = nvars * nx * ny * nz * sizeof(prec);
                printf("Expected file size: %d \n", size);
        }


        int size[3] = {nx, ny, nz};
        int part[2] = {px, py};
        int alloc = 1;
        int err = 0;
        prec *f;
        mpi_cart(&m, size, part);
        
        if (m.rank == 0 && m.nxt * px != nx) {
                fprintf(stderr,
                        "Number of grid points nx is not divisible by px.\n");
        }

        if (m.rank == 0 && m.nyt * py != ny) {
                fprintf(stderr,
                        "Number of grid points ny is not divisible by py.\n");
        }

        if (m.nxt * px != nx || m.nyt * py != ny) {
                MPI_Finalize();
                return -1;
        }

        err |= topo_read_serial(input, m.rank, px, py, m.coord, m.nxt, m.nyt,
                                alloc, &f);


        MPI_Datatype readtype = data_type(&m);

        MPI_File     fh;
        MPI_Status   filestatus;
        MPICHK(MPI_File_open(m.MCW, output, MPI_MODE_WRONLY | MPI_MODE_CREATE,
                            MPI_INFO_NULL, &fh));

        MPICHK(MPI_File_set_view(fh, 0, MPI_FLOAT, readtype, "native", 
                          MPI_INFO_NULL));

        float *buffer;
        int buffer_size = m.nxt * m.nyt * nvars;
        buffer = malloc(sizeof buffer * buffer_size);

        for (int j = 0; j < m.nyt; ++j) {
        for (int i = 0; i < m.nxt; ++i) {
                buffer[0 + nvars * i + j * nvars * m.nxt] = i * h;
                buffer[1 + nvars * i + j * nvars * m.nxt] = j * h;
        }
        }

        int show_info = (int) (nz / 10);
        show_info = show_info == 0 ? 1 : show_info;

        int len = m.nxt * m.nyt * nvars;
        if (m.rank == 0) printf("Processing...\n");
        for (int k = 0; k < nz; ++k) {
                // Depth,
                double H = (nz - 1) * h;
                for (int j = 0; j < m.nyt; ++j) {
                        for (int i = 0; i < m.nxt; ++i) {
                                // Evaluate the topography function f at
                                // position f(x_i, y_j) and compute the
                                // coordinate
                                // z_k = (H + f(x_i, y_j)) *( 1 - r_k) - H,
                                // where 0 <= r_k <= 1.
                                // This function maps to z = f(x_i, y_j) at the
                                // free surface and to z = -H at the bottom.
                                size_t lmy = 4 + m.nyt + 2 * ngsl + 2 * align;
                                size_t local_pos = 2 + align + (j + ngsl) +
                                                   (2 + i + ngsl) * lmy;
                                double rk = (double) k / (double) (nz - 1);
                                double mapping =
                                    (H + f[local_pos]) * (1 - rk) - H;
                                buffer[2 + nvars * i + j * nvars * m.nxt] =
                                        (prec)mapping;
                        }
                }
                MPICHK(MPI_File_write_all(fh, buffer, len, MPI_FLOAT,
                                          &filestatus));
                if (m.rank == 0)
                if (k % show_info == 0 && m.rank == 0)
                printf(" Slice z = %d out of nz = %d \n", k + 1, nz);
        }
        free(buffer);
        MPICHK(MPI_File_close(&fh));
        if (m.rank == 0) printf("done.\n");


        MPI_Finalize();
        return err;
}

void mpi_init(struct Mpi *m, int argc, char **argv)
{

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&m->rank);
  MPI_Comm_size(MPI_COMM_WORLD,&m->size);
  MPI_Comm_dup(MPI_COMM_WORLD, &m->MCW);
  m->coord[0] = 0;
  m->coord[1] = 0;

}


void mpi_cart(struct Mpi *m, const int *size, const int *part)
{
  m->nxt       = size[0] / part[0];
  m->nyt       = size[1] / part[1];
  m->nzt       = size[2];
  m->dim[0]    = part[0];
  m->dim[1]    = part[1];
  m->period[0] = 0;
  m->period[1] = 0;
  m->reorder   = 0;
  MPICHK(MPI_Cart_create(m->MCW, 2, m->dim, m->period, m->reorder, 
                             &m->MC1));
  MPICHK(MPI_Cart_coords(m->MC1, m->rank, 2, m->coord));
}


MPI_Datatype data_type(const struct Mpi *mpi)
{
  int old[3], new[3], offset[3];
  old[0] = nz;
  old[1] = ny;
  old[2] = nx * nvars;
  new[0] = nz;
  new[1] = mpi->nyt;
  new[2] = mpi->nxt * nvars;
  offset[0] = 0;
  offset[1] = mpi->nyt * mpi->coord[1];
  offset[2] = mpi->nxt * mpi->coord[0] * nvars;

  MPI_Datatype readtype;
  MPICHK(MPI_Type_create_subarray(3, old, new, offset, MPI_ORDER_C, 
                                 MPI_FLOAT, &readtype));
  MPICHK(MPI_Type_commit(&readtype));
  return readtype;
}
