/*
 * This program reads in a topography binary file and produces a binary file
 * that contains the grid coordinates (x_i, y_j, z_k) for each grid point in the
 * curvilinear grid.
 *
 */ 
#define VERSION_MAJOR 2
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
static int mz;
static prec h;
static int px;
static int py;
static int rpt;
const char *input;
const char *output;
const char *property;
const char *mesh;
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
MPI_Datatype data_type(const struct Mpi *mpi, int nz);

int main(int argc, char **argv)
{
        struct Mpi m;
        mpi_init(&m, argc, argv);

        if (argc < 12 && m.rank == 0) {
                printf(
                    "usage: %s <input> <output> <prop> <mesh> <nx> "
                    "<ny> <nz> <mz> <h> <px> <py> <rpt>\n",
                    argv[0]);
                printf("AWP curvilinear grid writer, v%d.%d.%d\n",
                       VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
                printf("\n");
                printf("Args:\n");
                printf(" input          Topography binary file\n");
                printf(
                    " output         Binary file to write containing "
                    "curvilinear grid coordinates\n");
                printf(
                    " property       Material property binary file to read\n");
                printf(
                    " mesh           Mesh binary file to write containing "
                    "material properties in the curvilinear grid \n");
                printf(" nx int         Number of grid points in the "
                    "x-direction\n");
                printf(" ny int         Number of grid points in the "
                    "y-direction\n");
                printf(" nz int         Number of grid points in the "
                    "z-direction\n");
                printf(" mz int         Number of grid points in the "
                    "z-direction of the property grid\n");
                printf(" h float        Grid spacing\n");
                printf(" px int         Number of MPI partitions in "
                    "the x-direction\n");
                printf(" py int         Number of MPI partitions in "
                    "the y-direction\n");
                printf(" rpt int        Whether repeat top layer when "
                    "write (1 = True, 0 = False) \n");
                printf(" Expect at least %d argc, got %d\n", 12, argc);

                MPI_Finalize();
                return -1;
        }

        input = argv[1];
        output = argv[2];
        property = argv[3];
        mesh = argv[4];
        nx = atoi(argv[5]);
        ny = atoi(argv[6]);
        nz = atoi(argv[7]);
        mz = atoi(argv[8]);
        h = atof(argv[9]);
        px = atoi(argv[10]);
        py = atoi(argv[11]);
        rpt = argc < 13 ? 1 : atoi(argv[12]);

        if (m.rank == 0) {
                printf("AWP curvilinear grid writer, v%d.%d.%d\n",
                       VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
                printf(
                    "input = %s output = %s property file = %s "
                    "mesh file = %s nx = %d ny = %d nz = %d "
                    "mz = %d h = %g px = %d py = %d rpt = %d\n",
                    input, output, property, mesh, nx, ny, nz, mz, h, px,
                    py, rpt);
                int size = nvars * nx * ny * nz * sizeof(prec);
                printf("Expected file size: %d \n", size);
                if (rpt > 1 || rpt < 0) {
                    printf("rpt should be either 0 (False) or "
                           "1 (True)\n");
                    MPI_Finalize();
                    return -1;
                }
                else if (rpt == 1) {
                    printf("The top layer will be repeated twice "
                           " in terms of properties\n");
                }
        }


        int size[3] = {nx, ny, nz};
        int part[2] = {px, py};
        int alloc = 1;
        int err = 0;
        int k0;
        char mpiErrStr[100];
        int mpiErrStrLen;

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


        MPI_Datatype readtype = data_type(&m, nz);
        MPI_Datatype readtype_m = data_type(&m, mz);

        MPI_File     fh, fm, fp;
        MPI_Status   filestatus;
        MPICHK(MPI_File_open(m.MCW, output, MPI_MODE_WRONLY | MPI_MODE_CREATE,
                            MPI_INFO_NULL, &fh));

        MPICHK(MPI_File_set_view(fh, 0, MPI_FLOAT, readtype, "native", 
                          MPI_INFO_NULL));
        
        MPICHK(MPI_File_open(m.MCW, mesh, MPI_MODE_WRONLY | MPI_MODE_CREATE,
                            MPI_INFO_NULL, &fm));
        MPICHK(MPI_File_set_view(fm, 0, MPI_FLOAT, readtype, "native", 
                          MPI_INFO_NULL));

        int buffer_size = m.nxt * m.nyt * nvars;
        float *buffer = (float*) calloc(buffer_size * nz, sizeof(float));

        for (int j = 0; j < m.nyt; ++j) {
        for (int i = 0; i < m.nxt; ++i) {
                buffer[0 + nvars * i + j * nvars * m.nxt] = i * h;
                buffer[1 + nvars * i + j * nvars * m.nxt] = j * h;
        }
        }

        float *prop = (float*) calloc(buffer_size * mz, sizeof(float));
        float *buffer_m = (float*) calloc(buffer_size * nz, sizeof(float));
        MPICHK(MPI_File_open(m.MCW, property, MPI_MODE_RDONLY,
                            MPI_INFO_NULL, &fp));
        
        err = MPI_File_set_view(fp, 0, MPI_FLOAT, readtype_m, "native", 
                                MPI_INFO_NULL);
        if (err != MPI_SUCCESS) {
            MPICHK(MPI_Error_string(err, mpiErrStr, &mpiErrStrLen));
            printf("%d) ERROR! MPI-IO reading property file set view: %s\n",
                        m.rank, mpiErrStr); 
        }
        err = MPI_File_read_all(fp, prop, buffer_size * mz, 
                        MPI_FLOAT, &filestatus);
        if (err != MPI_SUCCESS) {
            MPICHK(MPI_Error_string(err, mpiErrStr, &mpiErrStrLen));
            printf("%d) ERROR! MPI-IO reading property file read: %s\n",
                        m.rank, mpiErrStr);
        }

        int show_info = (int) (nz / 10);
        show_info = show_info == 0 ? 1 : show_info;
        double H = (nz - 1 - rpt) * h;

        int len = buffer_size * nz;
        if (m.rank == 0) printf("Processing...\n");

        for (int k = 0; k < nz; ++k) {
            // If k > 0 and we need repeat (rpt == 1), 
            // we shift the domain up by 1
            k0 = k == 0 ? k : k - rpt;  
            double rk = (double) k0 / (double) (nz - 1 - rpt);
            for (int i = 0; i < m.nxt; ++i) {
                for (int j = 0; j < m.nyt; ++j) {
                    size_t lmy = 4 + m.nyt + 2 * ngsl + 2 * align;
                    size_t local_pos = 2 + align + (j + ngsl) +
                                       (2 + i + ngsl) * lmy;
                    // Depth, k=0 is the surface
                    double mapping =
                        (H + f[local_pos]) * (1 - rk) - H;
                    buffer[2 + nvars * i + j * nvars * m.nxt] =
                            (prec)mapping;
                    
                    // For fp reading and mesh writing, we  start from the
                    // the surface, to keep compatible with the queried
                    // by depth CVM.
                    // k = 0  <--->  idx_z = 0
                    // idx_z should not exceed mz
                    int idx_z = (int) rintf((f[local_pos] - mapping)/ h);
                    if (idx_z >= mz) {
                        printf("Error! Curvilinear grids deeper than property "
                                "mesh\nAborting...\n");
                        MPI_Finalize();
                        return(-1);
                    }
                    size_t pos = nvars * (idx_z * m.nxt * m.nyt +
                                          j * m.nxt + i);
                    memcpy(buffer_m + nvars * (k * m.nxt * m.nyt +  
                                               j * m.nxt + i),
                           prop + pos, nvars * sizeof(float));
                }
            }
                MPICHK(MPI_File_write_all(fh, buffer, buffer_size, MPI_FLOAT,
                                          &filestatus));
        }
        MPICHK(MPI_File_write_all(fm, buffer_m, len,  MPI_FLOAT,
                                  &filestatus));
        free(buffer);
        MPICHK(MPI_File_close(&fh));
        free(buffer_m);
        MPICHK(MPI_File_close(&fm));
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


MPI_Datatype data_type(const struct Mpi *mpi, int num_z)
{
  int old[3], new[3], offset[3];
  old[0] = num_z;
  old[1] = ny;
  old[2] = nx * nvars;
  new[0] = num_z;
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
