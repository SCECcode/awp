#include <stdio.h>
#include <time.h>
#include <math.h>
#include <errno.h>
#include <assert.h>

#include <awp/definitions.h>
#include <test/test.h>
#include <topography/sources/source.h>
#include <topography/sources/sources.h>
#include <readers/input.h>

static int use;

// Moment tensors
static source_t Mxx;
static source_t Myy;
static source_t Mzz;
static source_t Mxy;
static source_t Mxz;
static source_t Myz;

static input_t input;

static int myrank;

void sources_init(const char *filename, const grids_t *grids, int ngrids,
                  const f_grid_t *f, const MPI_Comm comm, const int rank,
                  const int size) 
{
        use = strcmp(filename, "") != 0 ? 1 : 0;

        myrank = rank;

        if (!use) return;

        // FIXME: Add support for multiple grids

       if (rank == 0) { 
               AWPCHK(input_init(&input, filename));
       }
       AWPCHK(input_broadcast(&input, rank, 0, comm));


       Mxx = source_init("xx", XX, &input, grids, ngrids, f, rank, comm);
       Myy = source_init("yy", YY, &input, grids, ngrids, f, rank, comm);
       Mzz = source_init("zz", ZZ, &input, grids, ngrids, f, rank, comm);
       Mxy = source_init("xy", XY, &input, grids, ngrids, f, rank, comm);
       Mxz = source_init("xz", XZ, &input, grids, ngrids, f, rank, comm);
       Myz = source_init("yz", YZ, &input, grids, ngrids, f, rank, comm);
}

void sources_read(size_t step)
{
        if (!use) return;

        source_read(&Mxx, step);
        source_read(&Myy, step);
        source_read(&Mzz, step);
        source_read(&Mxy, step);
        source_read(&Mxz, step);
        source_read(&Myz, step);
}

void sources_add_cartesian(prec *d_xx, prec *d_yy, prec *d_zz, prec *d_xy,
                           prec *d_xz, prec *d_yz, const size_t step,
                           const prec h, const prec dt, const int grid_num) 
{
        if (!use) return;

        source_add_cartesian(d_xx, &Mxx, step, h, dt, grid_num);
        source_add_cartesian(d_yy, &Myy, step, h, dt, grid_num);
        source_add_cartesian(d_zz, &Mzz, step, h, dt, grid_num);
        source_add_cartesian(d_xy, &Mxy, step, h, dt, grid_num);
        source_add_cartesian(d_xz, &Mxz, step, h, dt, grid_num);
        source_add_cartesian(d_yz, &Myz, step, h, dt, grid_num);
}

void sources_add_curvilinear(prec *d_xx, prec *d_yy, prec *d_zz, prec *d_xy,
                           prec *d_xz, prec *d_yz, 
                           const size_t step,
                           const prec h, const prec dt,
                           const f_grid_t *f, const g_grid_t *g, 
                           const int grid_num) 
{
        if (!use) return;

        int ny = f->size[1];
        // last argument specifies if the grid is cell-centered in the z-direction
        source_add_curvilinear(d_xx, &Mxx, step, h, dt, f->d_f_c, ny, g->d_g3_c,
                               grid_num, 1);
        source_add_curvilinear(d_yy, &Myy, step, h, dt, f->d_f_c, ny, g->d_g3_c,
                               grid_num, 1);
        source_add_curvilinear(d_zz, &Mzz, step, h, dt, f->d_f_c, ny, g->d_g3_c,
                               grid_num, 1);
        source_add_curvilinear(d_xy, &Mxy, step, h, dt, f->d_f, ny, g->d_g3_c,
                               grid_num, 1);
        source_add_curvilinear(d_xz, &Mxz, step, h, dt, f->d_f_1, ny, g->d_g3,
                               grid_num, 0);
        source_add_curvilinear(d_yz, &Myz, step, h, dt, f->d_f_2, ny, g->d_g3,
                               grid_num, 0);
}

source_t sources_get_source(enum grid_types grid_type)
{
        switch (grid_type)
        {
                case XX:
                        return Mxx;
                        break;
                case YY:
                        return Myy;
                        break;
                case ZZ:
                        return Mzz;
                        break;
                case XY:
                        return Mxy;
                        break;
                case XZ:
                        return Mxz;
                        break;
                case YZ:
                        return Myz;
                        break;
                case SX:
                        fprintf(stderr, "No source can exist on grid SX\n");
                        break;
                case SY:
                        fprintf(stderr, "No source can exist on grid SY\n");
                        break;
                case SZ:
                        fprintf(stderr, "No source can exist on grid SZ\n");
                        break;
                case X:
                        fprintf(stderr, "No source can exist on grid X\n");
                        break;
                case Y:
                        fprintf(stderr, "No source can exist on grid Y\n");
                        break;
                case Z:
                        fprintf(stderr, "No source can exist on grid Z\n");
                        break;
                case NODE:
                        fprintf(stderr, "No source can exist on grid NODE\n");
                        break;
        }

        return Mxx; 
}

void sources_finalize(void)
{
        if (!use) return;

        source_finalize(&Mxx);
        source_finalize(&Myy);
        source_finalize(&Mzz);
        source_finalize(&Mxy);
        source_finalize(&Mxz);
        source_finalize(&Myz);
}

