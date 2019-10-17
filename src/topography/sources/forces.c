#include <stdio.h>
#include <time.h>
#include <math.h>
#include <errno.h>
#include <assert.h>

#include <awp/definitions.h>
#include <test/test.h>
#include <topography/sources/source.h>
#include <topography/sources/forces.h>
#include <readers/input.h>

static int use;

// Force components
static source_t Fx;
static source_t Fy;
static source_t Fz;

static input_t input;

static int myrank;

void forces_init(const char *filename, const grids_t *grids, int ngrids,
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


       Fx = source_init("fx", X, &input, grids, ngrids, f, rank, comm);
       Fy = source_init("fy", Y, &input, grids, ngrids, f, rank, comm);
       Fz = source_init("fz", Z, &input, grids, ngrids, f, rank, comm);

       AWPCHK(forces_boundary_check(&Fx));

}

int forces_boundary_check(const source_t *Fx)
{
       // Check that the source type is set to "1" so that the force is applied
       // on the boundary
       int type = 1;
       for (size_t i = 0; i < Fx->lengths[0]; ++i) {
               if (Fx->type[0][i] != type)
                       type = Fx->type[0][i];
        }
       if (type != 1)
               return ERR_INCOMPATIBLE_SOURCE_TYPE;
       return SUCCESS; 
}

void forces_read(size_t step)
{
        if (!use) return;

        source_read(&Fx, step);
        source_read(&Fy, step);
        source_read(&Fz, step);
}

void forces_add(prec *d_u1, prec *d_v1, prec *d_w1, const prec *d_d1, const size_t step,
                const prec h, const prec dt, const f_grid_t *f,
                const g_grid_t *g, const int grid_num) 
{
        if (!use) return;

        int nx = f->size[0];
        int ny = f->size[1];
        int nz = g->size;

        // SBP quadrature weights at the boundary (inverse norm coefficient Hi)
        // q : nodal grid
        // qh : node-centered grid
        // These weights must match SBP implementation. If the weights are
        // wrong, the amplitude will be wrong. 
        prec q = 3.55599789310935;
        prec qh = 2.9022824945274315;

        source_add_force(d_u1, d_d1, &Fx, step, h, dt, qh, f->d_f_1, nx, ny, nz,
                         g->d_g3_c, grid_num);
        source_add_force(d_v1, d_d1, &Fy, step, h, dt, qh, f->d_f_2, nx, ny, nz,
                         g->d_g3_c, grid_num);
        source_add_force(d_w1, d_d1, &Fz, step, h, dt, q, f->d_f_c, nx, ny, nz,
                         g->d_g3, grid_num);
}

void forces_finalize(void)
{
        if (!use) return;

        source_finalize(&Fx);
        source_finalize(&Fy);
        source_finalize(&Fz);
}

