#include <stdio.h>
#include <time.h>
#include <math.h>
#include <errno.h>
#include <assert.h>

#include <awp/definitions.h>
#include <test/test.h>
#include <topography/sources/source.h>
#include <topography/sources/forces.h>
#include <topography/mapping.h>
#include <grid/shift.h>
#include <readers/input.h>
#include "interpolation/interpolation.h"

static int use;

// Force components
static source_t Fx;
static source_t Fy;
static source_t Fz;

// Density at force location
static float *d_rho_interp_x, *d_rho_interp_y, *d_rho_interp_z;

static input_t input;

static int myrank;

void interpolate_density(float **d_rho_interp, const float *d_rho,
                         const source_t *F, const grids_t *grids, const int degree);
void interpolate_density(float **d_rho_interp, const float *d_rho,
                         const source_t *F, const grids_t *grids, const int degree) {
        if (!F->use) return;

        cu_interp_t d_interp;
       // Interpolate density to the force location
       size_t num_bytes = sizeof(float) * F->lengths[0]; 
       cudaMalloc((void**)d_rho_interp, num_bytes); 

       grid3_t grid = grid_init_full_grid(grids->z.inner_size,
                                          grid_node(), grids->z.coordinate, grids->z.boundary1,
                                          grids->z.boundary2, grids->z.gridspacing);
        grid_data_t xyz;
        grid_data_init(&xyz, grid, 0);
        AWPCHK(cuinterp_init(&d_interp, xyz.x, xyz.y, xyz.z,
                                     grid, F->x[0], F->y[0], F->z[0],
                                     F->global_indices[0],
                                     F->lengths[0], degree));
        cuinterp_interp_H(&d_interp, *d_rho_interp, d_rho);
        cuinterp_finalize(&d_interp);

}

void forces_init(const char *filename, const grids_t *grids, const struct mapping *map, int ngrids,
                  const f_grid_t *f, const g_grid_t *g, const MPI_Comm comm, const int rank,
                  const int size, const float *rho, const int istopo) 
{
        use = strcmp(filename, "") != 0 ? 1 : 0;

        myrank = rank;

        if (!use) return;

       if (rank == 0) { 
               AWPCHK(input_init(&input, filename));
       }
       AWPCHK(input_broadcast(&input, rank, 0, comm));


       Fx = source_init("fx", SX, &input, grids, map, ngrids, f, rank, comm, FORCE);
       Fy = source_init("fy", SY, &input, grids, map, ngrids, f, rank, comm, FORCE);
       Fz = source_init("fz", SZ, &input, grids, map, ngrids, f, rank, comm, FORCE);

       if (Fx.use) AWPCHK(forces_boundary_check(&Fx));
       if (Fy.use) AWPCHK(forces_boundary_check(&Fy));
       if (Fz.use) AWPCHK(forces_boundary_check(&Fz));

       interpolate_density(&d_rho_interp_x, rho, &Fx, grids, input.degree);
       interpolate_density(&d_rho_interp_y, rho, &Fy, grids, input.degree);
       interpolate_density(&d_rho_interp_z, rho, &Fz, grids, input.degree);

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

        source_add_force(d_u1, d_rho_interp_x, &Fx, step, h, dt, qh, f->d_f_1, nx, ny, nz,
                         g->d_g3_c, grid_num, 0, 1);
        source_add_force(d_v1, d_rho_interp_y, &Fy, step, h, dt, qh, f->d_f_2, nx, ny, nz,
                         g->d_g3_c, grid_num, 0, 2);
        source_add_force(d_w1, d_rho_interp_z, &Fz, step, h, dt, q, f->d_f_c, nx, ny, nz,
                         g->d_g3, grid_num, 0, 3);
}

void forces_add_cartesian(prec *d_xz, prec *d_yz, prec *d_zz, const size_t step,
                const int nx, const int ny, const int nz, const prec h, const prec dt, const int grid_num) 
{
        
        if (!use) return;

        source_add_force(d_xz, d_rho_interp_x, &Fx, step, h, dt, 1.0, NULL, nx, ny, nz, NULL, grid_num, 1, 1);
        source_add_force(d_yz, d_rho_interp_y, &Fy, step, h, dt, 1.0, NULL, nx, ny, nz, NULL, grid_num, 1, 2);
        source_add_force(d_zz, d_rho_interp_z, &Fz, step, h, dt, 1.0, NULL, nx, ny, nz, NULL, grid_num, 1, 3);
}

void forces_add_cartesian_velocity(prec *d_vx, prec *d_vy, prec *d_vz, const size_t step,
                const int nx, const int ny, const int nz, const prec h, const prec dt, const int grid_num) 
{
        
        if (!use) return;

        source_add_force(d_vx, d_rho_interp_x, &Fx, step, h, dt, 1.0, NULL, nx, ny, nz, NULL, grid_num, 2, 1);
        source_add_force(d_vy, d_rho_interp_y, &Fy, step, h, dt, 1.0, NULL, nx, ny, nz, NULL, grid_num, 2, 2);
        source_add_force(d_vz, d_rho_interp_z, &Fz, step, h, dt, 1.0, NULL, nx, ny, nz, NULL, grid_num, 2, 3);
}

void forces_finalize(void)
{
        if (!use) return;

        source_finalize(&Fx);
        source_finalize(&Fy);
        source_finalize(&Fz);

        if (d_rho_interp_x != NULL) cudaFree(d_rho_interp_x);
        if (d_rho_interp_y != NULL) cudaFree(d_rho_interp_y);
        if (d_rho_interp_z != NULL) cudaFree(d_rho_interp_z);
}




