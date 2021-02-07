#include <stdio.h>
#include <time.h>
#include <math.h>
#include <errno.h>
#include <assert.h>

#include <awp/definitions.h>
#include <test/test.h>
#include <topography/sources/source.h>
#include <topography/sources/forces.h>
#include <grid/shift.h>
#include <readers/input.h>
#include "interpolation/interpolation.h"

static int use;

// Force components
static source_t Fx;
static source_t Fy;
static source_t Fz;

// Density at force location
static float *d_rho_interp;

static input_t input;

static int myrank;
static cu_interp_t d_interp;


int interpolate_density(float *out, const float *in,
                        const float *x, const float *y, const float *z,
                        grid3_t grid, const float *qx,
                        const float *qy, const float *qz, const int m, const int deg);


int interpolate_density(float *out, const float *in,
                        const float *x, const float *y, const float *z,
                        grid3_t grid, const float *qx,
                        const float *qy, const float *qz, const int m, const int deg) 
{
        int err = 0;
        prec *lx, *ly, *lz, *xloc, *yloc, *zloc;
        lx = calloc(sizeof(lx), (deg + 1));
        ly = calloc(sizeof(ly), (deg + 1));
        lz = calloc(sizeof(lz), (deg + 1));
        xloc = calloc(sizeof(xloc), (deg + 1));
        yloc = calloc(sizeof(yloc), (deg + 1));
        zloc = calloc(sizeof(zloc), (deg + 1));

        fcn_print_info(grid);
        printf("x = %g y = %g \n", qx[0], qy[0]);

        int ny = grid.size.y - 4 - ngsl2;
        int nz = grid.size.z - 2 * align;

        #define _rho(i, j, k)                                                   \
        in[(k) + align + (2 * align + nz) * (i) * (2 * ngsl + ny + 4) + \
           (2 * align + nz) * (j)]
        for (int q = 0; q < m; ++q) { 
                int ix = 0; int iy = 0; int iz = 0;
                err = interp_lagrange1_coef(
                    xloc, lx, &ix, x, grid.size.x, qx[q], deg);
                err = interp_lagrange1_coef(
                    yloc, ly, &iy, y, grid.size.y, qy[q], deg);
                err = interp_lagrange1_coef(
                    zloc, lz, &iz, z, grid_boundary_size(grid).z, qz[q], deg);
                out[q] = 0.0;
                printf("ix = %d iy = %d iz = %d \n", ix, iy, iz);
                for (int i = 0; i < deg + 1; ++i) {
                for (int j = 0; j < deg + 1; ++j) {
                for (int k = 0; k < deg + 1; ++k) {
                        int pos = align + iz + k + grid.line * (align + 10 + ngsl + iy + j) + grid.slice * (ix + i);
                        out[q] += lx[i] * ly[j] * lz[k] * in[pos];
                }
                }
                }
                printf("out = %g rho = %g \n", out[0], _rho(ix, iy, 107));
        }
        exit(-1);


        free(lx);
        free(ly);
        free(lz);
        free(xloc);
        free(yloc);
        free(zloc);

        return err;
}


void forces_init(const char *filename, const grids_t *grids, int ngrids,
                  const f_grid_t *f, const g_grid_t *g, const MPI_Comm comm, const int rank,
                  const int size, const float *rho) 
{
        use = strcmp(filename, "") != 0 ? 1 : 0;

        myrank = rank;

        if (!use) return;

       if (rank == 0) { 
               AWPCHK(input_init(&input, filename));
       }
       AWPCHK(input_broadcast(&input, rank, 0, comm));


       Fx = source_init("fx", SX, &input, grids, ngrids, f, rank, comm, FORCE);
       Fy = source_init("fy", SY, &input, grids, ngrids, f, rank, comm, FORCE);
       Fz = source_init("fz", SZ, &input, grids, ngrids, f, rank, comm, FORCE);

       if (Fx.use) AWPCHK(forces_boundary_check(&Fz));

       // Interpolate density to the force location
       size_t num_bytes = sizeof d_rho_interp * Fz.lengths[0]; 
       cudaMalloc((void**)&d_rho_interp, num_bytes); 

       grid3_t grid = grid_init_full_grid(grids->z.inner_size,
                                          grid_node(), grids->z.coordinate, grids->z.boundary1,
                                          grids->z.boundary2, grids->z.gridspacing);
        grid_data_t xyz;
        grid_data_init(&xyz, grid);
        AWPCHK(cuinterp_init(&d_interp, xyz.x, xyz.y, xyz.z,
                                     grid, Fz.x[0], Fz.y[0], Fz.z[0],
                                     Fz.global_indices[0],
                                     Fz.lengths[0], input.degree));
        cuinterp_interp_H(&d_interp, d_rho_interp, rho);
       //size_t num_bytes = sizeof rho_interp * Fz.lengths[0]; 
       //rho_interp = malloc(num_bytes);
       //F_interp = malloc(num_bytes);
       //grid3_t grid = grid_init_full_grid(grids->z.inner_size,
       //                                   grid_node(), grids->z.coordinate, grids->z.boundary1,
       //                                   grids->z.boundary2, grids->z.gridspacing);
       //grid1_t x_grid = grid_grid1_x(grid);
       //grid1_t y_grid = grid_grid1_y(grid);
       //grid1_t z_grid = grid_grid1_z(grid);
       //prec *x1 = malloc(sizeof x1 * x_grid.size);
       //prec *y1 = malloc(sizeof y1 * y_grid.size);
       //prec *z1 = malloc(sizeof z1 * z_grid.size);
       //grid_fill1(x1, x_grid);
       //grid_fill1(y1, y_grid);
       //grid_fill1(z1, z_grid);
       //metrics_interpolate_jacobian(f, F_interp, f->f, g->g3, x1, y1, z1,
       //                             grid, Fz.x[0], Fz.y[0], Fz.z[0], 
       //                             Fz.lengths[0], input.degree);
       //interpolate_density(rho_interp, rho, x1, y1, z1,
       //                    grid, Fz.x[0], Fz.y[0], Fz.z[0], 
       //                    Fz.lengths[0], input.degree);
       //cudaMalloc((void**)&d_F_interp, num_bytes); 
       //cudaMalloc((void**)&d_rho_interp, num_bytes); 
       //cudaMemcpy(d_F_interp, F_interp, num_bytes, cudaMemcpyHostToDevice);
       //cudaMemcpy(d_rho_interp, rho_interp, num_bytes, cudaMemcpyHostToDevice);
       //free(x1);
       //free(y1);
       //free(z1);

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

        // Interpolates the Jacobian to the force location (simpler, less accurate)
        //source_add_force(d_u1, d_rho_interp, &Fx, step, h, dt, qh, d_F_interp, nx, ny, nz,
        //                 g->d_g3_c, grid_num);
        //source_add_force(d_v1, d_rho_interp, &Fy, step, h, dt, qh, d_F_interp, nx, ny, nz,
        //                 g->d_g3_c, grid_num);
        //source_add_force(d_w1, d_rho_interp, &Fz, step, h, dt, q, d_F_interp, nx, ny, nz,
        //                 g->d_g3, grid_num);

        source_add_force(d_u1, d_rho_interp, &Fx, step, h, dt, qh, f->d_f_1, nx, ny, nz,
                         g->d_g3_c, grid_num);
        source_add_force(d_v1, d_rho_interp, &Fy, step, h, dt, qh, f->d_f_2, nx, ny, nz,
                         g->d_g3_c, grid_num);
        source_add_force(d_w1, d_rho_interp, &Fz, step, h, dt, q, f->d_f_c, nx, ny, nz,
                         g->d_g3, grid_num);
}

void forces_finalize(void)
{
        if (!use) return;

        source_finalize(&Fx);
        source_finalize(&Fy);
        source_finalize(&Fz);

        cuinterp_finalize(&d_interp);
}




