#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <awp/definitions.h>
#include <awp/error.h>
#include <grid/grid_3d.h>
#include <topography/geometry/geometry.h>
#include <topography/topography.h>
#include <topography/readers/serial_reader.h>
#include <topography/topography.cuh>
#include <test/test.h>

topo_t topo_init(const int USETOPO, 
                 const char *INTOPO, 
                 const int rank,
                 const int x_rank_l,
                 const int x_rank_r,
                 const int y_rank_f,
                 const int y_rank_b,
                 const int coord[2],
                 int px,
                 int py,
                 int nxt,
                 int nyt,
                 int nzt,
                 const _prec dt,
                 const _prec h,
                 cudaStream_t stream_1,
                 cudaStream_t stream_2,
                 cudaStream_t stream_i
                     )
{

        int mxt;
        int myt;
        int mzt;
        topo_set_memory_size(nxt, nyt, nzt, &mxt, &myt, &mzt);
        int gridsize = mxt * myt * mzt;
        int slice = myt * mzt;
        int line = mzt;
        int slice_gl = ngsl * mzt;

        topo_t T = {.use = USETOPO, .dbg = TOPO_DBG, 
                    .verbose = TOPO_VERBOSE,
                    .rank = rank, 
                    .topography_file = INTOPO,
                    .x_rank_l = x_rank_l, .x_rank_r = x_rank_r,
                    .y_rank_f = y_rank_f, .y_rank_b = y_rank_b,
                    .coord = {coord[0], coord[1]},
                    .px = px, .py = py,
                    .nx = nxt, .ny = nyt, .nz = nzt, 
                    .mx = mxt, .my = myt, .mz = mzt, 
                    // Alignment by 2 is for extra padding needed for absorbing
                    // boundary layers
                    .off_x = {2, 2 + ngsl, 2 + ngsl + nxt, 2 + ngsl2 + nxt},
                    .off_y = {2, 2 + ngsl, 2 + ngsl + nyt, 2 + ngsl2 + nyt},
                    .off_z = {0, align, align + nzt, 2*align + nzt},
                    // Grid affinity
                    .sxx = {0, 1, 1}, .syy = {0, 1, 1}, .szz = {0, 1, 1},
                    .sxy = {1, 0, 1}, .sxz = {1, 1, 0}, .syz = {0, 0, 0},
                    .su1 = {1, 1, 1}, .sv1 = {0, 0, 1}, .sw1 = {0, 1, 0},
                    .gridsize = gridsize,
                    .slice = slice, .line = line,
                    .slice_gl = slice_gl,
                    .dth = dt/h,
                    .timestep = 1,
                    .gridspacing = h,
                    .stream_1 = stream_1,
                    .stream_2 = stream_2,
                    .stream_i = stream_i
                   };

        if (rank == 0 && T.verbose && T.use) printf("Topography:: enabled\n");
        if (T.dbg && rank == 0 && T.use)
                printf("Topography:: debugging enabled\n");

        if (T.dbg && rank == 0 && T.use)
                printf("Topography block size:: %d %d %d\n", TBX, TBY, TBZ);

        topo_set_bounds(&T);


        int3_t boundary1 = {.x = 0, .y = 0, .z = 0};
        int3_t boundary2 = {.x = 0, .y = 0, .z = 1};
        int3_t coordinate = {.x = coord[0], .y = coord[1], .z = 0};
        int3_t size = {.x = nxt, .y = nyt, .z = nzt};
        int3_t shift = {.x = 1, .y = 0, .z = 0};    
        T.topography_grid = grid_init_metric_grid(
            size, shift, coordinate, boundary1, boundary2, h);
        T.velocity_grid = grid_init_velocity_grid(
            size, shift, coordinate, boundary1, boundary2, h);
        T.stress_grid = grid_init_stress_grid(
            size, shift, coordinate, boundary1, boundary2, h);

        return T;

}

void topo_set_bounds(topo_t *T)
{
        topo_set_velocity_bounds_left(T, T->velocity_bounds_left);
        topo_set_velocity_bounds_right(T, T->velocity_bounds_right);
        topo_set_velocity_bounds_front(T, T->velocity_bounds_front);
        topo_set_velocity_bounds_back(T, T->velocity_bounds_back);
        topo_set_velocity_bounds_xdir(T, T->velocity_bounds_xdir);
        topo_set_velocity_grid_interior_size(T, &T->velocity_grid_interior);
        topo_set_velocity_grid_front_size(T, &T->velocity_grid_front);
        topo_set_velocity_grid_back_size(T, &T->velocity_grid_back);
        topo_set_velocity_offset_x(T, T->velocity_offset_x);
        topo_set_velocity_offset_y(T, T->velocity_offset_y);

        assert(!topo_check_grid_size(T, &T->velocity_grid_interior));
        assert(!topo_check_grid_size(T, &T->velocity_grid_front));
        assert(!topo_check_grid_size(T, &T->velocity_grid_back));

        topo_set_stress_bounds_left(T, T->stress_bounds_left);
        topo_set_stress_bounds_right(T, T->stress_bounds_right);
        topo_set_stress_bounds_ydir(T, T->stress_bounds_ydir);
        topo_set_stress_grid_interior_size(T, &T->stress_grid_interior);
        topo_set_stress_grid_left_size(T,  &T->stress_grid_left);
        topo_set_stress_grid_right_size(T, &T->stress_grid_right);
        topo_set_stress_offset_x(T, T->stress_offset_x);
        topo_set_stress_offset_y(T, T->stress_offset_y);

        assert(!topo_check_grid_size(T, &T->stress_grid_interior));
        assert(!topo_check_grid_size(T, &T->stress_grid_left));
        assert(!topo_check_grid_size(T, &T->stress_grid_right));
        //FIXME: Handle error code
        topo_check_block_size(T);
}

void topo_d_malloc(topo_t *T)
{
        // This function is only used for testing purposes (main AWP application
        // is responsible for memory allocation). Here, for simplicity, much
        // larger chunks of memory are allocated than necessary.
        int num_bytes = sizeof(_prec)*T->mx*T->my*T->mz;
        CUCHK(cudaMalloc((void**)&T->rho, num_bytes));
        CUCHK(cudaMalloc((void**)&T->lami, num_bytes));
        CUCHK(cudaMalloc((void**)&T->mui, num_bytes));
        CUCHK(cudaMalloc((void**)&T->xx, num_bytes));
        CUCHK(cudaMalloc((void**)&T->yy, num_bytes));
        CUCHK(cudaMalloc((void**)&T->zz, num_bytes));
        CUCHK(cudaMalloc((void**)&T->xy, num_bytes));
        CUCHK(cudaMalloc((void**)&T->xz, num_bytes));
        CUCHK(cudaMalloc((void**)&T->yz, num_bytes));
        CUCHK(cudaMalloc((void**)&T->r1, num_bytes));
        CUCHK(cudaMalloc((void**)&T->r2, num_bytes));
        CUCHK(cudaMalloc((void**)&T->r3, num_bytes));
        CUCHK(cudaMalloc((void**)&T->r4, num_bytes));
        CUCHK(cudaMalloc((void**)&T->r5, num_bytes));
        CUCHK(cudaMalloc((void**)&T->r6, num_bytes));
        CUCHK(cudaMalloc((void**)&T->u1, num_bytes));
        CUCHK(cudaMalloc((void**)&T->v1, num_bytes));
        CUCHK(cudaMalloc((void**)&T->w1, num_bytes));
        CUCHK(cudaMalloc((void**)&T->f_u1, num_bytes));
        CUCHK(cudaMalloc((void**)&T->f_v1, num_bytes));
        CUCHK(cudaMalloc((void**)&T->f_w1, num_bytes));
        CUCHK(cudaMalloc((void**)&T->b_u1, num_bytes));
        CUCHK(cudaMalloc((void**)&T->b_v1, num_bytes));
        CUCHK(cudaMalloc((void**)&T->b_w1, num_bytes));
        CUCHK(cudaMalloc((void**)&T->dcrjx, num_bytes));
        CUCHK(cudaMalloc((void**)&T->dcrjy, num_bytes));
        CUCHK(cudaMalloc((void**)&T->dcrjz, num_bytes));
        CUCHK(cudaMalloc((void**)&T->qpi, num_bytes));
        CUCHK(cudaMalloc((void**)&T->qsi, num_bytes));
        CUCHK(cudaMalloc((void**)&T->wwo, num_bytes));
        CUCHK(cudaMalloc((void**)&T->ww, num_bytes));
        CUCHK(cudaMalloc((void**)&T->vx1, num_bytes));
        CUCHK(cudaMalloc((void**)&T->vx2, num_bytes));
        CUCHK(cudaMalloc((void**)&T->coeff, num_bytes));
        CUCHK(cudaMalloc((void**)&T->lam_mu, num_bytes));
}

void topo_d_free(topo_t *T)
{
        CUCHK(cudaFree(T->rho)); 
        CUCHK(cudaFree(T->lami));
        CUCHK(cudaFree(T->mui));
        CUCHK(cudaFree(T->xx));
        CUCHK(cudaFree(T->yy));
        CUCHK(cudaFree(T->zz));
        CUCHK(cudaFree(T->xy));
        CUCHK(cudaFree(T->xz));
        CUCHK(cudaFree(T->yz));
        CUCHK(cudaFree(T->r1));
        CUCHK(cudaFree(T->r2));
        CUCHK(cudaFree(T->r3));
        CUCHK(cudaFree(T->r4));
        CUCHK(cudaFree(T->r5));
        CUCHK(cudaFree(T->r6));
        CUCHK(cudaFree(T->u1));
        CUCHK(cudaFree(T->v1));
        CUCHK(cudaFree(T->w1));
        CUCHK(cudaFree(T->f_u1));
        CUCHK(cudaFree(T->f_v1));
        CUCHK(cudaFree(T->f_w1));
        CUCHK(cudaFree(T->b_u1));
        CUCHK(cudaFree(T->b_v1));
        CUCHK(cudaFree(T->b_w1));
        CUCHK(cudaFree(T->dcrjx));
        CUCHK(cudaFree(T->dcrjy));
        CUCHK(cudaFree(T->dcrjz));
}

void topo_init_geometry(topo_t *T)
{
        int err = 0;
        int alloc = 0;

        err |= topo_read_serial(T->topography_file, T->rank, T->px, T->py,
                                T->coord, T->nx, T->ny, alloc, &T->metrics_f_init.f);
        geom_no_grid_stretching(&T->metrics_g);
        geom_custom(&T->metrics_f_init, T->topography_grid, T->px, T->py,
                    T->metrics_f_init.f);

        if (err > 0) {
                printf("%s \n", error_message(err));
                MPI_Abort(MPI_COMM_WORLD, err);
                exit(1);
        }
}

void topo_init_metrics(topo_t *T)
{
        if (!T->use) return;
        int size[3] = {T->nx, T->ny, T->nz};
        T->metrics_f_init = metrics_init_f(size, T->gridspacing, metrics_padding);
        T->metrics_f = metrics_init_f(size, T->gridspacing, ngsl);
        T->metrics_g = metrics_init_g(size, T->gridspacing);
}

void topo_build(topo_t *T)
{
        if (!T->use) return;

        metrics_build_f(&T->metrics_f_init);
        metrics_shift_f(&T->metrics_f, &T->metrics_f_init);
        metrics_d_copy_f(&T->metrics_f);
        metrics_free_f(&T->metrics_f_init);

        metrics_build_g(&T->metrics_g);

        #if TOPO_USE_CONST_MATERIAL
        if (T->rank == 0) {
                printf("Topography:: Overriding material properties.\n");
        }
        topo_init_material_H(T);
        #endif

}


void topo_free(topo_t *T)
{
        if (!T->use) return;
        metrics_free_f(&T->metrics_f);
        metrics_free_g(&T->metrics_g);
        free(T->x1);
        free(T->y1);
        free(T->z1);
        return;
}

int topo_check_block_size(const topo_t *T)
{
        int err = 0;
        if (T->rank!=0) {
                return err;
        }
        int tb[3] = {TBX, TBY, TBZ};
        int size[3] = {T->nx, T->ny, T->nz};
        char dirs[3] = {'x', 'y', 'z'};
        char dirscap[3] = {'X', 'Y', 'Z'};
        for (int i = 0; i < 3; i++) {
                if (tb[i] > size[i]) {
                        fprintf(
                            stderr,
                            "Too few grid points in the %c-direction. "
                            "Increase the number of grid points, or decrease"
                            " the number of threads per block (TB%c)\n.",
                            dirs[i], dirscap[i]);
                        err = 1;
                }
        }
        return err;
}

int topo_check_grid_size(const topo_t *T, const dim3 *grid)
{
        if (grid->x > (size_t)T->mx || grid->y > (size_t)T->my ||
            grid->z > (size_t)T->mz) {
                return 1;
        }
        return 0;
}

void topo_set_memory_size(const int nx, const int ny, const int nz, int *mx,
                          int *my, int *mz) 
{
        *mx = nx + 4 + ngsl2;
        *my = ny + 4 + ngsl2;
        *mz = nz + 2*align;
}

void topo_set_velocity_bounds_left(const topo_t *T, int *bounds)
{
        bounds[0] = 0;
        bounds[1] = ngsl;
}

void topo_set_velocity_bounds_right(const topo_t *T, int *bounds)
{
        bounds[0] = T->nx - ngsl;
        bounds[1] = T->nx;
}

void topo_set_velocity_bounds_front(const topo_t *T, int *bounds)
{
        bounds[0] = 0;
        bounds[1] = ngsl;
}

void topo_set_velocity_bounds_back(const topo_t *T, int *bounds)
{
        bounds[0] = T->ny - ngsl;
        bounds[1] = T->ny;
}

void topo_set_velocity_bounds_xdir(const topo_t *T, int *bounds)
{
        bounds[0] = 0;
        bounds[1] = T->nx;
}

void topo_set_velocity_grid_interior_size(const topo_t *T, dim3 *interior)
{
        interior->x = T->nx;
        interior->y = T->ny;
        interior->z = T->nz;
}

void topo_set_velocity_grid_front_size(const topo_t *T, dim3 *front)
{
        front->x = T->nx;
        front->y = ngsl;
        front->z = T->nz;
}

void topo_set_velocity_grid_back_size(const topo_t *T, dim3 *back)
{
        back->x = T->nx;
        back->y = ngsl;
        back->z = T->nz;
}

void topo_set_velocity_offset_x(const topo_t *T, int *offset)
{
        offset[0] = 2;
        int shift = T->off_x[1];
        offset[1] = shift + T->velocity_bounds_left[0];
        offset[2] = shift + T->velocity_bounds_left[1];
        offset[3] = shift + T->velocity_bounds_right[0];
        offset[4] = shift + T->velocity_bounds_right[1];
        offset[5] = T->off_x[3];
}

void topo_set_velocity_offset_y(const topo_t *T, int *offset)
{
        offset[0] = 2;
        int shift = T->off_y[1];
        offset[1] = shift + T->velocity_bounds_front[0];
        offset[2] = shift + T->velocity_bounds_front[1];
        offset[3] = shift + T->velocity_bounds_back[0];
        offset[4] = shift + T->velocity_bounds_back[1];
        offset[5] = T->off_y[3];
}

void topo_set_stress_bounds_left(const topo_t *T, int *bounds)
{
        // If the process is on the boundary, then there is no extra ghost layer
        // The bound is negative because the kernels are constructed such that i
        // = 0 is the first position in the velocity array. 
        if (T->x_rank_l < 0) {
                bounds[0] = ngsl / 2;
                bounds[1] = ngsl / 2;
        }
        // Process is in the interior. 
        else {
                bounds[0] = -ngsl / 2;
                bounds[1] = ngsl / 2;
        }
}

void topo_set_stress_bounds_right(const topo_t *T, int *bounds)
{
        if (T->x_rank_r < 0) {
                bounds[0] = T->nx - ngsl / 2;
                bounds[1] = T->nx - ngsl / 2;
        }
        else {
                bounds[0] = T->nx - ngsl / 2;
                bounds[1] = T->nx + ngsl / 2;
        }
}

void topo_set_stress_bounds_ydir(const topo_t *T, int *bounds)
{
        if (T->y_rank_f < 0) { 
                bounds[0] = ngsl / 2;
        } 
        else {
                bounds[0] = -ngsl / 2;
        }

        if (T->y_rank_b < 0) { 
                bounds[1] = T->ny - ngsl / 2;
        } 
        else {
                bounds[1] = T->ny + ngsl / 2;
        }
}

void topo_set_stress_grid_interior_size(const topo_t *T, dim3 *interior)
{
        interior->x = T->stress_bounds_right[0] - T->stress_bounds_left[0];
        interior->y = T->stress_bounds_ydir[1] - T->stress_bounds_ydir[0];
        interior->z = T->nz;
}

void topo_set_stress_grid_left_size(const topo_t *T, dim3 *left)
{
        left->x = T->stress_bounds_left[1] - T->stress_bounds_left[0];
        left->y = T->stress_bounds_ydir[1] - T->stress_bounds_ydir[0];
        left->z = T->nz;
}

void topo_set_stress_grid_right_size(const topo_t *T, dim3 *right)
{
        right->x = T->stress_bounds_right[1] - T->stress_bounds_right[0];
        right->y = T->stress_bounds_ydir[1] - T->stress_bounds_ydir[0];
        right->z = T->nz;
}

void topo_set_stress_offset_x(const topo_t *T, int *offset)
{
        offset[0] = 2;
        offset[1] = T->off_x[1] + T->stress_bounds_left[0];
        offset[2] = T->off_x[1] + T->stress_bounds_left[1];
        offset[3] = T->off_x[1] + T->stress_bounds_right[0];
        offset[4] = T->off_x[1] + T->stress_bounds_right[1];
        offset[5] = T->off_x[3];
}

void topo_set_stress_offset_y(const topo_t *T, int *offset)
{
        offset[0] = 2;
        int shift = T->off_y[1];
        offset[1] = shift + T->stress_bounds_ydir[0];
        offset[2] = shift + T->stress_bounds_ydir[1];
        offset[3] = T->off_x[3];
}

void topo_bind(topo_t *T,
               _prec*  __restrict__ rho, 
               _prec*  __restrict__ lami, 
               _prec*  __restrict__ mui,
               _prec *__restrict__  qp,
               _prec *__restrict__  coeff, 
               _prec *__restrict__  qs, 
               _prec *__restrict__ d_vx1, 
               _prec *__restrict__ d_vx2, 
               int *__restrict__ d_ww, 
               _prec *__restrict__ d_wwo,
               _prec*  __restrict__ xx, 
               _prec*  __restrict__ yy, 
               _prec*  __restrict__ zz,
               _prec*  __restrict__ xy, 
               _prec*  __restrict__ xz, 
               _prec*  __restrict__ yz,
               _prec*  __restrict__ r1, 
               _prec*  __restrict__ r2,  
               _prec*  __restrict__ r3, 
               _prec*  __restrict__ r4, 
               _prec*  __restrict__ r5,  
               _prec*  __restrict__ r6,
               _prec*  __restrict__ u1, 
               _prec*  __restrict__ v1,    
               _prec*  __restrict__ w1,
               _prec*  __restrict__ f_u1, 
               _prec*  __restrict__ f_v1,    
               _prec*  __restrict__ f_w1,
               _prec*  __restrict__ b_u1, 
               _prec*  __restrict__ b_v1,    
               _prec*  __restrict__ b_w1,
               _prec*  __restrict__ dcrjx, 
               _prec*  __restrict__ dcrjy,    
               _prec*  __restrict__ dcrjz)
{
                    T->rho = rho; 
                    T->lami = lami; 
                    T->mui = mui;
                    T->qpi     =  qp;     
                    T->coeff  = coeff; 
                    T->qsi     = qs;    
                    T->vx1  = d_vx1; 
                    T->vx2  = d_vx2; 
                    T->ww   = d_ww;  
                    T->wwo  = d_wwo; 
                    T->xx = xx; 
                    T->yy = yy; 
                    T->zz = zz;
                    T->xy = xy; 
                    T->xz = xz; 
                    T->yz = yz;
                    T->r1 = r1; 
                    T->r2 = r2; 
                    T->r3 = r3; 
                    T->r4 = r4; 
                    T->r5 = r5; 
                    T->r6 = r6;
                    T->u1 = u1; 
                    T->v1 = v1; 
                    T->w1 = w1;
                    T->f_u1 = f_u1; 
                    T->f_v1 = f_v1; 
                    T->f_w1 = f_w1;
                    T->b_u1 = b_u1; 
                    T->b_v1 = b_v1; 
                    T->b_w1 = b_w1;
                    T->dcrjx = dcrjx; 
                    T->dcrjy = dcrjy; 
                    T->dcrjz = dcrjz;
}

