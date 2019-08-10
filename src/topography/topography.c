#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <awp/definitions.h>
#include "topography/topography.h"
#include <topography/topography.cuh>

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
        int num_bytes = sizeof(_prec)*T->mx*T->my*T->mz;
        CUCHK(cudaMalloc((void**)&T->u1, num_bytes));
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

