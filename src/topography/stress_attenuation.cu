#include <cuda.h>
#include <nvToolsExt.h>
#include <stdio.h>

#include <topography/kernels/stress_attenuation.cuh>
#include <topography/kernels/optimized_launch_config.cuh>
#include <topography/stress_attenuation.cuh>
#include <test/test.h>

inline dim3 set_grid(const dim3 block, const int3_t size, const dim3 loop)
{
        dim3 out;
        out.x = ((1 - loop.x) * size.z + block.x - 1 + loop.x) / block.x;
        out.y = ((1 - loop.y) * size.y + block.y - 1 + loop.y) / block.y;
        out.z = ((1 - loop.z) * size.x + block.z - 1 + loop.z) / block.z;
        return out;
}

void topo_set_constants(topo_t *T)
{
        set_constants(T->gridspacing, T->dth * T->gridspacing, T->nx, T->ny,
                      T->nz);
}

void topo_stress_interior_H(topo_t *T)
{

        if (!T->use) return;
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }


     int shift = ngsl + 2;
     {
     dim3 block(DTOPO_STR_111_X, DTOPO_STR_111_Y,
                 DTOPO_STR_111_Z);
     int3_t size = {T->stress_bounds_right[0] - T->stress_bounds_left[0], 
                    T->stress_bounds_ydir[1] -  T->stress_bounds_ydir[0],
                    (int)T->stress_grid_interior.z};
     dim3 loop(0, 0, DTOPO_STR_112_LOOP_Z);
     dim3 grid = set_grid(block, size, loop);

        dtopo_str_111<<<grid, block, 0, T->stream_i>>>
                         (
                          T->xx, T->yy, T->zz, 
                          T->xy, T->xz, T->yz,
                          T->r1, T->r2, T->r3,
                          T->r4, T->r5, T->r6,
                          T->u1, T->v1, T->w1, 
                          T->metrics_f.d_f,
                          T->metrics_f.d_f1_1,
                          T->metrics_f.d_f1_2,
                          T->metrics_f.d_f1_c,
                          T->metrics_f.d_f2_1,
                          T->metrics_f.d_f2_2,
                          T->metrics_f.d_f2_c,
                          T->metrics_f.d_f_1,
                          T->metrics_f.d_f_2,
                          T->metrics_f.d_f_c,
                          T->metrics_g.d_g,
                          T->metrics_g.d_g3,
                          T->metrics_g.d_g3_c,
                          T->metrics_g.d_g_c,
                          T->lami,
                          T->mui, 
                          T->qpi,
                          T->coeff,
                          T->qsi,
                          T->dcrjx, T->dcrjy, T->dcrjz,
                          T->vx1,
                          T->vx2,
                          T->ww,
                          T->wwo,
                          T->nx, T->ny, T->nz, T->coord[0], T->coord[1], T->nz,
                          T->stress_bounds_left[1] + shift, 
                          T->stress_bounds_right[0]+ shift, 
                          T->stress_bounds_ydir[0] + shift, 
                          T->stress_bounds_ydir[1] + shift);

        CUCHK(cudaGetLastError());
        }

        {
        dim3 block(DTOPO_STR_112_X, DTOPO_STR_112_Y,
                    DTOPO_STR_112_Z);
        int3_t size = {(int)T->stress_bounds_right[0] - T->stress_bounds_left[0], 
                       (int)T->stress_bounds_ydir[1] -  T->stress_bounds_ydir[0],
                       TOP_BOUNDARY_SIZE};
        dim3 loop(0, 0, DTOPO_STR_112_LOOP_Z);
        dim3 grid = set_grid(block, size, loop);
        dtopo_str_112<<<grid, block, 0, T->stream_i>>>
                         (
                          T->xx, T->yy, T->zz, 
                          T->xy, T->xz, T->yz,
                          T->r1, T->r2, T->r3,
                          T->r4, T->r5, T->r6,
                          T->u1, T->v1, T->w1, 
                          T->metrics_f.d_f,
                          T->metrics_f.d_f1_1,
                          T->metrics_f.d_f1_2,
                          T->metrics_f.d_f1_c,
                          T->metrics_f.d_f2_1,
                          T->metrics_f.d_f2_2,
                          T->metrics_f.d_f2_c,
                          T->metrics_f.d_f_1,
                          T->metrics_f.d_f_2,
                          T->metrics_f.d_f_c,
                          T->metrics_g.d_g,
                          T->metrics_g.d_g3,
                          T->metrics_g.d_g3_c,
                          T->metrics_g.d_g_c,
                          T->lami,
                          T->mui, 
                          T->qpi,
                          T->coeff,
                          T->qsi,
                          T->dcrjx, T->dcrjy, T->dcrjz,
                          T->vx1,
                          T->vx2,
                          T->ww,
                          T->wwo,
                          T->nx, T->ny, T->nz, T->coord[0], T->coord[1], T->nz,
                          T->stress_bounds_left[1]  + shift, 
                          T->stress_bounds_right[0] + shift, 
                          T->stress_bounds_ydir[0]  + shift, 
                          T->stress_bounds_ydir[1]  + shift);
        CUCHK(cudaGetLastError());
        }
}

void topo_velocity_interior_H(topo_t *T)
{

        if (!T->use) return;
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }
        dim3 block (TBX, TBY, TBZ);
        dim3 grid ((T->velocity_grid_interior.x+TBX-1)/TBX, 
                   (T->velocity_grid_interior.y+TBY-1)/TBY,
                   (T->velocity_grid_interior.z+TBZ-1)/TBZ);
        // Compute velocities in the front send buffer region. 
        dtopo_vel_111<<<grid, block, 0, T->stream_1>>>(
                                                   T->u1, T->v1, T->w1,
                                                   T->dcrjx, T->dcrjy, T->dcrjz,
                                                   T->metrics_f.d_f,
                                                   T->metrics_f.d_f1_1,
                                                   T->metrics_f.d_f1_2,
                                                   T->metrics_f.d_f1_c,
                                                   T->metrics_f.d_f2_1,
                                                   T->metrics_f.d_f2_2,
                                                   T->metrics_f.d_f2_c,
                                                   T->metrics_f.d_f_1,
                                                   T->metrics_f.d_f_2,
                                                   T->metrics_f.d_f_c,
                                                   T->metrics_g.d_g,
                                                   T->metrics_g.d_g3,
                                                   T->metrics_g.d_g3_c,
                                                   T->metrics_g.d_g_c,
                                                   T->rho,
                                                   T->xx, T->xy, T->xz, 
                                                   T->yy, T->yz, T->zz,
                                                   T->timestep, T->dth,
                                                   T->nx, T->ny, T->nz,
                                                   T->velocity_bounds_left[0],
                                                   T->velocity_bounds_front[0], 
                                                   T->velocity_bounds_right[1],
                                                   T->velocity_bounds_front[1]);
        CUCHK(cudaGetLastError());

        // Compute interior part excluding send buffer regions
        dtopo_vel_111<<<grid, block, 0, T->stream_i>>>(
                                                   T->u1, T->v1, T->w1,
                                                   T->dcrjx, T->dcrjy, T->dcrjz,
                                                   T->metrics_f.d_f,
                                                   T->metrics_f.d_f1_1,
                                                   T->metrics_f.d_f1_2,
                                                   T->metrics_f.d_f1_c,
                                                   T->metrics_f.d_f2_1,
                                                   T->metrics_f.d_f2_2,
                                                   T->metrics_f.d_f2_c,
                                                   T->metrics_f.d_f_1,
                                                   T->metrics_f.d_f_2,
                                                   T->metrics_f.d_f_c,
                                                   T->metrics_g.d_g,
                                                   T->metrics_g.d_g3,
                                                   T->metrics_g.d_g3_c,
                                                   T->metrics_g.d_g_c,
                                                   T->rho,
                                                   T->xx, T->xy, T->xz, 
                                                   T->yy, T->yz, T->zz,
                                                   T->timestep, T->dth,
                                                   T->nx, T->ny, T->nz,
                                                   T->velocity_bounds_left[0],
                                                   T->velocity_bounds_front[1], 
                                                   T->velocity_bounds_right[1],
                                                   T->velocity_bounds_back[0]);
        CUCHK(cudaGetLastError());

        // Compute back send buffer region
        dtopo_vel_111<<<grid, block, 0, T->stream_2>>>(
                                                   T->u1, T->v1, T->w1,
                                                   T->dcrjx, T->dcrjy, T->dcrjz,
                                                   T->metrics_f.d_f,
                                                   T->metrics_f.d_f1_1,
                                                   T->metrics_f.d_f1_2,
                                                   T->metrics_f.d_f1_c,
                                                   T->metrics_f.d_f2_1,
                                                   T->metrics_f.d_f2_2,
                                                   T->metrics_f.d_f2_c,
                                                   T->metrics_f.d_f_1,
                                                   T->metrics_f.d_f_2,
                                                   T->metrics_f.d_f_c,
                                                   T->metrics_g.d_g,
                                                   T->metrics_g.d_g3,
                                                   T->metrics_g.d_g3_c,
                                                   T->metrics_g.d_g_c,
                                                   T->rho,
                                                   T->xx, T->xy, T->xz, 
                                                   T->yy, T->yz, T->zz,
                                                   T->timestep, T->dth,
                                                   T->nx, T->ny, T->nz,
                                                   T->velocity_bounds_left[0],
                                                   T->velocity_bounds_back[0], 
                                                   T->velocity_bounds_right[1],
                                                   T->velocity_bounds_back[1]);
        CUCHK(cudaGetLastError());

        // Adjust grid size for boundary computation
        grid.z = (TOP_BOUNDARY_SIZE+TBZ-1)/TBZ;
        // Boundary stencils near free surface
        
        dtopo_vel_112<<<grid, block, 0, T->stream_1>>>(
                                                   T->u1, T->v1, T->w1,
                                                   T->dcrjx, T->dcrjy, T->dcrjz,
                                                   T->metrics_f.d_f,
                                                   T->metrics_f.d_f1_1,
                                                   T->metrics_f.d_f1_2,
                                                   T->metrics_f.d_f1_c,
                                                   T->metrics_f.d_f2_1,
                                                   T->metrics_f.d_f2_2,
                                                   T->metrics_f.d_f2_c,
                                                   T->metrics_f.d_f_1,
                                                   T->metrics_f.d_f_2,
                                                   T->metrics_f.d_f_c,
                                                   T->metrics_g.d_g,
                                                   T->metrics_g.d_g3,
                                                   T->metrics_g.d_g3_c,
                                                   T->metrics_g.d_g_c,
                                                   T->rho,
                                                   T->xx, T->xy, T->xz, 
                                                   T->yy, T->yz, T->zz,
                                                   T->timestep, T->dth,
                                                   T->nx, T->ny, T->nz,
                                                   T->velocity_bounds_left[0],
                                                   T->velocity_bounds_front[0], 
                                                   T->velocity_bounds_right[1],
                                                   T->velocity_bounds_front[1]);
        CUCHK(cudaGetLastError());

        dtopo_vel_112<<<grid, block, 0, T->stream_i>>>(
                                                   T->u1, T->v1, T->w1,
                                                   T->dcrjx, T->dcrjy, T->dcrjz,
                                                   T->metrics_f.d_f,
                                                   T->metrics_f.d_f1_1,
                                                   T->metrics_f.d_f1_2,
                                                   T->metrics_f.d_f1_c,
                                                   T->metrics_f.d_f2_1,
                                                   T->metrics_f.d_f2_2,
                                                   T->metrics_f.d_f2_c,
                                                   T->metrics_f.d_f_1,
                                                   T->metrics_f.d_f_2,
                                                   T->metrics_f.d_f_c,
                                                   T->metrics_g.d_g,
                                                   T->metrics_g.d_g3,
                                                   T->metrics_g.d_g3_c,
                                                   T->metrics_g.d_g_c,
                                                   T->rho,
                                                   T->xx, T->xy, T->xz, 
                                                   T->yy, T->yz, T->zz,
                                                   T->timestep, T->dth,
                                                   T->nx, T->ny, T->nz,
                                                   T->velocity_bounds_left[0],
                                                   T->velocity_bounds_front[1], 
                                                   T->velocity_bounds_right[1],
                                                   T->velocity_bounds_back[0]);
        CUCHK(cudaGetLastError());

        dtopo_vel_112<<<grid, block, 0, T->stream_2>>>(
                                                   T->u1, T->v1, T->w1,
                                                   T->dcrjx, T->dcrjy, T->dcrjz,
                                                   T->metrics_f.d_f,
                                                   T->metrics_f.d_f1_1,
                                                   T->metrics_f.d_f1_2,
                                                   T->metrics_f.d_f1_c,
                                                   T->metrics_f.d_f2_1,
                                                   T->metrics_f.d_f2_2,
                                                   T->metrics_f.d_f2_c,
                                                   T->metrics_f.d_f_1,
                                                   T->metrics_f.d_f_2,
                                                   T->metrics_f.d_f_c,
                                                   T->metrics_g.d_g,
                                                   T->metrics_g.d_g3,
                                                   T->metrics_g.d_g3_c,
                                                   T->metrics_g.d_g_c,
                                                   T->rho,
                                                   T->xx, T->xy, T->xz, 
                                                   T->yy, T->yz, T->zz,
                                                   T->timestep, T->dth,
                                                   T->nx, T->ny, T->nz,
                                                   T->velocity_bounds_left[0],
                                                   T->velocity_bounds_back[0], 
                                                   T->velocity_bounds_right[1],
                                                   T->velocity_bounds_back[1]);
        CUCHK(cudaGetLastError());
}

void topo_stress_left_H(topo_t *T)
{

        if (!T->use) return;
        if (T->x_rank_l < 0) {
                return;
        }

        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }
        dim3 block(DTOPO_STR_111_X, DTOPO_STR_111_Y,
                    DTOPO_STR_111_Z);
        int3_t size = {(int)T->stress_bounds_left[1] - T->stress_bounds_left[0],
                       (int)T->stress_bounds_ydir[1] - T->stress_bounds_ydir[0],
                       (int)T->stress_grid_interior.z};
        dim3 loop(0, 0, DTOPO_STR_111_LOOP_Z);
        dim3 grid = set_grid(block, size, loop);

        int shift = ngsl + 2;
        dtopo_str_111<<<grid, block, 0, T->stream_1>>>
                         (
                          T->xx, T->yy, T->zz, 
                          T->xy, T->xz, T->yz,
                          T->r1, T->r2, T->r3,
                          T->r4, T->r5, T->r6,
                          T->u1, T->v1, T->w1, 
                          T->metrics_f.d_f,
                          T->metrics_f.d_f1_1,
                          T->metrics_f.d_f1_2,
                          T->metrics_f.d_f1_c,
                          T->metrics_f.d_f2_1,
                          T->metrics_f.d_f2_2,
                          T->metrics_f.d_f2_c,
                          T->metrics_f.d_f_1,
                          T->metrics_f.d_f_2,
                          T->metrics_f.d_f_c,
                          T->metrics_g.d_g,
                          T->metrics_g.d_g3,
                          T->metrics_g.d_g3_c,
                          T->metrics_g.d_g_c,
                          T->lami,
                          T->mui, 
                          T->qpi,
                          T->coeff,
                          T->qsi,
                          T->dcrjx, T->dcrjy, T->dcrjz,
                          T->vx1,
                          T->vx2,
                          T->ww,
                          T->wwo,
                          T->nx, T->ny, T->nz, T->coord[0], T->coord[1], T->nz,
                          T->stress_bounds_left[0] + shift, 
                          T->stress_bounds_left[1] + shift, 
                          T->stress_bounds_ydir[0] + shift, 
                          T->stress_bounds_ydir[1] + shift);
        CUCHK(cudaGetLastError());


        {
        dim3 block(DTOPO_STR_112_X, DTOPO_STR_112_Y,
                    DTOPO_STR_112_Z);
        int3_t size = {(int)T->stress_bounds_left[1] - T->stress_bounds_left[0],
                       (int)T->stress_bounds_ydir[1] - T->stress_bounds_ydir[0],
                       (int)T->stress_grid_interior.z};
        dim3 loop(0, 0, DTOPO_STR_112_LOOP_Z);
        dim3 grid = set_grid(block, size, loop);
        dtopo_str_112<<<grid, block, 0, T->stream_1>>>
                         (
                          T->xx, T->yy, T->zz, 
                          T->xy, T->xz, T->yz,
                          T->r1, T->r2, T->r3,
                          T->r4, T->r5, T->r6,
                          T->u1, T->v1, T->w1, 
                          T->metrics_f.d_f,
                          T->metrics_f.d_f1_1,
                          T->metrics_f.d_f1_2,
                          T->metrics_f.d_f1_c,
                          T->metrics_f.d_f2_1,
                          T->metrics_f.d_f2_2,
                          T->metrics_f.d_f2_c,
                          T->metrics_f.d_f_1,
                          T->metrics_f.d_f_2,
                          T->metrics_f.d_f_c,
                          T->metrics_g.d_g,
                          T->metrics_g.d_g3,
                          T->metrics_g.d_g3_c,
                          T->metrics_g.d_g_c,
                          T->lami,
                          T->mui, 
                          T->qpi,
                          T->coeff,
                          T->qsi,
                          T->dcrjx, T->dcrjy, T->dcrjz,
                          T->vx1,
                          T->vx2,
                          T->ww,
                          T->wwo,
                          T->nx, T->ny, T->nz, T->coord[0], T->coord[1], T->nz,
                          T->stress_bounds_left[0] + shift, 
                          T->stress_bounds_left[1] + shift, 
                          T->stress_bounds_ydir[0] + shift, 
                          T->stress_bounds_ydir[1] + shift);
        CUCHK(cudaGetLastError());
        }
}

void topo_stress_right_H(topo_t *T)
{

        if (!T->use) return;
        if (T->x_rank_r < 0) {
                return;
        }
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }

        int shift = ngsl + 2;
        {
        dim3 block(DTOPO_STR_111_X, DTOPO_STR_111_Y,
                    DTOPO_STR_111_Z);
        int3_t size = {(int)T->stress_bounds_right[1] - T->stress_bounds_left[0],
                       (int)T->stress_bounds_ydir[1] - T->stress_bounds_ydir[0],
                       (int)T->stress_grid_interior.z};
        dim3 loop(0, 0, DTOPO_STR_111_LOOP_Z);
        dim3 grid = set_grid(block, size, loop);
        dtopo_str_111<<<grid, block, 0, T->stream_2>>>
                         (
                          T->xx, T->yy, T->zz, 
                          T->xy, T->xz, T->yz,
                          T->r1, T->r2, T->r3,
                          T->r4, T->r5, T->r6,
                          T->u1, T->v1, T->w1, 
                          T->metrics_f.d_f,
                          T->metrics_f.d_f1_1,
                          T->metrics_f.d_f1_2,
                          T->metrics_f.d_f1_c,
                          T->metrics_f.d_f2_1,
                          T->metrics_f.d_f2_2,
                          T->metrics_f.d_f2_c,
                          T->metrics_f.d_f_1,
                          T->metrics_f.d_f_2,
                          T->metrics_f.d_f_c,
                          T->metrics_g.d_g,
                          T->metrics_g.d_g3,
                          T->metrics_g.d_g3_c,
                          T->metrics_g.d_g_c,
                          T->lami,
                          T->mui, 
                          T->qpi,
                          T->coeff,
                          T->qsi,
                          T->dcrjx, T->dcrjy, T->dcrjz,
                          T->vx1,
                          T->vx2,
                          T->ww,
                          T->wwo,
                          T->nx, T->ny, T->nz, T->coord[0], T->coord[1], T->nz,
                          T->stress_bounds_right[0] + shift, 
                          T->stress_bounds_right[1] + shift, 
                          T->stress_bounds_ydir[0]  + shift, 
                          T->stress_bounds_ydir[1]  + shift);
        CUCHK(cudaGetLastError());
        }

        {
        dim3 block(DTOPO_STR_112_X, DTOPO_STR_112_Y,
                    DTOPO_STR_112_Z);
        int3_t size = {(int)T->stress_bounds_right[1] - T->stress_bounds_left[0],
                       (int)T->stress_bounds_ydir[1] - T->stress_bounds_ydir[0],
                       TOP_BOUNDARY_SIZE};
        dim3 loop(0, 0, DTOPO_STR_112_LOOP_Z);
        dim3 grid = set_grid(block, size, loop);
        dtopo_str_112<<<grid, block, 0, T->stream_2>>>
                         (
                          T->xx, T->yy, T->zz, 
                          T->xy, T->xz, T->yz,
                          T->r1, T->r2, T->r3,
                          T->r4, T->r5, T->r6,
                          T->u1, T->v1, T->w1, 
                          T->metrics_f.d_f,
                          T->metrics_f.d_f1_1,
                          T->metrics_f.d_f1_2,
                          T->metrics_f.d_f1_c,
                          T->metrics_f.d_f2_1,
                          T->metrics_f.d_f2_2,
                          T->metrics_f.d_f2_c,
                          T->metrics_f.d_f_1,
                          T->metrics_f.d_f_2,
                          T->metrics_f.d_f_c,
                          T->metrics_g.d_g,
                          T->metrics_g.d_g3,
                          T->metrics_g.d_g3_c,
                          T->metrics_g.d_g_c,
                          T->lami,
                          T->mui, 
                          T->qpi,
                          T->coeff,
                          T->qsi,
                          T->dcrjx, T->dcrjy, T->dcrjz,
                          T->vx1,
                          T->vx2,
                          T->ww,
                          T->wwo,
                          T->nx, T->ny, T->nz, T->coord[0], T->coord[1], T->nz,
                          T->stress_bounds_right[0] + shift, 
                          T->stress_bounds_right[1] + shift, 
                          T->stress_bounds_ydir[0]  + shift, 
                          T->stress_bounds_ydir[1]  + shift);
        CUCHK(cudaGetLastError());
        }
}
