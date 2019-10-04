#include <cuda.h>
#include <nvToolsExt.h>
#include <stdio.h>

#include <topography/kernels/optimized_stress.cuh>
#include <topography/kernels/optimized_launch_config.cuh>
#include <topography/stress.cuh>

inline dim3 set_grid(const dim3 block, const int3_t size, const dim3 loop)
{
        dim3 out;
        out.x = ((1 - loop.x) * size.z + block.x - 1 + loop.x) / block.x;
        out.y = ((1 - loop.y) * size.y + block.y - 1 + loop.y) / block.y;
        out.z = ((1 - loop.z) * size.x + block.z - 1 + loop.z) / block.z;
        return out;
}

void topo_stress_interior_H(topo_t *T)
{

        if (!T->use) return;
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }

        {
        dim3 block(DTOPO_STR_111_X, DTOPO_STR_111_Y,
                    DTOPO_STR_111_Z);
        int3_t size = {(int)T->stress_bounds_right[0] - T->stress_bounds_left[1], 
                       (int)T->stress_bounds_ydir[1] -  T->stress_bounds_ydir[0],
                       (int)T->stress_grid_interior.z};
        dim3 loop(0, 0, DTOPO_STR_111_LOOP_Z);
        dim3 grid = set_grid(block, size, loop);
        dtopo_str_111<<<grid, block, 0, T->stream_i>>>
                         (
                          T->xx, T->xy, T->xz, 
                          T->yy, T->yz, T->zz,
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
                          T->lami,
                          T->mui, T->timestep,  
                          T->dth, 
                          T->nx, T->ny, T->nz,
                          T->stress_bounds_left[1], T->stress_bounds_ydir[0], 
                          T->stress_bounds_right[0], T->stress_bounds_ydir[1]);
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
                          T->xx, T->xy, T->xz, 
                          T->yy, T->yz, T->zz,
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
                          T->lami,
                          T->mui, T->timestep,  
                          T->dth, 
                          T->nx, T->ny, T->nz,
                          T->stress_bounds_left[1], T->stress_bounds_ydir[0], 
                          T->stress_bounds_right[0], T->stress_bounds_ydir[1]);

        CUCHK(cudaGetLastError());
        }

        if (TOPO_DBG) {
        dim3 block(DTOPO_STR_110_X, DTOPO_STR_110_Y,
                    DTOPO_STR_110_Z);
        int3_t size = {(int)T->stress_bounds_right[0] - T->stress_bounds_left[0], 
                       (int)T->stress_bounds_ydir[1] -  T->stress_bounds_ydir[0],
                       TOP_BOUNDARY_SIZE};
        dim3 loop(0, 0, DTOPO_STR_110_LOOP_Z);
        dim3 grid = set_grid(block, size, loop);
                dtopo_str_110<<<grid, block, 0, T->stream_i>>>
                         (
                          T->xx, T->xy, T->xz, 
                          T->yy, T->yz, T->zz,
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
                          T->lami,
                          T->mui, T->timestep,  
                          T->dth, 
                          T->nx, T->ny, T->nz,
                          T->stress_bounds_left[1], T->stress_bounds_ydir[0], 
                          T->stress_bounds_right[0], T->stress_bounds_ydir[1]);
                CUCHK(cudaGetLastError());
        }
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

        dtopo_str_111<<<grid, block, 0, T->stream_1>>>
                         (
                          T->xx, T->xy, T->xz, 
                          T->yy, T->yz, T->zz,
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
                          T->lami,
                          T->mui, T->timestep,  
                          T->dth, 
                          T->nx, T->ny, T->nz,
                          T->stress_bounds_left[0], T->stress_bounds_ydir[0], 
                          T->stress_bounds_left[1], T->stress_bounds_ydir[1]);
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
                          T->xx, T->xy, T->xz, 
                          T->yy, T->yz, T->zz,
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
                          T->lami,
                          T->mui, T->timestep,  
                          T->dth, 
                          T->nx, T->ny, T->nz,
                          T->stress_bounds_left[0], T->stress_bounds_ydir[0], 
                          T->stress_bounds_left[1], T->stress_bounds_ydir[1]);
        CUCHK(cudaGetLastError());
        }

        if (TOPO_DBG) {
                dim3 block(DTOPO_STR_110_X, DTOPO_STR_110_Y, DTOPO_STR_110_Z);
                int3_t size = {
                    (int)T->stress_bounds_left[1] - T->stress_bounds_left[0],
                    (int)T->stress_bounds_ydir[1] - T->stress_bounds_ydir[0],
                    (int)T->stress_grid_interior.z};
                dim3 loop(0, 0, DTOPO_STR_110_LOOP_Z);
                dim3 grid = set_grid(block, size, loop);
                dtopo_str_110<<<grid, block, 0, T->stream_1>>>
                         (
                          T->xx, T->xy, T->xz, 
                          T->yy, T->yz, T->zz,
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
                          T->lami,
                          T->mui, T->timestep,  
                          T->dth, 
                          T->nx, T->ny, T->nz,
                          T->stress_bounds_left[0], T->stress_bounds_ydir[0], 
                          T->stress_bounds_left[1], T->stress_bounds_ydir[1]);
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
                          T->xx, T->xy, T->xz, 
                          T->yy, T->yz, T->zz,
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
                          T->lami,
                          T->mui, T->timestep,  
                          T->dth, 
                          T->nx, T->ny, T->nz,
                          T->stress_bounds_right[0], T->stress_bounds_ydir[0], 
                          T->stress_bounds_right[1], T->stress_bounds_ydir[1]);
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
                          T->xx, T->xy, T->xz, 
                          T->yy, T->yz, T->zz,
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
                          T->lami,
                          T->mui, T->timestep,  
                          T->dth, 
                          T->nx, T->ny, T->nz,
                          T->stress_bounds_right[0], T->stress_bounds_ydir[0], 
                          T->stress_bounds_right[1], T->stress_bounds_ydir[1]);
        CUCHK(cudaGetLastError());
        }

        if (TOPO_DBG) {
                dim3 block(DTOPO_STR_110_X, DTOPO_STR_110_Y, DTOPO_STR_110_Z);
                int3_t size = {
                    (int)T->stress_bounds_right[1] - T->stress_bounds_left[0],
                    (int)T->stress_bounds_ydir[1] - T->stress_bounds_ydir[0],
                    TOP_BOUNDARY_SIZE};
                dim3 loop(0, 0, DTOPO_STR_110_LOOP_Z);
                dim3 grid = set_grid(block, size, loop);
                dtopo_str_110<<<grid, block, 0, T->stream_2>>>
                         (
                          T->xx, T->xy, T->xz, 
                          T->yy, T->yz, T->zz,
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
                          T->lami,
                          T->mui, T->timestep,  
                          T->dth, 
                          T->nx, T->ny, T->nz,
                          T->stress_bounds_right[0], T->stress_bounds_ydir[0], 
                          T->stress_bounds_right[1], T->stress_bounds_ydir[1]);
                CUCHK(cudaGetLastError());
        }
}
