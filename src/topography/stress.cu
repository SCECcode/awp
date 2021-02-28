#include <cuda.h>
#include <nvToolsExt.h>
#include <stdio.h>

#include <topography/stress.cuh>
#include <test/test.h>


// Threads in x, y, z
#ifndef STRIU_TX
#define STRIU_TX 32
#endif      
            
#ifndef STRIU_TY
#define STRIU_TY 1
#endif      
            
#ifndef STRIU_TZ
#define STRIU_TZ 4
#endif

// Unroll factor in CUDA x
#ifndef STRIU_RX
#define STRIU_RX 1
#endif

// Unroll factor in CUDA y
#ifndef STRIU_RY
#define STRIU_RY 2
#endif

// Number of threads per block to use for interior stress kernel
#ifndef STR_INT_X
#define STR_INT_X 32
#endif
#ifndef STR_INT_Y
#define STR_INT_Y 4
#endif
#ifndef STR_INT_Z
#define STR_INT_Z 1
#endif

#ifndef DTOPO_STR_110_X
#define DTOPO_STR_110_X STR_INT_X
#endif

#ifndef DTOPO_STR_110_Y
#define DTOPO_STR_110_Y STR_INT_Y
#endif

#ifndef DTOPO_STR_110_Z
#define DTOPO_STR_110_Z STR_INT_Z
#endif

#ifndef DTOPO_STR_111_X
#define DTOPO_STR_111_X STR_INT_X
#endif

#ifndef DTOPO_STR_111_Y
#define DTOPO_STR_111_Y STR_INT_Y
#endif

#ifndef DTOPO_STR_111_Z
#define DTOPO_STR_111_Z STR_INT_Z
#endif

#ifndef DTOPO_STR_112_X
#define DTOPO_STR_112_X STR_INT_X
#endif

#ifndef DTOPO_STR_112_Y
#define DTOPO_STR_112_Y STR_INT_Y
#endif

#ifndef DTOPO_STR_112_Z
#define DTOPO_STR_112_Z STR_INT_Z
#endif

#define DTOPO_STR_110_LOOP_Z 1
#define DTOPO_STR_111_LOOP_Z 1
#define DTOPO_STR_112_LOOP_Z 1

#include "kernels/stress.cu"
#include "kernels/stress_index_unroll.cu"

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
     int3_t size = {T->stress_bounds_right[0] - T->stress_bounds_left[0], 
                    T->stress_bounds_ydir[1] -  T->stress_bounds_ydir[0],
                    (int)T->stress_grid_interior.z};

        dim3 threads (STRIU_TX, STRIU_TY, STRIU_TZ);
        dim3 blocks((size.z - 4) / (STRIU_RX * threads.x) + 1,
                    (size.y - 1) / (STRIU_RY * threads.y) + 1,
                    (size.x - 1) / (threads.z) + 1);

        dtopo_str_111_index_unroll<STRIU_TX, STRIU_TY, STRIU_TZ, STRIU_RX, STRIU_RY><<<blocks, threads, 0, T->stream_i>>>
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

void topo_stress_left_H(topo_t *T)
{

        if (!T->use) return;
        if (T->x_rank_l < 0) {
                return;
        }

        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }
        int3_t size = {(int)T->stress_bounds_left[1] - T->stress_bounds_left[0],
                       (int)T->stress_bounds_ydir[1] - T->stress_bounds_ydir[0],
                       (int)T->stress_grid_interior.z};

        int shift = ngsl + 2;

        dim3 threads (STRIU_TX, STRIU_TY, STRIU_TZ);
        dim3 blocks((size.z - 4) / (STRIU_RX * threads.x) + 1,
                    (size.y - 1) / (STRIU_RY * threads.y) + 1,
                    (size.x - 1) / (threads.z) + 1);

        dtopo_str_111_index_unroll<STRIU_TX, STRIU_TY, STRIU_TZ, STRIU_RX, STRIU_RY><<<blocks, threads, 0, T->stream_1>>>
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
        int3_t size = {(int)T->stress_bounds_right[1] - T->stress_bounds_left[0],
                       (int)T->stress_bounds_ydir[1] - T->stress_bounds_ydir[0],
                       (int)T->stress_grid_interior.z};

        dim3 threads (STRIU_TX, STRIU_TY, STRIU_TZ);
        dim3 blocks((size.z - 4) / (STRIU_RX * threads.x) + 1,
                    (size.y - 1) / (STRIU_RY * threads.y) + 1,
                    (size.x - 1) / (threads.z) + 1);

        dtopo_str_111_index_unroll<STRIU_TX, STRIU_TY, STRIU_TZ, STRIU_RX, STRIU_RY><<<blocks, threads, 0, T->stream_2>>>
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
