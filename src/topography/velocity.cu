#include <cuda.h>
#include <nvToolsExt.h>
#include <stdio.h>

#include <topography/kernels/optimized_velocity.cuh>
#include <topography/kernels/optimized_launch_config.cuh>
#include <topography/velocity.cuh>
#include "kernels/velocity_unroll.cu"

inline dim3 set_grid(const dim3 block, const int3_t size, const dim3 loop)
{
        dim3 out;
        out.x = ((1 - loop.x) * size.z + block.x - 1 + loop.x) / block.x;
        out.y = ((1 - loop.y) * size.y + block.y - 1 + loop.y) / block.y;
        out.z = ((1 - loop.z) * size.x + block.z - 1 + loop.z) / block.z;
        return out;
}


void topo_velocity_interior_H(topo_t *T)
{

        if (!T->use) return;
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }


        // Compute velocities in the front send buffer region. 
        {
        nvtxRangePushA("velocity_interior_front");
        int3_t size = {
            .x = T->velocity_bounds_right[1] - T->velocity_bounds_left[0],
            .y = T->velocity_bounds_front[1] - T->velocity_bounds_front[0],
            .z = (int)T->velocity_grid_interior.z};


#if sm_61
#define nq 2
#define nr 2
      dim3 threads (64, 2, 2);
      dim3 blocks ((size.z-1)/(nr*threads.x)+1, 
                   (size.y-1)/(nq*threads.y)+1,
                   (size.x-1)/threads.z+1);
#else
#define nq 2
#define nr 4
      dim3 threads (32, 2, 2);
      dim3 blocks ((size.z-1)/(nr*threads.x)+1, 
                   (size.y-1)/(nq*threads.y)+1,
                   (size.x-1)/threads.z+1);
#endif
        dtopo_vel_111_unroll<nq, nr><<<blocks, threads, 0, T->stream_1>>>(
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
        cudaDeviceSynchronize();
        nvtxRangePop();
        CUCHK(cudaGetLastError());
        }

        // Compute interior part excluding send buffer regions
        {
                int3_t size = {
                    T->velocity_bounds_right[1] - T->velocity_bounds_left[0],
                    T->velocity_bounds_back[0] - T->velocity_bounds_front[1],
                    (int)T->velocity_grid_interior.z};
#if sm_61
#define nq 2
#define nr 2
      dim3 threads (64, 2, 2);
      dim3 blocks ((size.z-1)/(nr*threads.x)+1, 
                   (size.y-1)/(nq*threads.y)+1,
                   (size.x-1)/threads.z+1);
#else
#define nq 2
#define nr 4
      dim3 threads (32, 2, 2);
      dim3 blocks ((size.z-1)/(nr*threads.x)+1, 
                   (size.y-1)/(nq*threads.y)+1,
                   (size.x-1)/threads.z+1);
#endif
                nvtxRangePushA("velocity_interior_interior");
                dtopo_vel_111_unroll<nq, nr><<<blocks, threads, 0, T->stream_i>>>(
                    T->u1, T->v1, T->w1, T->dcrjx, T->dcrjy, T->dcrjz,
                    T->metrics_f.d_f, T->metrics_f.d_f1_1, T->metrics_f.d_f1_2,
                    T->metrics_f.d_f1_c, T->metrics_f.d_f2_1,
                    T->metrics_f.d_f2_2, T->metrics_f.d_f2_c,
                    T->metrics_f.d_f_1, T->metrics_f.d_f_2, T->metrics_f.d_f_c,
                    T->metrics_g.d_g, T->metrics_g.d_g3, T->metrics_g.d_g3_c,
                    T->metrics_g.d_g_c, T->rho, T->xx, T->xy, T->xz, T->yy,
                    T->yz, T->zz, T->timestep, T->dth, T->nx, T->ny, T->nz,
                    T->velocity_bounds_left[0], T->velocity_bounds_front[1],
                    T->velocity_bounds_right[1], T->velocity_bounds_back[0]);
                cudaDeviceSynchronize();
                nvtxRangePop();
                CUCHK(cudaGetLastError());
        }

        // Compute back send buffer region
        {
        nvtxRangePushA("velocity_interior_back");
        dim3 block (DTOPO_VEL_111_X, DTOPO_VEL_111_Y, DTOPO_VEL_111_Z);
        int3_t size = {T->velocity_bounds_right[1] - T->velocity_bounds_left[0],
                       T->velocity_bounds_back[1] - T->velocity_bounds_back[0],
                       (int)T->velocity_grid_interior.z};
        dim3 loop (0, 0, DTOPO_VEL_111_LOOP_Z);
        dim3 grid = set_grid(block, size, loop);
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
        cudaDeviceSynchronize();
        nvtxRangePop();
        CUCHK(cudaGetLastError());
        }
        // Boundary stencils near free surface
        {
        dim3 block (DTOPO_VEL_112_X, DTOPO_VEL_112_Y, DTOPO_VEL_112_Z);
        int3_t size = {T->velocity_bounds_right[1] - T->velocity_bounds_left[0],
                    T->velocity_bounds_front[1] - T->velocity_bounds_front[0],
                    TOP_BOUNDARY_SIZE};
        dim3 loop (0, 0, DTOPO_VEL_112_LOOP_Z);
        dim3 grid = set_grid(block, size, loop);
        
        nvtxRangePushA("velocity_interior_boundary_front");
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
        cudaDeviceSynchronize();
        nvtxRangePop();
        CUCHK(cudaGetLastError());
        }

        {
        dim3 block (DTOPO_VEL_112_X, DTOPO_VEL_112_Y, DTOPO_VEL_112_Z);
        int3_t size = {T->velocity_bounds_right[1] - T->velocity_bounds_left[0],
                       T->velocity_bounds_back[0] - T->velocity_bounds_front[1],
                       TOP_BOUNDARY_SIZE};
        dim3 loop (0, 0, DTOPO_VEL_112_LOOP_Z);
        dim3 grid = set_grid(block, size, loop);
        nvtxRangePushA("velocity_interior_boundary_interior");
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
        cudaDeviceSynchronize();
        nvtxRangePop();
        CUCHK(cudaGetLastError());
        }

        {
        nvtxRangePushA("velocity_interior_boundary_back");
        dim3 block (VEL_BND_X, VEL_BND_Y, VEL_BND_Z);
        int3_t size = {T->velocity_bounds_right[1] - T->velocity_bounds_left[0],
                       T->velocity_bounds_back[1] - T->velocity_bounds_back[0],
                       TOP_BOUNDARY_SIZE};
        dim3 loop (0, 0, DTOPO_VEL_112_LOOP_Z);
        dim3 grid = set_grid(block, size, loop);
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
        cudaDeviceSynchronize();
        nvtxRangePop();
        CUCHK(cudaGetLastError());
        }
}

void topo_velocity_front_H(topo_t *T)
{

        if (!T->use) return;
        if (T->y_rank_f < 0) {
                return;
        }

        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }

        {
                dim3 block(DTOPO_BUF_VEL_111_X, DTOPO_BUF_VEL_111_Y,
                            DTOPO_BUF_VEL_111_Z);
                int3_t size = {(int)T->nx, (int)T->velocity_grid_front.y,
                               (int)T->velocity_grid_interior.z};
                dim3 loop(0, 0, DTOPO_BUF_VEL_111_LOOP_Z);
                dim3 grid = set_grid(block, size, loop);
                nvtxRangePushA("velocity_front_interior");
                dtopo_buf_vel_111<<<grid, block, 0, T->stream_1>>>(
                    T->f_u1, T->f_v1, T->f_w1, T->dcrjx, T->dcrjy, T->dcrjz,
                    T->metrics_f.d_f, T->metrics_f.d_f1_1, T->metrics_f.d_f1_2,
                    T->metrics_f.d_f1_c, T->metrics_f.d_f2_1,
                    T->metrics_f.d_f2_2, T->metrics_f.d_f2_c,
                    T->metrics_f.d_f_1, T->metrics_f.d_f_2, T->metrics_f.d_f_c,
                    T->metrics_g.d_g, T->metrics_g.d_g3, T->metrics_g.d_g3_c,
                    T->metrics_g.d_g_c, T->rho, T->xx, T->xy, T->xz, T->yy,
                    T->yz, T->zz, T->u1, T->v1, T->w1, T->timestep, T->dth,
                    T->nx, T->ny, T->nz, 0, T->velocity_grid_front.y,
                    T->velocity_bounds_front[0]);
                cudaDeviceSynchronize();
                nvtxRangePop();
                CUCHK(cudaGetLastError());
        }

        // Boundary stencils near free surface
        // Adjust grid size for boundary computation
        {
                dim3 block(DTOPO_BUF_VEL_112_X, DTOPO_BUF_VEL_112_Y,
                            DTOPO_BUF_VEL_112_Z);
                int3_t size = {(int)T->nx, (int)T->velocity_grid_front.y,
                               (int)T->velocity_grid_interior.z};
                dim3 loop(0, 0, DTOPO_BUF_VEL_112_LOOP_Z);
                dim3 grid = set_grid(block, size, loop);
                nvtxRangePushA("velocity_front_boundary");
                dtopo_buf_vel_112<<<grid, block, 0, T->stream_1>>>
                                 (T->f_u1, T->f_v1, T->f_w1, 
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
                                  T->u1, T->v1, T->w1, 
                                  T->timestep, T->dth,
                                  T->nx, T->ny, T->nz,
                                  0, T->velocity_grid_front.y,
                                  T->velocity_bounds_front[0]);  
                cudaDeviceSynchronize();
                nvtxRangePop();
                CUCHK(cudaGetLastError());
        }

        // This kernel only runs in debug mode because it applies one-sided
        // stencils at depth
        if (TOPO_DBG) { 
                dim3 block(DTOPO_BUF_VEL_110_X, DTOPO_BUF_VEL_110_Y,
                            DTOPO_BUF_VEL_110_Z);
                int3_t size = {(int)T->nx, (int)T->velocity_grid_front.y,
                               (int)T->velocity_grid_interior.z};
                dim3 loop(0, 0, DTOPO_BUF_VEL_110_LOOP_Z);
                dim3 grid = set_grid(block, size, loop);
                dtopo_buf_vel_110<<<grid, block, 0, T->stream_1>>>
                         (T->f_u1, T->f_v1, T->f_w1, 
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
                          T->u1, T->v1, T->w1, 
                          T->timestep, T->dth,
                          T->nx, T->ny, T->nz,
                          0, T->velocity_grid_front.y,
                          T->velocity_bounds_front[0]);  
                CUCHK(cudaGetLastError());
        }
}

void topo_velocity_back_H(topo_t *T)
{
        if (!T->use) return;
        if (T->y_rank_b < 0) {
                return;
        }

        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }

        {
        dim3 block(DTOPO_BUF_VEL_111_X, DTOPO_BUF_VEL_111_Y,
                    DTOPO_BUF_VEL_111_Z);
        int3_t size = {(int)T->nx, (int)T->velocity_grid_back.y,
                       (int)T->velocity_grid_interior.z};
        dim3 loop(0, 0, DTOPO_BUF_VEL_111_LOOP_Z);
        dim3 grid = set_grid(block, size, loop);
        nvtxRangePushA("velocity_back_interior");
        dtopo_buf_vel_111<<<grid, block, 0, T->stream_2>>>
                         (T->b_u1, T->b_v1, T->b_w1, 
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
                          T->u1, T->v1, T->w1, 
                          T->timestep, T->dth,
                          T->nx, T->ny, T->nz,
                          0, T->velocity_grid_back.y,
                          T->velocity_bounds_back[0]);  
        cudaDeviceSynchronize();
        nvtxRangePop();
        CUCHK(cudaGetLastError());
        }

        // Boundary stencils near free surface
        {
        dim3 block(DTOPO_BUF_VEL_112_X, DTOPO_BUF_VEL_112_Y,
                    DTOPO_BUF_VEL_112_Z);
        int3_t size = {(int)T->nx, (int)T->velocity_grid_back.y,
                       (int)T->velocity_grid_interior.z};
        dim3 loop(0, 0, DTOPO_BUF_VEL_112_LOOP_Z);
        dim3 grid = set_grid(block, size, loop);
        nvtxRangePushA("velocity_back_boundary");
        dtopo_buf_vel_112<<<grid, block, 0, T->stream_2>>>
                         (T->b_u1, T->b_v1, T->b_w1, 
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
                          T->u1, T->v1, T->w1, 
                          T->timestep, T->dth,
                          T->nx, T->ny, T->nz,
                          0, T->velocity_grid_back.y,
                          T->velocity_bounds_back[0]);  
        CUCHK(cudaGetLastError());
        cudaDeviceSynchronize();
        nvtxRangePop();
        }

        // This kernel only runs in debug mode because it applies one-sided
        // stencils at depth
        if (TOPO_DBG) { 
                dim3 block(DTOPO_BUF_VEL_110_X, DTOPO_BUF_VEL_110_Y,
                            DTOPO_BUF_VEL_110_Z);
                int3_t size = {(int)T->nx, (int)T->velocity_grid_back.y,
                               (int)T->velocity_grid_interior.z};
                dim3 loop(0, 0, DTOPO_BUF_VEL_110_LOOP_Z);
                dim3 grid = set_grid(block, size, loop);
                dtopo_buf_vel_110<<<grid, block, 0, T->stream_2>>>
                         (T->b_u1, T->b_v1, T->b_w1, 
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
                          T->u1, T->v1, T->w1, 
                          T->timestep, T->dth,
                          T->nx, T->ny, T->nz,
                          0, T->velocity_grid_back.y,
                          T->velocity_bounds_back[0]);  
                CUCHK(cudaGetLastError());
        }
}

