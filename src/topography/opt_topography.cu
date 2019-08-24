#include <cuda.h>
#include <nvToolsExt.h>
#include <stdio.h>

#include <topography/topography.h>
#include <topography/opt_topography.cuh>
#include <topography/kernels/optimized.cuh>
#include <awp/definitions.h>

void topo_init_material_H(topo_t *T)
{
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }
        dim3 block (VEL_INT_X, VEL_INT_Y, VEL_INT_Z);
        dim3 grid ((T->mz+VEL_INT_X-1)/VEL_INT_X, 
                   (T->my+VEL_INT_Y-1)/VEL_INT_Y,
                   (T->mx+VEL_INT_Z-1)/VEL_INT_Z);


        // Apply material properties inside and outside ghost region
        dtopo_init_material_111<<<grid, block>>>(T->lami, T->mui, T->rho,
                                                 T->mx, T->my, T->mz);

        CUCHK(cudaGetLastError());
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
        dim3 block (VEL_INT_X, VEL_INT_Y, VEL_INT_Z);
        int grid_x = T->velocity_bounds_right[1] - T->velocity_bounds_left[0];
        int grid_y = T->velocity_bounds_front[1] - T->velocity_bounds_front[0];
        int grid_z = T->velocity_grid_interior.z;
        dim3 grid((grid_z + VEL_INT_X - 1) / VEL_INT_X,
                  (grid_y + VEL_INT_Y - 1) / VEL_INT_Y,
                  (grid_x + VEL_INT_Z - 1) / VEL_INT_Z);
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
        cudaDeviceSynchronize();
        nvtxRangePop();
        CUCHK(cudaGetLastError());
        }

        // Compute interior part excluding send buffer regions
        {
        dim3 block (VEL_INT_X, VEL_INT_Y, VEL_INT_Z);
        int grid_x = T->velocity_bounds_right[1] - T->velocity_bounds_left[0];
        int grid_y = T->velocity_bounds_back[0] - T->velocity_bounds_front[1];
        int grid_z = T->velocity_grid_interior.z;
        dim3 grid((grid_z + VEL_INT_X - 1) / VEL_INT_X,
                  (grid_y + VEL_INT_Y - 1) / VEL_INT_Y,
                  (grid_x + VEL_INT_Z - 1) / VEL_INT_Z);
        nvtxRangePushA("velocity_interior_interior");
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
        cudaDeviceSynchronize();
        nvtxRangePop();
        CUCHK(cudaGetLastError());
        }

        // Compute back send buffer region
        {
        dim3 block (VEL_INT_X, VEL_INT_Y, VEL_INT_Z);
        int grid_x = T->velocity_bounds_right[1] - T->velocity_bounds_left[0];
        int grid_y = T->velocity_bounds_back[1] - T->velocity_bounds_back[0];
        int grid_z = T->velocity_grid_interior.z;
        dim3 grid((grid_z + VEL_INT_X - 1) / VEL_INT_X,
                  (grid_y + VEL_INT_Y - 1) / VEL_INT_Y,
                  (grid_x + VEL_INT_Z - 1) / VEL_INT_Z);
        nvtxRangePushA("velocity_interior_back");
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
        dim3 block (VEL_BND_X, VEL_BND_Y, VEL_BND_Z);
        int grid_x = T->velocity_bounds_right[1] - T->velocity_bounds_left[0];
        int grid_y = T->velocity_bounds_front[1] - T->velocity_bounds_front[0];
        int grid_z = TOP_BOUNDARY_SIZE;
        dim3 grid((grid_z + VEL_BND_X - 1) / VEL_BND_X,
                  (grid_y + VEL_BND_Y - 1) / VEL_BND_Y,
                  (grid_x + VEL_BND_Z - 1) / VEL_BND_Z);
        
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
        dim3 block (VEL_BND_X, VEL_BND_Y, VEL_BND_Z);
        int grid_x = T->velocity_bounds_right[1] - T->velocity_bounds_left[0];
        int grid_y = T->velocity_bounds_back[0] - T->velocity_bounds_front[1];
        int grid_z = TOP_BOUNDARY_SIZE;
        dim3 grid((grid_z + VEL_BND_X - 1) / VEL_BND_X,
                  (grid_y + VEL_BND_Y - 1) / VEL_BND_Y,
                  (grid_x + VEL_BND_Z - 1) / VEL_BND_Z);
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
        int grid_x = T->velocity_bounds_right[1] - T->velocity_bounds_left[0];
        int grid_y = T->velocity_bounds_back[1] - T->velocity_bounds_back[0];
        int grid_z = TOP_BOUNDARY_SIZE;
        dim3 grid((grid_z + VEL_BND_X - 1) / VEL_BND_X,
                  (grid_y + VEL_BND_Y - 1) / VEL_BND_Y,
                  (grid_x + VEL_BND_Z - 1) / VEL_BND_Z);
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

        // This kernel only runs in debug mode because it applies one-sided
        // stencils at depth
        if (TOPO_DBG) {
                dim3 block(VEL_INT_X, VEL_INT_Y, VEL_INT_Z);
                int grid_x =
                    T->velocity_bounds_right[1] - T->velocity_bounds_left[0];
                int grid_y =
                    T->velocity_bounds_back[1] - T->velocity_bounds_front[0];
                int grid_z = T->velocity_grid_interior.z;
                dim3 grid((grid_z + VEL_INT_X - 1) / VEL_INT_X,
                          (grid_y + VEL_INT_Y - 1) / VEL_INT_Y,
                          (grid_x + VEL_INT_Z - 1) / VEL_INT_Z);
                dtopo_vel_110<<<grid, block, 0, T->stream_i>>>(
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
                                                   T->velocity_bounds_back[1]);
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
        dim3 block (VEL_INT_X, VEL_INT_Y, VEL_INT_Z);
        dim3 grid ((T->velocity_grid_interior.z+VEL_INT_X-1)/VEL_INT_X, 
                   (T->velocity_grid_interior.y+VEL_INT_Y-1)/VEL_INT_Y,
                   (T->velocity_grid_interior.x+VEL_INT_Z-1)/VEL_INT_Z);

        dim3 bnd_block (VEL_BND_X, VEL_BND_Y, VEL_BND_Z);
        dim3 bnd_grid (
                    (TOP_BOUNDARY_SIZE + VEL_BND_X-1)/VEL_BND_X,
                   (T->velocity_grid_interior.y+VEL_BND_Y-1)/VEL_BND_Y,
                   (T->velocity_grid_interior.x+VEL_BND_Z-1)/VEL_BND_Z);

        {
        nvtxRangePushA("velocity_front_interior");
        dtopo_buf_vel_111<<<grid, block, 0, T->stream_1>>>
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

        // Boundary stencils near free surface
        // Adjust grid size for boundary computation
        nvtxRangePushA("velocity_front_boundary");
        dtopo_buf_vel_112<<<bnd_grid, bnd_block, 0, T->stream_1>>>
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

        // This kernel only runs in debug mode because it applies one-sided
        // stencils at depth
        if (TOPO_DBG) { 
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
        dim3 block (VEL_INT_X, VEL_INT_Y, VEL_INT_Z);
        dim3 grid ((T->velocity_grid_front.x+VEL_INT_Z-1)/VEL_INT_Z, 
                   (T->velocity_grid_front.y+VEL_INT_Y-1)/VEL_INT_Y,
                   (T->velocity_grid_front.z+VEL_INT_X-1)/VEL_INT_X);

        dim3 bnd_block (VEL_BND_X, VEL_BND_Y, VEL_BND_Z);
        dim3 bnd_grid (
                    (TOP_BOUNDARY_SIZE + VEL_BND_X-1)/VEL_BND_X,
                   (T->velocity_grid_interior.y+VEL_BND_Y-1)/VEL_BND_Y,
                   (T->velocity_grid_interior.z+VEL_BND_Z-1)/VEL_BND_Z);

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

        // Boundary stencils near free surface
        nvtxRangePushA("velocity_back_boundary");
        dtopo_buf_vel_112<<<bnd_grid, bnd_block, 0, T->stream_2>>>
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

        // This kernel only runs in debug mode because it applies one-sided
        // stencils at depth
        if (TOPO_DBG) { 
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

void topo_stress_interior_H(topo_t *T)
{

        if (!T->use) return;
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }
        //FIXME: Adjust grid size for boundary
        dim3 block (TBX, TBY, TBZ);
        dim3 grid ((T->stress_grid_interior.x+TBX-1)/TBX, 
                   (T->stress_grid_interior.y+TBY-1)/TBY,
                   (T->stress_grid_interior.z+TBZ-1)/TBZ);


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

        // Adjust grid size for boundary computation
        grid.z = (TOP_BOUNDARY_SIZE+TBZ-1)/TBZ;
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

        if (TOPO_DBG) {
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

        //FIXME: Adjust grid size for boundary
        dim3 block (TBX, TBY, TBZ);
        dim3 grid ((T->stress_grid_left.x+TBX-1)/TBX, 
                   (T->stress_grid_left.y+TBY-1)/TBY,
                   (T->stress_grid_left.z+TBZ-1)/TBZ);

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


        grid.z = (TOP_BOUNDARY_SIZE+TBZ-1)/TBZ;
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

        if (TOPO_DBG) {
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

        //FIXME: Adjust grid size for boundary
        dim3 block (TBX, TBY, TBZ);
        dim3 grid ((T->stress_grid_right.x+TBX-1)/TBX, 
                   (T->stress_grid_right.y+TBY-1)/TBY,
                   (T->stress_grid_right.z+TBZ-1)/TBZ);


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

        grid.z = (TOP_BOUNDARY_SIZE+TBZ-1)/TBZ;
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

        if (TOPO_DBG) {
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
