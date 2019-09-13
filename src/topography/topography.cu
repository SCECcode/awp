#include <cuda.h>
#include <stdio.h>

#include <topography/topography.h>
#include <topography/topography.cuh>
#include <topography/kernels/unoptimized.cuh>
#include <awp/definitions.h>

void topo_init_material_H(topo_t *T)
{
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }
        dim3 block (TBX, TBY, TBZ);
        dim3 grid ((T->mx+TBX-1)/TBX, 
                   (T->my+TBY-1)/TBY,
                   (T->mz+TBZ-1)/TBZ);


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
        dim3 block (TBX, TBY, TBZ);
        dim3 grid ((T->velocity_grid_interior.x+TBX-1)/TBX, 
                   (T->velocity_grid_interior.y+TBY-1)/TBY,
                   (T->velocity_grid_interior.z+TBZ-1)/TBZ);

        if (TOPO_DBG) {
        printf("grid: %d %d %d block: %d %d %d \n", 
                        grid.x, grid.y, grid.z,
                        block.x, block.y, block.z);
        printf("n = %d %d %d \n", T->nx, T->ny, T->nz);
        }

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

        // This kernel only runs in debug mode because it applies one-sided
        // stencils at depth
        if (TOPO_DBG) { 
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
        //FIXME: Adjust grid size for boundary
        dim3 block (TBX, TBY, TBZ);
        dim3 grid ((T->velocity_grid_front.x+TBX-1)/TBX, 
                   (T->velocity_grid_front.y+TBY-1)/TBY,
                   (T->velocity_grid_front.z+TBZ-1)/TBZ);

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
        CUCHK(cudaGetLastError());

        // Boundary stencils near free surface
        // Adjust grid size for boundary computation
        grid.z = (TOP_BOUNDARY_SIZE+TBZ-1)/TBZ;
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
        //FIXME: Adjust grid size for boundary
        dim3 block (TBX, TBY, TBZ);
        dim3 grid ((T->velocity_grid_back.x+TBX-1)/TBX, 
                   (T->velocity_grid_back.y+TBY-1)/TBY,
                   (T->velocity_grid_back.z+TBZ-1)/TBZ);

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
        CUCHK(cudaGetLastError());

        // Boundary stencils near free surface
        // Adjust grid size for boundary computation
        grid.z = (TOP_BOUNDARY_SIZE+TBZ-1)/TBZ;
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



        printf("Launch unoptimized stress kernel.\n");
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
        return;

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
