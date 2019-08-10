#ifndef TOPOGRAPHY_H
#define TOPOGRAPHY_H


#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


#include <topography/metrics/metrics.h>
#include <awp/definitions.h>
#include <test/test.h>


// TOPO: Enable topography calls. If disabled, then no topography function calls
// will be included in the compiled binary.
#ifndef TOPO
#define TOPO 1
#endif
// TOPO_DBG: Display debugging information
#ifndef TOPO_DBG
#define TOPO_DBG 0
#endif
// Set this flag to true to receive messages from certain functions (can be
// useful for debugging)
#ifndef TOPO_VERBOSE
#define TOPO_VERBOSE 1
#endif
// Override material properties with predefined, constant properties
#ifndef TOPO_USE_CONST_MATERIAL
#define TOPO_USE_CONST_MATERIAL 0
#endif

/*
 * FIXME: Below follows a description of how the memory layout is defined and
 * how the overlapping of communication and computation works. Some of these
 * details are incorrect and will be revised.
 *
 * Memory layout for velocities:
 *
 *         back 
 *
 *         0 1 2
 *  left   3 4 5   right      bottom    0  1  2    top  
 *         6 7 8
 *                           
 * y ^     front
 *   |
 *   |----> x                         ----------> z
 *
 *  The numbers above indicate the blocks of certain parts of memory that are
 *  allocated for each variable `u1, v1, w1` (velocities) and `xx, yy, zz, xy,
 *  xz, yz` (stresses). The same size and layout of the block of memory is 
 *  allocated for each variable, and for each process. In the (x,y) plane, index
 *  4 is the `write to` block, and the surrounding blocks are ghost layers. In
 *  the z-direction, blocks 0 and 2 are alignments that contain unused memory,
 *  only block 1 contains data.
 *
 *
 *  AWP computation and communication pattern:
 *
 *  1. Post recv messages for receiving velocity data in the front and back
 *  ghost layers labeled (7) and (1) in the figure. The host data buffers are
 *  called `RF_vel` and `RB_vel`.
 *
 *  2. Compute velocities in (4) in three steps that run concurrently. 
 *  Imagine (4) being split into:
 *  --------------------------------
 *   (4b)  inner ghost layer, back       (nx * ngsl * nz)
 *  --------------------------------
 *   (4i)  interior                      (nx * (ny - 2*ngsl) * nz)
 *  --------------------------------
 *   (4f)  inner ghost layer, front      (nx * ngsl * nz)
 *  --------------------------------
 *  Then (4f), (4b), and (4) (yes, (4) and not (4i)) are all computed
 *  concurrently at this step using streams `stream1`, `stream2`, and `streami`.
 *  The computation of (4f) and
 *  (4b) is placed in the arrays `f_u1`, f_v1`, `f_w1`, `b_u1`, `b_v1`, `b_w1`,
 *  on the device, and then copied to host in the buffers `SF_vel, SB_vel`. The
 *  computation of (4) is placed in the arrays `u1`, `v1`, and `w1`.
 *
 *  3. Post send messages for sending data `SF_vel` and `SB_vel` to neighbors in
 *  the y-direction. Wait until send and receives have completed and then copy
 *  received data from host to device. To clarify, the received velocity data is
 *  copied from the buffers `RF_vel` and `RB_vel` to `f_u1`, etc, and then a
 *  kernel is launched to copy from `f_u1`, etc, to `u1`, `v1`, `w1`.
 *
 *  4. After a lot of other computation, (mostly DM), post recv messages for
 *  receiving velocity data in the left and right directions. The host data
 *  buffers are called `RL_vel` and `RR_vel` and covers blocks (0, 3, 6) and (2,
 *  5, 8) in the figure above. Hence, the corners are also requested. This
 *  request is possible because the send and receive of data in y-direction has
 *  completed.
 *
 *  5. Perform stress computation, while concurrently executing steps 6 - 7.
 *  The stress computation is split into three parts, left, right, and interior.
 *  The interior computation takes place while waiting to receive from left and
 *  right. The left and right stress computation takes place after left and
 *  right receives have completed.
 *
 *  6. Copy velocity device data from `u1`, etc, to the left and right send
 *  buffers, `SL_vel` and `SB_vel`. In the process, part of the front and back
 *  data received in step 2 is copied (this data is now located in ghost layers
 *  1 and 7 in the figure).
 *
 *  7. Post send messages in the x-direction for left and right send buffers.
 *  Wait for send and receives to complete and then copy from host to device.
 *  The received data is copied from host buffers `RL_vel` and `RR_vel` into
 *  device data arrays `u1`, `v1`, and `w1`.  
 *
 */

/* topo_t is a data structure that contains pointers to allocated device memory,
 * and parameters that define the size of the allocated memory.
 *
 *  Members:
 *
 *  Currently `off_x, off_y, off_z` are only used for testing (see topography_test)
 *  off_x : records the starting positions of each block in the
 *      x-direction, and also includes the end position + 1. In addition to the
 *      ghost layers, there is an alignment of 2 on either side.
 *  off_y : is defined in the same manner as `off_x`.
 *  off_z : Does not contain any ghost layers, but an alignment of `align` on
 *      either side.
 *
 *  sxx, syy, szz, ... : An array of boolean flags that indicates if the field
 *      is shifted a half-step in the direction defined by the index. For
 *      example, `su1[0] == 0` would imply that the field `u1` is not shifted by
 *      a half in the x-direction.
 *
 */
#ifdef __cplusplus
extern "C" {
#endif
typedef struct 
{
        int use;
        int dbg;
        int verbose;
        int rank;
        const char *topography_file;
        // this rank is set to -1 if the process is on either the left, right,
        // front or back boundary
        int x_rank_l;
        int x_rank_r;
        int y_rank_b;
        int y_rank_f;
        // 2D-Coordinate in the domain decomposition grid
        int coord[2];
        // Number of partitions in each grid direction
        int px;
        int py;
        // Grid size
        int nx;
        int ny;
        int nz;
        // Memory size
        int mx;
        int my;
        int mz;
        int off_x[4];
        int off_y[4];
        int off_z[4];
        // Bounds, offsets, and grid sizes for velocities
        int velocity_bounds_left[2];
        int velocity_bounds_right[2];
        int velocity_bounds_front[2];
        int velocity_bounds_back[2];
        int velocity_bounds_xdir[2];
        int velocity_offset_x[6];
        int velocity_offset_y[6];
        int boundary_top_size;
        int boundary_bottom_size;
        dim3 velocity_grid_interior;
        dim3 velocity_grid_front;
        dim3 velocity_grid_back;
        // Bounds, offsets, and grid sizes for stresses
        int stress_bounds_left[2];
        int stress_bounds_right[2];
        int stress_bounds_ydir[2];
        int stress_offset_x[6];
        int stress_offset_y[4];
        dim3 stress_grid_interior;
        dim3 stress_grid_left;
        dim3 stress_grid_right;
        int sxx[3];
        int syy[3];
        int szz[3];
        int sxy[3];
        int sxz[3];
        int syz[3];
        int su1[3];
        int sv1[3];
        int sw1[3];
        int gridsize;
        int slice;
        int slice_gl;
        int line;
        // Parameters
        // dth : dt/h
        // timestep : set to zero if the timestepping should be disabled.
        // In this case, the kernels only compute the spatial part.
        _prec dth;
        _prec timestep;
        _prec gridspacing;
        // Material properties
        _prec*  __restrict__ rho;
        _prec*  __restrict__ lami;
        _prec*  __restrict__ mui;
        // Stresses
        _prec*  __restrict__ xx;
        _prec*  __restrict__ yy;
        _prec*  __restrict__ zz;
        _prec*  __restrict__ xy;
        _prec*  __restrict__ xz;
        _prec*  __restrict__ yz;
        // Memory variables
        _prec*  __restrict__ r1;
        _prec*  __restrict__ r2;
        _prec*  __restrict__ r3;
        _prec*  __restrict__ r4;
        _prec*  __restrict__ r5;
        _prec*  __restrict__ r6;
        // Velocities
        _prec*  __restrict__ u1;
        _prec*  __restrict__ v1;
        _prec*  __restrict__ w1;
        // Velocities on front and back faces
        // The size of these are mx * ngls * my
        _prec*  __restrict__ f_u1;
        _prec*  __restrict__ f_v1;
        _prec*  __restrict__ f_w1;
        _prec*  __restrict__ b_u1;
        _prec*  __restrict__ b_v1;
        _prec*  __restrict__ b_w1;

        // Topography function
        f_grid_t metrics_f;
        // Grid stretching function
        g_grid_t metrics_g;
        grid3_t topography_grid;
        grid3_t stress_grid;
        grid3_t velocity_grid;
        // 1D grid data
        prec *x1;
        prec *y1;
        prec *z1;

        // Sponge layer variables
        _prec* __restrict__ dcrjx;
        _prec* __restrict__ dcrjy;
        _prec* __restrict__ dcrjz;
        // Streams
        // stream_1 : front 
        // stream_2 : back
        // stream_i : interior
        cudaStream_t stream_1;
        cudaStream_t stream_2;
        cudaStream_t stream_i;

} topo_t;                 
                   
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
                 );

void topo_set_bounds(topo_t *T);
void topo_init_metrics(topo_t *T);
void topo_init_geometry(topo_t *T);
void topo_build(topo_t *T);

void topo_free(topo_t *T);

void topo_d_malloc(topo_t *T);
void topo_d_free(topo_t *T);

int topo_check_block_size(const topo_t *T);
int topo_check_grid_size(const topo_t *T, const dim3 *grid);

void topo_init_grid(topo_t *T);
void topo_init_metrics(topo_t *T);

void topo_set_memory_size(const int nx, const int ny, const int nz, int *mx,
                          int *my, int *mz);
void topo_set_velocity_bounds_left(const topo_t *T, int *bounds);
void topo_set_velocity_bounds_right(const topo_t *T, int *bounds);
void topo_set_velocity_bounds_front(const topo_t *T, int *bounds);
void topo_set_velocity_bounds_back(const topo_t *T, int *bounds);
void topo_set_velocity_bounds_xdir(const topo_t *T, int *bounds);
void topo_set_velocity_grid_interior_size(const topo_t *T, dim3 *interior);
void topo_set_velocity_grid_front_size(const topo_t *T, dim3 *front);
void topo_set_velocity_grid_back_size(const topo_t *T, dim3 *back);
void topo_set_velocity_offset_x(const topo_t *T, int *offset);
void topo_set_velocity_offset_y(const topo_t *T, int *offset);

void topo_set_stress_bounds_left(const topo_t *T, int *bounds);
void topo_set_stress_bounds_right(const topo_t *T, int *bounds);
void topo_set_stress_bounds_ydir(const topo_t *T, int *bounds);
void topo_set_stress_grid_interior_size(const topo_t *T, dim3 *interior);
void topo_set_stress_grid_left_size(const topo_t *T, dim3 *left);
void topo_set_stress_grid_right_size(const topo_t *T, dim3 *right);
void topo_set_stress_offset_x(const topo_t *T, int *offset);
void topo_set_stress_offset_y(const topo_t *T, int *offset);




#ifdef __cplusplus
}
#endif
#endif

