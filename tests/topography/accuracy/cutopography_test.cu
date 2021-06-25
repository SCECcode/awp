#include <cuda.h>
#include <stdio.h>


#include "cutopography_test.cuh"

#define BLOCK_SIZE_X 1
#define BLOCK_SIZE_Y 1
#define BLOCK_SIZE_Z 32
#define TBX 1
#define TBY 1
#define TBZ 32


void topo_test_diffx_H(topo_t *T, _prec *out, const _prec *in)
{
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }
        dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, BLOCK_SIZE_X);
        dim3 grid ((T->nz+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, 
                   (T->ny+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,
                   (T->nx+BLOCK_SIZE_X-1)/BLOCK_SIZE_X);
        dtopo_test_diffx<<<grid, block, 0, T->stream_i>>>(
                                               out, in, 
                                               T->off_x[1], T->off_x[2],
                                               T->off_y[1], T->off_y[2],
                                               T->off_z[1], T->off_z[2],
                                               T->off_x[1], T->off_x[2],
                                               T->off_y[1], T->off_y[2],
                                               T->off_z[1], T->off_z[2],
                                               T->line, T->slice,
                                               T->line, T->slice
                                              );
        CUCHK(cudaGetLastError());
        
        return;
}

void topo_test_cgdiffx_H(topo_t *T, _prec *out, const _prec *in)
{
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }
        dim3 block (TBX, TBY, TBZ);
        dim3 grid ((T->nx+TBX-1)/TBX, 
                   (T->ny+TBY-1)/TBY,
                   (T->nz+TBZ-1)/TBZ);
        CUCHK(cudaGetLastError());
        if (TOPO_DBG > 1) { 
                printf("Grid: %d %d %d \n", grid.x, grid.y, grid.z);
        }
        dtopo_test_diffx_111<<<grid, block, 0, T->stream_i>>>(
                                               out, in, 
                                               T->nx, T->ny, T->nz);
        CUCHK(cudaGetLastError());
        
        return;
}

void topo_test_diffy_H(topo_t *T, _prec *out, const _prec *in)
{
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }
        dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, BLOCK_SIZE_X);
        dim3 grid ((T->nz+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, 
                   (T->ny+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,
                   (T->nx+BLOCK_SIZE_X-1)/BLOCK_SIZE_X);
        dtopo_test_diffy<<<grid, block, 0, T->stream_i>>>(
                                               out, in, 
                                               T->off_x[1], T->off_x[2],
                                               T->off_y[1], T->off_y[2],
                                               T->off_z[1], T->off_z[2],
                                               T->off_x[1], T->off_x[2],
                                               T->off_y[1], T->off_y[2],
                                               T->off_z[1], T->off_z[2],
                                               T->line, T->slice,
                                               T->line, T->slice
                                              );
        CUCHK(cudaGetLastError());
        
        return;
}

void topo_test_diffz_H(topo_t *T, _prec *out, const _prec *in)
{
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }
        dim3 block (TBX, TBY, TBZ);
        dim3 grid ((T->nx+TBX-1)/TBX, 
                   (T->ny+TBY-1)/TBY,
                   (T->nz+TBZ-1)/TBZ);
        CUCHK(cudaGetLastError());
        dtopo_test_diffz_111<<<grid, block, 0, T->stream_i>>>(
                                               out, in, 
                                               T->nx, T->ny, T->nz);
        dtopo_test_diffz_112<<<grid, block, 0, T->stream_i>>>(
                                               out, in, 
                                               T->nx, T->ny, T->nz);
        CUCHK(cudaGetLastError());
        
        return;
}

void topo_test_poly_H(topo_t *T, _prec *out, const _prec *coef,
                      const _prec *deg, const int *shift)
{
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }

        dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, BLOCK_SIZE_X);
        dim3 grid ((T->nz+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, 
                   (T->ny+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,
                   (T->nx+BLOCK_SIZE_X-1)/BLOCK_SIZE_X);

        CUCHK(cudaGetLastError());

        // Initialize the end result (yy) to something else than zero to make
        // sure that the test is not trivially passed
        dtopo_test_poly<<<grid, block>>>(
                                         out, 
                                         T->off_x[1], T->off_x[2],
                                         T->off_y[1], T->off_y[2],
                                         T->off_z[1], T->off_z[2],
                                         T->off_x[1], T->off_x[2],
                                         T->off_y[1], T->off_y[2],
                                         T->off_z[1], T->off_z[2],
                                         T->nx, T->ny, T->nz,
                                         T->line, T->slice,
                                         T->coord[0], T->coord[1],
                                         coef[0], coef[1], coef[2],
                                         deg[0], deg[1], deg[2],
                                         shift[0], shift[1], shift[2]
                                         );
        CUCHK(cudaGetLastError());
}

void topo_test_polystr_H(topo_t *T, _prec *out, const _prec *coef,
                      const _prec *deg, const int *shift)
{
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }

        dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, BLOCK_SIZE_X);
        dim3 grid ((T->nz+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, 
                   (T->ny+ngsl+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,
                   (T->nx+ngsl+BLOCK_SIZE_X-1)/BLOCK_SIZE_X);

        CUCHK(cudaGetLastError());

        // Initialize the end result (yy) to something else than zero to make
        // sure that the test is not trivially passed
        dtopo_test_poly<<<grid, block>>>(
                                         out, 
                                         T->off_x[1]-ngsl/2, T->off_x[2]+ngsl/2,
                                         T->off_y[1]-ngsl/2, T->off_y[2]+ngsl/2,
                                         T->off_z[1], T->off_z[2],
                                         T->off_x[1]-ngsl/2, T->off_x[2]+ngsl/2,
                                         T->off_y[1]-ngsl/2, T->off_y[2]+ngsl/2,
                                         T->off_z[1], T->off_z[2],
                                         T->nx, T->ny, T->nz,
                                         T->line, T->slice,
                                         T->coord[0], T->coord[1],
                                         coef[0], coef[1], coef[2],
                                         deg[0], deg[1], deg[2],
                                         shift[0], shift[1], shift[2]
                                         );
        CUCHK(cudaGetLastError());
}

void topo_test_polyzbnd_H(topo_t *T, _prec *out, const _prec *coef,
                      const _prec *deg, const int *shift)
{
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }

        dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, BLOCK_SIZE_X);
        dim3 grid ((T->nz+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, 
                   (T->ny+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,
                   (T->nx+BLOCK_SIZE_X-1)/BLOCK_SIZE_X);

        CUCHK(cudaGetLastError());

        dtopo_test_polyzbnd<<<grid, block>>>(
                                         out, 
                                         T->off_x[1], T->off_x[2],
                                         T->off_y[1], T->off_y[2],
                                         T->off_z[1], T->off_z[2],
                                         T->off_x[1], T->off_x[2],
                                         T->off_y[1], T->off_y[2],
                                         T->off_z[1], T->off_z[2],
                                         T->nx, T->ny, T->nz,
                                         T->line, T->slice,
                                         T->coord[0], T->coord[1],
                                         coef[0], coef[1], coef[2],
                                         deg[0], deg[1], deg[2],
                                         shift[0], shift[1], shift[2]
                                         );
        CUCHK(cudaGetLastError());
}

void topo_test_polystrzbnd_H(topo_t *T, _prec *out, const _prec *coef,
                      const _prec *deg, const int *shift)
{
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }

        dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, BLOCK_SIZE_X);
        dim3 grid ((T->nz+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, 
                   (T->ny+ngsl+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,
                   (T->nx+ngsl+BLOCK_SIZE_X-1)/BLOCK_SIZE_X);

        CUCHK(cudaGetLastError());

        dtopo_test_polyzbnd<<<grid, block>>>(
                                         out, 
                                         T->off_x[1]-ngsl/2, T->off_x[2]+ngsl/2,
                                         T->off_y[1]-ngsl/2, T->off_y[2]+ngsl/2,
                                         T->off_z[1], T->off_z[2],
                                         T->off_x[1]-ngsl/2, T->off_x[2]+ngsl/2,
                                         T->off_y[1]-ngsl/2, T->off_y[2]+ngsl/2,
                                         T->off_z[1], T->off_z[2],
                                         T->nx, T->ny, T->nz,
                                         T->line, T->slice,
                                         T->coord[0], T->coord[1],
                                         coef[0], coef[1], coef[2],
                                         deg[0], deg[1], deg[2],
                                         shift[0], shift[1], shift[2]
                                         );
        CUCHK(cudaGetLastError());
}

void topo_test_polyf_H(topo_t *T, _prec *out, const _prec *coef, const _prec
                *deg, const int *shift)
{
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }
        dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, BLOCK_SIZE_X);
        dim3 grid ((T->nz+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, 
                   (T->ny+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,
                   (T->nx+BLOCK_SIZE_X-1)/BLOCK_SIZE_X);
        dtopo_test_poly<<<grid, block, 0, T->stream_1>>>(
                                         out,
                                         T->off_x[1], T->off_x[2],
                                         0, ngsl,
                                         T->off_z[1], T->off_z[2],
                                         T->off_x[1], T->off_x[2],
                                         T->off_y[1], T->off_y[1] + ngsl,
                                         T->off_z[1], T->off_z[2],
                                         T->nx, T->ny, T->nz,
                                         T->line, T->slice_gl,
                                         T->coord[0], T->coord[1],
                                         coef[0], coef[1], coef[2],
                                         deg[0], deg[1], deg[2],
                                         shift[0], shift[1], shift[2]
                                         );
        CUCHK(cudaGetLastError());
        
        return;
}

void topo_test_polyzbndf_H(topo_t *T, _prec *out, const _prec *coef, const _prec
                *deg, const int *shift)
{
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }
        dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, BLOCK_SIZE_X);
        dim3 grid ((T->nz+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, 
                   (T->ny+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,
                   (T->nx+BLOCK_SIZE_X-1)/BLOCK_SIZE_X);
        dtopo_test_polyzbnd<<<grid, block, 0, T->stream_1>>>(
                                         out,
                                         T->off_x[1], T->off_x[2],
                                         0, ngsl,
                                         T->off_z[1], T->off_z[2],
                                         T->off_x[1], T->off_x[2],
                                         T->off_y[1], T->off_y[1] + ngsl,
                                         T->off_z[1], T->off_z[2],
                                         T->nx, T->ny, T->nz,
                                         T->line, T->slice_gl,
                                         T->coord[0], T->coord[1],
                                         coef[0], coef[1], coef[2],
                                         deg[0], deg[1], deg[2],
                                         shift[0], shift[1], shift[2]
                                         );
        CUCHK(cudaGetLastError());
        
        return;
}

void topo_test_polyb_H(topo_t *T, _prec *out, const _prec *coef, const _prec
                       *deg, const int *shift)
{
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }
        dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, BLOCK_SIZE_X);
        dim3 grid ((T->nz+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, 
                   (T->ny+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,
                   (T->nx+BLOCK_SIZE_X-1)/BLOCK_SIZE_X);
        //Differentiate  (`sxx = 0`)
        dtopo_test_poly<<<grid, block, 0, T->stream_2>>>(
                                         out, 
                                         T->off_x[1], T->off_x[2],
                                         0, ngsl,
                                         T->off_z[1], T->off_z[2],
                                         T->off_x[1], T->off_x[2],
                                         T->off_y[2] - ngsl, T->off_y[2],
                                         T->off_z[1], T->off_z[2],
                                         T->nx, T->ny, T->nz,
                                         T->line, T->slice_gl,
                                         T->coord[0], T->coord[1],
                                         coef[0], coef[1], coef[2],
                                         deg[0], deg[1], deg[2],
                                         shift[0], shift[1], shift[2]
                                         );
        CUCHK(cudaGetLastError());
        
        return;
}

void topo_test_polyzbndb_H(topo_t *T, _prec *out, const _prec *coef, const _prec
                       *deg, const int *shift)
{
        if (TOPO_DBG) {
                printf("launching %s(%d)\n", __func__, T->rank);
        }
        dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, BLOCK_SIZE_X);
        dim3 grid ((T->nz+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, 
                   (T->ny+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,
                   (T->nx+BLOCK_SIZE_X-1)/BLOCK_SIZE_X);
        //Differentiate  (`sxx = 0`)
        dtopo_test_polyzbnd<<<grid, block, 0, T->stream_2>>>(
                                         out, 
                                         T->off_x[1], T->off_x[2],
                                         0, ngsl,
                                         T->off_z[1], T->off_z[2],
                                         T->off_x[1], T->off_x[2],
                                         T->off_y[2] - ngsl, T->off_y[2],
                                         T->off_z[1], T->off_z[2],
                                         T->nx, T->ny, T->nz,
                                         T->line, T->slice_gl,
                                         T->coord[0], T->coord[1],
                                         coef[0], coef[1], coef[2],
                                         deg[0], deg[1], deg[2],
                                         shift[0], shift[1], shift[2]
                                         );
        CUCHK(cudaGetLastError());
        
        return;
}

__global__ void dtopo_test_diffx(_prec *xx, const _prec *u1,
                                    const int wi0, const int win,
                                    const int wj0, const int wjn,
                                    const int wk0, const int wkn,
                                    const int ri0, const int rin,
                                    const int rj0, const int rjn,
                                    const int rk0, const int rkn,
                                    const int wline, const int wslice,
                                    const int rline, const int rslice)

{
     const _prec dx[2] = {-0.0416666666666667, 1.1250000000000000};

     const int wk = threadIdx.x + blockIdx.x*blockDim.x + wk0;
     const int wj = threadIdx.y + blockIdx.y*blockDim.y + wj0;
     const int wi = threadIdx.z + blockIdx.z*blockDim.z + wi0;
     const int rk = threadIdx.x + blockIdx.x*blockDim.x + wk0;
     const int rj = threadIdx.y + blockIdx.y*blockDim.y + wj0;
     const int ri = threadIdx.z + blockIdx.z*blockDim.z + wi0;
     if (wk >= wkn || wj >= wjn || wi >= win) {
             return;
     }
     if (rk >= rkn || rj >= rjn || ri >= rin) {
             return;
     }

     int pos = wk + wline*wj + wslice*wi;
     xx[pos] = dx[0]*( u1[rk + rline*rj + rslice*(ri + 2)] 
                     - u1[rk + rline*rj + rslice*(ri - 1)]
                     )
             + dx[1]*(  u1[rk + rline*rj + rslice*(ri + 1)]       
                      - u1[rk + rline*rj + rslice*(ri + 0)] 
                     );                                   
}

__global__ void dtopo_test_diffx_111(_prec *xx, const _prec *u1, const int nx, const int ny, const int nz)
{
     const _prec dx[4] = {0.0416666666666667, -1.1250000000000000, 1.1250000000000000, -0.0416666666666667};
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if ( i >= nx) return;
     const int j = threadIdx.y + blockIdx.y*blockDim.y;
     if ( j >= ny) return;
     const int k = threadIdx.z + blockIdx.z*blockDim.z;
     if ( k >= nz) return;
     xx[align + k + (2*align + nz)*(i + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2)] = 
           + dx[0]*u1[align + k + (2*align + nz)*(i + ngsl + 1)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2)] 
           + dx[1]*u1[align + k + (2*align + nz)*(i + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2)] 
           + dx[2]*u1[align + k + (2*align + nz)*(i + ngsl + 3)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2)] 
           + dx[3]*u1[align + k + (2*align + nz)*(i + ngsl + 4)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2)];
     
}


__global__ void dtopo_test_diffy(_prec *yy, const _prec *v1,
                                    const int wi0, const int win,
                                    const int wj0, const int wjn,
                                    const int wk0, const int wkn,
                                    const int ri0, const int rin,
                                    const int rj0, const int rjn,
                                    const int rk0, const int rkn,
                                    const int wline, const int wslice,
                                    const int rline, const int rslice)

{
     const _prec dy[2] = {-0.0416666666666667, 1.1250000000000000};

     const int wk = threadIdx.x + blockIdx.x*blockDim.x + wk0;
     const int wj = threadIdx.y + blockIdx.y*blockDim.y + wj0;
     const int wi = threadIdx.z + blockIdx.z*blockDim.z + wi0;
     const int rk = threadIdx.x + blockIdx.x*blockDim.x + wk0;
     const int rj = threadIdx.y + blockIdx.y*blockDim.y + wj0;
     const int ri = threadIdx.z + blockIdx.z*blockDim.z + wi0;
     if (wk >= wkn || wj >= wjn || wi >= win) {
             return;
     }
     if (rk >= rkn || rj >= rjn || ri >= rin) {
             return;
     }

     int pos = wk + wline*wj + wslice*wi;
     yy[pos] = dy[0]*( v1[rk + rline*(rj + 2) + rslice*ri] 
                     - v1[rk + rline*(rj - 1) + rslice*ri]
                     )
             + dy[1]*(  v1[rk + rline*(rj + 1) + rslice*ri]       
                      - v1[rk + rline*(rj + 0) + rslice*ri] 
                     );                                   
}
__global__ void dtopo_test_diffz_111(_prec *xz, const _prec *u1, const int nx, const int ny, const int nz)
{
     const _prec dz[4] = {0.0416666666666667, -1.1250000000000000, 1.1250000000000000, -0.0416666666666667};
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if ( i >= nx) return;
     const int j = threadIdx.y + blockIdx.y*blockDim.y;
     if ( j >= ny) return;
     const int k = threadIdx.z + blockIdx.z*blockDim.z;
     if ( k >= nz - 5) return;
     // Hack used to only update the interior point for which there is data.
     if ( k <= 2) return;
     #define _xz(p,q,r) xz[align + (r) + (2*align + nz)*((p) + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*((q) + ngsl + 2)]
     #define _u1(p,q,r) u1[align + (r) + (2*align + nz)*((p) + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*((q) + ngsl + 2)]

     _xz(i,j,k) = dz[0]*_u1(i,j,k-1) + dz[1]*_u1(i,j,k) \
                + dz[2]*_u1(i,j,k+1) + dz[3]*_u1(i,j,k+2);

     #undef _xz
     #undef _u1
     
}

__global__ void dtopo_test_diffz_112(_prec *xz, const _prec *u1, const int nx, const int ny, const int nz)
{
     const _prec dzr[5][6] = {{0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000}, {2.4843703320382104, -2.6581943725716441, 0.1054629150477628, 0.0683611254856712, 0.0000000000000000, 0.0000000000000000}, {0.0788758473205719, 0.8521077862739277, -0.9014051908492852, -0.0295784427452145, 0.0000000000000000, 0.0000000000000000}, {-0.0147185348696016, -0.0162224835422866, 1.1130610406813668, -1.1259397586922681, 0.0438197364227896, 0.0000000000000000}, {-0.0040598373854470, 0.0051290309438727, -0.0391885057638776, 1.1187625510387915, -1.1222064403269296, 0.0415632014935900}};
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if ( i >= nx) return;
     const int j = threadIdx.y + blockIdx.y*blockDim.y;
     if ( j >= ny) return;
     const int k = threadIdx.z + blockIdx.z*blockDim.z;
     if ( k >= 5) return;
     xz[align + nz + (2*align + nz)*(i + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2) - 1 - k] = 
     dzr[k][5]*u1[align + nz + (2*align + nz)*(i + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2) - 6] + dzr[k][4]*u1[align + nz + (2*align + nz)*(i + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2) - 5] + dzr[k][3]*u1[align + nz + (2*align + nz)*(i + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2) - 4] + dzr[k][2]*u1[align + nz + (2*align + nz)*(i + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2) - 3] + dzr[k][1]*u1[align + nz + (2*align + nz)*(i + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2) - 2] + dzr[k][0]*u1[align + nz + (2*align + nz)*(i + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2) - 1];

     _prec out = xz[align + nz + (2*align + nz)*(i + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2) - 1 - k];
     if (TOPO_DBG > 1 && i == 10 && j == 10) {
             printf("out[%d] = %g in = %g %g %g %g %g %g \n", k, out,
                             u1[align + nz + (2*align + nz)*(i + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2) - 6],
                             u1[align + nz + (2*align + nz)*(i + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2) - 5],
                             u1[align + nz + (2*align + nz)*(i + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2) - 4],
                             u1[align + nz + (2*align + nz)*(i + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2) - 3],
                             u1[align + nz + (2*align + nz)*(i + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2) - 2],
                             u1[align + nz + (2*align + nz)*(i + ngsl + 2)*(2*ngsl + ny + 4) + (2*align + nz)*(j + ngsl + 2) - 1]
                             );
     }
     
}

__global__ void dtopo_test_poly(_prec *out, 
                                const int wi0, const int win,
                                const int wj0, const int wjn,
                                const int wk0, const int wkn,
                                const int ri0, const int rin,
                                const int rj0, const int rjn,
                                const int rk0, const int rkn,
                                const int nx, const int ny, const int nz,
                                const int line, const int slice,
                                const int rx, const int ry,
                                const _prec a0, const _prec a1, const _prec a2,
                                const _prec p0, const _prec p1, const _prec p2, 
                                const _prec s0, const _prec s1, const _prec s2)
{
     // Indices used for output
     const int wk = threadIdx.x + blockIdx.x*blockDim.x + wk0;
     if ( wk >= wkn) return;
     const int wj = threadIdx.y + blockIdx.y*blockDim.y + wj0;
     if ( wj >= wjn) return;
     const int wi = threadIdx.z + blockIdx.z*blockDim.z + wi0;
     if ( wi >= win) return;

     // Indices used for input
     const int rk = threadIdx.x + blockIdx.x*blockDim.x + rk0;
     if ( rk >= rkn) return;
     const int rj = threadIdx.y + blockIdx.y*blockDim.y + rj0;
     if ( rj >= rjn) return;
     const int ri = threadIdx.z + blockIdx.z*blockDim.z + ri0;
     if ( ri >= rin) return;
     
     const int pos = wk + wj*line + wi*slice;
     out[pos] = a0*pow((_prec)(ri + nx*rx - 0.5*s0), (_prec)p0) 
              + a1*pow((_prec)(rj + ny*ry - 0.5*s1), (_prec)p1) 
              + a2*pow((_prec)(rk         - 0.5*s2), (_prec)p2);
}


__global__ void dtopo_test_polyzbnd(_prec *out, 
                                    const int wi0, const int win,
                                    const int wj0, const int wjn,
                                    const int wk0, const int wkn,
                                    const int ri0, const int rin,
                                    const int rj0, const int rjn,
                                    const int rk0, const int rkn,
                                    const int nx, const int ny, const int nz,
                                    const int line, const int slice,
                                    const int rx, const int ry,
                                    const _prec a0, const _prec a1, const _prec a2,
                                    const _prec p0, const _prec p1, const _prec p2, 
                                    const _prec s0, const _prec s1, const _prec s2)
{
     // Indices used for output
     const int wk = threadIdx.x + blockIdx.x*blockDim.x + wk0;
     if ( wk >= wkn) return;
     const int wj = threadIdx.y + blockIdx.y*blockDim.y + wj0;
     if ( wj >= wjn) return;
     const int wi = threadIdx.z + blockIdx.z*blockDim.z + wi0;
     if ( wi >= win) return;

     // Indices used for input
     const int rk = threadIdx.x + blockIdx.x*blockDim.x + rk0;
     if ( rk >= rkn) return;
     const int rj = threadIdx.y + blockIdx.y*blockDim.y + rj0;
     if ( rj >= rjn) return;
     const int ri = threadIdx.z + blockIdx.z*blockDim.z + ri0;
     if ( ri >= rin) return;



/*
 *                                       n-4  n-3   n-2  n-1  
 *   z    ------o-----o-|---o-----o--|---o----o-----o---*
 *                      |            |     
 *                      |            |     
 *   zh   ---o-----o----|o-----o-----|^----o-----o--o
 *                      |            |n-4  n-3   n-2 n-1
 *
 *           Bottom           Interior           Top 
 */


     _prec zkp = 0.0; 
     if (rk == rkn - 1 && s2 == 1) {
           zkp = pow((_prec)(rkn - 2), (_prec)p2);
     } 
     else if (rk == rk0) {
        zkp = pow((_prec)rk, (_prec)p2);
     }   
     else if (rk == rkn - 1 && s2 == 0) {
           zkp = 0;
     } 
     else {
        zkp = pow((_prec)(rk- 0.5*s2), (_prec)p2);
     }
     
     const int pos = wk + wj*line + wi*slice;
     out[pos] = a0*pow((_prec)(ri + nx*rx - 0.5*s0),(_prec)p0) 
              + a1*pow((_prec)(rj + ny*ry - 0.5*s1),(_prec)p1) 
              + a2*zkp;
}
