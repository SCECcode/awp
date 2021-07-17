#include <topography/energy.cuh>

__global__ void energy_kernel(double *temp, const float *vxp, const float *vyp, const float *vzp,
        const float *xxp, const float *yyp, const float *zzp, const float *xyp, const float *xzp,
        const float *yzp, const float *vx, const float *vy, const float *vz, const
        float *xx, const float *yy, const float *zz, const float *xy, const float *xz, const float
        *yz, const float *rho, const float *mui, const float *lami, const int nx, const int ny,
        const int nz) {
                int idz = threadIdx.x + blockDim.x * blockIdx.x;
                int idy = threadIdx.y + blockDim.y * blockIdx.y;

                int my = ny + 4 + 2 * ngsl;
                int mz = nz + 2 * align;

                int block = my * mz;
                int offset = idz + mz * (2 + ngsl + idy) + align + block * (2 + ngsl);

                double kinetic_E = 0.0;
                double strain_E = 0.0;

                int pos = offset;
                for (int i = 0; i < nx; ++i) {
                    pos += block;
                    double rhox = rho[pos];
                    double Hxp = mui[pos];
                    kinetic_E += vx[pos] * rhox * (vx[pos] - vxp[pos]);
                    strain_E += xyp[pos] * Hxp * (xy[pos] - xyp[pos]);
                }

                double partial_E = kinetic_E + strain_E;

                __shared__ double spartial_E[1024];

                double val = partial_E;
                for (int i = 16; i > 0; i /= 2)
                        val += __shfl_down_sync(0xffffffff, val, i);

                if (threadIdx.x == 0) spartial_E[threadIdx.y] = val;

                __syncthreads();

                if (threadIdx.x == 0 && threadIdx.y == 0) {
                        double block_E = 0.0;
                        for (int i = 0; i < blockDim.y; ++i) {
                                block_E += spartial_E[i];
                        }

                        atomicAdd(temp, block_E);
                }
}

void energy_rate(energy_t *e, int step, const float *d_vx, const float *d_vy, const float *d_vz, const float *d_xx, const float *d_yy, const float *d_zz, const float *d_xy, const float *d_xz, const float *d_yz, const float *d_rho, const float *d_mui, const float *d_lami, const int nx, const int ny, const int nz) {
    if (!e->use || step >= e->num_steps) return;

        double out[1] = {1.0};
    CUCHK(cudaMemset(e->rate, 0, sizeof(double)));
    
    dim3 threads (32, 4, 1);
    //printf("n = %d %d %d \n", nz, ny, nx);
    dim3 blocks ( (nz - 1) / threads.x  + 1 , (ny - 1) / threads.y + 1, 1);
    //printf("blocks = %d %d %d \n", blocks.x, blocks.y, blocks.z);
    energy_kernel<<<blocks, threads>>>(e->rate, e->d_vxp, e->d_vyp, e->d_vzp, e->d_xxp, e->d_yyp, e->d_zzp, e->d_xyp, e->d_xzp, e->d_yzp, d_vx, d_vy, d_vz, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, d_rho, d_mui, d_lami, nx, ny, nz);
    CUCHK(cudaGetLastError());
    cudaMemcpy(out, e->rate, sizeof(double), cudaMemcpyDeviceToHost);
    CUCHK(cudaGetLastError());
    e->kinetic_energy_rate[step] = out[0];
    printf("energy = %g \n", out[0]);

}
