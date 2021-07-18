#include <topography/energy.cuh>

__global__ void energy_kernel(double *kinetic_rate, double *strain_rate, const float *vxp, const float *vyp, const float *vzp,
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

                float Hz_hat = 1.0f;
                float Hz = 1.0f;

                const float hhzr[5] = {0.3445563972099920, 0.4372900885645984,
                                       1.3056954965124901, 0.9124580177129197,
                                       1.0000000000000000};
                const float hzr[5] = {
                    0.0000000000000000, 0.2812150147607664, 1.4480216223843674,
                    0.6769783776156325, 1.0937849852392336};

                int k = nz - idz - 1;
                if (k < 5 && k >= 0) {
                    Hz_hat = hhzr[k];
                    Hz = hzr[k];
                }


                int pos = offset;
                for (int i = 0; i < nx; ++i) {
                    pos += block;
                    float rhox = rho[pos];
                    float rhoy = rho[pos];
                    float rhoz = rho[pos];
                    float muxy = mui[pos];
                    float muxz = mui[pos];
                    float muyz = mui[pos];
                    float lam = 1.0f / lami[pos];
                    float mu = 1.0f / mui[pos];
                    float lam_mu = 0.5f * lam / (mu * (3.0f * lam + 2.0f * mu));
                    float trace = (xx[pos] - xxp[pos]) + (yy[pos] - yyp[pos]) + (zz[pos] - zzp[pos]);

                    float exx = 0.5f * mui[pos] * (xx[pos] - xxp[pos]) - lam_mu * trace;
                    float eyy = 0.5f * mui[pos] * (yy[pos] - yyp[pos]) - lam_mu * trace;
                    float ezz = 0.5f * mui[pos] * (zz[pos] - zzp[pos]) - lam_mu * trace;
                    float exy = 0.5f * mui[pos] * (xy[pos] - xyp[pos]);
                    float exz = 0.5f * mui[pos] * (xz[pos] - xzp[pos]);
                    float eyz = 0.5f * mui[pos] * (yz[pos] - yzp[pos]);

                    kinetic_E += 
                        0.5f * Hz_hat * vx[pos] * rhox * (vx[pos] - vxp[pos]) +
                        0.5f * Hz_hat * vy[pos] * rhoy * (vy[pos] - vyp[pos]) +
                        0.5f * Hz * vz[pos] * rhoz * (vz[pos] - vzp[pos]);
                    strain_E += 0.5f * xxp[pos] * Hz_hat * exx + 
                                0.5f * yyp[pos] * Hz_hat * eyy +
                                0.5f * zzp[pos] * Hz_hat * ezz + 
                                1.0f * xyp[pos] * Hz_hat * exy +
                                1.0f * xzp[pos] * Hz * exz + 
                                1.0f * yzp[pos] * Hz * eyz;
                }

                if (idz > nz - 1 || idz < 0) {
                    kinetic_E = 0;
                    strain_E = 0;
                }

                //if (idz < 8 || idz > nz - 8) { // || idy < 100 || idy > ny - 100) {
                //    kinetic_E = 0;
                //    strain_E = 0;
                //}

                //if (idz < 21 || idz > nz - 21 || idy < 101 || idy > ny - 101) {
                //}

                __shared__ double spartial_kinetic[1024];
                __shared__ double spartial_strain[1024];

                double kin = kinetic_E;
                double str = strain_E;
                for (int i = 16; i > 0; i /= 2) {
                        kin += __shfl_down_sync(0xffffffff, kin, i);
                        str += __shfl_down_sync(0xffffffff, str, i);
                }

                if (threadIdx.x == 0) {
                    spartial_kinetic[threadIdx.y] = kin;
                    spartial_strain[threadIdx.y] = str;

                }   
                __syncthreads();

                if (threadIdx.x == 0 && threadIdx.y == 0) {
                        double block_strain_rate = 0.0;
                        double block_kinetic_rate = 0.0;
                        for (int i = 0; i < blockDim.y; ++i) {
                                block_kinetic_rate += spartial_kinetic[i];
                                block_strain_rate += spartial_strain[i];
                        }

                        atomicAdd(strain_rate, block_strain_rate);
                        atomicAdd(kinetic_rate, block_kinetic_rate);
                }
}

void energy_rate(energy_t *e, int step, const float *d_vx, const float *d_vy, const float *d_vz, const float *d_xx, const float *d_yy, const float *d_zz, const float *d_xy, const float *d_xz, const float *d_yz, const float *d_rho, const float *d_mui, const float *d_lami, const int nx, const int ny, const int nz) {
    if (!e->use || step >= e->num_steps) return;

        double out_kinetic[1] = {0.0};
        double out_strain[1] = {0.0};
    CUCHK(cudaMemset(e->kinetic_rate, 0, sizeof(double)));
    CUCHK(cudaMemset(e->strain_rate, 0, sizeof(double)));
    
    dim3 threads (32, 4, 1);
    //printf("n = %d %d %d \n", nz, ny, nx);
    dim3 blocks ( (nz - 4) / threads.x  + 1 , (ny - 1) / threads.y + 1, 1);
    //printf("blocks = %d %d %d \n", blocks.x, blocks.y, blocks.z);
    energy_kernel<<<blocks, threads>>>(e->kinetic_rate, e->strain_rate, e->d_vxp, e->d_vyp, e->d_vzp, e->d_xxp, e->d_yyp, e->d_zzp, e->d_xyp, e->d_xzp, e->d_yzp, d_vx, d_vy, d_vz, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, d_rho, d_mui, d_lami, nx, ny, nz);
    CUCHK(cudaGetLastError());
    cudaMemcpy(out_kinetic, e->kinetic_rate, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_strain, e->strain_rate, sizeof(double), cudaMemcpyDeviceToHost);
    CUCHK(cudaGetLastError());
    e->kinetic_energy_rate[step] = out_kinetic[0];
    e->strain_energy_rate[step] = out_strain[0];
    printf("kinetic = %g strain = %g , sum = %g \n", out_kinetic[0], out_strain[0], out_kinetic[0] + out_strain[0]);

}
