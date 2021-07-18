#include <topography/energy.cuh>

__global__ void energy_kernel(
    double *kinetic_rate, double *strain_rate, const float *vxp,
    const float *vyp, const float *vzp, const float *xxp, const float *yyp,
    const float *zzp, const float *xyp, const float *xzp, const float *yzp,
    const float *vx, const float *vy, const float *vz, const float *xx,
    const float *yy, const float *zz, const float *xy, const float *xz,
    const float *yz, const float *f, const float *f_1, const float *f_2,
    const float *f_c, const float *g3, const float *g3_c,
    const float *rho, const float *mui, const float *lami,
    const int nx, const int ny, const int nz) {
    int idz = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    int my = ny + 4 + 2 * ngsl;
    int mz = nz + 2 * align;

    int line = mz;
    int slice = my * mz;
    int offset = idz + align + line * (2 + ngsl + idy) + slice * (2 + ngsl);

    int fline = 2 * align + 4 + ny + 2 * ngsl;
    int f_offset = 2 + ngsl + idy + align + fline * (2 + ngsl);
    int g_offset = align + idz;

    double kinetic_E = 0.0;
    double strain_E = 0.0;

    float Hz_hat = 1.0f;
    float Hz = 1.0f;

    const float hhzr[5] = {0.3445563972099920, 0.4372900885645984,
                           1.3056954965124901, 0.9124580177129197,
                           1.0000000000000000};
    const float hzr[5] = {0.0000000000000000, 0.2812150147607664,
                          1.4480216223843674, 0.6769783776156325,
                          1.0937849852392336};


    int k = nz - idz - 1;
    if (k < 5 && k >= 0) {
        Hz_hat = hhzr[k];
        Hz = hzr[k];
    }

    int pos = offset;
    int fpos = f_offset;
    int gpos = g_offset;

    if (idz >= nz) {
        pos = 0;
        fpos = 0;
        gpos = 0;
    }

    for (int i = 0; i < nx; ++i) {


        float Jx = f_1[fpos] * g3_c[gpos];
        float Jy = f_2[fpos] * g3_c[gpos];
        float Jz = f_c[fpos] * g3[gpos];
        float Jxx = f_c[fpos] * g3_c[gpos];
        float Jxy = f[fpos] * g3_c[gpos];
        float Jxz = f_1[fpos] * g3[gpos];
        float Jyz = f_2[fpos] * g3[gpos];

        float rhox = 0.25f * (rho[pos - 1] + rho[pos - line - 1] + rho[pos] + rho[pos - line]);
        float rhoy = 0.25f * (rho[pos - 1] + rho[pos + slice - 1] + rho[pos] + rho[pos + slice]);
        float rhoz = 0.25f * (rho[pos] + rho[pos + slice] + rho[pos - line] + rho[pos + slice - line]);


        float muixy = 0.5f * (mui[pos] + mui[pos-1]);
        float muixz = 0.5f * (mui[pos] + mui[pos-line]);
        float muiyz = 0.5f * (mui[pos] + mui[pos+slice]);
        float lamixx = 
            (lami[pos - 1] + lami[pos - 1 + slice] + lami[pos - 1 + slice - line] +
             lami[pos - line - 1] + lami[pos] + lami[pos + slice] +
             lami[pos + slice - line] + lami[pos - line]) / 8.f;
        float lam = 1.0f / lamixx;
        float muixx =
            (mui[pos - 1] + mui[pos - 1 + slice] + mui[pos - 1 + slice - line] +
             mui[pos - line - 1] + mui[pos] + mui[pos + slice] +
             mui[pos + slice - line] + mui[pos - line]) / 8.f;
        float mu = 1.0f / muixx;
        float lam_mu = 0.5f * lam / (mu * (3.0f * lam + 2.0f * mu));
        float trace =
            (xx[pos] - xxp[pos]) + (yy[pos] - yyp[pos]) + (zz[pos] - zzp[pos]);

        double exx = 0.5f * muixx * ((double)xx[pos] - (double)xxp[pos]) - lam_mu * trace;
        double eyy = 0.5f * muixx * ((double)yy[pos] - (double)yyp[pos]) - lam_mu * trace;
        double ezz = 0.5f * muixx * ((double)zz[pos] - (double)zzp[pos]) - lam_mu * trace;
        double exy = 0.5f * muixy * ((double)xy[pos] - (double)xyp[pos]);
        double exz = 0.5f * muixz * ((double)xz[pos] - (double)xzp[pos]);
        double eyz = 0.5f * muiyz * ((double)yz[pos] - (double)yzp[pos]);

        kinetic_E += 0.5f * Jx * Hz_hat * vx[pos] * rhox * ((double)vx[pos] - (double)vxp[pos]) +
                     0.5f * Jy * Hz_hat * vy[pos] * rhoy * ((double)vy[pos] - (double)vyp[pos]) +
                     0.5f * Jz * Hz * vz[pos] * rhoz * ((double)vz[pos] - (double)vzp[pos]);
        strain_E +=
            0.5f * xxp[pos] * Jxx * Hz_hat * exx +
            0.5f * yyp[pos] * Jxx * Hz_hat * eyy +
            0.5f * zzp[pos] * Jxx * Hz_hat * ezz +
            1.0f * xyp[pos] * Jxy * Hz_hat * exy +
            1.0f * xzp[pos] * Jxz * Hz * exz + 
            1.0f * yzp[pos] * Jyz * Hz * eyz;

        pos += slice;
        fpos += fline;
    }

    if (idz > nz - 1 || idz < 0) {
        kinetic_E = 0;
        strain_E = 0;
    }

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

void energy_rate(energy_t *e, int step, const float *d_vx, const float *d_vy,
                 const float *d_vz, const float *d_xx, const float *d_yy,
                 const float *d_zz, const float *d_xy, const float *d_xz,
                 const float *d_yz, const float *d_rho, const float *d_mui,
                 const float *d_lami, 
                 const f_grid_t *metrics_f,
                 const g_grid_t *metrics_g,
                 const int nx, const int ny, const int nz)
{
    if (!e->use || step >= e->num_steps) return;

        double out_kinetic[1] = {0.0};
        double out_strain[1] = {0.0};
    CUCHK(cudaMemset(e->kinetic_rate, 0, sizeof(double)));
    CUCHK(cudaMemset(e->strain_rate, 0, sizeof(double)));
    
    dim3 threads (32, 4, 1);
    dim3 blocks ( (nz - 4) / threads.x  + 1 , (ny - 1) / threads.y + 1, 1);
    energy_kernel<<<blocks, threads>>>(e->kinetic_rate, e->strain_rate, e->d_vxp, e->d_vyp, e->d_vzp, e->d_xxp, e->d_yyp, e->d_zzp, e->d_xyp, e->d_xzp, e->d_yzp, d_vx, d_vy, d_vz, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, metrics_f->d_f, metrics_f->d_f_1, metrics_f->d_f_2, metrics_f->d_f_c, metrics_g->d_g3, metrics_g->d_g3_c, d_rho, d_mui, d_lami, nx, ny, nz);
    CUCHK(cudaGetLastError());
    cudaMemcpy(out_kinetic, e->kinetic_rate, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_strain, e->strain_rate, sizeof(double), cudaMemcpyDeviceToHost);
    CUCHK(cudaGetLastError());
    e->kinetic_energy_rate[step] = out_kinetic[0];
    e->strain_energy_rate[step] = out_strain[0];
    printf("kinetic = %g strain = %g , sum = %g \n", out_kinetic[0], out_strain[0], out_kinetic[0] + out_strain[0]);

}
