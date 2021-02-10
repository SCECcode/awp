#include <stdio.h>
#include <stdlib.h>
#include <topography/mms.cuh>
#include <awp/pmcl3d_cons.h>

// Background values (P-wave speed, S-wave speed, density)
static float scp0, scs0, srho0;
// Perturbation values (P-wave speed, S-wave speed, density)
static float sdcp, sdcs, sdrho;
// Wave mode
static float smode;

// Background values (velocities and stresses)
static float svx0, svy0, svz0, sxx0, syy0, szz0, sxy0, sxz0, syz0;
// Perturbation values (velocities and stresses)
static float sdvx, sdvy, sdvz, sdxx, sdyy, sdzz, sdxy, sdxz, sdyz;

// Plane wave position
static float szc;


__inline__ __device__ int in_bounds_stress(int nx, int ny, int nz, int i, int j, int k) {
                if (i < ngsl / 2 + 2 || i >= nx + 3 * ngsl / 2 + 2) return 0;
                if (j < ngsl / 2 + 2 || j >= ny + 3 * ngsl / 2 + 2) return 0;
                if (k >= align + nz) return 0;
                return 1;
}

__inline__ __device__ int in_bounds_velocity(int nx, int ny, int nz, int i, int j, int k) {
                if (i < ngsl + 2 || i >= nx + ngsl + 2) return 0;
                if (j < ngsl + 2 || j >= ny + ngsl + 2) return 0;
                if (k >= align + nz) return 0;
                return 1;
}

__inline__ __device__ int in_bounds(int i, int j, int k, int bi, int bj, int bk,
                                    int ei, int ej, int ek) {
        if (i < bi || j < bj || k - align < bk) return 0;
        if (i >= ei || j >= ej || k - align >= ek) return 0;
        return 1;
}

__inline__ __device__ float length_x(int nx, float h) {
        return (nx - 1) * h;
}

__inline__ __device__ float length_y(int ny, float h) {
        return (ny - 1) * h;
}

__inline__ __device__ float length_z(int nz, float h) {
        return (nz - 2) * h;
}

__inline__ __device__ float xi(int i, int px, float Lx, float h, int hat=0) {
                return (i - ngsl - 2 - hat * 0.5f) * h + px * Lx;
}

__inline__ __device__ float yj(int j, int py, float Ly, float h, int hat=0) {
                float shift = hat == 0? 0.0f : 0.5f;
                return (j - ngsl - 2 - hat * 0.5f) * h + py * Ly;
}

__inline__ __device__ float zk(int k, int pz, float Lz, float h, int hat=0) {
                return (k - align - hat * 0.5f) * h + pz * Lz;
}

__inline__ __device__ float wavenumber(float mode, float L) {
                return M_PI * mode / L;
}

__inline__ __device__ float material_perturbation(float x, float y, float z, float kx, float ky, float kz) {
        return sin(kx * x) * sin(ky * y) * sin(kz * z);
}

__inline__ __device__ float gaussian(float z, float z0, float t, float k, float om) {
        float tau = k * (z - z0)  + om * t;
        return exp( - tau * tau );
}



__global__ void material_properties(
              float *d_d1, float *d_lam,
              float *d_mu, float *d_qp, float *d_qs,
              const float lam0, const float mu0, const float rho0, 
              const float dlam, const float dmu, const float drho, 
              const float mode, 
              const int nx, const int ny,
              const int nz, 
              const int px, const int py, const int pz, 
              const float h
              ) {

                int i = threadIdx.z + blockDim.z * blockIdx.z;
                int j = threadIdx.y + blockDim.y * blockIdx.y;
                int k = align + threadIdx.x + blockDim.x * blockIdx.x;

                if (!in_bounds_stress(nx, ny, nz, i, j, k)) return;
                
                float Lx = length_x(nx, h);
                float Ly = length_y(ny, h);
                float Lz = length_z(nz, h);

                float x = xi(i, px, Lx, h);
                float y = yj(j, py, Ly, h);
                float z = zk(k, pz, Lz, h);

                int line = 2 * align + nz;
                int slice = line * (4 + 2 * ngsl + ny);
                int pos = k + line * j + slice * i;

                float kx = wavenumber(mode, Lx);
                float ky = wavenumber(mode, Ly);
                float kz = wavenumber(mode, Lz);

                float S = material_perturbation(x, y, z, kx, ky, kz);
                
                d_d1[pos] = rho0 + drho * S;
                d_lam[pos] = 1.0f / (lam0 + dlam * S);
                d_mu[pos] = 1.0f /  (mu0 + dmu * S);
                d_qp[pos] = 1e-10;
                d_qs[pos] = 1e-10;
                                        
}

__global__ void exact_velocity(
              float *d_vx, float *d_vy, float *d_vz, 
              const float vx0, const float vy0, const float vz0, 
              const float xx0, const float yy0, const float zz0, 
              const float xy0, const float xz0, const float yz0, 
              const float dvx, const float dvy, const float dvz, 
              const float dxx, const float dyy, const float dzz, 
              const float dxy, const float dxz, const float dyz, 
              const float cp0, const float cs0, const float rho0,
              const float dcp, const float dcs, const float drho,
              const float zc, 
              const float mode,
              const int nx, const int ny, const int nz, 
              const int px, const int py, const int pz, 
              const int bi, const int bj, const int bk, 
              const int ei, const int ej, const int ek, 
              const float h, const float t, 
              const int apply_in_interior
              ) {

                int i = threadIdx.z + blockDim.z * blockIdx.z;
                int j = threadIdx.y + blockDim.y * blockIdx.y;
                int k = align + threadIdx.x + blockDim.x * blockIdx.x;

                if (!in_bounds_velocity(nx, ny, nz, i, j, k)) return;
                int is_in_bounds = in_bounds(i, j, k, bi, bj, bk, ei, ej, ek);
                if (apply_in_interior && !is_in_bounds) return;
                if (!apply_in_interior && is_in_bounds) return;
                
                float Lx =  length_x(nx, h);
                float Ly =  length_y(ny, h);
                float Lz =  length_z(nz, h);

                float x = xi(i, px, Lx, h);
                float y = yj(j, py, Ly, h);
                float z = zk(k, pz, Lz, h);
                float zh = zk(k, pz, Lz, h, 1);

                int line = 2 * align + nz;
                int slice = line * (4 + 2 * ngsl + ny);
                int pos = k + line * j + slice * i;

                float kx = wavenumber(mode, Lx);
                float ky = wavenumber(mode, Ly);
                float kz = wavenumber(mode, Lz);

                float S = material_perturbation(x, y, z, kx, ky, kz);
                float cp = cp0 + dcp * S;
                float cs = cs0 + dcs * S;

                float om_p = cp * kz;


                d_vx[pos] = 0.0f;
                d_vy[pos] = 0.0f;
                d_vz[pos] = vz0 + dvz * gaussian(z, zc, t, kz, om_p);
                
                                        
}

__global__ void exact_stress(
              float *d_xx, float *d_yy, float *d_zz, 
              float *d_xy, float *d_xz, float *d_yz, 
              const float vx0, const float vy0, const float vz0, 
              const float xx0, const float yy0, const float zz0, 
              const float xy0, const float xz0, const float yz0, 
              const float dvx, const float dvy, const float dvz, 
              const float dxx, const float dyy, const float dzz, 
              const float dxy, const float dxz, const float dyz, 
              const float cp0, const float cs0, const float rho0,
              const float dcp, const float dcs, const float drho, 
              const float zc, 
              const float mode, 
              const int nx, const int ny, const int nz, 
              const int px, const int py, const int pz, 
              const int bi, const int bj, const int bk, 
              const int ei, const int ej, const int ek, 
              const float h, const float t, const int apply_in_interior
              ) {

                int i = threadIdx.z + blockDim.z * blockIdx.z;
                int j = threadIdx.y + blockDim.y * blockIdx.y;
                int k = align + threadIdx.x + blockDim.x * blockIdx.x;

                if (!in_bounds_stress(nx, ny, nz, i, j, k)) return;
                int is_in_bounds = in_bounds(i, j, k, bi, bj, bk, ei, ej, ek);
                if (apply_in_interior && !is_in_bounds) return;
                if (!apply_in_interior && is_in_bounds) return;
                
                float Lx =  length_x(nx, h);
                float Ly =  length_y(ny, h);
                float Lz =  length_z(nz, h);

                float x = xi(i, px, Lx, h);
                float y = yj(j, py, Ly, h);
                float z = zk(k, pz, Lz, h);
                float zh = zk(k, pz, Lz, h, 1);

                int line = 2 * align + nz;
                int slice = line * (4 + 2 * ngsl + ny);
                int pos = k + line * j + slice * i;

                float kx = wavenumber(mode, Lx);
                float ky = wavenumber(mode, Ly);
                float kz = wavenumber(mode, Lz);

                float S = material_perturbation(x, y, z, kx, ky, kz);
                float cp = cp0 + dcp * S;
                float cs = cs0 + dcs * S;
                float rho = rho0;

                float om_p = cp * kz;

                d_xx[pos] = 0.0f;
                d_yy[pos] = 0.0f;
                d_zz[pos] = rho * cp * ( zz0 + dzz * gaussian(z, zc, t, kz, om_p) );

                
                d_xy[pos] = 0.0f;
                d_xz[pos] = 0.0f;
                d_yz[pos] = 0.0f;
                                        
}

__global__ void force_velocity(
                float *d_vx, float *d_vy, float *d_vz,
                const float vx0, const float vy0, const float vz0, 
                const float xx0, const float yy0, const float zz0, 
                const float xy0, const float xz0, const float yz0, 
                const float dvx, const float dvy, const float dvz, 
                const float dxx, const float dyy, const float dzz, 
                const float dxy, const float dxz, const float dyz, 
                const float cp0, const float cs0, const float rho0,
                const float dcp, const float dcs, const float drho, 
                const float mode,
                const int nx, const int ny, const int nz,
                const int px, const int py, const int pz,
                const float h, const float t, const float dt) {

                int i = threadIdx.z + blockDim.z * blockIdx.z;
                int j = threadIdx.y + blockDim.y * blockIdx.y;
                int k = align + threadIdx.x + blockDim.x * blockIdx.x;

                if (!in_bounds_velocity(nx, ny, nz, i, j, k)) return;

                float Lx =  length_x(nx, h);
                float Ly =  length_y(ny, h);
                float Lz =  length_z(nz, h);
 
                float kx = wavenumber(mode, Lx);
                float ky = wavenumber(mode, Ly);
                float kz = wavenumber(mode, Lz);

                float x = xi(i, px, Lx, h);
                float y = yj(j, py, Ly, h);
                float z = zk(k, pz, Lz, h);

                int line = 2 * align + nz;
                int slice = line * (4 + 2 * ngsl + ny);
                int pos = k + line * j + slice * i;

                float S = material_perturbation(x, y, z, kx, ky, kz);

                float cp = cp0 + dcp * S;
                float cs = cs0 + dcs * S;
                float rho = rho0 + drho * S;
                float om_p = cp * kz;
}

__global__ void force_stress(
                float *d_xx, float *d_yy, float *d_zz, 
                float *d_xy, float *d_xz, float *d_yz,
                const float vx0, const float vy0, const float vz0, 
                const float xx0, const float yy0, const float zz0, 
                const float xy0, const float xz0, const float yz0, 
                const float dvx, const float dvy, const float dvz, 
                const float dxx, const float dyy, const float dzz, 
                const float dxy, const float dxz, const float dyz, 
                const float cp0, const float cs0, const float rho0,
                const float dcp, const float dcs, const float drho,
                const float mode,
                const int nx, const int ny, const int nz,
                const int px, const int py, const int pz,
                const float h, const float t, const float dt) {

                int i = threadIdx.z + blockDim.z * blockIdx.z;
                int j = threadIdx.y + blockDim.y * blockIdx.y;
                int k = align + threadIdx.x + blockDim.x * blockIdx.x;

                if (!in_bounds_stress(nx, ny, nz, i, j, k)) return;

                float Lx =  length_x(nx, h);
                float Ly =  length_y(ny, h);
                float Lz =  length_z(nz, h);
 
                float kx = wavenumber(mode, Lx);
                float ky = wavenumber(mode, Ly);
                float kz = wavenumber(mode, Lz);

                float x = xi(i, px, Lx, h);
                float y = yj(j, py, Ly, h);
                float z = zk(k, pz, Lz, h);
                float zh = zk(k, pz, Lz, h, 1);

                int line = 2 * align + nz;
                int slice = line * (4 + 2 * ngsl + ny);
                int pos = k + line * j + slice * i;

                float S = material_perturbation(x, y, z, kx, ky, kz);
                float cp = cp0 + dcp * S;
                float cs = cs0 + dcs * S;
                float rho = rho0 + drho * S;

                float om_p = cp * kz;

                float lam = rho * (cp * cp  - 2 * cs * cs);
                float mu = rho * cs * cs;
}

void mms_init(const char *MMSFILE,
        const int *nxt, const int *nyt,
              const int *nzt, const int ngrids, float **d_d1, float **d_lam,
              float **d_mu,
              float **d_qp, float **d_qs,
              float **d_vx, float **d_vy, float **d_vz,
              float **d_xx, float **d_yy, float **d_zz, float **d_xy,
              float **d_xz, float **d_yz, int px, int py, const float *h, const float dt)
{


        FILE *fh = fopen(MMSFILE, "r");
        if (!fh)  {
         if (px == 0 && py == 0) {
                 fprintf(stderr, "Failed to open: %s \n", MMSFILE);
                 exit(-1);
        }
                return; 
        }

        int mode = 0;

        int parsed = fscanf(fh,
                            "%f %f %f %f %f %f | %d | %f %f %f %f %f %f %f %f "
                            "%f | %f %f %f %f %f %f %f %f %f | %f \n",
                            &scp0, &scs0, &srho0, &sdcp, &sdcs, &sdrho, &mode,
                            &svx0, &svy0, &svz0, &sxx0, &syy0, &szz0, &sxy0,
                            &sxz0, &syz0, &sdvx, &sdvy, &sdvz, &sdxx, &sdyy,
                            &sdzz, &sdxy, &sdxz, &sdyz, &szc);
        if (parsed != 26 && px == 0 && py == 0)
                 fprintf(stderr, "Failed to parse: %s \n", MMSFILE);

        smode = (float)mode;

        if (px == 0 && py == 0) {
                printf("Done reading mms input file\n");
                printf("Settings: \n");
                printf("        cp0 = %g cs0 = %g rho0 = %g \n", scp0, scs0, srho0);
                printf("        dcp = %g dcs = %g drho = %g \n", sdcp, sdcs, sdrho);
                printf("        mode = %g \n", smode);
                printf("        vx0 = %g vy0 = %g vz0 = %g \n", svx0, svy0, svz0);
                printf("        xx0 = %g yy0 = %g zz0 = %g \n", sxx0, syy0, szz0);
                printf("        xy0 = %g xz0 = %g yz0 = %g \n", sxy0, sxz0, syz0);
                printf("        dvx = %g dvy = %g dvz = %g \n", sdvx, sdvy, sdvz);
                printf("        dxx = %g dyy = %g dzz = %g \n", sdxx, sdyy, sdzz);
                printf("        dxy = %g dxz = %g dyz = %g \n", sdxy, sdxz, sdyz);
                printf("        zc = %g \n", szc);
        }


        const int INTERIOR = 1;
        dim3 threads (32, 4, 1);
        for (int p = 0; p < ngrids; ++p) {
        
                int mz = nzt[p];
                int my = nyt[p] + 2 * ngsl + 4;
                int mx = nxt[p] + 2 * ngsl + 4;

                float mu0 = scs0 * scs0 * srho0;
                float dmu = sdcs * sdcs * sdrho;
                float lam0 = scp0 * scp0 * srho0 - 2.0 * scs0 * scs0 * srho0;
                float dlam = sdcp * sdcp * sdrho - 2.0 * sdcs * sdcs * sdrho;
                printf("mu0 = %g lam0 = %g dlam = %g dmu = %g \n", mu0, lam0, dlam, dmu);

                if (px == 0 && py == 0) printf("Setting material properties for grid = %d \n", p);
                // Set material properties
                dim3 blocks( (mz - 1) / threads.x + 1, (my - 1) / threads.y + 1, (mx - 1) / threads.z + 1);
                material_properties<<<blocks, threads>>>(
                    d_d1[p], d_lam[p], d_mu[p], d_qp[p], d_qs[p], lam0, mu0,
                    srho0, dlam, dmu, sdrho, smode, nxt[p], nyt[p], nzt[p], px,
                    py, p, h[p]);

                if (px == 0 && py == 0) printf("Setting velocity initial conditions for grid = %d \n", p);
                // Set initial conditions for velocity
                mms_exact_velocity(
                                d_vx[p], d_vy[p], d_vz[p],
                                nxt[p], nyt[p], nzt[p], 
                                px, py, p, 
                                0, 0, 0, 
                                2 + 2 * ngsl + nxt[p], 4 + 2 * ngsl + nyt[p], nzt[p], 
                                h[p], 0.0f, INTERIOR);

                if (px == 0 && py == 0) printf("Setting stress initial conditions for grid = %d \n", p);
                // Set initial conditions for stress
                mms_exact_stress(
                                d_xx[p], d_yy[p], d_zz[p], 
                                d_xy[p], d_xz[p], d_yz[p], 
                                nxt[p], nyt[p], nzt[p], 
                                px, py, p, 
                                0, 0, 0, 
                                4 + 2 * ngsl + nxt[p], 4 + 2 * ngsl + nyt[p], nzt[p], 
                                h[p], 0.0f - 0.5f * dt, INTERIOR);
                CUCHK(cudaGetLastError());



        }

                if (px == 0 && py == 0) printf("MMS initialization done. \n");
}

void mms_exact_velocity(
              float *d_vx, float *d_vy, float *d_vz,
              const int nx, const int ny, const int nz, 
              const int px, const int py, const int pz, 
              const int bi, const int bj, const int bk, 
              const int ei, const int ej, const int ek, 
              const float h, const float t, const int apply_in_interior)
{
                int mz = nz;
                int my = ny + 2 * ngsl + 4;
                int mx = nx + 2 * ngsl + 4;
                dim3 threads(32, 4, 1);
                dim3 blocks((mz - 1) / threads.x + 1, (my - 1) / threads.y + 1,
                            (mx - 1) / threads.z + 1);

                exact_velocity<<<blocks, threads>>>(
                                d_vx, d_vy, d_vz,
                                svx0, svy0, svz0, 
                                sxx0, syy0, szz0,
                                sxy0, sxz0, syz0,
                                sdvx, sdvy, sdvz, 
                                sdxx, sdyy, sdzz,
                                sdxy, sdxz, sdyz,
                                scp0, scs0, srho0,
                                sdcp, sdcs, sdrho, 
                                szc, 
                                smode, 
                                nx, ny, nz, 
                                px, py, pz, 
                                bi, bj, bk,
                                ei, ej, ek,
                                h, t, apply_in_interior);
                CUCHK(cudaGetLastError());
}

void mms_exact_stress(
              float *d_xx, float *d_yy, float *d_zz, 
              float *d_xy, float *d_xz, float *d_yz, 
              const int nx, const int ny, const int nz, 
              const int px, const int py, const int pz, 
              const int bi, const int bj, const int bk, 
              const int ei, const int ej, const int ek, 
              const float h, const float t, const int apply_in_interior)
{
                int mz = nz;
                int my = ny + 2 * ngsl + 4;
                int mx = nx + 2 * ngsl + 4;
                dim3 threads(32, 4, 1);
                dim3 blocks((mz - 1) / threads.x + 1, (my - 1) / threads.y + 1,
                            (mx - 1) / threads.z + 1);
                exact_stress<<<blocks, threads>>>(
                                d_xx, d_yy, d_zz, 
                                d_xy, d_xz, d_yz, 
                                svx0, svy0, svz0, 
                                sxx0, syy0, szz0,
                                sxy0, sxz0, syz0,
                                sdvx, sdvy, sdvz, 
                                sdxx, sdyy, sdzz,
                                sdxy, sdxz, sdyz,
                                scp0, scs0, srho0,
                                sdcp, sdcs, sdrho, 
                                szc, 
                                smode, 
                                nx, ny, nz, 
                                px, py, pz, 
                                bi, bj, bk, 
                                ei, ej, ek, 
                                h, t, apply_in_interior);
                CUCHK(cudaGetLastError());

}

void mms_force_velocity(float *d_vx, float *d_vy, float *d_vz, const int nx, const int ny, const int nz, const float h, const int px, const int py, const int pz, const float t, const float dt)
{
                int mz = nz;
                int my = ny + 2 * ngsl + 4;
                int mx = nx + 2 * ngsl + 4;
                dim3 threads (32, 4, 1);
                dim3 blocks( (mz - 1) / threads.x + 1, (my - 1) / threads.y + 1, (mx - 1) / threads.z + 1);
                force_velocity<<<blocks, threads>>>(
                    d_vx, d_vy, d_vz,
                    svx0, svy0, svz0, 
                    sxx0, syy0, szz0,
                    sxy0, sxz0, syz0,
                    sdvx, sdvy, sdvz, 
                    sdxx, sdyy, sdzz,
                    sdxy, sdxz, sdyz,
                    scp0, scs0, srho0,
                    sdcp, sdcs, sdrho, 
                    smode, 
                    nx, ny, nz, 
                    px, py, pz, h, t, dt);
                CUCHK(cudaGetLastError());
}
void mms_force_stress(float *d_xx, float *d_yy, float *d_zz, float *d_xy,
                      float *d_xz, float *d_yz, const int nx, const int ny, const int nz,
                      const float h, const int px, const int py, const int pz, const float t, const float dt) {
        int mz = nz;
        int my = ny + 2 * ngsl + 4;
        int mx = nx + 2 * ngsl + 4;
        dim3 threads(32, 4, 1);
        dim3 blocks( (mz - 1) / threads.x + 1, (my - 1) / threads.y + 1, (mx - 1) / threads.z + 1);
        force_stress<<<blocks, threads>>>(
                    d_xx, d_yy, d_zz, 
                    d_xy, d_xz, d_yz, 
                    svx0, svy0, svz0, 
                    sxx0, syy0, szz0,
                    sxy0, sxz0, syz0,
                    sdvx, sdvy, sdvz, 
                    sdxx, sdyy, sdzz,
                    sdxy, sdxz, sdyz,
                    scp0, scs0, srho0,
                    sdcp, sdcs, sdrho, 
                    smode, 
                    nx, ny, nz, 
                    px, py, pz, h, t, dt);
        CUCHK(cudaGetLastError());
}

