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

// Solution values
static float svx0, svy0, svz0, sxx0, syy0, szz0, sxy0, sxz0, syz0;
static float sdvx, sdvy, sdvz, sdxx, sdyy, sdzz, sdxy, sdxz, sdyz;


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

__inline__ __device__ float length_x(int nx, float h) {
        return (nx - 1) * h;
}

__inline__ __device__ float length_y(int ny, float h) {
        return (ny - 1) * h;
}

__inline__ __device__ float length_z(int nz, float h) {
        return (nz - 1) * h;
}

__inline__ __device__ float xi(int i, int px, float Lx, float h) {
                return (i - ngsl - 2) * h + px * Lx;
}

__inline__ __device__ float yj(int j, int py, float Ly, float h) {
                return (j - ngsl - 2) * h + py * Ly;
}

__inline__ __device__ float zk(int k, int pz, float Lz, float h) {
                return (k - align) * h + pz * Lz;
}

__inline__ __device__ float wavenumber(float mode, float L) {
                return M_PI * mode / L;
}

__inline__ __device__ float material_perturbation(float x, float y, float z, float kx, float ky, float kz) {
        return sin(kx * x) * sin(ky * y) * sin(kz * z);
}


__global__ void material_properties(
              const int nx, const int ny,
              const int nz, float *d_d1, float *d_lam,
              float *d_mu, float *d_qp, float *d_qs,
              const float lam0, const float mu0, const float rho0, 
              const float dlam, const float dmu, const float drho, 
              const float mode, const float h, const int px, const int py, const int pz
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
              const int nx, const int ny, const int nz, 
              float *d_vx, float *d_vy, float *d_vz, 
              float *d_xx, float *d_yy, float *d_zz, 
              float *d_xy, float *d_xz, float *d_yz, 
              float vx0, float vy0, float vz0, 
              float xx0, float yy0, float zz0, 
              float xy0, float xz0, float yz0, 
              float dvx, float dvy, float dvz, 
              float dxx, float dyy, float dzz, 
              float dxy, float dxz, float dyz, 
              float cp0, float cs0, 
              float dcp, float dcs, 
              float mode, const float h, const int px, const int py, const int pz, 
              float t
              ) {

                int i = threadIdx.z + blockDim.z * blockIdx.z;
                int j = threadIdx.y + blockDim.y * blockIdx.y;
                int k = align + threadIdx.x + blockDim.x * blockIdx.x;

                if (!in_bounds_velocity(nx, ny, nz, i, j, k)) return;
                
                float Lx =  length_x(nx, h);
                float Ly =  length_y(ny, h);
                float Lz =  length_z(nz, h);

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
                float cp = cp0 + dcp * S;
                float cs = cs0 + dcs * S;

                float om_p = cp / k;

                d_vz[pos] = vz0 + dvz * sin(kz * z) * sin(om_p * t);
                
                                        
}

__global__ void exact_stress(
              const int nx, const int ny, const int nz, 
              float *d_vx, float *d_vy, float *d_vz, 
              float *d_xx, float *d_yy, float *d_zz, 
              float *d_xy, float *d_xz, float *d_yz, 
              float vx0, float vy0, float vz0, 
              float xx0, float yy0, float zz0, 
              float xy0, float xz0, float yz0, 
              float dvx, float dvy, float dvz, 
              float dxx, float dyy, float dzz, 
              float dxy, float dxz, float dyz, 
              float cp0, float cs0, 
              float dcp, float dcs, 
              float mode, const float h, const int px, const int py, const int pz, 
              float t
              ) {

                int i = threadIdx.z + blockDim.z * blockIdx.z;
                int j = threadIdx.y + blockDim.y * blockIdx.y;
                int k = align + threadIdx.x + blockDim.x * blockIdx.x;

                if (!in_bounds_stress(nx, ny, nz, i, j, k)) return;
                
                float Lx =  length_x(nx, h);
                float Ly =  length_y(ny, h);
                float Lz =  length_z(nz, h);

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
                float cp = cp0 + dcp * S;
                float cs = cs0 + dcs * S;

                float om_p = cp / k;

                d_zz[pos] = zz0 + dzz * sin(kz * z) * cos(om_p * t);
                
                                        
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
                            "%f | %f %f %f %f %f %f %f %f %f \n",
                            &scp0, &scs0, &srho0, &sdcp, &sdcs, &sdrho, &mode,
                            &svx0, &svy0, &svz0, &sxx0, &syy0, &szz0, &sxy0,
                            &sxz0, &syz0, &sdvx, &sdvy, &sdvz, &sdxx, &sdyy,
                            &sdzz, &sdxy, &sdxz, &sdyz);
        if (parsed != 19 && px == 0 && py == 0)
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
        }


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
                material_properties<<<blocks, threads>>>(nxt[p], nyt[p], nzt[p], d_d1[p], d_lam[p], d_mu[p], d_qp[p], d_qs[p],
                                lam0, mu0, srho0, dlam, dmu, sdrho, smode, h[p], px, py, p);

                if (px == 0 && py == 0) printf("Setting velocity initial conditions for grid = %d \n", p);
                // Set initial conditions for velocity
                exact_velocity<<<blocks, threads>>>(nxt[p], nyt[p], nzt[p], 
                                d_vx[p], d_vy[p], d_vz[p],
                                d_xx[p], d_yy[p], d_zz[p], 
                                d_xy[p], d_xz[p], d_yz[p], 
                                svx0, svy0, svz0, 
                                sxx0, syy0, szz0,
                                sxy0, sxz0, syz0,
                                sdvx, sdvy, sdvz, 
                                sdxx, sdyy, sdzz,
                                sdxy, sdxz, sdyz,
                                scp0, scs0, sdcp, sdcs, smode, h[p], px, py, p, 0.0f);

                if (px == 0 && py == 0) printf("Setting stress initial conditions for grid = %d \n", p);
                // Set initial conditions for stress
                exact_stress<<<blocks, threads>>>(nxt[p], nyt[p], nzt[p], 
                                d_vx[p], d_vy[p], d_vz[p],
                                d_xx[p], d_yy[p], d_zz[p], 
                                d_xy[p], d_xz[p], d_yz[p], 
                                svx0, svy0, svz0, 
                                sxx0, syy0, szz0,
                                sxy0, sxz0, syz0,
                                sdvx, sdvy, sdvz, 
                                sdxx, sdyy, sdzz,
                                sdxy, sdxz, sdyz,
                                scp0, scs0, sdcp, sdcs, smode, h[p], px, py, p, 0.0f + 0.5f * dt);

                cudaError_t cerr;
                CUCHK(cerr=cudaGetLastError());



        }

                if (px == 0 && py == 0) printf("MMS initialization done. \n");
}

