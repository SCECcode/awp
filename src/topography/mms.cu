#include <stdio.h>
#include <stdlib.h>
#include <topography/mms.cuh>
#include <awp/pmcl3d_cons.h>

// Background values (P-wave speed, S-wave speed, density)
static float scp0, scs0, srho0;
// Perturbation values (P-wave speed, S-wave speed, density)
static float sdcp, sdcs, sdrho;
// Wave mode
static int smode;

__global__ void material_properties(
              const int nxt, const int nyt,
              const int nzt, float *d_d1, float *d_lam,
              float *d_mu,
              const float lam0, const float mu0, const float rho0, 
              const float dlam, const float dmu, const float drho, 
              const float mode, const float h, const int px, const int py
              ) {

                int i = threadIdx.z + blockDim.z * blockIdx.z;
                int j = threadIdx.y + blockDim.y * blockIdx.y;
                int k = align + threadIdx.x + blockDim.x * blockIdx.x;

                if (i < ngsl / 2 + 2 || i >= nxt - ngsl / 2 + 2) return;
                if (j < ngsl / 2 + 2 || j >= nyt - ngsl / 2 + 2) return;
                if (k >= align + nzt) return;
                
                float Lx =  (nxt - 1) * h;
                float Ly =  (nyt - 1) * h;
                float Lz =  (nzt - 1) * h;

                float x = (i - ngsl - 2) * h + px * Lx;
                float y = (j - ngsl - 2) * h + py * Ly;
                float z = h * k;

                int line = 2 * align + nzt;
                int slice = line * (4 + ngsl2 + nyt);
                int pos = k + line * j + slice * i;

                float S = sin(M_PI * mode * x / Lx) *
                          sin(M_PI * mode * y / Ly) * sin(M_PI * mode * z / Lz);

                d_d1[pos] = rho0 + drho * S;
                d_lam[pos] = lam0 + dlam * S;
                d_mu[pos] = mu0 + dmu * S;
                                        
                                        
}

void mms_init(const char *MMSFILE,
        const int *nxt, const int *nyt,
              const int *nzt, const int ngrids, float **d_d1, float **d_lam,
              float **d_mu, float **d_vx, float **d_vy, float **d_vz,
              float **d_xx, float **d_yy, float **d_zz, float **d_xy,
              float **d_xz, float **d_yz, int px, int py, const float *h)
{


        FILE *fh = fopen(MMSFILE, "r");
        if (!fh)  {
         if (px == 0 && py == 0) {
                 fprintf(stderr, "Failed to open: %s \n", MMSFILE);
                 exit(-1);
        }
                return; 
        }

 

        int parsed = fscanf(fh, "%f %f %f %f %f %f %d\n", &scp0, &scs0, &srho0, &sdcp, &sdcs, &sdrho, &smode);
        if (parsed != 7 && px == 0 && py == 0)
                 fprintf(stderr, "Failed to parse: %s \n", MMSFILE);

        if (px == 0 && py == 0) {
                printf("Done reading mms input file\n");
                printf("Settings: \n");
                printf("        cp0 = %g cs0 = %g rho0 = %g \n", scp0, scs0, srho0);
                printf("        dcp = %g dcs = %g drho = %g \n", sdcp, sdcs, sdrho);
                printf("        mode = %d \n", smode);


        }


        dim3 threads (32, 4, 1);
        for (int p = 0; p < ngrids; ++p) {
        
                int mz = nzt[p];
                int my = nyt[p] + 2 * ngsl;
                int mx = nxt[p] + 2 * ngsl;

                float mu0 = 1.0 / ( scs0 * scs0 * srho0);
                float dmu = 1.0 / ( sdcs * sdcs * sdrho);
                float lam0 = 1.0 / ( scp0 * scp0 * srho0 - 2.0 * scs0 * scs0 * srho0);
                float dlam = 1.0 / ( sdcp * sdcp * sdrho - 2.0 * sdcs * sdcs * sdrho);

                if (px == 0 && py == 0) printf("Setting material properties for grid = %d \n", p);
                // Set material properties
                dim3 blocks( (mz - 1) / threads.x + 1, (my - 1) / threads.y + 1, (mx - 1) / threads.z + 1);
                material_properties<<<blocks, threads>>>(nxt[p], nyt[p], nzt[p], d_d1[p], d_lam[p], d_mu[p],
                                lam0, mu0, srho0, dlam, dmu, sdrho, smode, h[p], px, py);

                cudaError_t cerr;
                CUCHK(cerr=cudaGetLastError());



        }

        exit(-1);

}

