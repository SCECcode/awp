#include <math.h>

#ifndef DEBUG_H
#define DEBUG_H

void zeros(float *out, int nx, int ny, int nz);
void check_values(const char *label, const float *in, int nx, int ny, int nz, int rank);
void zeros(float *out, int nx, int ny, int nz)
{
        for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
                int offset = k + j*nz + i*ny*nz;
                out[offset] = 0.0f;
        }
        }
        }
}

void check_values(const char *label, const float *in, int nx, int ny, int nz, int rank)
{
                int is_big = 0;
                int i0 = 0;
                int j0 = 0;
                int k0 = 0;
                float value = 0.0;
                for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                        int offset = k + j*nz + i*ny*nz;
                        if ( fabs(in[offset]) > 1e6) {
                                i0 = i;
                                j0 = j;
                                k0 = k;
                                value = in[offset];
                                is_big = 1;
                                break;
                        }
                }
                }
                }
                if (is_big) {
                        printf("rank = %d %s[%d,%d,%d] = %g \n", rank, label, i0, j0, k0,
                                        value); 
                        exit(1);
                }

} 

#endif

