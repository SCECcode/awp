#include <math.h>
#include <functions/norm.h>

double l2norm(const float *a_in, 
              const int nx, const int ny, const int nz,
              const int i0, const int in, 
              const int j0, const int jn, 
              const int k0, const int kn)
{
        double sum = 0.0;
        for (int i = i0; i < in; ++i) {
        for (int j = j0; j < jn; ++j) {
        for (int k = k0; k < kn; ++k) {
                sum += (double)pow(a_in[k + j * nz + i * nz * ny], 2);
        }
        }
        }
        return sqrt(sum);
}
