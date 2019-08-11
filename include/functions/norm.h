#ifndef FUNCTION_NORM_H
#define FUNCTION_NORM_H
#ifdef __cplusplus
extern "C" {
#endif

double l2norm(const float *a_in, 
              const int nx, const int ny, const int nz,
              const int i0, const int in, 
              const int j0, const int jn, 
              const int k0, const int kn);

#ifdef __cplusplus
}
#endif
#endif

