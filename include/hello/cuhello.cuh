#ifndef HELLO_CUH
#define HELLO_CUH

#include <stdio.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif
        void hello_h(void);
#ifdef __cplusplus
}
#endif

__global__ void hello_d(double *a);
#endif

#define cudachk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}
