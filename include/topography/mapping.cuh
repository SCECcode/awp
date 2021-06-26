#ifndef _TOPOGRAPHY_MAPPING_CUH
#define _TOPOGRAPHY_MAPPING_CUH
#include <topography/mapping.h>
#define TOL 1e-4

__device__ __host__ __inline__ float topo_mapping0(const float f, const float r,
                                                  const float h, const int n) {
        float l = (n - 2) * h;
        float d1 = h * 6.0;

        if (r < h * MAPPING_START_POINT) return r;
        else 
                return f * (r - h * MAPPING_START_POINT) + h * MAPPING_START_POINT;
}                             

// Differentiate mapping with respect to r1, r2  
__device__ __host__ __inline__ float topo_mapping(const float f_1, const float r,
                                                  const float h, const int n) {
        float l = (n - 2) * h;
        float d1 = h * 6.0;

        //return 0.0;
        return f_1 * (r - h * MAPPING_START_POINT);
        //return r*(d1 - r)*f_1/(d1 - l);
}

// Differentiate mapping with respect to r3
__device__ __host__ __inline__ float topo_diff_mapping(const float f,
                                                       const float r,
                                                       const float h,
                                                       const int n) {
        float l = (n - 2) * h;
        float d1 = h * 6.0;
        return f;
        //return  (- d1*f + l + 2*r*f - 2*r)/( l - d1);
}

#endif


