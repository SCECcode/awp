#ifndef _TOPOGRAPHY_MAPPING_H
#define _TOPOGRAPHY_MAPPING_H

#define TOL 1e-4
#define MAPPING_START_POINT 7

__device__ __host__ __inline__ float topo_mapping0(const float f, const float r,
                                                  const float h, const int n) {
        float l = (n - 2) * h;
        float d1 = h * 6.0;

        //printf("f = %g \n", f); 
        if (r < h * MAPPING_START_POINT) return (r - h * MAPPING_START_POINT);
        else 
                return f * (r - h * MAPPING_START_POINT);
        //return (r*(-l + r) - r*(-d1 + r)*f)/(d1 - l);
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


