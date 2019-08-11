#ifndef _TOPOGRAPHY_H
#define _TOPOGRAPHY_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <topography/topography.h>
#include <awp/definitions.h>
#ifdef __cplusplus
extern "C" {
#endif
void topo_init_material_H(topo_t *T);
void topo_velocity_interior_H(topo_t *T);
void topo_velocity_front_H(topo_t *T);
void topo_velocity_back_H(topo_t *T);
void topo_stress_interior_H(topo_t *T);
void topo_stress_left_H(topo_t *T);
void topo_stress_right_H(topo_t *T);
#ifdef __cplusplus
}
#endif

// Number of threads per block to use
#ifndef TBX
#define TBX 1
#endif
#ifndef TBY
#define TBY 1
#endif
#ifndef TBZ
#define TBZ 16
#endif

// Number of boundary points. Must match the number of boundary stencils used in
// the interpolation and differentiation stencils
#define BOTTOM_BOUNDARY_SIZE 7
#define TOP_BOUNDARY_SIZE 7



#endif

