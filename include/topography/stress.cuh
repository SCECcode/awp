#ifndef _TOPOGRAPHY_STRESS_H
#define _TOPOGRAPHY_STRESS_H

#include <awp/definitions.h>
#include <topography/topography.h>
#include <topography/opt_topography.cuh>
#include <topography/stress.cuh>

#ifdef __cplusplus
extern "C" {
#endif
void topo_set_constants(topo_t *T);
void topo_stress_interior_H(topo_t *T);
void topo_stress_left_H(topo_t *T);
void topo_stress_right_H(topo_t *T);
#ifdef __cplusplus
}
#else
void topo_set_constants(topo_t *T);
void topo_stress_interior_H(topo_t *T);
void topo_stress_left_H(topo_t *T);
void topo_stress_right_H(topo_t *T);
#endif

#endif
