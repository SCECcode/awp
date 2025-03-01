#ifndef _TOPOGRAPHY_VELOCITY_H
#define _TOPOGRAPHY_VELOCITY_H
#include <cuda.h>
#include <nvToolsExt.h>
#include <stdio.h>

#include <awp/definitions.h>
#include <topography/topography.h>

#ifdef __cplusplus
extern "C" {
#endif
void topo_velocity_interior_H(topo_t *T);
void topo_velocity_front_H(topo_t *T);
void topo_velocity_back_H(topo_t *T);
#ifdef __cplusplus
}
#endif

#endif
