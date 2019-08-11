#pragma once
#ifndef TOPOGRAPHY_HOST_H
#define TOPOGRAPHY_HOST_H
#ifdef __cplusplus
extern "C" {
#endif

#include <topography/topography.h>

void topo_h_malloc(topo_t *host);
void topo_dtoh(topo_t *host, const topo_t *device);
void topo_h_free(topo_t *host);

#ifdef __cplusplus
}
#endif
#endif

