#pragma once
#ifndef TOPO_INIT_RANDOM_H
#define TOPO_INIT_RANDOM_H


#include <awp/definitions.h>

#ifdef __cplusplus
extern "C" {
#endif

void topo_d_random(topo_t *T, const int seed, prec *d_field);

#ifdef __cplusplus
}
#endif
#endif

