#pragma once
#ifndef TOPO_INIT_CONSTANT_H
#define TOPO_INIT_CONSTANT_H

#ifdef __cplusplus
extern "C" {
#endif
void topo_d_zero_init(topo_t *T);

void topo_d_constant(topo_t *T, const prec value, prec *d_field);
void topo_d_constanti(topo_t *T, const int value, int *d_field);

#ifdef __cplusplus
}
#endif
#endif
