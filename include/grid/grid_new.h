#pragma once
#include <awp/definitions.h>


_prec grid_overlap(const _prec h);
_prec grid_height(const int nz, const _prec h, const int istopo);

void global_to_local(_prec *zloc, int *block_index, const _prec z,
                     const _prec h, const int *nz, const int num_grids,
                     const int istopo);
