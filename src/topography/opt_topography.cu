#include <cuda.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <stdio.h>

#include <topography/topography.h>
#include <topography/opt_topography.cuh>
#include <topography/kernels/optimized_launch_config.cuh>
#include <topography/kernels/optimized_velocity.cuh>
#include <awp/definitions.h>

void topo_init_material_H(topo_t *T)
{
        fprintf(stderr, "Not Implemented\n");
}


