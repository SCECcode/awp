#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <awp/definitions.h>
#include <functions/random.h>
#include <topography/topography.h>
#include <topography/initializations/random.h>

void topo_d_random(topo_t *T, const int seed, prec *d_field)
{
        int num_bytes = sizeof(_prec) * T->mx * T->my * T->mz;
        prec *tmp = (_prec*)malloc(num_bytes);
        set_seed(seed);
        for(int i = 0; i < T->mx; ++i) {
        for(int j = 0; j < T->my; ++j) {
        for(int k = 0; k < T->mz; ++k) {
                tmp[k + j * T->mz + i * T->mz * T->my] = randomf();
        }
        }
        }

        CUCHK(cudaMemcpy(d_field, tmp, num_bytes, cudaMemcpyHostToDevice));

        free(tmp);
}
