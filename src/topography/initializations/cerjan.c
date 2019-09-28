#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <topography/topography.h>
#include <topography/initializations/constant.h>
#include <topography/initializations/cerjan.h>

void topo_d_cerjan_disable(topo_t *T)
{
        // Setting all cerjan sponge layer coefficients to unity disables the
        // layer
        prec *ones =  (prec*)malloc(sizeof(prec) * T->mx * T->my * T->mz);
        for (int i = 0; i < T->mx * T->my * T->mz; ++i) {
                ones[i] = 1.0;
        }
        CUCHK(cudaMemcpy(T->dcrjx, ones, T->mx, cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(T->dcrjy, ones, T->my, cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(T->dcrjz, ones, T->mz, cudaMemcpyHostToDevice));
        free(ones);
}

