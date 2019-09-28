#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <awp/definitions.h>
#include <topography/topography.h>
#include <topography/initializations/quadratic.h>

void topo_d_quadratic_i(topo_t *T, prec *d_field)
{
        int num_bytes = sizeof(_prec) * T->mx * T->my * T->mz;
        prec *tmp = (_prec*)malloc(num_bytes);
        printf("size: %d %d %d\n",T->mx, T->my, T->mz);
        for(int i = 0; i < T->mx; ++i) {
        for(int j = 0; j < T->my; ++j) {
        for(int k = 0; k < T->mz; ++k) {
                tmp[k + j * T->mz + i * T->mz * T->my] =  i * i;
        }
        }
        }

        CUCHK(cudaMemcpy(d_field, tmp, num_bytes, cudaMemcpyHostToDevice));

        free(tmp);
}

void topo_d_quadratic_j(topo_t *T, prec *d_field)
{
        int num_bytes = sizeof(_prec) * T->mx * T->my * T->mz;
        prec *tmp = (_prec*)malloc(num_bytes);
        for(int i = 0; i < T->mx; ++i) {
        for(int j = 0; j < T->my; ++j) {
        for(int k = 0; k < T->mz; ++k) {
                tmp[k + j * T->mz + i * T->mz * T->my] = j * j;
        }
        }
        }

        CUCHK(cudaMemcpy(d_field, tmp, num_bytes, cudaMemcpyHostToDevice));

        free(tmp);
}

void topo_d_quadratic_k(topo_t *T, prec *d_field)
{
        int num_bytes = sizeof(_prec) * T->mx * T->my * T->mz;
        prec *tmp = (_prec*)malloc(num_bytes);
        for(int i = 0; i < T->mx; ++i) {
        for(int j = 0; j < T->my; ++j) {
        for(int k = 0; k < T->mz; ++k) {
                tmp[k + j * T->mz + i * T->mz * T->my] = k * k;
        }
        }
        }

        CUCHK(cudaMemcpy(d_field, tmp, num_bytes, cudaMemcpyHostToDevice));

        free(tmp);
}
