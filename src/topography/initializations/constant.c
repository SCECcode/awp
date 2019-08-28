#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include <awp/definitions.h>
#include <topography/topography.h>
#include <topography/initializations/constant.h>

void topo_d_zero_init(topo_t *T)
{
        prec *ptrs[] = {T->rho,   T->lami,  T->mui,  T->xx,   T->yy,   T->zz,
                        T->xy,    T->xz,    T->yz,   T->r1,   T->r2,   T->r3,
                        T->r4,    T->r5,    T->r6,   T->u1,   T->v1,   T->w1,
                        T->f_u1,  T->f_v1,  T->f_w1, T->b_u1, T->b_v1, T->b_w1
                        };
        int num_bytes = sizeof(_prec) * T->mx * T->my * T->mz;
        for (int i = 0; i < 24; ++i) {
                cudaMemset(ptrs[i], 0, num_bytes);
        }
}

void topo_d_constant(topo_t *T, const prec value, prec *d_field)
{
        int num_bytes = sizeof(_prec) * T->mx * T->my * T->mz;
        prec *tmp = (_prec*)malloc(num_bytes);
        for(int i = 0; i < T->mx; ++i) {
        for(int j = 0; j < T->my; ++j) {
        for(int k = 0; k < T->mz; ++k) {
                tmp[k + j * T->mz + i * T->mz * T->my] = value;
        }
        }
        }

        CUCHK(cudaMemcpy(d_field, tmp, num_bytes, cudaMemcpyHostToDevice));

        free(tmp);
}

void topo_d_constanti(topo_t *T, const int value, int *d_field)
{
        int num_bytes = sizeof(int) * T->mx * T->my * T->mz;
        int *tmp = (int*)malloc(num_bytes);
        for(int i = 0; i < T->mx; ++i) {
        for(int j = 0; j < T->my; ++j) {
        for(int k = 0; k < T->mz; ++k) {
                tmp[k + j * T->mz + i * T->mz * T->my] = value;
        }
        }
        }

        CUCHK(cudaMemcpy(d_field, tmp, num_bytes, cudaMemcpyHostToDevice));

        free(tmp);
}
