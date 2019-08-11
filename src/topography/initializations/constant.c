#include <cuda_runtime.h>
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

void topo_d_constant(topo_t *T)
{


}
