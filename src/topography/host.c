#include <stdio.h>
#include <cuda_runtime.h>

#include <awp/definitions.h>
#include <topography/host.h>

void topo_h_malloc(topo_t *host)
{
        int num_bytes = sizeof(_prec) * host->mx * host->my * host->mz;
        host->u1 = (prec*)malloc(num_bytes);
        host->v1 = (prec*)malloc(num_bytes);
        host->w1 = (prec*)malloc(num_bytes);
        host->xx = (prec*)malloc(num_bytes);
        host->yy = (prec*)malloc(num_bytes);
        host->zz = (prec*)malloc(num_bytes);
        host->xy = (prec*)malloc(num_bytes);
        host->xz = (prec*)malloc(num_bytes);
        host->yz = (prec*)malloc(num_bytes);
        
}

void topo_dtoh(topo_t *host, const topo_t *device)
{
        int num_bytes = sizeof(_prec) * device->mx * device->my * device->mz;
        CUCHK(cudaMemcpy(host->u1, device->u1, num_bytes,
                         cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(host->v1, device->v1, num_bytes,
                         cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(host->w1, device->w1, num_bytes,
                         cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(host->xx, device->xx, num_bytes,
                         cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(host->yy, device->yy, num_bytes,
                         cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(host->zz, device->zz, num_bytes,
                         cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(host->xy, device->xy, num_bytes,
                         cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(host->xz, device->xz, num_bytes,
                         cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(host->yz, device->yz, num_bytes,
                         cudaMemcpyDeviceToHost));
}

void topo_h_free(topo_t *host)
{
        free(host->u1);
        free(host->v1);
        free(host->w1);
        free(host->xx);
        free(host->yy);
        free(host->zz);
        free(host->xy);
        free(host->xz);
        free(host->yz);
}

