#include <awp/init.h>
#include <test/test.h>

/*
 * Determine GPU rank based on the number of available GPUs.
 *
 */ 
int init_gpu_rank(const int rank)
{
        int ngpu = 0;
        CUCHK(cudaGetDeviceCount(&ngpu));
        if (ngpu == 0) {
                printf ("No GPUs available\n");
                return -1;
        }
        int rank_gpu = rank % ngpu;
        CUCHK(cudaSetDevice(rank_gpu));
        return rank_gpu;
}

