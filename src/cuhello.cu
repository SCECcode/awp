#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <hello/cuhello.cuh>

void hello_h(void)
{
        unsigned int size = 16;
        // dim3 block = {size, 1, 1};
        // dim3 grid = {1, 1, 1};
				//CUDA-9.0 syntax
				dim3 block(size, 1, 1);
				dim3 grid(1, 1, 1);
        double *d_data;
        int num_bytes = size*sizeof(double);

        cudaMalloc(&d_data, num_bytes);
        double *h_data = (double*)malloc(num_bytes);

        hello_d<<<grid, block>>>(d_data);
        cudachk( cudaPeekAtLastError() );

        cudaMemcpy(h_data, d_data, num_bytes, cudaMemcpyDeviceToHost);
        cudachk( cudaPeekAtLastError() );

        cudachk( cudaDeviceSynchronize() );
        for (int i = 0; i < size; ++i) {
                printf("a[%d] = %g\n", i, h_data[i]);
        }

        cudaFree(d_data);
        free(h_data);
}

__global__ void hello_d(double *a)
{
        int thread = threadIdx.x;
        a[thread] = (double)thread;
        printf("GPU[%d]: Hello world!\n", thread);
}
