#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <awp/definitions.h>
#include <buffers/buffer.h>
#include <test/test.h>

buffer_t buffer_init(size_t num_elements, size_t num_device,
                         size_t num_host, size_t stride) 
{
        buffer_t out = {
            .num_device = num_device,
            .num_host = num_host,
            .stride = stride,
            .num_elements = num_elements,
            .d_offset = 0,
            .h_offset = 0,
        };

        size_t num_bytes =
            sizeof(prec) * num_elements * num_device;
        out.d_buffer_bytes = num_bytes;
        CUCHK(cudaMalloc((void**)&out.d_buffer, num_bytes));
        CUCHK(cudaMemset(out.d_buffer, 0, num_bytes));
        num_bytes *= num_host;
        size_t num_elem = num_elements * num_device * num_host;
        out.h_buffer_bytes = sizeof(prec) * num_elem;
        out.h_buffer = calloc(sizeof out.h_buffer, num_elem);


        return out;
}

void buffer_finalize(buffer_t *buffer)
{
       CUCHK(cudaFree(buffer->d_buffer));
       free(buffer->h_buffer); 
}

prec* buffer_get_device_ptr(buffer_t *buffer, size_t step)
{
        buffer->d_offset = ((step / buffer->stride) % buffer->num_device) *
                           buffer->num_elements;
        return &buffer->d_buffer[buffer->d_offset];
}

prec* buffer_get_host_ptr(buffer_t *buffer, size_t step)
{
        buffer->h_offset =
            ((step / buffer->stride / buffer->num_device) % buffer->num_host) *
            buffer->num_device * buffer->num_elements;
        prec *ptr = &buffer->h_buffer[buffer->h_offset];

        return ptr;
}

int buffer_is_device_ready(const buffer_t *buffer, size_t step)
{
        return (step % buffer->stride == 0);
}

int buffer_is_device_full(const buffer_t *buffer, size_t step)
{
        return step % buffer->stride == 0 &&
               (step / buffer->stride) % buffer->num_device ==
                   (buffer->num_device - 1);
}

int buffer_is_device_empty(const buffer_t *buffer, size_t step)
{
        return step % buffer->stride == 0 &&
               (step / buffer->stride) % buffer->num_device == 0;
}

int buffer_is_host_full(const buffer_t *buffer, size_t step)
{
        return buffer_is_device_full(buffer, step) &&
               (step / buffer->stride / buffer->num_device) % buffer->num_host ==
                   (buffer->num_host - 1);
}

int buffer_is_host_empty(const buffer_t *buffer, size_t step)
{
        return buffer_is_device_empty(buffer, step) &&
               (step / buffer->stride / buffer->num_device) %
                       buffer->num_host ==
                   0;
}

void buffer_copy_to_device(buffer_t *buffer, size_t step)
{
       if (!buffer_is_device_empty(buffer, step)) return;

       CUCHK(cudaMemcpy(buffer->d_buffer, buffer_get_host_ptr(buffer, step),
                        buffer->d_buffer_bytes, cudaMemcpyHostToDevice));
}

void buffer_copy_to_host(buffer_t *buffer, size_t step)
{
       if (!buffer_is_device_full(buffer, step)) return;

       CUCHK(cudaMemcpy(buffer_get_host_ptr(buffer, step), buffer->d_buffer,
                        buffer->d_buffer_bytes, cudaMemcpyDeviceToHost));
}

