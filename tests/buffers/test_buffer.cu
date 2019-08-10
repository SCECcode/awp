#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <awp/definitions.h>
#include <buffers/buffer.h>
#include <test/test.h>
#include <test/check.h>
#include <test/array.h>

#define PRINT 0


int test_is_device_ready(void);
int test_is_device_empty(void);
int test_is_device_full(void);
int test_is_host_full(void);
int test_is_host_empty(void);
int test_get_device_ptr(void);
int test_get_host_ptr(void);
int test_copy_to_host(void);
int test_copy_to_device(void);


int main(int argc, char **argv)
{
        int err = 0;
        test_divider();
        printf("Testing buffer.c\n");
        err |= test_is_device_ready();
        err |= test_is_device_full();
        err |= test_is_device_empty();
        err |= test_is_host_full();
        err |= test_is_host_empty();
        err |= test_get_device_ptr();
        err |= test_get_host_ptr();
        err |= test_copy_to_host();
        err |= test_copy_to_device();
        printf("Testing completed.\n");
        test_divider();
        return err;
}

int test_is_device_ready(void)
{
        int err = 0;
        int stride = 2;
        int num_elements = 10;
        int num_host = 2;
        int num_device = 2;
        buffer_t buffer = buffer_init(num_elements, num_device, num_host,
                                          stride);

        test_t test = test_init(" * buffer_is_device_ready", 0, 0);
        err |= s_assert(buffer_is_device_ready(&buffer, 0) == 1);
        err |= s_assert(buffer_is_device_ready(&buffer, 1) == 0);
        err |= s_assert(buffer_is_device_ready(&buffer, stride - 1) == 0);
        err |= s_assert(buffer_is_device_ready(&buffer, stride) == 1);
        err |= s_assert(buffer_is_device_ready(&buffer, 2 * stride) == 1);
        
        buffer_finalize(&buffer);

        err = test_finalize(&test, err);

        return test_last_error();
}

int test_is_device_full(void)
{
        int err = 0;
        int stride = 2;
        int num_elements = 10;
        int num_host = 2;

        test_t test = test_init(" * buffer_is_device_full", 0, 0);

        {
        int num_device = 1;
        buffer_t buffer = buffer_init(num_elements, num_device, num_host,
                                          stride);
        
        err |= s_assert(buffer_is_device_full(&buffer, 0) == 1);
        err |= s_assert(buffer_is_device_full(&buffer, 1) == 0);
        err |= s_assert(buffer_is_device_full(&buffer, stride) == 1);
        err |= s_assert(buffer_is_device_full(&buffer, 2 * stride) == 1);
        
        buffer_finalize(&buffer);
        }

        {
        int num_device = 4;
        buffer_t buffer = buffer_init(num_elements, num_device, num_host,
                                          stride);
        
        err |= s_assert(buffer_is_device_full(&buffer, 0) == 0);
        err |= s_assert(buffer_is_device_full(&buffer, 1) == 0);
        err |= s_assert(buffer_is_device_full(&buffer, stride) == 0);
        err |= s_assert(buffer_is_device_full(&buffer, 3 * stride) == 1);
        err |= s_assert(buffer_is_device_full(&buffer, 7 * stride) == 1);
        
        buffer_finalize(&buffer);
        }

        err = test_finalize(&test, err);

        return test_last_error();
}

int test_is_device_empty(void)
{
        int err = 0;
        int stride = 2;
        int num_elements = 10;
        int num_host = 2;

        test_t test = test_init(" * buffer_is_device_empty", 0, 0);

        {
        int num_device = 1;
        buffer_t buffer = buffer_init(num_elements, num_device, num_host,
                                         stride);

        
        err |= s_assert(buffer_is_device_empty(&buffer, 0) == 1);
        err |= s_assert(buffer_is_device_empty(&buffer, 1) == 0);
        err |= s_assert(buffer_is_device_empty(&buffer, stride) == 1);
        err |= s_assert(buffer_is_device_empty(&buffer, 2 * stride) == 1);
        
        buffer_finalize(&buffer);
        }

        {
        int num_device = 4;
        buffer_t buffer = buffer_init(num_elements, num_device, num_host,
                                          stride);

        
        err |= s_assert(buffer_is_device_empty(&buffer, 0) == 1);
        err |= s_assert(buffer_is_device_empty(&buffer, 1) == 0);
        err |= s_assert(buffer_is_device_empty(&buffer, stride) == 0);
        err |= s_assert(buffer_is_device_empty(&buffer, 4 * stride) == 1);
        err |= s_assert(buffer_is_device_empty(&buffer, 8 * stride) == 1);
        
        buffer_finalize(&buffer);
        }

        err = test_finalize(&test, err);

        return test_last_error();
}

int test_is_host_full(void)
{
        int err = 0;
        int stride = 2;
        int num_elements = 10;

        test_t test = test_init(" * buffer_is_host_full", 0, 0);

        {

        int num_device = 2;
        int num_host = 1;
        buffer_t buffer = buffer_init(num_elements, num_device, num_host,
                                          stride);

        err |= s_assert(buffer_is_host_full(&buffer, 0) == 0);
        err |= s_assert(buffer_is_host_full(&buffer, 1) == 0);
        err |= s_assert(buffer_is_host_full(&buffer, stride) == 1);
        err |= s_assert(buffer_is_host_full(&buffer, 3 * stride) == 1);
        err |= s_assert(buffer_is_host_full(&buffer, 3 * stride + 1) == 0);
        
        buffer_finalize(&buffer);

        }


        {

        int num_device = 2;
        int num_host = 2;
        buffer_t buffer = buffer_init(num_elements, num_device, num_host,
                                          stride);

        err |= s_assert(buffer_is_host_full(&buffer, 0) == 0);
        err |= s_assert(buffer_is_host_full(&buffer, 1) == 0);
        err |= s_assert(buffer_is_host_full(&buffer, stride) == 0);
        err |= s_assert(buffer_is_host_full(&buffer, 3 * stride) == 1);
        err |= s_assert(buffer_is_host_full(&buffer, 3 * stride + 1) == 0);
        
        buffer_finalize(&buffer);

        }

        err = test_finalize(&test, err);

        return test_last_error();
}

int test_is_host_empty(void)
{
        int err = 0;
        int stride = 2;
        int num_elements = 10;

        test_t test = test_init(" * buffer_is_host_empty", 0, 0);

        {

        int num_device = 2;
        int num_host = 1;
        buffer_t buffer = buffer_init(num_elements, num_device, num_host,
                                         stride);

        err |= s_assert(buffer_is_host_empty(&buffer, 0) == 1);
        err |= s_assert(buffer_is_host_empty(&buffer, 1) == 0);
        err |= s_assert(buffer_is_host_empty(&buffer, stride) == 0);
        err |= s_assert(buffer_is_host_empty(&buffer, 2 * stride) == 1);
        err |= s_assert(buffer_is_host_empty(&buffer, 2 * stride + 1) == 0);
        
        buffer_finalize(&buffer);

        }


        {

        int num_device = 2;
        int num_host = 2;
        buffer_t buffer = buffer_init(num_elements, num_device, num_host,
                                         stride);

        err |= s_assert(buffer_is_host_empty(&buffer, 0) == 1);
        err |= s_assert(buffer_is_host_empty(&buffer, 1) == 0);
        err |= s_assert(buffer_is_host_empty(&buffer, stride) == 0);
        err |= s_assert(buffer_is_host_empty(&buffer, 4 * stride) == 1);
        err |= s_assert(buffer_is_host_empty(&buffer, 4 * stride + 1) == 0);
        err |= s_assert(buffer_is_host_empty(&buffer, 8 * stride) == 1);
        err |= s_assert(buffer_is_host_empty(&buffer, 8 * stride + 1) == 0);
        
        buffer_finalize(&buffer);

        }

        err = test_finalize(&test, err);

        return test_last_error();
}

int test_get_device_ptr(void)
{
        int err = 0;
        int stride = 2;
        size_t num_elements = 10;
        int num_host = 2;
        int num_device = 2;
        buffer_t buffer = buffer_init(num_elements, num_device, num_host,
                                          stride);

        test_t test = test_init(" * buffer_get_device_ptr", 0, 0);
        prec *ptr1, *ptr2;

        prec *ptr = buffer_get_device_ptr(&buffer, 0);
        err |= s_assert(buffer.d_offset == 0);

        ptr = buffer_get_device_ptr(&buffer, stride - 1);
        err |= s_assert(buffer.d_offset == 0);

        ptr1 = buffer_get_device_ptr(&buffer, stride);
        err |= s_assert(buffer.d_offset == num_elements);

        ptr2 = buffer_get_device_ptr(&buffer, stride + 1);
        err |= s_assert(buffer.d_offset == num_elements);

        err |= s_assert(ptr1 == ptr2);


        buffer_finalize(&buffer);

        err = test_finalize(&test, err);

        return test_last_error();
}
int test_get_host_ptr(void)
{
        int err = 0;
        int stride = 2;
        size_t num_elements = 10;
        int num_host = 3;
        int num_device = 2;
        buffer_t buffer = buffer_init(num_elements, num_device, num_host,
                                          stride);

        test_t test = test_init(" * buffer_get_host_ptr", 0, 0);
        

        prec *ptr = buffer_get_host_ptr(&buffer, 0);
        err |= s_assert(buffer.h_offset == 0);

        ptr = buffer_get_host_ptr(&buffer, stride * num_device);
        err |= s_assert(buffer.h_offset == num_device * num_elements);

        ptr = buffer_get_host_ptr(&buffer, 2 * stride * num_device);
        err |= s_assert(buffer.h_offset == 2 * num_device * num_elements);

        ptr = buffer_get_host_ptr(&buffer, 2 * stride * num_device + 1);
        err |= s_assert(buffer.h_offset == 2 * num_device * num_elements);
        
        buffer_finalize(&buffer);

        err = test_finalize(&test, err);

        return test_last_error();
}

int test_copy_to_host(void)
{
        // Fill up the host buffer twice, and check that the result is correct
        // for the second fill.
        int err = 0;
        int stride = 4;
        size_t num_elements = 2;
        size_t num_steps = 25*2;
        size_t num_device = 2;
        size_t num_host = 3;

        buffer_t buffer = buffer_init(num_elements, num_device, num_host,
                                          stride);

        size_t num_bytes = num_elements * sizeof(prec);
        prec *tmp = (prec*)malloc(num_bytes);

        test_t test = test_init(" * buffer_copy_to_host", 0, 0);

        for (size_t step = 1; step < num_steps; ++step) {

                // Order of operations is important, first do device operations,
                // then host, and finally disk.
                if (buffer_is_device_ready(&buffer, step)) {
                        array_fill(tmp, (prec)step, num_elements);
                        cudaMemcpy(buffer_get_device_ptr(&buffer, step), tmp,
                                   num_bytes, cudaMemcpyHostToDevice);
                }
                
                if (buffer_is_device_full(&buffer, step)) {
                        buffer_copy_to_host(&buffer, step);
                }

                if (buffer_is_host_full(&buffer, step)) {
                        // Write to disk ...
                }

        }
        free(tmp);

        // previous: 
        // {0, 0, 4, 4, 8, 8, 12, 12, 12, 16, 16, 20, 20};
        prec ans[12] = {24, 24, 28, 28, 32, 32, 36, 36, 40, 40, 44, 44};

        double ferr = chk_inf(ans, buffer.h_buffer,
                       num_elements * num_device * num_host);
        err |= s_assert(ferr == 0.0);

        buffer_finalize(&buffer);

        err |= test_finalize(&test, err);
        return err;
}

int test_copy_to_device(void)
{
        // Fill up the host buffer twice, and check that the result is correct
        // for the second fill.
        int err = 0;
        int stride = 4;
        size_t num_elements = 2;
        size_t num_steps = 25*2;
        size_t num_device = 2;
        size_t num_host = 3;
        size_t total_host = num_device * num_elements;
        size_t total_size = num_elements * num_device;

        buffer_t buffer = buffer_init(num_elements, num_device, num_host,
                                         stride);

        size_t num_bytes = total_size * sizeof(prec);
        prec *tmp = (prec*)malloc(num_bytes);
        prec *ans = (prec*)malloc(sizeof ans * num_device);

        test_t test = test_init(" * buffer_copy_to_device", 0, 0);

        array_fill(buffer_get_host_ptr(&buffer, 0), 0, total_host);
        buffer_copy_to_device(&buffer, 0);
        array_fill(ans, 0, num_device);

        cudaMemcpy(tmp, buffer_get_device_ptr(&buffer, 0),
                                   num_bytes, cudaMemcpyDeviceToHost);

        for (size_t step = 1; step < num_steps; ++step) {


                if (buffer_is_device_empty(&buffer, step)) {
                        array_fill(buffer_get_host_ptr(&buffer, step), step,
                                   total_host);
                        array_fill(ans, step, num_device);
                }

                buffer_copy_to_device(&buffer, step);

                if (buffer_is_device_ready(&buffer, step)) {
                        cudaMemcpy(tmp, buffer_get_device_ptr(&buffer, step),
                                   num_bytes, cudaMemcpyDeviceToHost);
                        double ferr = chk_inf(ans, tmp, num_device);
                        err |= s_assert(ferr == 0.0);
                }

        }
        free(ans);
        free(tmp);

        buffer_finalize(&buffer);

        err |= test_finalize(&test, err);
        return test_last_error();
}

