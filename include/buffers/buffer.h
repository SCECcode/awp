#ifndef BUFFER_H
#define BUFFER_H
#ifdef __cplusplus
extern "C" {
#endif

#include <awp/definitions.h>

/*
 * This module provides data structures and functions for buffering data on the
 * device and host.
 *
 * Usage:
 * - Initialize the buffer using `buffer_init()`.
 * 
 * - Check whether host and device buffers are either empty or full by using the
 *   functions:
 *   buffer_is_device_ready()
 *   buffer_is_device_empty()
 *   buffer_is_device_full()
 *   buffer_is_host_empty()
 *   buffer_is_host_full()
 *
 * - Obtain buffer pointers by calling the functions:
 *   buffer_out_device(), buffer_out_host().
 *
 * - Pass the pointer returned by buffer_out_device to the kernel that should
 *   write to the device buffer.  There is no need to keep track of the offset
 *   to each buffer index. This is handled internally.
 *
 *   Call buffer_out_copy_to_host() to copy data from the device buffer to the
 *   host buffer when the device buffer becomes full.
 *
 * - Call `buffer_finalize` to free allocated memory.
 *
 *   Example:  Use buffer for writing data to disk.
 *
 *       buffer_t buffer = buffer_init(num_elements, num_device, num_host,
 *                                     stride);
 *       for (int step = 0; step < num_steps; ++step) {
 *               if (buffer_is_device_ready(&buffer, step)) {
 *                      float *device_ptr = buffer_get_device_ptr(&buffer,
 *                                                                step);
 *                      // write to device buffer
 *                      // ...
 *               }
 *
 *               if (buffer_is_device_full(&buffer, step)) {
 *                       // Copy buffered data on device to host
 *                       buffer_out_copy_to_host(&buffer, step);
 *               }
 *
 *               if (buffer_is_host_full(&buffer, step)) {
 *                       float *host_ptr = buffer_get_host_ptr(&buffer, step);
 *                       // Write to host buffer to disk ...
 *
 *               }
 *       }
 
 *       buffer_finalize(&buffer);
 *
 *   Example:  Use buffer for reading data from disk.
 *
 *       buffer_t buffer = buffer_init(num_elements, num_device, num_host,
 *                                     stride);
 *       for (int step = 0; step < num_steps; ++step) {
 *               if (buffer_is_host_empty(&buffer, step)) {
 *                      float *host_ptr = buffer_get_host_ptr(&buffer, step);
 *                      // write to host buffer
 *                      // ...
 *               }
 *
 *               if (buffer_is_device_empty(&buffer, step)) {
 *                       // Copy buffered data on host to device
 *                       buffer_copy_to_device(&buffer, step);
 *               }
 *
 *               if (buffer_is_device_ready(&buffer, step)) {
 *                       float *device_ptr = buffer_get_device_ptr(&buffer, 
 *                                                                 step);
 *                       // Do something with the device data
 *                       // ...        
 *
 *               }
 *       }
 *
 *       buffer_finalize(&buffer);
 *
 *
 */
typedef struct
{
        prec *d_buffer;
        prec *h_buffer;
        size_t stride;
        size_t num_elements;
        size_t num_device;
        size_t num_host;
        size_t d_buffer_bytes;
        size_t h_buffer_bytes;
        size_t d_offset;
        size_t h_offset;
} buffer_t;

/*
 * Initialize buffer
 *
 * Arguments:
 *
 * num_elements: The number of elements written to the device buffer per write.
 * num_device: The capacity of the device buffer in terms of the number of
 *      writes. If `num_device = 10` then it is possible to write 10 times to
 *      the device buffer before it becomes full.
 * num_host: The capacity of the host buffer in terms of the number of writes.
 *      If `num_host = 10` then it is possible to copy the device buffer 10
 *      times to the host buffer before it becomes full.
 *
 * Return value:
 *  Returns a buffer data structure.
 *
 * Notes:
 *
 * The total size of the device buffer in number of elements is:
 *      element_size x num_device
 *
 * The total size of the host buffer in number of elements is:
 *     device_size x num_host = element_size x num_device x num_host
 *
 */
buffer_t buffer_init(size_t num_elements, size_t num_device, size_t num_host,
                     size_t stride);

void buffer_finalize(buffer_t *buffer);

// Get pointer to device buffer
prec* buffer_get_device_ptr(buffer_t *buffer, size_t step);

// Get pointer to host buffer
prec* buffer_get_host_ptr(buffer_t *buffer, size_t step);


/* Check if the device buffer is ready for use.
 *
 * Arguments:
 *      buffer: Buffer data structure.
 *      step: Time step to query buffer at.
 *
 * Return value:
 *      0 : Device buffer is not ready.
 *      1 : Device buffer is ready.
 */
int buffer_is_device_ready(const buffer_t *buffer, size_t step);

/* Check if device buffer is full.
 *
 * Arguments:
 *      buffer: Buffer data structure.
 *      step: Time step to query buffer at.
 *
 * Return value:
 *      0 : Device buffer is not full.
 *      1 : Device buffer is full.
 */
int buffer_is_device_full(const buffer_t *buffer, size_t step);

/* Check if device buffer is empty.
 *
 * Arguments:
 *      buffer: Buffer data structure.
 *      step: Time step to query buffer at.
 *
 * Return value:
 *      0 : Device buffer is not empty.
 *      1 : Device buffer is empty.
 */
int buffer_is_device_empty(const buffer_t *buffer, size_t step);

/* Check if host buffer is full.
 *
 * Arguments:
 *      buffer: Buffer data structure.
 *      step: Time step to query buffer at.
 *
 * Return value:
 *      0 : Host buffer is not full.
 *      1 : Host buffer is full.
 */
int buffer_is_host_full(const buffer_t *buffer, size_t step);

/* Check if host buffer is empty.
 *
 * Arguments:
 *      buffer: Buffer data structure.
 *      step: Time step to query buffer at.
 *
 * Return value:
 *      0 : Host buffer is not empty.
 *      1 : Host buffer is empty.
 */
int buffer_is_host_empty(const buffer_t *buffer, size_t step);


/* Copy contents of device buffer to host buffer.
 *
 * Arguments:
 *      buffer: Buffer data structure.
 *      step: Time step to query buffer at.
 */
void buffer_copy_to_device(buffer_t *buffer, size_t step);

/* Copy one element in host buffer to device buffer.
 *
 * Arguments:
 *      buffer: Buffer data structure.
 *      step: Time step to query buffer at.
 */
void buffer_copy_to_host(buffer_t *buffer, int step);

#ifdef __cplusplus
}
#endif
#endif

