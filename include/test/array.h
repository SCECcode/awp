#ifndef ARRAY_H
#define ARRAY_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/*
 * Transpose an array `b` having `m` rows and `n` columns
 */ 
#define array_transpose(a, b, m, n) {\
{ \
        for (size_t i = 0; i < (size_t)n; i++) { \
                for (size_t j = 0; j < (size_t)m; ++j) { \
                        a[j + i * m] = b[i + j * n]; \
                } \
        } \
} \
} \

#define array_fill(a, val, n) {\
{ \
        for (size_t j = 0; j < (size_t)n; j++) { \
                a[j] = val;\
        }\
}\
}\

#define array_range(a, n) {\
{ \
        for (size_t j = 0; j < (size_t)n; j++) { \
                a[j] = j;\
        }\
}\
}\

#define array_addc(a, n, b) {\
{ \
        for (size_t j = 0; j < (size_t)n; j++) { \
                a[j] += b;\
        }\
}\
}\

#define array_reduce(T, a, n) {\
{ \
        T out; \
        for (size_t j = 0; j < (size_t)n; j++) { \
                out += a[j];\
        }\
        return out;\
}\
}\

#ifdef __cplusplus
}
#endif
#endif

