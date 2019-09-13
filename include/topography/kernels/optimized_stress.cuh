#ifndef OPTIMIZED_STRESS_H
#define OPTIMIZED_STRESS_H
#include <awp/definitions.h>
#include <math.h>

__global__ void dtopo_str_110(
    float *__restrict__ s11, float *__restrict__ s12, float *__restrict__ s13,
    float *__restrict__ s22, float *__restrict__ s23, float *__restrict__ s33,
    float *__restrict__ u1, float *__restrict__ u2, float *__restrict__ u3,
    const float *__restrict__ dcrjx, const float *__restrict__ dcrjy,
    const float *__restrict__ dcrjz, const float *__restrict__ f,
    const float *__restrict__ f1_1, const float *__restrict__ f1_2,
    const float *__restrict__ f1_c, const float *__restrict__ f2_1,
    const float *__restrict__ f2_2, const float *__restrict__ f2_c,
    const float *__restrict__ f_1, const float *__restrict__ f_2,
    const float *__restrict__ f_c, const float *__restrict__ g,
    const float *__restrict__ g3, const float *__restrict__ g3_c,
    const float *__restrict__ g_c, const float *__restrict__ lami,
    const float *__restrict__ mui, const float a, const float nu, const int nx,
    const int ny, const int nz, const int bi, const int bj, const int ei,
    const int ej);
__global__ void dtopo_str_111(
    float *__restrict__ s11, float *__restrict__ s12, float *__restrict__ s13,
    float *__restrict__ s22, float *__restrict__ s23, float *__restrict__ s33,
    float *__restrict__ u1, float *__restrict__ u2, float *__restrict__ u3,
    const float *__restrict__ dcrjx, const float *__restrict__ dcrjy,
    const float *__restrict__ dcrjz, const float *__restrict__ f,
    const float *__restrict__ f1_1, const float *__restrict__ f1_2,
    const float *__restrict__ f1_c, const float *__restrict__ f2_1,
    const float *__restrict__ f2_2, const float *__restrict__ f2_c,
    const float *__restrict__ f_1, const float *__restrict__ f_2,
    const float *__restrict__ f_c, const float *__restrict__ g,
    const float *__restrict__ g3, const float *__restrict__ g3_c,
    const float *__restrict__ g_c, const float *__restrict__ lami,
    const float *__restrict__ mui, const float a, const float nu, const int nx,
    const int ny, const int nz, const int bi, const int bj, const int ei,
    const int ej);
__global__ void dtopo_str_112(
    float *__restrict__ s11, float *__restrict__ s12, float *__restrict__ s13,
    float *__restrict__ s22, float *__restrict__ s23, float *__restrict__ s33,
    float *__restrict__ u1, float *__restrict__ u2, float *__restrict__ u3,
    const float *__restrict__ dcrjx, const float *__restrict__ dcrjy,
    const float *__restrict__ dcrjz, const float *__restrict__ f,
    const float *__restrict__ f1_1, const float *__restrict__ f1_2,
    const float *__restrict__ f1_c, const float *__restrict__ f2_1,
    const float *__restrict__ f2_2, const float *__restrict__ f2_c,
    const float *__restrict__ f_1, const float *__restrict__ f_2,
    const float *__restrict__ f_c, const float *__restrict__ g,
    const float *__restrict__ g3, const float *__restrict__ g3_c,
    const float *__restrict__ g_c, const float *__restrict__ lami,
    const float *__restrict__ mui, const float a, const float nu, const int nx,
    const int ny, const int nz, const int bi, const int bj, const int ei,
    const int ej);
__global__ void dtopo_init_material_111(float *__restrict__ lami,
                                        float *__restrict__ mui,
                                        float *__restrict__ rho, const int nx,
                                        const int ny, const int nz);
#endif