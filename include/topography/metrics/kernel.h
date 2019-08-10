#ifndef METRICS_KERNEL_H
#define METRICS_KERNEL_H
#include <math.h>
#include <awp/definitions.h>

void metrics_f_interp_1_111(float *df1, const float *f, const int nx, const int ny, const int nz);
void metrics_f_interp_2_111(float *df1, const float *f, const int nx, const int ny, const int nz);
void metrics_f_interp_c_111(float *df1, const float *f, const int nx, const int ny, const int nz);
void metrics_f_diff_1_1_111(float *df1, const float *f, const float hi, const int nx, const int ny, const int nz);
void metrics_f_diff_1_2_111(float *df1, const float *f, const float hi, const int nx, const int ny, const int nz);
void metrics_f_diff_2_1_111(float *df1, const float *f, const float hi, const int nx, const int ny, const int nz);
void metrics_f_diff_2_2_111(float *df1, const float *f, const float hi, const int nx, const int ny, const int nz);
void metrics_g_interp_110(float *g3, const float *g, const int nx, const int ny, const int nz);
void metrics_g_interp_111(float *g3, const float *g, const int nx, const int ny, const int nz);
void metrics_g_interp_112(float *g3, const float *g, const int nx, const int ny, const int nz);
void metrics_g_diff_3_110(float *g3, const float *g, const float hi, const int nx, const int ny, const int nz);
void metrics_g_diff_3_111(float *g3, const float *g, const float hi, const int nx, const int ny, const int nz);
void metrics_g_diff_3_112(float *g3, const float *g, const float hi, const int nx, const int ny, const int nz);
void metrics_g_diff_c_110(float *g3, const float *g, const float hi, const int nx, const int ny, const int nz);
void metrics_g_diff_c_111(float *g3, const float *g, const float hi, const int nx, const int ny, const int nz);
void metrics_g_diff_c_112(float *g3, const float *g, const float hi, const int nx, const int ny, const int nz);
#endif

