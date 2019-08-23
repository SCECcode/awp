#ifndef CUMETRICS_KERNEL_H
#define CUMETRICS_KERNEL_H
#include "pmcl3d_cons.h"
#include <math.h>

void dmetrics_interp_1_111(float *df1, const float *f, const int nx,
                           const int ny, const int nz);
void dmetrics_interp_2_111(float *df1, const float *f, const int nx,
                           const int ny, const int nz);
void dmetrics_interp_y_1_111(float *df1, const float *f, const int nx,
                             const int ny, const int nz);
void dmetrics_interp_y_2_111(float *df1, const float *f, const int nx,
                             const int ny, const int nz);
void dmetrics_diff_x_1_111(float *df1, const float *f, const int nx,
                           const int ny, const int nz);
void dmetrics_diff_x_2_111(float *df1, const float *f, const int nx,
                           const int ny, const int nz);
void dmetrics_diff_y_1_111(float *df1, const float *f, const int nx,
                           const int ny, const int nz);
void dmetrics_diff_y_2_111(float *df1, const float *f, const int nx,
                           const int ny, const int nz);
#endif