#include "cumetrics_kernel.h"

void dmetrics_interp_1_111(float *df1, const float *f, const int nx,
                           const int ny, const int nz) {
  const float phy[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  for (int k = 0; k < 1; ++k) {
    for (int j = 0; j < ny - 4; ++j) {
      for (int i = 0; i < nx - 4; ++i) {
#define _f(i, j)                                                               \
  f[(i) + align + ngsl + ((j) + ngsl + 2) * (2 * ngsl + nx + 4) + 2]
#define _df1(i, j)                                                             \
  df1[(i) + align + ngsl + ((j) + ngsl + 2) * (2 * ngsl + nx + 4) + 2]
        _df1(i + 2, j + 2) = phy[0] * _f(i + 2, j) + phy[1] * _f(i + 2, j + 1) +
                             phy[2] * _f(i + 2, j + 2) +
                             phy[3] * _f(i + 2, j + 3);
#undef _f
#undef _df1
      }
    }
  }
}

void dmetrics_interp_2_111(float *df1, const float *f, const int nx,
                           const int ny, const int nz) {
  const float px[4] = {-0.0625000000000000, 0.5625000000000000,
                       0.5625000000000000, -0.0625000000000000};
  for (int k = 0; k < 1; ++k) {
    for (int j = 0; j < ny - 4; ++j) {
      for (int i = 0; i < nx - 4; ++i) {
#define _f(i, j)                                                               \
  f[(i) + align + ngsl + ((j) + ngsl + 2) * (2 * ngsl + nx + 4) + 2]
#define _df1(i, j)                                                             \
  df1[(i) + align + ngsl + ((j) + ngsl + 2) * (2 * ngsl + nx + 4) + 2]
        _df1(i + 2, j + 2) =
            px[0] * _f(i + 1, j + 2) + px[1] * _f(i + 2, j + 2) +
            px[2] * _f(i + 3, j + 2) + px[3] * _f(i + 4, j + 2);
#undef _f
#undef _df1
      }
    }
  }
}

void dmetrics_interp_y_1_111(float *df1, const float *f, const int nx,
                             const int ny, const int nz) {
  const float phy[4] = {-0.0625000000000000, 0.5625000000000000,
                        0.5625000000000000, -0.0625000000000000};
  for (int k = 0; k < 1; ++k) {
    for (int j = 0; j < ny - 4; ++j) {
      for (int i = 0; i < nx - 4; ++i) {
#define _f(i, j)                                                               \
  f[(i) + align + ngsl + ((j) + ngsl + 2) * (2 * ngsl + nx + 4) + 2]
#define _df1(i, j)                                                             \
  df1[(i) + align + ngsl + ((j) + ngsl + 2) * (2 * ngsl + nx + 4) + 2]
        _df1(i + 2, j + 2) = phy[0] * _f(i + 2, j) + phy[1] * _f(i + 2, j + 1) +
                             phy[2] * _f(i + 2, j + 2) +
                             phy[3] * _f(i + 2, j + 3);
#undef _f
#undef _df1
      }
    }
  }
}

void dmetrics_interp_y_2_111(float *df1, const float *f, const int nx,
                             const int ny, const int nz) {
  const float py[4] = {-0.0625000000000000, 0.5625000000000000,
                       0.5625000000000000, -0.0625000000000000};
  for (int k = 0; k < 1; ++k) {
    for (int j = 0; j < ny - 4; ++j) {
      for (int i = 0; i < nx - 4; ++i) {
#define _f(i, j)                                                               \
  f[(i) + align + ngsl + ((j) + ngsl + 2) * (2 * ngsl + nx + 4) + 2]
#define _df1(i, j)                                                             \
  df1[(i) + align + ngsl + ((j) + ngsl + 2) * (2 * ngsl + nx + 4) + 2]
        _df1(i + 2, j + 2) =
            py[0] * _f(i + 2, j + 1) + py[1] * _f(i + 2, j + 2) +
            py[2] * _f(i + 2, j + 3) + py[3] * _f(i + 2, j + 4);
#undef _f
#undef _df1
      }
    }
  }
}

void dmetrics_diff_x_1_111(float *df1, const float *f, const int nx,
                           const int ny, const int nz) {
  const float dhx[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  for (int k = 0; k < 1; ++k) {
    for (int j = 0; j < ny - 4; ++j) {
      for (int i = 0; i < nx - 4; ++i) {
#define _f(i, j)                                                               \
  f[(i) + align + ngsl + ((j) + ngsl + 2) * (2 * ngsl + nx + 4) + 2]
#define _df1(i, j)                                                             \
  df1[(i) + align + ngsl + ((j) + ngsl + 2) * (2 * ngsl + nx + 4) + 2]
        _df1(i + 2, j + 2) = dhx[0] * _f(i, j + 2) + dhx[1] * _f(i + 1, j + 2) +
                             dhx[2] * _f(i + 2, j + 2) +
                             dhx[3] * _f(i + 3, j + 2);
#undef _f
#undef _df1
      }
    }
  }
}

void dmetrics_diff_x_2_111(float *df1, const float *f, const int nx,
                           const int ny, const int nz) {
  const float dx[4] = {0.0416666666666667, -1.1250000000000000,
                       1.1250000000000000, -0.0416666666666667};
  for (int k = 0; k < 1; ++k) {
    for (int j = 0; j < ny - 4; ++j) {
      for (int i = 0; i < nx - 4; ++i) {
#define _f(i, j)                                                               \
  f[(i) + align + ngsl + ((j) + ngsl + 2) * (2 * ngsl + nx + 4) + 2]
#define _df1(i, j)                                                             \
  df1[(i) + align + ngsl + ((j) + ngsl + 2) * (2 * ngsl + nx + 4) + 2]
        _df1(i + 2, j + 2) =
            dx[0] * _f(i + 1, j + 2) + dx[1] * _f(i + 2, j + 2) +
            dx[2] * _f(i + 3, j + 2) + dx[3] * _f(i + 4, j + 2);
#undef _f
#undef _df1
      }
    }
  }
}

void dmetrics_diff_y_1_111(float *df1, const float *f, const int nx,
                           const int ny, const int nz) {
  const float dhy[4] = {0.0416666666666667, -1.1250000000000000,
                        1.1250000000000000, -0.0416666666666667};
  for (int k = 0; k < 1; ++k) {
    for (int j = 0; j < ny - 4; ++j) {
      for (int i = 0; i < nx - 4; ++i) {
#define _f(i, j)                                                               \
  f[(i) + align + ngsl + ((j) + ngsl + 2) * (2 * ngsl + nx + 4) + 2]
#define _df1(i, j)                                                             \
  df1[(i) + align + ngsl + ((j) + ngsl + 2) * (2 * ngsl + nx + 4) + 2]
        _df1(i + 2, j + 2) = dhy[0] * _f(i + 2, j) + dhy[1] * _f(i + 2, j + 1) +
                             dhy[2] * _f(i + 2, j + 2) +
                             dhy[3] * _f(i + 2, j + 3);
#undef _f
#undef _df1
      }
    }
  }
}

void dmetrics_diff_y_2_111(float *df1, const float *f, const int nx,
                           const int ny, const int nz) {
  const float dy[4] = {0.0416666666666667, -1.1250000000000000,
                       1.1250000000000000, -0.0416666666666667};
  for (int k = 0; k < 1; ++k) {
    for (int j = 0; j < ny - 4; ++j) {
      for (int i = 0; i < nx - 4; ++i) {
#define _f(i, j)                                                               \
  f[(i) + align + ngsl + ((j) + ngsl + 2) * (2 * ngsl + nx + 4) + 2]
#define _df1(i, j)                                                             \
  df1[(i) + align + ngsl + ((j) + ngsl + 2) * (2 * ngsl + nx + 4) + 2]
        _df1(i + 2, j + 2) =
            dy[0] * _f(i + 2, j + 1) + dy[1] * _f(i + 2, j + 2) +
            dy[2] * _f(i + 2, j + 3) + dy[3] * _f(i + 2, j + 4);
#undef _f
#undef _df1
      }
    }
  }
}
