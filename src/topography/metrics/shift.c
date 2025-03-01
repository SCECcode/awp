#include <topography/metrics/shift.h>
#include <stdio.h>

void metrics_shift_f_apply(float *fout, const float *fin, const int nx,
                           const int ny)
{
        int mx = nx + 2 * ngsl;
        int my = ny + 2 * ngsl;
        const int padding = 8;

#define _fout(i, j)               \
        fout[(j) + align + \
             ((i) + 2) * (2 * align + 2 * ngsl + ny + 4) + 2]
#define _fin(i, j)                  \
        fin[(j) + align + \
            ((i) + 2) * (2 * align + 2 * padding + ny + 4) + 2]
        for (int i = 0; i < mx; ++i) {
                for (int j = 0; j < my; ++j) {
                        _fout(i, j) =
                            _fin(i + padding - ngsl, j + padding - ngsl);
                }
        }
}
