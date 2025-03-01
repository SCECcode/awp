#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <awp/error.h>
#include <test/test.h>
#include <topography/readers/serial_reader.h>
#include <topography/metrics/metrics.h>


int topo_read_serial(const char *filename, const int rank, const int px,
                     const int py, const int *coord, const int nx,
                     const int ny, const int alloc, prec **out)
{

        FILE *fh = fopen(filename, "rb");
        if (fh == NULL) {
                return ERR_FILE_READ;
        }
        int gnx, gny, padding;
        int count;

        count = fread(&gnx, sizeof gnx, 1, fh);
        count = fread(&gny, sizeof gny, 1, fh);
        count = fread(&padding, sizeof padding, 1, fh);
        int gmx = gnx + 2 * padding;
        int gmy = gny + 2 * padding;

        assert(count > 0);
        assert(nx * px == gnx);
        assert(ny * py == gny);
        assert(padding >= metrics_padding);

        if (nx * px != gnx || ny * py != gny || padding < metrics_padding) {
                fclose(fh);
                return ERR_INCONSISTENT_SIZE;
        }

        float *data = malloc(sizeof data * gmx * gmy);
        count = fread(data, sizeof data, gmx * gmy, fh);

        int lmx = 4 + nx + 2 * metrics_padding;
        int lmy = 4 + ny + 2 * metrics_padding + 2 * align;

        if (alloc) {
                *out = malloc(sizeof out * lmx * lmy); 
        }

        for (int i = 0; i < (nx + 2 * metrics_padding); ++i) {
        for (int j = 0; j < (ny + 2 * metrics_padding); ++j) {
                size_t global_pos =   (ny * coord[1] + j) 
                                    + (nx * coord[0] + i) * gmy;
                size_t local_pos = 2 + align + j + (2 + i) * lmy;
                (*out)[local_pos] = data[global_pos];
        }
        }

        free(data);

        int err = 0;
        return err;
}

