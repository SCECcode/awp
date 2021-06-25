#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <awp/pmcl3d_cons.h>
#include <vtk/vtk.h>

size_t vtk_write_grid(const char *fname, 
                      const _prec *x,
                      const _prec *y,
                      const _prec *z,
                      const fcn_grid_t grid)
{
        FILE *fh;
        size_t count = 0;
        fh = fopen(fname, "w");
        
        if (!fh) {
                return count;
        }
        int zk = grid.exclude_top_row;
        size_t numpts = grid.size.x*grid.size.y*(grid.size.z - zk);

        // Header
        fprintf(fh, "# vtk DataFile Version 4.2\n");
        fprintf(fh, "vtk output\n");
        fprintf(fh, "ASCII\n");
        fprintf(fh, "DATASET STRUCTURED_GRID\n");
        fprintf(fh, "DIMENSIONS %d %d %d\n", grid.size.x, grid.size.y,
                grid.size.z - zk);
        fprintf(fh, "POINTS %ld float\n", numpts);
        
        // Coordinates
        for (int k = grid.offset1.z; k < grid.size.z + grid.offset1.z - zk; ++k) {
        for (int j = grid.offset1.y; j < grid.size.y + grid.offset1.y; ++j) {
        for (int i = grid.offset1.x; i < grid.size.x + grid.offset1.x; ++i) {
                int offset = k + j*grid.line + i*grid.slice;
                fprintf(fh, "%f %f %f\n", 
                        x[offset], y[offset], z[offset]);
                count++;
        }
        }
        }
        assert(count == numpts);

        fclose(fh);

        return count;
}

size_t vtk_write_grid_xz(const char *fname, 
                      const _prec *x,
                      const _prec *z,
                      const fcn_grid_t grid)
{
        FILE *fh;
        size_t count = 0;
        fh = fopen(fname, "w");
        
        if (!fh) {
                return count;
        }
        int zk = grid.exclude_top_row;
        size_t numpts = grid.size.x*(grid.size.z - zk);

        // Header
        fprintf(fh, "# vtk DataFile Version 4.2\n");
        fprintf(fh, "vtk output\n");
        fprintf(fh, "ASCII\n");
        fprintf(fh, "DATASET STRUCTURED_GRID\n");
        fprintf(fh, "DIMENSIONS %d %d %d\n", grid.size.x, 1,
                grid.size.z - zk);
        fprintf(fh, "POINTS %ld float\n", numpts);
        
        // Coordinates
        int j = grid.offset1.y + grid.size.y / 2;
        for (int k = grid.offset1.z; k < grid.size.z + grid.offset1.z - zk; ++k) {
        for (int i = grid.offset1.x; i < grid.size.x + grid.offset1.x; ++i) {
                int offset = k + j*grid.line + i*grid.slice;
                fprintf(fh, "%f %f %f\n", 
                        x[offset], 0.0, z[offset]);
                count++;
        }
        }
        assert(count == numpts);

        fclose(fh);

        return count;
}

/*
 * Append a scalar field to an already existing vtk file.
 * Call for example `vtk_write_grid` before calling this function.
 *
 * Arguments:
 * fname : Filename of output.
 * label : VTK label.
 * data : An array of length `n` that contains the scalar values. 
 * n : Number of grid points in each grid direction.
 * nmem : Number of coordinates in each grid direction as stored in memory.
 *      If the grid layout is the same as the memory layout, then `n = nmem`.
 */
size_t vtk_append_scalar(const char *fname,
                         const char *label, 
                         const _prec *data,
                         const fcn_grid_t grid
                         )
{
        size_t count = 0;
        int zk = grid.exclude_top_row;
        size_t numpts = grid.size.x*grid.size.y*(grid.size.z - zk);
        
        
        FILE *fh = fopen(fname, "a");
        
        if (!fh) {
                return count;
        }

        fprintf(fh, "POINT_DATA %ld \n", numpts);
        fprintf(fh, "FIELD scalar 1\n");
        fprintf(fh, "%s 1 %ld float\n", label, numpts);
        // Coordinates
        for (int k = grid.offset1.z; k < grid.size.z + grid.offset1.z - zk; ++k) {
        for (int j = grid.offset1.y; j < grid.size.y + grid.offset1.y; ++j) {
        for (int i = grid.offset1.x; i < grid.size.x + grid.offset1.x; ++i) {
                int offset = k + j*grid.line + i*grid.slice;
                fprintf(fh, "%f \n", data[offset]);
                count++;
        }
        }
        }
        assert(count == numpts);

        fclose(fh);

        return count;
}

/*
 * Append a scalar field to an already existing vtk file.
 * Call for example `vtk_write_grid_xz` before calling this function.
 *
 * Arguments:
 * fname : Filename of output.
 * label : VTK label.
 * data : An array of length `n` that contains the scalar values. 
 * n : Number of grid points in each grid direction.
 * nmem : Number of coordinates in each grid direction as stored in memory.
 *      If the grid layout is the same as the memory layout, then `n = nmem`.
 */
size_t vtk_append_scalar_xz(const char *fname,
                         const char *label, 
                         const _prec *data,
                         const fcn_grid_t grid
                         )
{
        size_t count = 0;
        int zk = grid.exclude_top_row;
        size_t numpts = grid.size.x*(grid.size.z - zk);
        
        
        FILE *fh = fopen(fname, "a");
        
        if (!fh) {
                return count;
        }

        fprintf(fh, "POINT_DATA %ld \n", numpts);
        fprintf(fh, "FIELD scalar 1\n");
        fprintf(fh, "%s 1 %ld float\n", label, numpts);
        // Coordinates
        int j = grid.offset1.y + grid.size.y / 2;
        for (int k = grid.offset1.z; k < grid.size.z + grid.offset1.z - zk; ++k) {
        for (int i = grid.offset1.x; i < grid.size.x + grid.offset1.x; ++i) {
                int offset = k + j*grid.line + i*grid.slice;
                fprintf(fh, "%f \n", data[offset]);
                count++;
        }
        }
        assert(count == numpts);

        fclose(fh);

        return count;
}

