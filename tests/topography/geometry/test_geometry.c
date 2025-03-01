#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include <test/test.h>
#include <functions/functions.h>
#include <topography/metrics/metrics.h>
#include <topography/geometry/geometry.h>
#include <grid/shift.h>
#include <vtk/vtk.h>

void test_gaussian(_prec **x, _prec **y, _prec **z, const int write_vtk,
                   const int rank);
void test_incline_plane(const int write_vtk,
                   const int rank);

int main(int argc, char **argv)
{

        int rank, mpi_size;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        assert(mpi_size == 4);

        if (rank == 0) {
                printf("===========================\n");
                printf("Testing geometry.c, vtk.c\n");
                printf("This test will produce several vtk files.\n");
                printf("Use rm *.vtk to remove when done.\n");
                printf("===========================\n");
        }
        //_prec *x, *y, *z;
        //test_gaussian(&x, &y, &z, 1, rank);
        test_incline_plane(1, rank);
        //free(x);
        //free(y);
        //free(z);
        MPI_Finalize();
        return 0;
}

void test_gaussian(_prec **x, _prec **y, _prec **z, const int write_vtk,
                   const int rank) {
        _prec amplitude = 10.0;
        _prec3_t width = {30, 30, 0.0};
        _prec3_t center = {128.0, 128.0, 0.0};
        _prec gridspacing = 1.0;
        _prec h = gridspacing;
        int gsize[3] = {128, 128, 32};

        f_grid_t metrics_f = metrics_init_f(gsize, gridspacing, 8);
        g_grid_t metrics_g = metrics_init_g(gsize, gridspacing);

        int3_t shift = grid_xx();
        
        int3_t coord = {.x = rank%2, .y = rank / 2, .z = 0};
        int3_t size = {gsize[0], gsize[1], gsize[2]};

        fcn_grid_t topography_grid = fcn_init_grid(size, shift, coord, ngsl, h);
        fcn_grid_t velocity_grid = fcn_init_grid(size, shift, coord, 0, h);
        fcn_grid_t stress_grid = fcn_init_grid(size, shift, coord, ngsl/2, h);

        if (rank == 0)
                fcn_print_info(topography_grid);

        *x = malloc(topography_grid.num_bytes);
        *y = malloc(topography_grid.num_bytes);
        *z = malloc(topography_grid.num_bytes);

        _prec *xp = *x;
        _prec *yp = *y;
        _prec *zp = *z;

        fcn_fill_grid(xp, topography_grid, shift, 0);
        fcn_fill_grid(yp, topography_grid, shift, 1);
        fcn_fill_grid(zp, topography_grid, shift, 2);

        // Center parameter coordinates at bottom left corner of the velocity
        // grid
        fcn_shift(xp, xp, topography_grid, -ngsl);
        fcn_shift(yp, yp, topography_grid, -ngsl);

        geom_no_grid_stretching(&metrics_g);
        geom_gaussian(&metrics_f, xp, yp, topography_grid, amplitude,
                      width, center);

        metrics_build_f(&metrics_f);
        metrics_build_g(&metrics_g);

        geom_mapping_z(zp, stress_grid, shift, &metrics_f,
                       &metrics_g);
        if (!write_vtk)
                return;
        char vtk_file[128];
        sprintf(vtk_file, "gaussian_stress_grid_%d%d.vtk", coord.x, coord.y);
        vtk_write_grid(vtk_file, xp, yp, zp, stress_grid);
        size_t count = vtk_append_scalar(vtk_file, "z", zp, stress_grid);
        printf("Wrote: %s <number of elements written: %ld> \n", vtk_file,
               count);
        sprintf(vtk_file, "gaussian_velocity_grid_%d%d.vtk", coord.x, coord.y);
        vtk_write_grid(vtk_file, xp, yp, zp, velocity_grid);
        count = vtk_append_scalar(vtk_file, "z", zp, velocity_grid);
        printf("Wrote: %s <number of elements written: %ld> \n", vtk_file,
               count);
}

void test_incline_plane(const int write_vtk, const int rank) 
{
        _prec phi_x = 0.3;
        _prec phi_y = 0.0;
        int px = 2;
        int py = 2;
        int gsize[3] = {12, 12, 12};

        _prec gridspacing = 1.0 / (gsize[2] - 2);

        f_grid_t metrics_f = metrics_init_f(gsize, gridspacing, 8);
        g_grid_t metrics_g = metrics_init_g(gsize, gridspacing);

        int3_t shift = grid_u3();
        
        int3_t coord = {.x = rank%2, .y = rank / 2, .z = 0};
        int3_t size = {gsize[0], gsize[1], gsize[2]};

        int3_t boundary1 = {.x = 0, .y = 0, .z = 0};
        int3_t boundary2 = {.x = 0, .y = 0, .z = 1};

        grid3_t topography_grid = grid_init_metric_grid(
            size, shift, coord, boundary1, boundary2, gridspacing);
        grid3_t velocity_grid = grid_init_velocity_grid(
            size, shift, coord, boundary1, boundary2, gridspacing);
        grid3_t stress_grid = grid_init_stress_grid(
            size, shift, coord, boundary1, boundary2, gridspacing);

        if (rank == 0)
                fcn_print_info(topography_grid);

        grid1_t x1_grid = grid_grid1_x(topography_grid);
        grid1_t y1_grid = grid_grid1_y(topography_grid);
        grid1_t z1_grid = grid_grid1_z(topography_grid);


        _prec *x1 = malloc(sizeof(x1) * x1_grid.size);
        _prec *y1 = malloc(sizeof(y1) * y1_grid.size);
        _prec *z1 = malloc(sizeof(z1) * z1_grid.size);

        grid_fill1(x1, x1_grid, 1);
        grid_fill1(y1, y1_grid, 0);
        grid_fill1(z1, z1_grid, 0);

        _prec *x = malloc(topography_grid.num_bytes);
        _prec *y = malloc(topography_grid.num_bytes);
        _prec *z = malloc(topography_grid.num_bytes);

        grid_fill3_x(x, x1, topography_grid);
        grid_fill3_y(y, y1, topography_grid);
        grid_fill3_z(z, z1, topography_grid);

        geom_no_grid_stretching(&metrics_g);
        geom_incline_plane(&metrics_f, x, y, topography_grid, phi_x, phi_y, px, py);

        metrics_build_f(&metrics_f);
        metrics_build_g(&metrics_g);

        geom_mapping_z(z, stress_grid, shift, &metrics_f, &metrics_g);

        if (write_vtk) {
                char vtk_file[128];
                sprintf(vtk_file, "incline_stress_grid_%d%d.vtk", coord.x,
                        coord.y);
                vtk_write_grid(vtk_file, x, y, z, stress_grid);
                size_t count = vtk_append_scalar(vtk_file, "z", z, stress_grid);
                printf("Wrote: %s <number of elements written: %ld> \n",
                       vtk_file, count);
                sprintf(vtk_file, "incline_velocity_grid_%d%d.vtk", coord.x,
                        coord.y);
                vtk_write_grid(vtk_file, x, y, z, velocity_grid);
                count = vtk_append_scalar(vtk_file, "z", z, velocity_grid);
                printf("Wrote: %s <number of elements written: %ld> \n",
                       vtk_file, count);
        }

        free(x);
        free(y);
        free(z);
        free(x1);
        free(y1);
        free(z1);
}

