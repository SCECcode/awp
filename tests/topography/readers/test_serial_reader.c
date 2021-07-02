#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include <test/test.h>
#include <functions/functions.h>
#include <grid/shift.h>
#include <topography/readers/serial_reader.h>
#include <topography/geometry/geometry.h>
#include <topography/metrics/metrics.h>

void init_geometry(prec **f, const int *gsize, const int3_t coord,
                   const int rank, int px, int py);

void init_global_grid(prec **f, const int *gsize);
void init_local_grid(prec **f, const int *lsize, const int3_t coord, const int gmy);
void write_geometry(const _prec *f, const int *gsize, int rank);
int test_read_grid(int rank, const _prec *local_f, const int *local_size,
                   const int px, const int py, const int3_t coord);

// This file will be temporarily created
const char *geometry_file = "geometry.bin";


int main(int argc, char **argv)
{

        int rank, mpi_size;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        int px = 2; 
        int py = 2; 
        assert(mpi_size == px * py);

        int3_t coord = { .x = rank % px, .y = rank / px, .z = 0};

        int err = 0;
        int local_grid[3] = {3, 2, 4};
        int global_grid[3] = {local_grid[0] * px, local_grid[1] * py,
                              local_grid[2]};
        prec * global_f;
        prec * local_f;


        if (rank == 0) {
                printf("===========================\n");
                printf("Testing topography/serial_reader.c\n");
                printf("===========================\n");
        }
        if (rank == 0) {
                //init_geometry(&global_f, global_grid, coord, rank, 1, 1);
                init_global_grid(&global_f, global_grid);
        }
            
        int gmy = global_grid[1] + 4 + 2 * align + 2 * metrics_padding;

        init_local_grid(&local_f, local_grid, coord, gmy);
        if (rank == 0) {
            printf("%d %d %d \n", rank, coord.x, coord.y);
        init_local_grid(&local_f, local_grid, coord, gmy);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 1) {
            printf("%d %d %d \n", rank, coord.x, coord.y);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        //init_geometry(&local_f, local_grid, coord, rank, px, py);
        write_geometry(global_f, global_grid, rank);


        if (rank == 0) {
        printf("global grid\n\n");
        for (int i = 0; i < global_grid[0]; ++i) {
        for (int j = 0; j < global_grid[1]; ++j) {
                size_t pos = align + 2 + j + metrics_padding + (2 *align + global_grid[1] + 4 + 2 * metrics_padding) * ( 2 + i + metrics_padding);
                    printf(" %-4.f", global_f[pos]);
        }
            printf("\n");
        }

        }

        MPI_Barrier(MPI_COMM_WORLD);
        for (int d = 0; d < 2; ++d) {
        if (rank == d) {
            printf("local_f, rank = %d\n", d);
            printf("\n");
            printf("\n");
        for (int i = 0; i < local_grid[0]; ++i) {
        for (int j = 0; j < local_grid[1]; ++j) {
                size_t pos = align + 2 + j + metrics_padding + (2 *align + local_grid[1] + 4 + 2 * metrics_padding) * ( 2 + i + metrics_padding);
                    printf(" %-4.f", local_f[pos]);
        }
            printf("\n");
        }
            printf("\n");
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        err = test_read_grid(rank, local_f, local_grid, px, py, coord);
        if (rank == 0) {
                free(global_f);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        free(local_f);
        MPI_Finalize();
        return err;
}

void init_global_grid(prec **f, const int *gsize) {

        int nx = gsize[0];
        int ny = gsize[1];

        int mxp = nx + 2 * metrics_padding;
        int myp = ny + 2 * metrics_padding;
        int mx = 4 + mxp;
        int my = 4 + myp + 2 * align;
        *f = malloc(mx * my * sizeof(prec)); 

        prec *global_f = *f;

        for (int i = 0; i < mxp; ++i) {
        for (int j = 0; j < myp; ++j) {
            size_t pos = align + 2 + j + my * (2 + i);
            //global_f[pos] = 0;
            global_f[pos] = j + my * i;
        }
        }

        for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
                size_t pos = align + 2 + metrics_padding + j + my * (2 + i + metrics_padding);
        }
        }

        for (int i = 0; i < mxp; ++i) {
        for (int j = 0; j < myp; ++j) {
            size_t pos = align + 2 + j + my * (2 + i);
            printf(" %-4.f", global_f[pos]);
        }
            printf("\n");
        }
}

void init_local_grid(prec **f, const int *lsize, const int3_t coord, const int gmy) {

        int nx = lsize[0];
        int ny = lsize[1];
        int mxp = nx + 2 * metrics_padding;
        int myp = ny + 2 * metrics_padding;
        int mx = 4 + mxp;
        int my = 4 + myp + 2 * align;
        size_t num_bytes = mx * my * sizeof(prec);
        *f = malloc(num_bytes); 

        prec *local_f = *f;
        memset(local_f, 0, num_bytes);


        for (int i = 0; i < mxp; ++i) {
        for (int j = 0; j < myp; ++j) {
                size_t pos = align + 2 + j + my * (2 + i);
                local_f[pos] = (j + ny * coord.y) + gmy * (i + nx * coord.x);
                //printf(" %-4.f", local_f[pos]);
        }
            //printf("\n");
        }

        //for (int i = 0; i < nx; ++i) {
        //for (int j = 0; j < ny; ++j) {
        //        size_t pos = align + 2 + metrics_padding + j + my * (2 + i + metrics_padding);
        //        //global_f[pos] = j + my * i;
        //        local_f[pos] = (j + ny * coord.y) + gmy * (i + nx * coord.x);
        //}
        //}

        for (int i = 0; i < mxp; ++i) {
        for (int j = 0; j < myp; ++j) {
            size_t pos = align + 2 + j + my * (2 + i);
            printf(" %-4.f", local_f[pos]);
        }
            printf("\n");
        }


}

void init_geometry(prec **f, const int *gsize, const int3_t coord,
                   const int rank, int px, int py) {
        _prec gridspacing = 0.1; 

        f_grid_t metrics_f = metrics_init_f(gsize, gridspacing, metrics_padding);
        g_grid_t metrics_g = metrics_init_g(gsize, gridspacing);

        int3_t shift = grid_u3();
        
        int3_t size = {gsize[0], gsize[1], gsize[2]};

        int3_t boundary1 = {.x = 0, .y = 0, .z = 0};
        int3_t boundary2 = {.x = 0, .y = 0, .z = 1};

        grid3_t topography_grid = grid_init_metric_grid(
            size, shift, coord, boundary1, boundary2, gridspacing);

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

         _prec3_t hill_width = {.x = 0.5, .y = 0.5, .z = 0};
         _prec hill_height = 1.0;
         _prec3_t hill_center = {.x = 0.0, .y = 0.0, .z = 0};
         _prec3_t canyon_width = {.x = 0.5, .y = 0.5, .z = 0};
         _prec canyon_height = 0.0;
         _prec3_t canyon_center = {.x = 2, .y = 2, .z = 0};

        geom_gaussian_hill_and_canyon(&metrics_f, x1, y1, topography_grid, 
                        hill_width, hill_height, hill_center,
                        canyon_width, canyon_height, canyon_center,
                        px, py);

        int mxy = metrics_f.mem[0] * metrics_f.mem[1];
        *f = malloc(metrics_sizeof_f(&metrics_f)); 
        for (int i = 0; i < metrics_f.mem[0]; ++i) {
        for (int j = 0; j < metrics_f.mem[1]; ++j) {
                size_t pos = j + metrics_f.mem[1] * i;
                (*f)[pos] = y1[i];// + mxy * coord.x + mxy * px * coord.y;
                //(*f)[pos] = metrics_f.f[pos];
        }
        }

        //geom_custom(&metrics_f, topography_grid, px,
        //         py, *f); 

        metrics_build_f(&metrics_f);
        metrics_build_g(&metrics_g);


        free(x);
        free(y);
        free(z);
        free(x1);
        free(y1);
        free(z1);
}

void write_geometry(const prec *f, const int *gsize, int rank) {
        if (rank != 0)
                return;
        int nx = gsize[0];
        int ny = gsize[1];
        int mx = nx + 2 * metrics_padding;
        int my = ny + 2 * metrics_padding;
        FILE *fh = fopen(geometry_file, "wb");
        float *data;
        data = malloc(sizeof data * mx * my);
        fwrite(&nx, sizeof nx, 1, fh);
        fwrite(&ny, sizeof ny, 1, fh);
        fwrite(&metrics_padding, sizeof metrics_padding, 1, fh);
        int slice = 4 + my + 2 * align;

        for (int i = 0; i < mx; ++i) {
        for (int j = 0; j < my; ++j) {
                size_t fpos = align + 2 + j + (2 + i) * slice;
                data[j + i * my] = f[fpos];
        }
        }

        fwrite(data, sizeof(float), mx * my, fh);
        fclose(fh);

}

int test_read_grid(int rank, const _prec *local_f, const int *local_size, const
                int px, const int py, const int3_t coord)
{
        MPI_Barrier(MPI_COMM_WORLD);
        int err = 0;

        test_t test = test_init("read_topography", rank, px * py);

        int lnx, lny;
        lnx = local_size[0];
        lny = local_size[1];
        int lmx = 4 + lnx + 2 * metrics_padding;
        int lmy = 4 + lny + 2 * metrics_padding + 2 * align;
        prec *read_f;
        int icoord[2] = {coord.x, coord.y};
        int alloc = 1;

        read_f = malloc(sizeof read_f * lmx * lmy); 

        for (int i = 0; i < lmx * lmy; ++i)
                read_f[i] = 0.0;

        err |= topo_read_serial(geometry_file, rank, px, py, icoord, lnx, lny,
                                alloc, &read_f);

        MPI_Barrier(MPI_COMM_WORLD);

        // Compare data read from file with locally computed data
        float sum = 0;
            
        for (int i = 0; i < (lnx + 2 * metrics_padding); ++i) {
        for (int j = 0; j < (lny + 2 * metrics_padding); ++j) {
                size_t local_pos = 2 + align + j + (i + 2) * lmy;
                double val = fabs(read_f[local_pos] - local_f[local_pos]);
                sum += val; 
        }
        }

        free(read_f);
        remove(geometry_file);

        err |= !(sum == 0);
        err |= test_finalize(&test, err);

        return err;

}

