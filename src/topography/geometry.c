#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <awp/definitions.h>
#include <vtk/vtk.h>
#include <utils/copy.h>
#include <grid/shift.h>
#include <topography/topography.h>
#include <topography/geometry.h>
#include <topography/geometry/geometry.h>

void topo_init_grid(topo_t *T)
{
        if (!T->use) return;
        //FIXME: Handle proper grid initialization
        geom_cartesian_topography(&T->metrics_f);
        geom_no_grid_stretching(&T->metrics_g);

        grid1_t x1_grid = grid_grid1_x(T->topography_grid);
        grid1_t y1_grid = grid_grid1_y(T->topography_grid);
        grid1_t z1_grid = grid_grid1_z(T->topography_grid);

        T->x1 = malloc(sizeof(T->x1) * x1_grid.size);
        T->y1 = malloc(sizeof(T->y1) * y1_grid.size);
        T->z1 = malloc(sizeof(T->z1) * z1_grid.size);

        grid_fill1(T->x1, x1_grid);
        grid_fill1(T->y1, y1_grid);
        grid_fill1(T->z1, z1_grid);
}

void topo_init_gaussian_hill_and_canyon_xz(topo_t *T, const _prec3_t hill_width,
                                        const _prec hill_height,
                                        const _prec3_t hill_center,
                                        const _prec3_t canyon_width,
                                        const _prec canyon_height,
                                        const _prec3_t canyon_center)
{
        if (!T->use) return;
        geom_no_grid_stretching(&T->metrics_g);
        geom_gaussian_hill_and_canyon_xz(
            &T->metrics_f, T->x1, T->y1, T->topography_grid, hill_width,
            hill_height, hill_center, canyon_width, canyon_height,
            canyon_center, T->px, T->py);
}

void topo_init_gaussian_hill_and_canyon(topo_t *T, const _prec3_t hill_width,
                                        const _prec hill_height, 
                                        const _prec3_t hill_center,
                                        const _prec3_t canyon_width,
                                        const _prec canyon_height,
                                        const _prec3_t canyon_center)
{
        if (!T->use) return;
        geom_no_grid_stretching(&T->metrics_g);
        geom_gaussian_hill_and_canyon(
            &T->metrics_f, T->x1, T->y1, T->topography_grid, hill_width,
            hill_height, hill_center, canyon_width, canyon_height,
            canyon_center, T->px, T->py);
}

void topo_write_geometry_vtk(topo_t *T, const int mode)
{
        if (!T->use) return;
#if !(TOPO_USE_VTK)
        return;
#endif
        printf("Writing geometry\n");

        _prec *x = malloc(T->topography_grid.num_bytes);
        _prec *y = malloc(T->topography_grid.num_bytes);
        _prec *z = malloc(T->topography_grid.num_bytes);

        grid_fill3_x(x, T->x1, T->stress_grid);
        grid_fill3_y(y, T->y1, T->stress_grid);
        grid_fill3_z(z, T->z1, T->stress_grid);

        geom_mapping_z(z, T->stress_grid, grid_node(), &T->metrics_f,
                       &T->metrics_g);

        char vtk_file[256];

        mkdir("vtk", 0700);
        sprintf(vtk_file, "vtk/geometry_%d%d.vtk", T->coord[0], T->coord[0]);
        switch (mode) {
                case 0:
                vtk_write_grid(vtk_file, x, y, z, T->velocity_grid);
                break;
                case 1:
                vtk_write_grid_xz(vtk_file, x, z, T->velocity_grid);
                break;
        }

        free(x);
        free(y);
        free(z);
}

void topo_write_vtk(topo_t *T, const int step, int mode)
{
        if (!T->use) return;
#if !(TOPO_USE_VTK)
        return;
#endif
        char vtk_vx[256];
        char vtk_vy[256];
        char vtk_vz[256];
        char geom_file[256];
        mkdir("vtk", 0700);
        sprintf(vtk_vx, "vtk/vx_%d%d_%04d.vtk", T->coord[0], T->coord[0], step);
        sprintf(vtk_vy, "vtk/vy_%d%d_%04d.vtk", T->coord[0], T->coord[0], step);
        sprintf(vtk_vz, "vtk/vz_%d%d_%04d.vtk", T->coord[0], T->coord[0], step);
        sprintf(geom_file, "vtk/geometry_%d%d.vtk", T->coord[0], T->coord[0]);
        copyfile(vtk_vx, geom_file);
        copyfile(vtk_vy, geom_file);
        copyfile(vtk_vz, geom_file);

        size_t num_bytes = sizeof(_prec) * T->mx * T->my * T->mz;
        prec *vx = malloc(sizeof vx * num_bytes);
        prec *vy = malloc(sizeof vy * num_bytes);
        prec *vz = malloc(sizeof vz * num_bytes);
        CUCHK(cudaMemcpy(vx, T->u1, num_bytes, cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(vy, T->v1, num_bytes, cudaMemcpyDeviceToHost));
        CUCHK(cudaMemcpy(vz, T->w1, num_bytes, cudaMemcpyDeviceToHost));
        switch (mode) {
                case 0:
                vtk_append_scalar(vtk_vx, "x", vx, T->velocity_grid);
                vtk_append_scalar(vtk_vy, "y", vy, T->velocity_grid);
                vtk_append_scalar(vtk_vz, "z", vz, T->velocity_grid);
                break;
                case 1:
                vtk_append_scalar_xz(vtk_vx, "x", vx, T->velocity_grid);
                vtk_append_scalar_xz(vtk_vz, "z", vz, T->velocity_grid);
                break;
        }

        free(vx);
        free(vy);
        free(vz);
}

