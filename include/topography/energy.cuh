#ifndef ENERGY_CUH
#define ENERGY_CUH


#include <mpi.h>
#include <stdio.h>
#include <awp/pmcl3d_cons.h>
#include <topography/metrics/metrics.h>

typedef struct {
    int use;
    int rank;
    MPI_Comm comm;
    double *kinetic_energy_rate;
    double *strain_energy_rate;
    double *kinetic_rate;
    double *strain_rate;
    int num_steps;
    size_t num_bytes;
    // Copies of velocity components at previous time step
    float *d_vxp;
    float *d_vyp;
    float *d_vzp;

    // Copies of stress components at previous time step
    float *d_xxp;
    float *d_yyp;
    float *d_zzp;
    float *d_xyp;
    float *d_xzp;
    float *d_yzp;

} energy_t;

#ifdef __cplusplus
extern "C" {
void energy_rate(energy_t *e, int step, const float *d_vx, const float *d_vy,
                 const float *d_vz, const float *d_xx, const float *d_yy,
                 const float *d_zz, const float *d_xy, const float *d_xz,
                 const float *d_yz, const float *d_rho, const float *d_mui,
                 const float *d_lami, const f_grid_t *metrics_f, const g_grid_t *metrics_g,
                 const int nx, const int ny, const int nz);
#endif
#ifdef __cplusplus
}
#endif

energy_t energy_init(int useenergy, const int rank, const MPI_Comm comm, const int num_steps, const int nx, const int ny, const int nz) {
    energy_t energy;
    energy.use = 0;
    energy.rank = -1;

    if (!useenergy) return energy;
    energy.use = 1;

    if (rank == 0)
    printf("Energy output:: enabled\n");


    energy.rank = rank;
    energy.comm = comm;
    energy.num_steps = num_steps;
    energy.kinetic_energy_rate = (double*)malloc(sizeof (double) * num_steps);
    energy.strain_energy_rate = (double*)malloc(sizeof (double) * num_steps);
    size_t num_bytes = (nx + 2 * ngsl + 4) * (ny + 2 * ngsl + 4) * (nz + 2 * align) * sizeof(float);
    energy.num_bytes = num_bytes;
    CUCHK(cudaMalloc((void**)&energy.d_vxp, num_bytes));
    CUCHK(cudaMalloc((void**)&energy.d_vyp, num_bytes));
    CUCHK(cudaMalloc((void**)&energy.d_vzp, num_bytes));
    CUCHK(cudaMalloc((void**)&energy.d_xxp, num_bytes));
    CUCHK(cudaMalloc((void**)&energy.d_yyp, num_bytes));
    CUCHK(cudaMalloc((void**)&energy.d_zzp, num_bytes));
    CUCHK(cudaMalloc((void**)&energy.d_xyp, num_bytes));
    CUCHK(cudaMalloc((void**)&energy.d_xzp, num_bytes));
    CUCHK(cudaMalloc((void**)&energy.d_yzp, num_bytes));
    CUCHK(cudaMalloc((void**)&energy.kinetic_rate, sizeof(double)));
    CUCHK(cudaMalloc((void**)&energy.strain_rate, sizeof(double)));

    return energy;

}

void energy_update_previous_solutions(energy_t *e, float *d_vx, float *d_vy, float *d_vz, float *d_xx, float *d_yy, float *d_zz, float *d_xy, float *d_xz, float *d_yz) {

    CUCHK(cudaMemcpy(e->d_vxp, d_vx, e->num_bytes, cudaMemcpyDeviceToDevice));
    CUCHK(cudaMemcpy(e->d_vyp, d_vy, e->num_bytes, cudaMemcpyDeviceToDevice));
    CUCHK(cudaMemcpy(e->d_vzp, d_vz, e->num_bytes, cudaMemcpyDeviceToDevice));
    CUCHK(cudaMemcpy(e->d_xxp, d_xx, e->num_bytes, cudaMemcpyDeviceToDevice));
    CUCHK(cudaMemcpy(e->d_yyp, d_yy, e->num_bytes, cudaMemcpyDeviceToDevice));
    CUCHK(cudaMemcpy(e->d_zzp, d_zz, e->num_bytes, cudaMemcpyDeviceToDevice));
    CUCHK(cudaMemcpy(e->d_xyp, d_xy, e->num_bytes, cudaMemcpyDeviceToDevice));
    CUCHK(cudaMemcpy(e->d_xzp, d_xz, e->num_bytes, cudaMemcpyDeviceToDevice));
    CUCHK(cudaMemcpy(e->d_yzp, d_yz, e->num_bytes, cudaMemcpyDeviceToDevice));

}

void energy_zero(energy_t *e, float *d_vx, float *d_vy, float *d_vz, float *d_xx, float *d_yy, float *d_zz, float *d_xy, float *d_xz, float *d_yz, int mode) {
        //cudaMemset(d_vx, 0, e->num_bytes);
        //cudaMemset(d_vy, 0, e->num_bytes);
        //cudaMemset(d_vz, 0, e->num_bytes);
        //cudaMemset(d_xx, 0, e->num_bytes);
        //cudaMemset(d_yy, 0, e->num_bytes);
        //cudaMemset(d_zz, 0, e->num_bytes);
        //cudaMemset(d_xy, 0, e->num_bytes);
        //cudaMemset(d_xz, 0, e->num_bytes);
        //cudaMemset(d_yz, 0, e->num_bytes);

    //if (mode == 0) {
    //    cudaMemset(d_vx, 0, e->num_bytes);
    //    cudaMemset(d_vy, 0, e->num_bytes);
    //    cudaMemset(d_vz, 0, e->num_bytes);
    //    cudaMemset(d_xx, 0, e->num_bytes);
    //    cudaMemset(d_yy, 0, e->num_bytes);
    //    cudaMemset(d_zz, 0, e->num_bytes);
    //    cudaMemset(d_xy, 0, e->num_bytes);
    //    cudaMemset(d_xz, 0, e->num_bytes);
    //    cudaMemset(d_yz, 0, e->num_bytes);
    //}

    //if (mode == 1) {
    //    cudaMemset(d_xx, 0, e->num_bytes);
    //    cudaMemset(d_yy, 0, e->num_bytes);
    //    cudaMemset(d_zz, 0, e->num_bytes);
    //    cudaMemset(d_xy, 0, e->num_bytes);
    //    cudaMemset(d_xz, 0, e->num_bytes);
    //    cudaMemset(d_yz, 0, e->num_bytes);
    //}

}

void energy_kinetic_rate(energy_t *e, int step) {
    if (!e->use || step >= e->num_steps) return;

    e->kinetic_energy_rate[step] = (double)step;
    e->strain_energy_rate[step] = (double)step;

}

void energy_output(energy_t *e, const char *filename) {
    if (!e->use || e->rank != 0) return;
        
    FILE *fh = fopen(filename, "w");
    printf("Writing energy output\n");

    if (e->rank == 0)
    printf("Energy output written to: %s number of steps written: %d \n", filename, e->num_steps);
    for (int i = 0; i < e->num_steps; ++i)
        fprintf(fh, "%g %g %g \n", 
                e->kinetic_energy_rate[i], e->strain_energy_rate[i],
                e->kinetic_energy_rate[i] + e->strain_energy_rate[i]
                );

    fclose(fh);
}

void energy_free(energy_t *e) {
    if (!e->use) return;
    free(e->kinetic_energy_rate);
    free(e->strain_energy_rate);
    CUCHK(cudaFree(e->d_vxp));
    CUCHK(cudaFree(e->d_vyp));
    CUCHK(cudaFree(e->d_vzp));
    CUCHK(cudaFree(e->d_xxp));
    CUCHK(cudaFree(e->d_yyp));
    CUCHK(cudaFree(e->d_zzp));
    CUCHK(cudaFree(e->d_xyp));
    CUCHK(cudaFree(e->d_xzp));
    CUCHK(cudaFree(e->d_yzp));
    CUCHK(cudaFree(e->strain_rate));
    CUCHK(cudaFree(e->kinetic_rate));
}

#endif
