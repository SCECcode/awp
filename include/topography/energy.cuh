#ifndef ENERGY_CUH
#define ENERGY_CUH

#ifdef __cplusplus
extern "C" {
#endif
    void testit();
#ifdef __cplusplus
}
#endif

#include <stdio.h>
#include <awp/pmcl3d_cons.h>

typedef struct {
    int use;
    int rank;
    double *kinetic_energy_rate;
    double *strain_energy_rate;
    int num_steps;
    // Copies of velocity components at previous time step
    float *d_vxp;

} energy_t;

energy_t energy_init(int useenergy, const int rank, const int num_steps, const int nx, const int ny, const int nz) {
    double *kinetic_energy_rate = NULL;
    double *strain_energy_rate = NULL;
    energy_t energy;
    energy.use = 0;
    energy.rank = -1;

    if (!useenergy) return energy;
    energy.use = 1;

    if (rank == 0)
    printf("Energy output:: enabled\n");

    energy.rank = rank;
    energy.num_steps = num_steps;
    energy.kinetic_energy_rate = (double*)malloc(sizeof (double) * num_steps);
    energy.strain_energy_rate = (double*)malloc(sizeof (double) * num_steps);
    size_t num_bytes = (nx + 2 * ngsl + 4) * (ny + 2 * ngsl + 4) * (nz + 2 * align) * sizeof(float);
    CUCHK(cudaMalloc((void**)&energy.d_vxp, num_bytes));

    return energy;

}


void energy_kinetic_rate(energy_t *e, int step) {
    if (!e->use || step >= e->num_steps) return;

    e->kinetic_energy_rate[step] = (double)step;
    e->strain_energy_rate[step] = (double)step;

}

void energy_output(energy_t *e, const char *filename) {
    if (!e->use) return;
        
    FILE *fh = fopen(filename, "w");
    printf("Writing energy output\n");

    if (e->rank == 0)
    printf("Energy output written to: %s number of steps written: %d \n", filename, e->num_steps);
    for (int i = 0; i < e->num_steps; ++i)
        fprintf(fh, "%g %g \n", e->kinetic_energy_rate[i], e->strain_energy_rate[i]);

    fclose(fh);
}

void energy_free(energy_t *e) {
    if (!e->use) return;
    free(e->kinetic_energy_rate);
    free(e->strain_energy_rate);
    cudaFree(e->d_vxp);
}

#endif
