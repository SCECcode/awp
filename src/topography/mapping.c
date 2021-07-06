#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <topography/mapping.h>

const int VERBOSE = 0;
#define EPSILON 1e-4

void hermite_cubic_basis(double b[4], const double t);
void hermite_cubic_basis_derivative(double db[4], const double t);
void adjust(double *m0, double *m1, const double s);
void grid_stretch(struct mapping *map);

void hermite_cubic_basis(double b[4], const double t) {
    b[0] = (1.0 + 2.0 * t) * (1.0 - t) * (1.0 - t);
    b[1] = t * (1.0 - t) * (1.0 - t);
    b[2] = t * t * (3.0 - 2.0 * t);
    b[3] = t * t * (t - 1.0);
}

void hermite_cubic_basis_derivative(double db[4], const double t) {
   db[0] = 6.0 * t * t - 6.0 * t;
   db[1] = 3.0 * t * t - 4.0 * t + 1.0;
   db[2] = -6.0 * t * t + 6.0 * t;
   db[3] = 3.0 * t * t - 2.0 * t;
}


void adjust(double *m0, double *m1, const double s) {
    double a = *m0 / s;
    double b = *m1 / s;

    if (a < 0 || b < 0)
        fprintf(stderr, "%s:%s():%d Non-monotonic mapping function data!\n",
                __FILE__, __func__, __LINE__);

    if (a * a + b * b > 9) {
        double v = 3.0 / sqrt(a * a + b * b);
        *m0 = v * a * s;
        *m1 = v * b * s;
    }
}

void grid_stretch(struct mapping *map) {
    const double dzb = map->dzb;
    const double dzt = map->dzt;
    const double h = map->h;

    double s0 = dzb / h;
    double s1 = (1.0 - dzb - dzt) / ( 1.0 - 2.0 * h);
    double s2 = dzt / h;

    double m0 = s0;
    double m1 = 0.5 * (s0 + s1);
    double m2 = 0.5 * (s1 + s2);
    double m3 = s2;

    adjust(&m0, &m1, s0);
    adjust(&m1, &m2, s1);
    adjust(&m2, &m3, s2);

    map->m[0] = m0;
    map->m[1] = m1;
    map->m[2] = m2;
    map->m[3] = m3;
}

double map_height(const int nz, const double dz) {
        return dz * (nz - 2 - MAPPING_START_POINT);
}


struct mapping map_init(const double dzb, const double dzt, const double h) {

    struct mapping map;
    map.dzb = dzb;
    map.dzt = dzt;
    map.h = h;
    map.r[0] = 0.0;
    map.r[1] = h;
    map.r[2] = 1.0 - h;
    map.r[3] = 1.0;
    map.z[0] = 0.0;
    map.z[1] = dzb;
    map.z[2] = 1.0 - dzt;
    map.z[3] = 1.0;

    grid_stretch(&map);

    return map;
}

int map_find_cell_r(const double r, const struct mapping *map) {
    if (r < -EPSILON) 
        fprintf(stderr, "%s:%s():%d Outside interval (r = %f, r < 0)!\n", 
                __FILE__, __func__, __LINE__, r);
    else if (r <= map->h) return 0;
    else if (r > map->h && r <= 1.0 - map->h) return 1;
    else if (r <= 1.0) return 2;
    if (r > 1.0 + EPSILON) 
        fprintf(stderr, "%s:%s():%d Outside interval (r = %f, r > 1)!\n",
                __FILE__, __func__, __LINE__, r);

    return -1;
}

int map_find_cell_z(const double z, const struct mapping *map) {
    if (z < -EPSILON) 
        fprintf(stderr, "%s:%s():%d Outside interval (z = %f, z < 0)!\n",
                __FILE__, __func__, __LINE__, z);
    else if (z <= map->dzb) return 0;
    else if (z > map->dzb && z <= 1.0 - map->dzt) return 1.0;
    else if (z <= 1.0) return 2;
    if (z > 1.0 + EPSILON) 
        fprintf(stderr, "%s:%s():%d Outside interval (z = %f, z > 1)!\n",
                __FILE__, __func__, __LINE__, z);
    return -1;

}

double map_eval(const double r, const struct mapping *map) {
    int c = map_find_cell_r(r, map);
    double b[4];
    double dr = map->r[c+1] - map->r[c];
    hermite_cubic_basis(b, (r - map->r[c]) / dr);
    return b[0] * map->z[c] + dr * map->m[c] * b[1] + b[2] * map->z[c+1] + dr * b[3] * map->m[c+1];
}

double map_eval_derivative(const double r, const struct mapping *map) {
    int c = map_find_cell_r(r, map);
    double b[4];
    double dr = map->r[c+1] - map->r[c];
    hermite_cubic_basis_derivative(b, (r - map->r[c]) / dr);
    double d =  b[0] * map->z[c] + dr * map->m[c] * b[1] + b[2] * map->z[c+1] + dr * b[3] * map->m[c+1];
    return d / dr;
}

double map_invert(const double z, const struct mapping *map, const double eps, const int maxiter) {

    double rk = z;
    double fk = map_eval(rk, map);
    double dfk = map_eval_derivative(z, map);
    double h = map->h;

    int k = 0;
    double rl = rk;
    while ( (fabs(z - fk) > eps * h || fabs(rk - rl) > eps * h) && k < maxiter) {
        if (VERBOSE)
        printf("k = %d rk = %f fk = %f dfk = %f \n", k, rk, fk, dfk);
        rl = rk;
        rk = rk - (fk - z) / dfk;
        rk = rk < 0 ? 0 : rk;
        rk = rk > 1 ? 1 : rk;
        fk = map_eval(rk, map);
        dfk = map_eval_derivative(rk, map);
        k = k + 1;
    }
    if (VERBOSE)
        printf("\n");
    if (k >= maxiter)
        printf(
            "WARNING: Mapping inversion failed to converge. Either increase "
            "the number of maximum iterations or decrease the tolerance. r = %g, |z - f(r)| = %g \n",
            rk, fabs(z - fk));

    return rk;
}
