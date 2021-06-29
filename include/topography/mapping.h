#ifndef _TOPOGRAPHY_MAPPING_H
#define _TOPOGRAPHY_MAPPING_H
#define MAPPING_START_POINT 7

struct mapping {
    double dzb;
    double dzt;
    double h;
    double r[4];
    double z[4];
    double m[4];
};



struct mapping map_init(const double dzb, const double dzt, const double h);
int map_find_cell_r(const double r, const struct mapping *map);
int map_find_cell_z(const double z, const struct mapping *map);
double map_eval(const double r, const struct mapping *map);
double map_eval_derivative(const double r, const struct mapping *map);
double map_invert(const double z, const struct mapping *map, const double eps, const int maxiter);

#endif
