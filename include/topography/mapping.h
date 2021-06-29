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

void hermite_cubic_basis(double b[4], const double t);
void hermite_cubic_basis_derivative(double db[4], const double t);


void adjust(double *m0, double *m1, const double s);
void grid_stretch(struct mapping *map);
struct mapping init_mapping(const double dzb, const double dzt, const double h);
int find_cell_r(const double r, const struct mapping *map);
int find_cell_z(const double z, const struct mapping *map);
double eval(const double r, const struct mapping *map);
double eval_derivative(const double r, const struct mapping *map);
double invert(const double z, const struct mapping *map, const double eps, const int maxiter);

#endif
