#ifndef DUMP_H
#define DUMP_H
#include <awp/pmcl3d_cons.h>
void dump_all_data(_prec *d_u1, _prec *d_v1, _prec *d_w1, 
    _prec *d_xx, _prec *d_yy, _prec *d_zz,_prec *d_xz,_prec *d_yz,_prec *d_xy, 
    int nel, int tstep, int tsub, int d_i, int rank, int ncpus);
void dump_nonzeros(_prec *var, int nx, int ny, int nz, char *varname, int desc, int tstep, int tsub, 
     int rank, int ncpus);
void dump_variable(_prec *var, long int nel, char *varname, int desc, int tstep, int tsub, int rank, int ncpus);
void dump_all_stresses(_prec *d_xx, _prec *d_yy, _prec *d_zz, _prec *d_xz, _prec *d_yz, _prec *d_xy, 
    long int nel, int desc, int tstep, int tsub, int rank, int ncpus);
void dump_all_vels(_prec *d_u1, _prec *d_v1, _prec *d_w1, long int nel, int desc, int tstep, int tsub, int rank, int ncpus);
#endif

