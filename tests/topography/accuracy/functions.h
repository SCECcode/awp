#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "cuda.h"

typedef void (*fcnp)(_prec *,
                     const int, const int, 
                     const int, const int, 
                     const int, const int, 
                     const int, const int, 
                     const _prec *);

typedef _prec (*fcn_gridp)(const _prec, const _prec, const _prec, const _prec *);

//FIXME: remove
//typedef struct
//{
//        int3_t size;
//        int3_t inner_size;
//        int3_t mem;
//        int3_t coordinate;
//        int3_t offset1;
//        int3_t offset2;
//        int3_t alignment;
//        _prec gridspacing;
//        int num_bytes;
//        int line;
//        int slice;
//        int exclude_top_row;
//} fcn_grid_t;


void fcn_fill_grid(_prec *out, const fcn_grid_t grid, const int3_t shift, const int axis);

void fcn_shift(_prec *out, _prec *in, const fcn_grid_t grid, const _prec shift);
void fcn_power(_prec *out, _prec *in, const fcn_grid_t grid,
               const _prec exponent);
void fcn_normalize(_prec *out, _prec *in, const fcn_grid_t grid);
void fcn_difference(_prec *out, _prec *in1, _prec *in2, const fcn_grid_t grid);
void fcn_apply(_prec *out, fcn_gridp fcn, const _prec *x, const _prec *y,
               const _prec *z, const _prec *properties, const fcn_grid_t grid);
void fcn_abs(_prec *out, _prec *in, const fcn_grid_t grid);

void fcn_constant(_prec *out, 
                  const int i0, const int in, 
                  const int j0, const int jn, 
                  const int k0, const int kn, 
                  const int line, const int slice, 
                  const _prec *args);

void fcn_poly(_prec *out, 
              const int i0, const int in, 
              const int j0, const int jn, 
              const int k0, const int kn, 
              const int line, const int slice, 
              const _prec *args);

void fcn_polynomial(_prec *out, const fcn_grid_t grid, const _prec *coef,
                    const _prec *deg, const int *shift, const int i0,
                    const int in, const int j0, const int jn, const int k0,
                    const int kn);

void fcn_polybndz(_prec *out, 
              const int i0, const int in, 
              const int j0, const int jn, 
              const int k0, const int kn, 
              const int line, const int slice, 
              const _prec *args);


#endif
