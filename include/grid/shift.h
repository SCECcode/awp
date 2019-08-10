#ifndef SHIFT_H
#define SHIFT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <awp/definitions.h>

enum eshift {
        GRID_U1,
        GRID_U2,
        GRID_U3,
        GRID_XX,
        GRID_YY,
        GRID_ZZ,
        GRID_XY,
        GRID_XZ,
        GRID_YZ
};

void shift_node(int *shift);

void shift_u1(int *shift);


void shift_u2(int *shift);

void shift_u3(int *shift);

void shift_xx(int *shift);

void shift_yy(int *shift);

void shift_zz(int *shift);

void shift_xy(int *shift);

void shift_xz(int *shift);

void shift_yz(int *shift);


int3_t grid_node(void);

int3_t grid_u1(void);

int3_t grid_u2(void);

int3_t grid_u3(void);

int3_t grid_x(void);

int3_t grid_y(void);

int3_t grid_z(void);


int3_t grid_xx(void);

int3_t grid_yy(void);

int3_t grid_zz(void);

int3_t grid_xy(void);

int3_t grid_xz(void);

int3_t grid_yz(void);

int3_t grid_shift(enum eshift gridtype);

const char *grid_shift_label(enum eshift gridtype);

#ifdef __cplusplus
}
#endif

#endif
