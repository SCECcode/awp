#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "topography_test.h"
#include "cutopography.cuh"
#include "cutopography_test.cuh"
#include "topography.h"
#include "functions.h"
#include "grid_check.h"


topo_test_t topo_test_init(topo_t *T)
{
        topo_test_t Tt = {.use = TOPO_TEST, .tol = TOPO_TEST_TOLERANCE};

        if (Tt.use && T->rank == 0) printf("Topography:: testing enabled\n");

#if TOPO_TEST_CONSTX || TOPO_TEST_CONSTY
        Tt.out = T->u1;
        for (int i = 0; i < 3; ++i) {
                Tt.out_shift[i] = T->su1[i];
        }
        Tt.cu1[0] = 1;
        Tt.cv1[0] = 2;
        Tt.cw1[0] = 3;
        topo_test_poly_H(T, T->u1, Tt.cu1, Tt.deg, Tt.out_shift);
        topo_test_poly_H(T, T->v1, Tt.cv1, Tt.deg, Tt.out_shift);
        topo_test_poly_H(T, T->w1, Tt.cw1, Tt.deg, Tt.out_shift);
#endif

#if TOPO_TEST_LINX
        Tt.coef[0] = 1.0;
        Tt.deg[0] = 1.0;
        Tt.out = T->u1;
        for (int i = 0; i < 3; ++i) {
                Tt.out_shift[i] = T->su1[i];
        }
        topo_test_poly_H(T, Tt.out, Tt.coef, Tt.deg, Tt.out_shift);
#endif

#if TOPO_TEST_LINY
        Tt.coef[1] = 1.0;
        Tt.deg[1] = 1.0;
        Tt.out = T->u1;
        Tt.velf = T->f_u1;
        Tt.velb = T->b_u1;
        for (int i = 0; i < 3; ++i) {
                Tt.out_shift[i] = T->su1[i];
                Tt.in_shift[i] = T->su1[i];
        }
        topo_test_poly_H(T, Tt.out, Tt.coef, Tt.deg, Tt.out_shift);
#endif

#if TOPO_TEST_DIFFCONSTX
        Tt.coef[0] = 1.0;
        Tt.out = T->xx;
        Tt.in = T->u1;
        for (int i = 0; i < 3; ++i) {
                Tt.out_shift[i] = T->sxx[i];
                Tt.in_shift[i] = T->su1[i];
        }
        topo_test_poly_H(T, Tt.in, Tt.coef, Tt.deg, Tt.in_shift);
        topo_test_poly_H(T, Tt.out, Tt.coef, Tt.deg, Tt.out_shift);
#endif

#if TOPO_TEST_DIFFCONSTY
        Tt.coef[1] = 1.0;
        Tt.out = T->yy;
        Tt.velf = T->f_v1;
        Tt.velb = T->b_v1;
        Tt.in = T->v1;
        for (int i = 0; i < 3; ++i) {
                Tt.out_shift[i] = T->syy[i];
                Tt.in_shift[i] = T->sv1[i];
        }
        topo_test_poly_H(T, Tt.in, Tt.coef, Tt.deg, Tt.in_shift);
        topo_test_poly_H(T, Tt.out, Tt.coef, Tt.deg, Tt.out_shift);
#endif

#if TOPO_TEST_DIFFCONSTZ
        Tt.coef[2] = 1.0;
        Tt.out = T->xz;
        Tt.in = T->u1;
        for (int i = 0; i < 3; ++i) {
                Tt.out_shift[i] = T->sxz[i];
                Tt.in_shift[i] = T->su1[i];
        }
        topo_test_polyzbnd_H(T, Tt.in, Tt.coef, Tt.deg, Tt.in_shift);
#endif

#if TOPO_TEST_DIFFLINX
        Tt.coef[0] = 1.0;
        Tt.deg[0] = 1.0;
        Tt.out = T->xx;
        Tt.in = T->u1;
        for (int i = 0; i < 3; ++i) {
                Tt.out_shift[i] = T->sxx[i];
                Tt.in_shift[i] = T->su1[i];
        }
        topo_test_poly_H(T, Tt.in, Tt.coef, Tt.deg, Tt.in_shift);
        topo_test_poly_H(T, Tt.out, Tt.coef, Tt.deg, Tt.out_shift);
#endif

#if TOPO_TEST_DIFFLINY
        Tt.coef[1] = 1.0;
        Tt.deg[1] = 1.0;
        Tt.out = T->yy;
        Tt.in = T->v1;
        Tt.velf = T->f_v1;
        Tt.velb = T->b_v1;
        for (int i = 0; i < 3; ++i) {
                Tt.out_shift[i] = T->syy[i];
                Tt.in_shift[i] = T->sv1[i];
        }
        topo_test_poly_H(T, Tt.in, Tt.coef, Tt.deg, Tt.in_shift);
        topo_test_poly_H(T, Tt.out, Tt.coef, Tt.deg, Tt.out_shift);
#endif

#if TOPO_TEST_DIFFLINZ
        Tt.coef[2] = 1.0;
        Tt.deg[2] = 1.0;
        Tt.out = T->xz;
        Tt.in = T->u1;
        for (int i = 0; i < 3; ++i) {
                Tt.out_shift[i] = T->sxz[i];
                Tt.in_shift[i] = T->su1[i];
        }
        topo_test_polyzbnd_H(T, Tt.in, Tt.coef, Tt.deg, Tt.in_shift);
        // Plug in answer in advance to make sure that points that do
        // not get updated have the correct answer (instead of adjusting
        // bounds of test function)
        _prec deg[3] = {0, 0, 0};
        _prec coef[3] = {0, 0, 1};
        topo_test_polyzbnd_H(T, Tt.out, coef, deg, Tt.out_shift);
#endif

#if TOPO_TEST_DIFFQUADX || TOPO_TEST_CGDIFFQUADX
        Tt.coef[0] = 1.0;
        Tt.deg[0] = 2.0;
        Tt.out = T->xx;
        Tt.in = T->u1;
        for (int i = 0; i < 3; ++i) {
                Tt.out_shift[i] = T->sxx[i];
                Tt.in_shift[i] = T->su1[i];
        }
        topo_test_poly_H(T, Tt.in, Tt.coef, Tt.deg, Tt.in_shift);
        topo_test_poly_H(T, Tt.out, Tt.coef, Tt.deg, Tt.out_shift);
#endif

#if TOPO_TEST_DIFFQUADY
        Tt.coef[1] = 1.0;
        Tt.deg[1] = 2.0;
        Tt.out = T->yz;
        Tt.in = T->w1;
        Tt.velf = T->f_w1;
        Tt.velb = T->b_w1;
        for (int i = 0; i < 3; ++i) {
                Tt.out_shift[i] = T->syz[i];
                Tt.in_shift[i] = T->sw1[i];
        }
        topo_test_poly_H(T, Tt.in, Tt.coef, Tt.deg, Tt.in_shift);
        topo_test_poly_H(T, Tt.out, Tt.coef, Tt.deg, Tt.out_shift);
#endif

#if TOPO_TEST_DIFFQUADZ
        Tt.coef[2] = 1.0;
        Tt.deg[2] = 2.0;
        Tt.out = T->xz;
        Tt.in = T->u1;
        for (int i = 0; i < 3; ++i) {
                Tt.out_shift[i] = T->sxz[i];
                Tt.in_shift[i] = T->su1[i];
        }
        topo_test_polyzbnd_H(T, Tt.in, Tt.coef, Tt.deg, Tt.in_shift);
        // Plug in answer in advance to make sure that points that do
        // not get updated have the correct answer (instead of adjusting
        // bounds of test function)
        _prec deg[3] = {0, 0, 1};
        _prec coef[3] = {0, 0, 2};
        topo_test_polyzbnd_H(T, Tt.out, coef, deg, Tt.out_shift);
#endif

#if TOPO_TEST_VELCONST
        Tt.cxx[0] = 1.0;
        Tt.cyy[0] = 1.0;
        Tt.czz[0] = 1.0;
        Tt.cxy[0] = 1.0;
        Tt.cxz[0] = 1.0;
        Tt.cyz[0] = 1.0;
        // Input
        topo_test_polystr_H(T, T->xx, Tt.cxx, Tt.deg, T->sxx);
        topo_test_polystr_H(T, T->yy, Tt.cyy, Tt.deg, T->syy);
        topo_test_polystr_H(T, T->zz, Tt.czz, Tt.deg, T->szz);
        topo_test_polystr_H(T, T->xy, Tt.cxy, Tt.deg, T->sxy);
        topo_test_polystr_H(T, T->xz, Tt.cxz, Tt.deg, T->sxz);
        topo_test_polystr_H(T, T->yz, Tt.cyz, Tt.deg, T->syz);

        // Output
        topo_test_poly_H(T, T->u1, Tt.cu1, Tt.deg, T->su1);
        topo_test_poly_H(T, T->v1, Tt.cv1, Tt.deg, T->sv1);
        topo_test_poly_H(T, T->w1, Tt.cw1, Tt.deg, T->sw1);
#endif

#if TOPO_TEST_VELLINX
        Tt.cxx[0] = 1.0;
        Tt.cyy[0] = 1.0;
        Tt.czz[0] = 1.0;
        Tt.cxy[0] = 1.0;
        Tt.cxz[0] = 1.0;
        Tt.cyz[0] = 1.0;
        Tt.deg[0] = 1.0;
        // Input
        topo_test_polystr_H(T, T->xx, Tt.cxx, Tt.deg, T->sxx);
        topo_test_polystr_H(T, T->yy, Tt.cyy, Tt.deg, T->syy);
        topo_test_polystr_H(T, T->zz, Tt.czz, Tt.deg, T->szz);
        topo_test_polystr_H(T, T->xy, Tt.cxy, Tt.deg, T->sxy);
        topo_test_polystr_H(T, T->xz, Tt.cxz, Tt.deg, T->sxz);
        topo_test_polystr_H(T, T->yz, Tt.cyz, Tt.deg, T->syz);

        // Output
        Tt.cu1[0] = 0.0;
        Tt.cv1[0] = 0.0;
        Tt.cw1[0] = 0.0;
        topo_test_poly_H(T, T->u1, Tt.cu1, Tt.deg, T->su1);
        topo_test_poly_H(T, T->v1, Tt.cv1, Tt.deg, T->sv1);
        topo_test_poly_H(T, T->w1, Tt.cw1, Tt.deg, T->sw1);
#endif

#if TOPO_TEST_VELLINY
        Tt.cxx[1] = 1.0;
        Tt.cyy[1] = 1.0;
        Tt.czz[1] = 1.0;
        Tt.cxy[1] = 1.0;
        Tt.cxz[1] = 1.0;
        Tt.cyz[1] = 1.0;
        Tt.deg[1] = 1.0;
        // Input
        topo_test_polystr_H(T, T->xx, Tt.cxx, Tt.deg, T->sxx);
        topo_test_polystr_H(T, T->yy, Tt.cyy, Tt.deg, T->syy);
        topo_test_polystr_H(T, T->zz, Tt.czz, Tt.deg, T->szz);
        topo_test_polystr_H(T, T->xy, Tt.cxy, Tt.deg, T->sxy);
        topo_test_polystr_H(T, T->xz, Tt.cxz, Tt.deg, T->sxz);
        topo_test_polystr_H(T, T->yz, Tt.cyz, Tt.deg, T->syz);

        // Output
        Tt.cu1[0] = 0.0;
        Tt.cv1[0] = 0.0;
        Tt.cw1[0] = 0.0;
        topo_test_poly_H(T, T->u1, Tt.cu1, Tt.deg, T->su1);
        topo_test_poly_H(T, T->v1, Tt.cv1, Tt.deg, T->sv1);
        topo_test_poly_H(T, T->w1, Tt.cw1, Tt.deg, T->sw1);
#endif

#if TOPO_TEST_VELLINZ
        Tt.cxx[2] = 1.0;
        Tt.cyy[2] = 1.0;
        Tt.czz[2] = 1.0;
        Tt.cxy[2] = 1.0;
        Tt.cxz[2] = 1.0;
        Tt.cyz[2] = 1.0;
        Tt.deg[2] = 1.0;
        // Input
        topo_test_polystrzbnd_H(T, T->xx, Tt.cxx, Tt.deg, T->sxx);
        topo_test_polystrzbnd_H(T, T->yy, Tt.cyy, Tt.deg, T->syy);
        topo_test_polystrzbnd_H(T, T->zz, Tt.czz, Tt.deg, T->szz);
        topo_test_polystrzbnd_H(T, T->xy, Tt.cxy, Tt.deg, T->sxy);
        topo_test_polystrzbnd_H(T, T->xz, Tt.cxz, Tt.deg, T->sxz);
        topo_test_polystrzbnd_H(T, T->yz, Tt.cyz, Tt.deg, T->syz);

        // Output
        Tt.cu1[0] = 0;
        Tt.cv1[0] = 0;
        Tt.cw1[0] = 0;
        topo_test_polyzbnd_H(T, T->u1, Tt.cu1, Tt.deg, T->su1);
        topo_test_polyzbnd_H(T, T->v1, Tt.cv1, Tt.deg, T->sv1);
        topo_test_polyzbnd_H(T, T->w1, Tt.cw1, Tt.deg, T->sw1);
#endif

#if TOPO_TEST_VELQUADX
        Tt.cxx[0] = 1.0;
        Tt.cyy[0] = 1.0;
        Tt.czz[0] = 1.0;
        Tt.cxy[0] = 1.0;
        Tt.cxz[0] = 1.0;
        Tt.cyz[0] = 1.0;
        Tt.deg[0] = 2.0;
        // Input
        topo_test_polystr_H(T, T->xx, Tt.cxx, Tt.deg, T->sxx);
        topo_test_polystr_H(T, T->yy, Tt.cyy, Tt.deg, T->syy);
        topo_test_polystr_H(T, T->zz, Tt.czz, Tt.deg, T->szz);
        topo_test_polystr_H(T, T->xy, Tt.cxy, Tt.deg, T->sxy);
        topo_test_polystr_H(T, T->xz, Tt.cxz, Tt.deg, T->sxz);
        topo_test_polystr_H(T, T->yz, Tt.cyz, Tt.deg, T->syz);

        // Output
        Tt.cu1[0] = 0;
        Tt.cv1[0] = 0;
        Tt.cw1[0] = 0;
        topo_test_poly_H(T, T->u1, Tt.cu1, Tt.deg, T->su1);
        topo_test_poly_H(T, T->v1, Tt.cv1, Tt.deg, T->sv1);
        topo_test_poly_H(T, T->w1, Tt.cw1, Tt.deg, T->sw1);
#endif

#if TOPO_TEST_VELQUADY
        Tt.cxx[1] = 1.0;
        Tt.cyy[1] = 1.0;
        Tt.czz[1] = 1.0;
        Tt.cxy[1] = 1.0;
        Tt.cxz[1] = 1.0;
        Tt.cyz[1] = 1.0;
        Tt.deg[1] = 2.0;
        // Input
        topo_test_polystr_H(T, T->xx, Tt.cxx, Tt.deg, T->sxx);
        topo_test_polystr_H(T, T->yy, Tt.cyy, Tt.deg, T->syy);
        topo_test_polystr_H(T, T->zz, Tt.czz, Tt.deg, T->szz);
        topo_test_polystr_H(T, T->xy, Tt.cxy, Tt.deg, T->sxy);
        topo_test_polystr_H(T, T->xz, Tt.cxz, Tt.deg, T->sxz);
        topo_test_polystr_H(T, T->yz, Tt.cyz, Tt.deg, T->syz);

        // Output
        Tt.cu1[0] = 0;
        Tt.cv1[0] = 0;
        Tt.cw1[0] = 0;
        topo_test_poly_H(T, T->u1, Tt.cu1, Tt.deg, T->su1);
        topo_test_poly_H(T, T->v1, Tt.cv1, Tt.deg, T->sv1);
        topo_test_poly_H(T, T->w1, Tt.cw1, Tt.deg, T->sw1);
#endif

#if TOPO_TEST_VELQUADZ
        Tt.cxx[2] = 1.0;
        Tt.cyy[2] = 1.0;
        Tt.czz[2] = 1.0;
        Tt.cxy[2] = 1.0;
        Tt.cxz[2] = 1.0;
        Tt.cyz[2] = 1.0;
        Tt.deg[2] = 2.0;
        // Input
        topo_test_polystrzbnd_H(T, T->xx, Tt.cxx, Tt.deg, T->sxx);
        topo_test_polystrzbnd_H(T, T->yy, Tt.cyy, Tt.deg, T->syy);
        topo_test_polystrzbnd_H(T, T->zz, Tt.czz, Tt.deg, T->szz);
        topo_test_polystrzbnd_H(T, T->xy, Tt.cxy, Tt.deg, T->sxy);
        topo_test_polystrzbnd_H(T, T->xz, Tt.cxz, Tt.deg, T->sxz);
        topo_test_polystrzbnd_H(T, T->yz, Tt.cyz, Tt.deg, T->syz);

        // Output
        Tt.cu1[0] = 0;
        Tt.cv1[0] = 0;
        Tt.cw1[0] = 0;
        topo_test_polyzbnd_H(T, T->u1, Tt.cu1, Tt.deg, T->su1);
        topo_test_polyzbnd_H(T, T->v1, Tt.cv1, Tt.deg, T->sv1);
        topo_test_polyzbnd_H(T, T->w1, Tt.cw1, Tt.deg, T->sw1);
#endif

#if TOPO_TEST_VELFRONTBACK
        Tt.cxx[1] = 1.0;
        Tt.cyy[1] = 1.0;
        Tt.czz[1] = 1.0;
        Tt.cxy[1] = 1.0;
        Tt.cxz[1] = 1.0;
        Tt.cyz[1] = 1.0;
        Tt.deg[1] = 2.0;
        // Input
        topo_test_polystr_H(T, T->xx, Tt.cxx, Tt.deg, T->sxx);
        topo_test_polystr_H(T, T->yy, Tt.cyy, Tt.deg, T->syy);
        topo_test_polystr_H(T, T->zz, Tt.czz, Tt.deg, T->szz);
        topo_test_polystr_H(T, T->xy, Tt.cxy, Tt.deg, T->sxy);
        topo_test_polystr_H(T, T->xz, Tt.cxz, Tt.deg, T->sxz);
        topo_test_polystr_H(T, T->yz, Tt.cyz, Tt.deg, T->syz);

        // Output
        Tt.cu1[0] = 0;
        Tt.cv1[0] = 0;
        Tt.cw1[0] = 0;
        topo_test_poly_H(T, T->u1, Tt.cu1, Tt.deg, T->su1);
        topo_test_poly_H(T, T->v1, Tt.cv1, Tt.deg, T->sv1);
        topo_test_poly_H(T, T->w1, Tt.cw1, Tt.deg, T->sw1);
#endif

#if TOPO_TEST_STRCONST
        // Input
        Tt.cu1[0] = 0;
        Tt.cv1[0] = 0;
        Tt.cw1[0] = 0;
        Tt.deg[0] = 0.0;
        topo_test_poly_H(T, T->u1, Tt.cu1, Tt.deg, T->su1);
        topo_test_poly_H(T, T->v1, Tt.cv1, Tt.deg, T->sv1);
        topo_test_poly_H(T, T->w1, Tt.cw1, Tt.deg, T->sw1);

#endif

#if TOPO_TEST_STRLINX
        // Input
        Tt.cu1[0] = 1;
        Tt.cv1[0] = 1;
        Tt.cw1[0] = 1;
        Tt.deg[0] = 1.0;
        topo_test_poly_H(T, T->u1, Tt.cu1, Tt.deg, T->su1);
        topo_test_poly_H(T, T->v1, Tt.cv1, Tt.deg, T->sv1);
        topo_test_poly_H(T, T->w1, Tt.cw1, Tt.deg, T->sw1);

        // Output
        Tt.cxx[0] = 0.0;
        Tt.cyy[0] = 0.0;
        Tt.czz[0] = 0.0;
        Tt.cxy[0] = 0.0;
        Tt.cxz[0] = 0.0;
        Tt.cyz[0] = 0.0;
        topo_test_polystr_H(T, T->xx, Tt.cxx, Tt.deg, T->sxx);
        topo_test_polystr_H(T, T->yy, Tt.cyy, Tt.deg, T->syy);
        topo_test_polystr_H(T, T->zz, Tt.czz, Tt.deg, T->szz);
        topo_test_polystr_H(T, T->xy, Tt.cxy, Tt.deg, T->sxy);
        topo_test_polystr_H(T, T->xz, Tt.cxz, Tt.deg, T->sxz);
        topo_test_polystr_H(T, T->yz, Tt.cyz, Tt.deg, T->syz);
#endif

#if TOPO_TEST_STRLINY
        // Input
        Tt.cu1[1] = 1;
        Tt.cv1[1] = 1;
        Tt.cw1[1] = 1;
        Tt.deg[1] = 1.0;
        topo_test_poly_H(T, T->u1, Tt.cu1, Tt.deg, T->su1);
        topo_test_poly_H(T, T->v1, Tt.cv1, Tt.deg, T->sv1);
        topo_test_poly_H(T, T->w1, Tt.cw1, Tt.deg, T->sw1);

        // Output
        Tt.cxx[0] = 0.0;
        Tt.cyy[0] = 0.0;
        Tt.czz[0] = 0.0;
        Tt.cxy[0] = 0.0;
        Tt.cxz[0] = 0.0;
        Tt.cyz[0] = 0.0;
        topo_test_polystr_H(T, T->xx, Tt.cxx, Tt.deg, T->sxx);
        topo_test_polystr_H(T, T->yy, Tt.cyy, Tt.deg, T->syy);
        topo_test_polystr_H(T, T->zz, Tt.czz, Tt.deg, T->szz);
        topo_test_polystr_H(T, T->xy, Tt.cxy, Tt.deg, T->sxy);
        topo_test_polystr_H(T, T->xz, Tt.cxz, Tt.deg, T->sxz);
        topo_test_polystr_H(T, T->yz, Tt.cyz, Tt.deg, T->syz);
#endif

#if TOPO_TEST_STRLINZ
        // Input
        Tt.cu1[2] = 1;
        Tt.cv1[2] = 1;
        Tt.cw1[2] = 1;
        Tt.deg[2] = 1.0;
        topo_test_polyzbnd_H(T, T->u1, Tt.cu1, Tt.deg, T->su1);
        topo_test_polyzbnd_H(T, T->v1, Tt.cv1, Tt.deg, T->sv1);
        topo_test_polyzbnd_H(T, T->w1, Tt.cw1, Tt.deg, T->sw1);

        // Output
        Tt.cxx[0] = 0.0;
        Tt.cyy[0] = 0.0;
        Tt.czz[0] = 0.0;
        Tt.cxy[0] = 0.0;
        Tt.cxz[0] = 0.0;
        Tt.cyz[0] = 0.0;
        topo_test_polystrzbnd_H(T, T->xx, Tt.cxx, Tt.deg, T->sxx);
        topo_test_polystrzbnd_H(T, T->yy, Tt.cyy, Tt.deg, T->syy);
        topo_test_polystrzbnd_H(T, T->zz, Tt.czz, Tt.deg, T->szz);
        topo_test_polystrzbnd_H(T, T->xy, Tt.cxy, Tt.deg, T->sxy);
        topo_test_polystrzbnd_H(T, T->xz, Tt.cxz, Tt.deg, T->sxz);
        topo_test_polystrzbnd_H(T, T->yz, Tt.cyz, Tt.deg, T->syz);
#endif

#if TOPO_TEST_STRQUADX
        // Input
        Tt.cu1[0] = 1;
        Tt.cv1[0] = 1;
        Tt.cw1[0] = 1;
        Tt.deg[0] = 2.0;
        topo_test_poly_H(T, T->u1, Tt.cu1, Tt.deg, T->su1);
        topo_test_poly_H(T, T->v1, Tt.cv1, Tt.deg, T->sv1);
        topo_test_poly_H(T, T->w1, Tt.cw1, Tt.deg, T->sw1);

        // Output
        Tt.cxx[0] = 0.0;
        Tt.cyy[0] = 0.0;
        Tt.czz[0] = 0.0;
        Tt.cxy[0] = 0.0;
        Tt.cxz[0] = 0.0;
        Tt.cyz[0] = 0.0;
        topo_test_polystr_H(T, T->xx, Tt.cxx, Tt.deg, T->sxx);
        topo_test_polystr_H(T, T->yy, Tt.cyy, Tt.deg, T->syy);
        topo_test_polystr_H(T, T->zz, Tt.czz, Tt.deg, T->szz);
        topo_test_polystr_H(T, T->xy, Tt.cxy, Tt.deg, T->sxy);
        topo_test_polystr_H(T, T->xz, Tt.cxz, Tt.deg, T->sxz);
        topo_test_polystr_H(T, T->yz, Tt.cyz, Tt.deg, T->syz);
#endif

#if TOPO_TEST_STRQUADY
        // Input
        Tt.cu1[1] = 1;
        Tt.cv1[1] = 1;
        Tt.cw1[1] = 1;
        Tt.deg[1] = 2.0;
        topo_test_poly_H(T, T->u1, Tt.cu1, Tt.deg, T->su1);
        topo_test_poly_H(T, T->v1, Tt.cv1, Tt.deg, T->sv1);
        topo_test_poly_H(T, T->w1, Tt.cw1, Tt.deg, T->sw1);

        // Output
        Tt.cxx[0] = 0.0;
        Tt.cyy[0] = 0.0;
        Tt.czz[0] = 0.0;
        Tt.cxy[0] = 0.0;
        Tt.cxz[0] = 0.0;
        Tt.cyz[0] = 0.0;
        topo_test_polystr_H(T, T->xx, Tt.cxx, Tt.deg, T->sxx);
        topo_test_polystr_H(T, T->yy, Tt.cyy, Tt.deg, T->syy);
        topo_test_polystr_H(T, T->zz, Tt.czz, Tt.deg, T->szz);
        topo_test_polystr_H(T, T->xy, Tt.cxy, Tt.deg, T->sxy);
        topo_test_polystr_H(T, T->xz, Tt.cxz, Tt.deg, T->sxz);
        topo_test_polystr_H(T, T->yz, Tt.cyz, Tt.deg, T->syz);
#endif

#if TOPO_TEST_STRQUADZ
        // Input
        Tt.cu1[2] = 1;
        Tt.cv1[2] = 1;
        Tt.cw1[2] = 1;
        Tt.deg[2] = 2.0;
        topo_test_polyzbnd_H(T, T->u1, Tt.cu1, Tt.deg, T->su1);
        topo_test_polyzbnd_H(T, T->v1, Tt.cv1, Tt.deg, T->sv1);
        topo_test_polyzbnd_H(T, T->w1, Tt.cw1, Tt.deg, T->sw1);

        // Output
        Tt.cxx[0] = 0.0;
        Tt.cyy[0] = 0.0;
        Tt.czz[0] = 0.0;
        Tt.cxy[0] = 0.0;
        Tt.cxz[0] = 0.0;
        Tt.cyz[0] = 0.0;
        topo_test_polystrzbnd_H(T, T->xx, Tt.cxx, Tt.deg, T->sxx);
        topo_test_polystrzbnd_H(T, T->yy, Tt.cyy, Tt.deg, T->syy);
        topo_test_polystrzbnd_H(T, T->zz, Tt.czz, Tt.deg, T->szz);
        topo_test_polystrzbnd_H(T, T->xy, Tt.cxy, Tt.deg, T->sxy);
        topo_test_polystrzbnd_H(T, T->xz, Tt.cxz, Tt.deg, T->sxz);
        topo_test_polystrzbnd_H(T, T->yz, Tt.cyz, Tt.deg, T->syz);
#endif

        return Tt;
}

void topo_test_velfront(topo_test_t *Tt, topo_t *T)
{
        if (T->y_rank_f < 0) {
                return;
        }

#if TOPO_TEST_CONSTY
        topo_test_polyf_H(T, T->f_u1, Tt->cu1, Tt->deg, Tt->in_shift);
        topo_test_polyf_H(T, T->f_v1, Tt->cv1, Tt->deg, Tt->in_shift);
        topo_test_polyf_H(T, T->f_w1, Tt->cw1, Tt->deg, Tt->in_shift);
#endif

#if TOPO_TEST_STRLINY || TOPO_TEST_STRQUADY
        topo_test_polyf_H(T, T->f_u1, Tt->cu1, Tt->deg, T->su1);
        topo_test_polyf_H(T, T->f_v1, Tt->cv1, Tt->deg, T->sv1);
        topo_test_polyf_H(T, T->f_w1, Tt->cw1, Tt->deg, T->sw1);
#endif

#if TOPO_TEST_STRLINZ || TOPO_TEST_STRQUADZ
        topo_test_polyzbndf_H(T, T->f_u1, Tt->cu1, Tt->deg, T->su1);
        topo_test_polyzbndf_H(T, T->f_v1, Tt->cv1, Tt->deg, T->sv1);
        topo_test_polyzbndf_H(T, T->f_w1, Tt->cw1, Tt->deg, T->sw1);
#endif

#if TOPO_TEST_LINY || TOPO_TEST_DIFFCONSTY || TOPO_TEST_DIFFLINY || \
    TOPO_TEST_DIFFQUADY
        topo_test_polyf_H(T, Tt->velf, Tt->coef, Tt->deg, Tt->in_shift);
#endif

#if TOPO_TEST_VELFRONTBACK
        topo_velocity_front_H(T);
#endif
}

void topo_test_velback(topo_test_t *Tt, topo_t *T)
{
        if (T->y_rank_b < 0) {
                return;
        }

#if TOPO_TEST_CONSTY
        topo_test_polyb_H(T, T->b_u1, Tt->cu1, Tt->deg, Tt->in_shift);
        topo_test_polyb_H(T, T->b_v1, Tt->cv1, Tt->deg, Tt->in_shift);
        topo_test_polyb_H(T, T->b_w1, Tt->cw1, Tt->deg, Tt->in_shift);
#endif

#if TOPO_TEST_STRLINY || TOPO_TEST_STRQUADY
        topo_test_polyb_H(T, T->b_u1, Tt->cu1, Tt->deg, T->su1);
        topo_test_polyb_H(T, T->b_v1, Tt->cv1, Tt->deg, T->sv1);
        topo_test_polyb_H(T, T->b_w1, Tt->cw1, Tt->deg, T->sw1);
#endif

#if TOPO_TEST_STRLINZ || TOPO_TEST_STRQUADZ
        topo_test_polyzbndb_H(T, T->b_u1, Tt->cu1, Tt->deg, T->su1);
        topo_test_polyzbndb_H(T, T->b_v1, Tt->cv1, Tt->deg, T->sv1);
        topo_test_polyzbndb_H(T, T->b_w1, Tt->cw1, Tt->deg, T->sw1);
#endif

#if TOPO_TEST_LINY || TOPO_TEST_DIFFCONSTY || TOPO_TEST_DIFFLINY || \
    TOPO_TEST_DIFFQUADY
        topo_test_polyb_H(T, Tt->velb, Tt->coef, Tt->deg, Tt->in_shift);
#endif

#if TOPO_TEST_VELFRONTBACK
        topo_velocity_back_H(T);
#endif
}

void topo_test_velx(const topo_test_t *Tt, topo_t *T)
{
#if TOPO_TEST_VELCONST || TOPO_TEST_VELLINX || TOPO_TEST_VELLINY ||  \
    TOPO_TEST_VELLINZ || TOPO_TEST_VELQUADX || TOPO_TEST_VELQUADY || \
    TOPO_TEST_VELQUADZ
        topo_velocity_interior_H(T);
#endif
}

void topo_test_stress(const topo_test_t *Tt, topo_t *T)
{
#if TOPO_TEST_DIFFCONSTX || TOPO_TEST_DIFFLINX || TOPO_TEST_DIFFQUADX
        topo_test_diffx_H(T, T->xx, T->u1);
#endif

#if TOPO_TEST_DIFFCONSTZ || TOPO_TEST_DIFFLINZ || TOPO_TEST_DIFFQUADZ
        topo_test_diffz_H(T, Tt->out, Tt->in);
#endif

#if TOPO_TEST_CGDIFFQUADX
        topo_test_cgdiffx_H(T, T->xx, T->u1);
#endif

#if TOPO_TEST_DIFFCONSTY || TOPO_TEST_DIFFLINY
        topo_test_diffy_H(T, T->yy, T->v1);
#endif

#if TOPO_TEST_DIFFQUADY
        topo_test_diffy_H(T, T->yz, T->w1);
#endif
}

void topo_test_stress_interior(const topo_test_t *Tt, topo_t *T)
{
#if TOPO_TEST_STRCONST || TOPO_TEST_STRLINX || TOPO_TEST_STRLINY ||  \
    TOPO_TEST_STRLINZ || TOPO_TEST_STRQUADX || TOPO_TEST_STRQUADY || \
    TOPO_TEST_STRQUADZ
        topo_stress_interior_H(T);
#endif
}

void topo_test_stress_sides(const topo_test_t *Tt, topo_t *T)
{
#if TOPO_TEST_STRCONST || TOPO_TEST_STRLINX || TOPO_TEST_STRLINY ||  \
    TOPO_TEST_STRLINZ || TOPO_TEST_STRQUADX || TOPO_TEST_STRQUADY || \
    TOPO_TEST_STRQUADZ
        topo_stress_left_H(T);
        topo_stress_right_H(T);
#endif
}

int topo_test_finalize(const topo_test_t *Tt, topo_t *T)
{
        if (!Tt->use) return 0;

        int err = 0;

#if TOPO_TEST_CONSTX
        err |= topo_test_constx(Tt, T);
#endif

#if TOPO_TEST_CONSTY
        err |= topo_test_consty(Tt, T);
#endif

#if TOPO_TEST_LINX
        err |= topo_test_linx(Tt, T);
#endif

#if TOPO_TEST_LINY
        err |= topo_test_liny(Tt, T);
#endif

#if TOPO_TEST_DIFFCONSTX
        err |= topo_test_diffconstx(Tt, T);
#endif

#if TOPO_TEST_DIFFCONSTY
        err |= topo_test_diffconsty(Tt, T);
#endif

#if TOPO_TEST_DIFFCONSTZ
        err |= topo_test_diffconstz(Tt, T);
#endif

#if TOPO_TEST_DIFFLINX
        err |= topo_test_difflinx(Tt, T);
#endif

#if TOPO_TEST_DIFFLINY
        err |= topo_test_diffliny(Tt, T);
#endif

#if TOPO_TEST_DIFFLINZ
        err |= topo_test_difflinz(Tt, T);
#endif

#if TOPO_TEST_DIFFQUADX || TOPO_TEST_CGDIFFQUADX
        err |= topo_test_diffquadx(Tt, T);
#endif

#if TOPO_TEST_DIFFQUADY
        err |= topo_test_diffquady(Tt, T);
#endif

#if TOPO_TEST_DIFFQUADZ
        err |= topo_test_diffquadz(Tt, T);
#endif

#if TOPO_TEST_VELCONST
        err |= topo_test_velconst(Tt, T);
#endif

#if TOPO_TEST_VELLINX
        err |= topo_test_vellinx(Tt, T);
#endif

#if TOPO_TEST_VELLINY
        err |= topo_test_velliny(Tt, T);
#endif

#if TOPO_TEST_VELLINZ
        err |= topo_test_vellinz(Tt, T);
#endif

#if TOPO_TEST_VELQUADX
        err |= topo_test_velquadx(Tt, T);
#endif

#if TOPO_TEST_VELQUADY
        err |= topo_test_velquady(Tt, T);
#endif

#if TOPO_TEST_VELQUADZ
        err |= topo_test_velquadz(Tt, T);
#endif

#if TOPO_TEST_VELFRONTBACK
        err |= topo_test_velfrontback(Tt, T);
#endif

#if TOPO_TEST_STRCONST
        err |= topo_test_strconst(Tt, T);
#endif

#if TOPO_TEST_STRLINX
        err |= topo_test_strlinx(Tt, T);
#endif

#if TOPO_TEST_STRLINY
        err |= topo_test_strliny(Tt, T);
#endif

#if TOPO_TEST_STRLINZ
        err |= topo_test_strlinz(Tt, T);
#endif

#if TOPO_TEST_STRQUADX
        err |= topo_test_strquadx(Tt, T);
#endif

#if TOPO_TEST_STRQUADY
        err |= topo_test_strquady(Tt, T);
#endif

#if TOPO_TEST_STRQUADZ
        err |= topo_test_strquadz(Tt, T);
#endif

        return err;
}

int topo_test_constx(const topo_test_t *Tt, const topo_t *T)
{
        // Select regions to test
        // 1 : region will be tested
        // 0 : region will not be tested
        // There are only two processes in this test, so MPI send-recv only
        // takes place in the x-direction.
        int regions[9] = {0, 0, 0,
                          1, 1, 1,
                          0, 0, 0};

        if (T->rank == 0) {
                regions[3] = 0;
        }
        if (T->rank == 1) {
                regions[5] = 0;
        }
        
        _prec *fields[3] = {T->u1, T->v1, T->w1};
        char *fields_str[3] = {"u1", "v1", "w1"};
        _prec ans[3] = {1.0, 2.0, 3.0};

        int err = 0;
        for (int i = 0; i < 3; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[1] = {(_prec)ans[i]};
                err |= topo_test_fcn(fcn_constant, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_consty(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 1, 0,
                          0, 1, 0,
                          0, 1, 0};

        if (T->rank == 0) {
                regions[7] = 0;
        }
        if (T->rank == 1) {
                regions[1] = 0;
        }
        
        _prec *fields[3] = {T->u1, T->v1, T->w1};
        char *fields_str[3] = {"u1", "v1", "w1"};
        _prec ans[3] = {1.0, 2.0, 3.0};

        int err = 0;
        for (int i = 0; i < 3; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[1] = {(_prec)ans[i]};
                err |= topo_test_fcn(fcn_constant, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_linx(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 0, 0,
                          1, 1, 1,
                          0, 0, 0};

        if (T->rank == 0) {
                regions[3] = 0;
        }
        if (T->rank == 1) {
                regions[5] = 0;
        }
        
        _prec *fields[3] = {T->u1};
        char *fields_str[3] = {"u1"};

        int err = 0;
        for (int i = 0; i < 1; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[13] = {1.0, 0.0, 0.0,
                                 1.0, 0.0, 0.0,
                                 T->su1[0], T->su1[1], T->su1[2],
                                 T->coord[0], T->coord[1],
                                 T->nx, T->ny};
                err |= topo_test_fcn(fcn_poly, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_liny(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 1, 0,
                          0, 1, 0,
                          0, 1, 0};

        if (T->rank == 0) {
                regions[7] = 0;
        }
        if (T->rank == 1) {
                regions[1] = 0;
        }
        
        _prec *fields[3] = {T->u1, T->v1, T->w1};
        char *fields_str[3] = {"u1", "v1", "w1"};

        int err = 0;
        for (int i = 0; i < 1; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[13] = {0.0, 1.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 T->su1[0], T->su1[1], T->su1[2],
                                 T->coord[0], T->coord[1],
                                 T->nx, T->ny};
                err |= topo_test_fcn(fcn_poly, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_diffconstx(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 0, 0,
                          1, 1, 1,
                          0, 0, 0};

        _prec *fields[1] = {T->xx};
        char *fields_str[1] = {"xx"};
       
        int err = 0;

        // Only check the rank in the middle because the ranks on the boundary
        // will not correctly compute the stencil due to applying an interior
        // stencil on the boundary
        if (T->rank != 1) {
                return err;
        }

        for (int i = 0; i < 1; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[1] = {0.0};
                err |= topo_test_fcn(fcn_constant, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_diffconsty(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 1, 0,
                          0, 1, 0,
                          0, 1, 0};

        _prec *fields[1] = {T->yy};
        char *fields_str[1] = {"yy"};
       
        int err = 0;

        // Only check the rank in the middle because the ranks on the boundary
        // will not correctly compute the stencil due to applying an interior
        // stencil on the boundary
        if (T->rank != 1) {
                return err;
        }

        for (int i = 0; i < 1; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[1] = {0.0};
                err |= topo_test_fcn(fcn_constant, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_diffconstz(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 1, 0,
                          0, 1, 0,
                          0, 1, 0};

        _prec *fields[1] = {Tt->out};
        char *fields_str[1] = {"xz"};
       
        int err = 0;

        // Only check the rank in the middle because the ranks on the boundary
        // will not correctly compute the stencil due to applying an interior
        // stencil on the boundary
        if (T->rank != 1) {
                return err;
        }

        for (int i = 0; i < 1; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[1] = {0.0};
                err |= topo_test_fcn(fcn_constant, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_difflinx(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 0, 0,
                          0, 1, 0,
                          0, 0, 0};

        _prec *fields[1] = {T->xx};
        char *fields_str[1] = {"xx"};
       
        int err = 0;

        // Only check the rank in the middle because the ranks on the boundary
        // will not correctly compute the stencil due to applying an interior
        // stencil on the boundary
        if (T->rank != 1) {
                return err;
        }

        for (int i = 0; i < 1; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[13] = {1.0, 0.0, 0.0,
                                  0, 0.0, 0.0,
                                  T->su1[0], T->su1[1], T->su1[2],
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_fcn(fcn_poly, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_diffliny(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 0, 0,
                          0, 1, 0,
                          0, 0, 0};

        _prec *fields[1] = {T->yy};
        char *fields_str[1] = {"yy"};
       
        int err = 0;

        // Only check the rank in the middle because the ranks on the boundary
        // will not correctly compute the stencil due to applying an interior
        // stencil on the boundary
        if (T->rank != 1) {
                return err;
        }

        for (int i = 0; i < 1; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[13] = {0.0, 1.0, 0.0,
                                  0, 0.0, 0.0,
                                  T->sv1[0], T->sv1[1], T->sv1[2],
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_fcn(fcn_poly, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_difflinz(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 0, 0,
                          0, 1, 0,
                          0, 0, 0};

        _prec *fields[1] = {Tt->out};
        char *fields_str[1] = {"xz"};
       
        int err = 0;

        for (int i = 0; i < 1; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[13] = {0.0, 0.0, 1.0,
                                  0, 0.0, 0.0,
                                  Tt->out_shift[0], Tt->out_shift[1], 
                                  Tt->out_shift[2],
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_fcn(fcn_polybndz, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_diffquadx(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 0, 0,
                          0, 1, 0,
                          0, 0, 0};

        _prec *fields[1] = {T->xx};
        char *fields_str[1] = {"xx"};
       
        int err = 0;

        // Only check the rank in the middle because the ranks on the boundary
        // will not correctly compute the stencil due to applying an interior
        // stencil on the boundary
        if (T->rank != 1) {
                return err;
        }

        for (int i = 0; i < 1; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[13] = {2.0, 0.0, 0.0,
                                  1.0, 0.0, 0.0,
                                  Tt->out_shift[0], Tt->out_shift[1], 
                                  Tt->out_shift[2],
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_fcn(fcn_poly, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_diffquady(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 0, 0,
                          0, 1, 0,
                          0, 0, 0};

        _prec *fields[1] = {T->yz};
        char *fields_str[1] = {"yz"};
       
        int err = 0;

        // Only check the rank in the middle because the ranks on the boundary
        // will not correctly compute the stencil due to applying an interior
        // stencil on the boundary
        if (T->rank != 1) {
                return err;
        }

        for (int i = 0; i < 1; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[13] = {0.0, 2.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  Tt->out_shift[0], Tt->out_shift[1], 
                                  Tt->out_shift[2],
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_fcn(fcn_poly, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_diffquadz(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 0, 0,
                          0, 1, 0,
                          0, 0, 0};

        _prec *fields[1] = {Tt->out};
        char *fields_str[1] = {"xz"};
       
        int err = 0;

        for (int i = 0; i < 1; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[13] = {0.0, 0.0, 2.0,
                                  0, 0.0, 1.0,
                                  Tt->out_shift[0], Tt->out_shift[1], 
                                  Tt->out_shift[2],
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_fcn(fcn_polybndz, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_velconst(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {1, 1, 1,
                          1, 1, 1,
                          1, 1, 1};

        _prec *fields[3] = {T->u1, T->v1, T->w1};
        char *fields_str[3] = {"u1", "v1", "w1"};
       
        int err = 0;

        if (T->rank != 1) {
                return err;
        }

        for (int i = 0; i < 3; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[13] = {0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0,
                                  Tt->out_shift[0], Tt->out_shift[1], 
                                  Tt->out_shift[2],
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_fcn(fcn_poly, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_vellinx(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 0, 0,
                          0, 1, 0,
                          0, 0, 0};

        _prec *fields[3] = {T->u1, T->v1, T->w1};
        xyz shift[3] = {
                {.x = T->su1[0], .y = T->su1[1], .z = T->su1[2]},
                {.x = T->sv1[0], .y = T->sv1[1], .z = T->sv1[2]},
                {.x = T->sw1[0], .y = T->sw1[1], .z = T->sw1[2]}
        };
        char *fields_str[3] = {"u1", "v1", "w1"};
       
        int err = 0;

        if (T->rank != 1) {
                return err;
        }

        for (int i = 0; i < 3; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[13] = {T->dth, 0.0, 0.0,
                                  0.0, 0.0, 0.0,
                                  shift[i].x, shift[i].y, shift[i].z,
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_fcn(fcn_poly, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_velliny(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 0, 0,
                          0, 1, 0,
                          0, 0, 0};

        _prec *fields[3] = {T->u1, T->v1, T->w1};
        xyz shift[3] = {
                {.x = T->su1[0], .y = T->su1[1], .z = T->su1[2]},
                {.x = T->sv1[0], .y = T->sv1[1], .z = T->sv1[2]},
                {.x = T->sw1[0], .y = T->sw1[1], .z = T->sw1[2]}
        };
        char *fields_str[3] = {"u1", "v1", "w1"};
       
        int err = 0;

        if (T->rank != 1) {
                return err;
        }

        for (int i = 0; i < 3; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[13] = {0.0, T->dth, 0.0,
                                  0.0, 0.0, 0.0,
                                  shift[i].x, shift[i].y, shift[i].z,
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_fcn(fcn_poly, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_vellinz(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 0, 0,
                          0, 1, 0,
                          0, 0, 0};

        _prec *fields[3] = {T->u1, T->v1, T->w1};
        xyz shift[3] = {
                {.x = T->su1[0], .y = T->su1[1], .z = T->su1[2]},
                {.x = T->sv1[0], .y = T->sv1[1], .z = T->sv1[2]},
                {.x = T->sw1[0], .y = T->sw1[1], .z = T->sw1[2]}
        };
        char *fields_str[3] = {"u1", "v1", "w1"};
       
        int err = 0;

        if (T->rank != 1) {
                return err;
        }

        for (int i = 0; i < 3; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[13] = {0.0, 0.0, T->dth,
                                  0.0, 0.0, 0.0,
                                  shift[i].x, shift[i].y, shift[i].z,
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_fcn(fcn_polybndz, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_velquadx(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 0, 0,
                          0, 1, 0,
                          0, 0, 0};

        _prec *fields[3] = {T->u1, T->v1, T->w1};
        xyz shift[3] = {
                {.x = T->su1[0], .y = T->su1[1], .z = T->su1[2]},
                {.x = T->sv1[0], .y = T->sv1[1], .z = T->sv1[2]},
                {.x = T->sw1[0], .y = T->sw1[1], .z = T->sw1[2]}
        };
        char *fields_str[3] = {"u1", "v1", "w1"};
       
        int err = 0;

        if (T->rank != 1) {
                return err;
        }

        for (int i = 0; i < 3; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[13] = {2*T->dth, 0.0, 0.0,
                                  1.0, 0.0, 0.0,
                                  shift[i].x, shift[i].y, shift[i].z,
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_fcn(fcn_poly, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_velquady(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 0, 0,
                          0, 1, 0,
                          0, 0, 0};

        _prec *fields[3] = {T->u1, T->v1, T->w1};
        xyz shift[3] = {
                {.x = T->su1[0], .y = T->su1[1], .z = T->su1[2]},
                {.x = T->sv1[0], .y = T->sv1[1], .z = T->sv1[2]},
                {.x = T->sw1[0], .y = T->sw1[1], .z = T->sw1[2]}
        };
        char *fields_str[3] = {"u1", "v1", "w1"};
       
        int err = 0;

        if (T->rank != 1) {
                return err;
        }

        for (int i = 0; i < 3; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[13] = {0.0, 2*T->dth, 0.0,
                                  0.0, 1.0, 0.0,
                                  shift[i].x, shift[i].y, shift[i].z,
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_fcn(fcn_poly, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_velquadz(const topo_test_t *Tt, const topo_t *T)
{
        int regions[9] = {0, 0, 0,
                          0, 1, 0,
                          0, 0, 0};

        _prec *fields[3] = {T->u1, T->v1, T->w1};
        xyz shift[3] = {
                {.x = T->su1[0], .y = T->su1[1], .z = T->su1[2]},
                {.x = T->sv1[0], .y = T->sv1[1], .z = T->sv1[2]},
                {.x = T->sw1[0], .y = T->sw1[1], .z = T->sw1[2]}
        };
        char *fields_str[3] = {"u1", "v1", "w1"};
       
        int err = 0;

        if (T->rank != 1) {
                return err;
        }

        for (int i = 0; i < 3; ++i) {
                _prec ferr[9] = {0.0};
                _prec args[13] = {0.0, 0.0, 2*T->dth,
                                  0.0, 0.0, 1.0,
                                  shift[i].x, shift[i].y, shift[i].z,
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_fcn(fcn_polybndz, T, fields[i], Tt->tol,
                                     args, regions, ferr);
                check_printerr(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}


int topo_test_velfrontback(const topo_test_t *Tt, const topo_t *T)
{
        int regions[25] = {0, 1, 1, 1, 0, 
                           0, 0, 0, 0, 0, 
                           0, 0, 0, 0, 0, 
                           0, 0, 0, 0, 0, 
                           0, 1, 1, 1, 0};

        _prec *fields[3] = {T->u1, T->v1, T->w1};
        xyz shift[3] = {
                {.x = T->su1[0], .y = T->su1[1], .z = T->su1[2]},
                {.x = T->sv1[0], .y = T->sv1[1], .z = T->sv1[2]},
                {.x = T->sw1[0], .y = T->sw1[1], .z = T->sw1[2]}
        };
        char *fields_str[3] = {"u1", "v1", "w1"};
       
        int err = 0;

        if (T->rank != 1) {
                return err;
        }

        for (int i = 0; i < 3; ++i) {
                _prec ferr[25] = {0.0};
                _prec args[13] = {0.0, 2*T->dth, 0.0,
                                  0.0, 1.0, 0.0,
                                  shift[i].x, shift[i].y, shift[i].z,
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_velocity_fcn(
                    fcn_poly, check_flinferr, T, fields[i], Tt->tol,
                    args, regions, ferr);
                check_printerr55(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_strconst(const topo_test_t *Tt, const topo_t *T)
{
        int regions[15] = {1, 1, 1, 1, 1, 
                           1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1};

        _prec *fields[6] = {T->xx, T->yy, T->zz, T->xy, T->xz, T->yz};
        xyz shift[6] = {
                {.x = T->sxx[0], .y = T->sxx[1], .z = T->sxx[2]},
                {.x = T->syy[0], .y = T->syy[1], .z = T->syy[2]},
                {.x = T->szz[0], .y = T->szz[1], .z = T->szz[2]},
                {.x = T->sxy[0], .y = T->sxy[1], .z = T->sxy[2]},
                {.x = T->sxz[0], .y = T->sxz[1], .z = T->sxz[2]},
                {.x = T->syz[0], .y = T->syz[1], .z = T->syz[2]}
        };
        char *fields_str[6] = {"xx", "yy", "zz", "xy", "xz", "yz"};
       
        int err = 0;

        if (T->rank != 1) {
                return err;
        }

        for (int i = 0; i < 6; ++i) {
                _prec ferr[15] = {0.0};
                _prec args[13] = {0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0,
                                  shift[i].x, shift[i].y, shift[i].z,
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_stress_fcn(fcn_poly, check_fl1err, T, fields[i],
                                            Tt->tol, args, regions, ferr);
                check_printerr53(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_strlinx(const topo_test_t *Tt, const topo_t *T)
{
        int regions[15] = {1, 1, 1, 1, 1, 
                           1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1};

        _prec *fields[6] = {T->xx, T->yy, T->zz, T->xy, T->xz, T->yz};
        xyz shift[6] = {
                {.x = T->sxx[0], .y = T->sxx[1], .z = T->sxx[2]},
                {.x = T->syy[0], .y = T->syy[1], .z = T->syy[2]},
                {.x = T->szz[0], .y = T->szz[1], .z = T->szz[2]},
                {.x = T->sxy[0], .y = T->sxy[1], .z = T->sxy[2]},
                {.x = T->sxz[0], .y = T->sxz[1], .z = T->sxz[2]},
                {.x = T->syz[0], .y = T->syz[1], .z = T->syz[2]}
        };
        char *fields_str[6] = {"xx", "yy", "zz", "xy", "xz", "yz"};
       
        int err = 0;

        if (T->rank != 1) {
                return err;
        }

        _prec err_coef[6] = {3*T->dth, T->dth, T->dth, T->dth, T->dth, 0};
        for (int i = 0; i < 6; ++i) {
                _prec ferr[15] = {0.0};
                _prec args[13] = {err_coef[i], 0.0, 0.0,
                                  0.0, 0.0, 0.0,
                                  shift[i].x, shift[i].y, shift[i].z,
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |=
                    topo_test_stress_fcn(fcn_poly, check_flinferr, T, fields[i],
                                         Tt->tol, args, regions, ferr);
                check_printerr53(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_strliny(const topo_test_t *Tt, const topo_t *T)
{
        int regions[15] = {1, 1, 1, 1, 1, 
                           1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1};

        _prec *fields[6] = {T->xx, T->yy, T->zz, T->xy, T->xz, T->yz};
        xyz shift[6] = {
                {.x = T->sxx[0], .y = T->sxx[1], .z = T->sxx[2]},
                {.x = T->syy[0], .y = T->syy[1], .z = T->syy[2]},
                {.x = T->szz[0], .y = T->szz[1], .z = T->szz[2]},
                {.x = T->sxy[0], .y = T->sxy[1], .z = T->sxy[2]},
                {.x = T->sxz[0], .y = T->sxz[1], .z = T->sxz[2]},
                {.x = T->syz[0], .y = T->syz[1], .z = T->syz[2]}
        };
        char *fields_str[6] = {"xx", "yy", "zz", "xy", "xz", "yz"};
       
        int err = 0;

        if (T->rank != 1) {
                return err;
        }

        _prec err_coef[6] = {T->dth, 3 * T->dth, T->dth, T->dth, 0, T->dth};
        for (int i = 0; i < 6; ++i) {
                _prec ferr[15] = {0.0};
                _prec args[13] = {0.0, err_coef[i], 0.0,
                                  0.0, 0.0, 0.0,
                                  shift[i].x, shift[i].y, shift[i].z,
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |=
                    topo_test_stress_fcn(fcn_poly, check_flinferr, T, fields[i],
                                         Tt->tol, args, regions, ferr);
                check_printerr53(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_strlinz(const topo_test_t *Tt, const topo_t *T)
{
        int regions[15] = {1, 1, 1, 1, 1, 
                           1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1};

        _prec *fields[6] = {T->xx, T->yy, T->zz, T->xy, T->xz, T->yz};
        xyz shift[6] = {
                {.x = T->sxx[0], .y = T->sxx[1], .z = T->sxx[2]},
                {.x = T->syy[0], .y = T->syy[1], .z = T->syy[2]},
                {.x = T->szz[0], .y = T->szz[1], .z = T->szz[2]},
                {.x = T->sxy[0], .y = T->sxy[1], .z = T->sxy[2]},
                {.x = T->sxz[0], .y = T->sxz[1], .z = T->sxz[2]},
                {.x = T->syz[0], .y = T->syz[1], .z = T->syz[2]}
        };
        char *fields_str[6] = {"xx", "yy", "zz", "xy", "xz", "yz"};
       
        int err = 0;

        if (T->rank != 1) {
                return err;
        }

        _prec err_coef[6] = {T->dth, T->dth, 3 * T->dth, 0, T->dth, T->dth};
        for (int i = 0; i < 6; ++i) {
                _prec ferr[15] = {0.0};
                _prec args[13] = {0.0, 0.0, err_coef[i],
                                  0.0, 0.0, 0.0,
                                  shift[i].x, shift[i].y, shift[i].z,
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_stress_fcn(fcn_polybndz, check_flinferr, T,
                                            fields[i], Tt->tol, args, regions,
                                            ferr);
                check_printerr53(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_strquadx(const topo_test_t *Tt, const topo_t *T)
{
        int regions[15] = {1, 1, 1, 1, 1, 
                           1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1};

        _prec *fields[6] = {T->xx, T->yy, T->zz, T->xy, T->xz, T->yz};
        xyz shift[6] = {
                {.x = T->sxx[0], .y = T->sxx[1], .z = T->sxx[2]},
                {.x = T->syy[0], .y = T->syy[1], .z = T->syy[2]},
                {.x = T->szz[0], .y = T->szz[1], .z = T->szz[2]},
                {.x = T->sxy[0], .y = T->sxy[1], .z = T->sxy[2]},
                {.x = T->sxz[0], .y = T->sxz[1], .z = T->sxz[2]},
                {.x = T->syz[0], .y = T->syz[1], .z = T->syz[2]}
        };
        char *fields_str[6] = {"xx", "yy", "zz", "xy", "xz", "yz"};
       
        int err = 0;

        if (T->rank != 1) {
                return err;
        }

        _prec err_coef[6] = {3*T->dth, T->dth, T->dth, T->dth, T->dth, 0};
        for (int i = 0; i < 6; ++i) {
                _prec ferr[15] = {0.0};
                _prec args[13] = {2 * err_coef[i], 0.0, 0.0,
                                  1.0, 0.0, 0.0,
                                  shift[i].x, shift[i].y, shift[i].z,
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |=
                    topo_test_stress_fcn(fcn_poly, check_flinferr, T, fields[i],
                                         Tt->tol, args, regions, ferr);
                check_printerr53(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_strquady(const topo_test_t *Tt, const topo_t *T)
{
        int regions[15] = {1, 1, 1, 1, 1, 
                           1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1};

        _prec *fields[6] = {T->xx, T->yy, T->zz, T->xy, T->xz, T->yz};
        xyz shift[6] = {
                {.x = T->sxx[0], .y = T->sxx[1], .z = T->sxx[2]},
                {.x = T->syy[0], .y = T->syy[1], .z = T->syy[2]},
                {.x = T->szz[0], .y = T->szz[1], .z = T->szz[2]},
                {.x = T->sxy[0], .y = T->sxy[1], .z = T->sxy[2]},
                {.x = T->sxz[0], .y = T->sxz[1], .z = T->sxz[2]},
                {.x = T->syz[0], .y = T->syz[1], .z = T->syz[2]}
        };
        char *fields_str[6] = {"xx", "yy", "zz", "xy", "xz", "yz"};
       
        int err = 0;

        if (T->rank != 1) {
                return err;
        }

        _prec err_coef[6] = {T->dth, 3 * T->dth, T->dth, T->dth, 0, T->dth};
        for (int i = 0; i < 6; ++i) {
                _prec ferr[15] = {0.0};
                _prec args[13] = {0.0, 2 * err_coef[i], 0.0,
                                  0.0, 1.0, 0.0,
                                  shift[i].x, shift[i].y, shift[i].z,
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |=
                    topo_test_stress_fcn(fcn_poly, check_flinferr, T, fields[i],
                                         Tt->tol, args, regions, ferr);
                check_printerr53(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}

int topo_test_strquadz(const topo_test_t *Tt, const topo_t *T)
{
        int regions[15] = {1, 1, 1, 1, 1, 
                           1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1};

        _prec *fields[6] = {T->xx, T->yy, T->zz, T->xy, T->xz, T->yz};
        xyz shift[6] = {
                {.x = T->sxx[0], .y = T->sxx[1], .z = T->sxx[2]},
                {.x = T->syy[0], .y = T->syy[1], .z = T->syy[2]},
                {.x = T->szz[0], .y = T->szz[1], .z = T->szz[2]},
                {.x = T->sxy[0], .y = T->sxy[1], .z = T->sxy[2]},
                {.x = T->sxz[0], .y = T->sxz[1], .z = T->sxz[2]},
                {.x = T->syz[0], .y = T->syz[1], .z = T->syz[2]}
        };
        char *fields_str[6] = {"xx", "yy", "zz", "xy", "xz", "yz"};
       
        int err = 0;

        if (T->rank != 1) {
                return err;
        }

        _prec err_coef[6] = {T->dth, T->dth, 3 * T->dth, 0, T->dth, T->dth};
        for (int i = 0; i < 6; ++i) {
                _prec ferr[15] = {0.0};
                _prec args[13] = {0.0, 0.0, 2 * err_coef[i],
                                  0.0, 0.0, 1.0,
                                  shift[i].x, shift[i].y, shift[i].z,
                                  T->coord[0], T->coord[1],
                                  T->nx, T->ny};
                err |= topo_test_stress_fcn(fcn_polybndz, check_flinferr, T,
                                            fields[i], Tt->tol, args, regions,
                                            ferr);
                check_printerr53(__func__, T->rank, fields_str[i], ferr);
        }

        return err;
}


int topo_test_fcn(fcnp fp, const topo_t *T, const _prec *dres, const _prec tol,
                  const _prec *args, const int *regions, _prec *ferr)
{
        int size = sizeof(_prec)*T->gridsize;
        _prec *res = malloc(size);
        _prec *ans = malloc(size);
        cudaMemcpy(res, dres, size, cudaMemcpyDeviceToHost);

        int err = 0;

        // Apply function everywhere (excluding alignment space and bottom
        // in z-direction region)
        fp(ans, 
           T->off_x[0], T->off_x[3],
           T->off_y[0], T->off_y[3],
           T->off_z[1], T->off_z[2],
           T->line, T->slice, 
           args);
        err = check_all(check_fl1err, res, ans, 
                        T->off_x, T->off_y, T->off_z, 3, 3,
                        T->line, T->slice, 
                        tol, regions, ferr);

        free(res);
        free(ans);

        return err;
}

int topo_test_stress_fcn(fcnp fp, check_fun check_fp,
                         const topo_t *T, const _prec *dres,
                         const _prec tol, const _prec *args, const int *regions,
                         _prec *ferr) 
{
        int size = sizeof(_prec)*T->gridsize;
        _prec *res = malloc(size);
        _prec *ans = malloc(size);
        cudaMemcpy(res, dres, size, cudaMemcpyDeviceToHost);

        int err = 0;

        fp(ans, 
           T->stress_offset_x[1], T->stress_offset_x[4],
           T->stress_offset_y[1], T->stress_offset_y[2],
           T->off_z[1], T->off_z[2],
           T->line, T->slice, 
           args);

        err = check_all(check_fp, res, ans, 
                        T->stress_offset_x, T->stress_offset_y, T->off_z, 5, 3,
                        T->line, T->slice, 
                        tol, regions, ferr);

        free(res);
        free(ans);

        return err;
}

int topo_test_velocity_fcn(fcnp fp, check_fun check_fp, const topo_t *T,
                           const _prec *dres, const _prec tol,
                           const _prec *args, const int *regions, _prec *ferr) 
{
        int size = sizeof(_prec)*T->gridsize;
        _prec *res = malloc(size);
        _prec *ans = malloc(size);
        cudaMemcpy(res, dres, size, cudaMemcpyDeviceToHost);

        int err = 0;

        fp(ans, 
           T->off_x[0], T->off_x[3],
           T->off_y[0], T->off_y[3],
           T->off_z[1], T->off_z[2],
           T->line, T->slice, 
           args);

        err = check_all(check_fp, res, ans, 
                        T->velocity_offset_x, T->velocity_offset_y, 
                        T->off_z, 5, 5,
                        T->line, T->slice, 
                        tol, regions, ferr);

        free(res);
        free(ans);

        return err;
}
