#ifndef TOPOGRAPHY_TEST_H
#define TOPOGRAPHY_TEST_H
/* 
 *  This module is used to test the module `topography`.
 *  While this module provides a number of tests (see below), only one test can
 *  run at a time. A test that fails will return an error code `err`. 
 *  If `err > 0` then the test has failed.
 *
 *  The testing procedure works in the following three steps:
 *  1. The test is initialized, and the particular chosen test is configured.
 *  This action takes place when calling `test_topo_init()`, and this function
 *  is called before entering the time stepping loop.
 *  For example, `TOPO_TEST_CONSTX` enables a test that checks if a constant
 *  function is correctly handled. The initialization step calls a CUDA kernel
 *  that will initialize one or more of the device arrays `u1, v1, w1, etc`. In
 *  this case, the kernel launch function is named `topo_test_const_H`.
 *
 *  2. Outside code then runs and modifies the device arrays as desired. To
 *  mimic the behavior of the velocity communication and computation, the
 *  functions `test_velfront`, `test_velback` are called at the time when the
 *  compute kernels for the front and back parts of the velocity field are
 *  called. Similarly, `test_velx` is called when the kernel for the interior
 *  computation of the velocity field takes place.
 *
 *  3. The test is finalized by copying the device data to host and comparing
 *  the result to a priori known answer. The l-2 norm is used for the
 *  comparison. This action takes place when calling `test_topo_finalize` and is
 *  executed after the time stepping loop has completed.
 *
 */ 


// TOPO_TEST: Enable testing
#ifndef TOPO_TEST
#define TOPO_TEST 0
#endif 

/*
 * Tests to run (choose only one).
 * These tests can be activated by specifying the flag `-D` and
 * variable name. For example,`-DTOPO_TEST_CONSTX=1` will run the test
 * `TOPO_TEST_CONSTX` (see below for available tests). 
 *
 * Description:
 *      TOPO_TEST_CONSTX: Check that a constant function is handled correctly
 *      with communication in the x-direction.
 *
 *      TOPO_TEST_CONSTY: Check that a constant function is handled correctly
 *      with communication in the y-direction.
 *
 *      TOPO_TEST_LINX: Check that a linear function is handled correctly
 *      with communication in the x-direction.
 *
 *      TOPO_TEST_DIFFCONSTX: Differentiate a constant function in the
 *      x-direction on the device and check that the zero function is produced
 *      on the host.
 *
 *      TOPO_TEST_DIFFCONSTY: Differentiate a constant function in the
 *      y-direction on the device and check that the zero function is produced
 *      on the host.
 *
 *      TOPO_TEST_DIFFCONSTZ: Differentiate a constant function in the
 *      z-direction on the device and check that the zero function is produced
 *      on the host. This test includes stencils for the top boundary.
 *
 *      TOPO_TEST_DIFFLINX: Differentiate a linear function in the x-direction
 *      on the device and check that the correct constant function is produced
 *      on the host.
 *
 *      TOPO_TEST_DIFFLINY: Differentiate a linear function in the y-direction
 *      on the device and check that the correct constant function is produced
 *      on the host.
 *
 *      TOPO_TEST_DIFFLINZ: Differentiate a linear function in the
 *      z-direction on the device and check that the correct constant function
 *      is produced on the host. This test includes stencils for the top
 *      boundary.
 *
 *      TOPO_TEST_DIFFQUADX: Differentiate a quadratic function in the
 *      x-direction on the device and check that the correct linear function is
 *      produced on the host.
 *
 *      TOPO_TEST_DIFFQUADY: Differentiate a quadratic function in the
 *      y-direction on the device and check that the correct linear function is
 *      produced on the host.
 *
 *      TOPO_TEST_DIFFQUADZ: Differentiate a quadratic function in the
 *      y-direction on the device and check that the correct linear function is
 *      produced on the host.
 *
 *      TOPO_TEST_VELCONST: Test the topography velocity kernel by using a
 *      constant function. 
 *
 *      TOPO_TEST_VELLINX: Test the topography velocity kernel by using a linear
 *      function in the x-direction. 
 *
 *      TOPO_TEST_VELLINY: Test the topography velocity kernel by using a linear
 *      function in the y-direction. 
 *
 *      TOPO_TEST_VELLINZ: Test the topography velocity kernel by using a linear
 *      function in the z-direction. 
 *
 *      TOPO_TEST_VELQUADX: Test the topography velocity kernel by using a
 *      quadratic function in the x-direction. 
 *
 *      TOPO_TEST_VELQUADY: Test the topography velocity kernel by using a
 *      quadratic function in the y-direction. 
 *
 *      TOPO_TEST_VELQUADZ: Test the topography velocity kernel by using a
 *      quadratic function in the z-direction. 
 *
 *      TOPO_TEST_STRCONST: Test the topography stress kernel by using a
 *      constant function. 
 * 
 *      TOPO_TEST_STRLINX: Test the topography stress kernel by using a linear
 *      function in the x-direction. 
 * 
 *      TOPO_TEST_STRLINY: Test the topography stress kernel by using a linear
 *      function in the y-direction. 
 *
 *      TOPO_TEST_STRLINZ: Test the topography stress kernel by using a linear
 *      function in the z-direction. 
 *
 *      TOPO_TEST_STRQUADX: Test the topography stress kernel by using a
 *      quadratic function in the x-direction. 
 *
 *      TOPO_TEST_STRQUADY: Test the topography stress kernel by using a
 *      quadratic function in the y-direction. 
 *
 *      TOPO_TEST_STRQUADZ: Test the topography stress kernel by using a
 *      quadratic function in the z-direction. 
 *
 *      TOPO_TEST_VELFRONTBACK: Test the topography front and back velocity
 *      kernels by using a quadratic function in the y-direction. 
 */
#ifndef TOPO_TEST_CONSTX
#define TOPO_TEST_CONSTX 0
#endif

#ifndef TOPO_TEST_CONSTY
#define TOPO_TEST_CONSTY 0
#endif

#ifndef TOPO_TEST_LINX
#define TOPO_TEST_LINX 0
#endif

#ifndef TOPO_TEST_DIFFCONSTX
#define TOPO_TEST_DIFFCONSTX 0
#endif

#ifndef TOPO_TEST_DIFFCONSTY
#define TOPO_TEST_DIFFCONSTY 0
#endif

#ifndef TOPO_TEST_DIFFCONSTZ
#define TOPO_TEST_DIFFCONSTZ 0
#endif

#ifndef TOPO_TEST_DIFFLINX
#define TOPO_TEST_DIFFLINX 0
#endif

#ifndef TOPO_TEST_DIFFLINY
#define TOPO_TEST_DIFFLINY 0
#endif

#ifndef TOPO_TEST_DIFFLINZ
#define TOPO_TEST_DIFFLINZ 0
#endif

#ifndef TOPO_TEST_DIFFQUADX
#define TOPO_TEST_DIFFQUADX 0
#endif

#ifndef TOPO_TEST_DIFFQUADY
#define TOPO_TEST_DIFFQUADY 0
#endif

#ifndef TOPO_TEST_DIFFQUADZ
#define TOPO_TEST_DIFFQUADZ 0
#endif

#ifndef TOPO_TEST_VELCONST
#define TOPO_TEST_VELCONST 0
#endif

#ifndef TOPO_TEST_VELLINX
#define TOPO_TEST_VELLINX 0
#endif

#ifndef TOPO_TEST_VELLINY
#define TOPO_TEST_VELLINY 0
#endif

#ifndef TOPO_TEST_VELLINZ
#define TOPO_TEST_VELLINZ 0
#endif

#ifndef TOPO_TEST_VELQUADX
#define TOPO_TEST_VELQUADX 0
#endif

#ifndef TOPO_TEST_VELQUADY
#define TOPO_TEST_VELQUADY 0
#endif

#ifndef TOPO_TEST_VELQUADZ
#define TOPO_TEST_VELQUADZ 0
#endif

#ifndef TOPO_TEST_VELFRONTBACK
#define TOPO_TEST_VELFRONTBACK 0
#endif

#ifndef TOPO_TEST_STRCONST
#define TOPO_TEST_STRCONST 0
#endif

#ifndef TOPO_TEST_STRLINX
#define TOPO_TEST_STRLINX 0
#endif

#ifndef TOPO_TEST_STRLINY
#define TOPO_TEST_STRLINY 0
#endif

#ifndef TOPO_TEST_STRLINZ
#define TOPO_TEST_STRLINZ 0
#endif

#ifndef TOPO_TEST_STRQUADX
#define TOPO_TEST_STRQUADX 0
#endif

#ifndef TOPO_TEST_STRQUADY
#define TOPO_TEST_STRQUADY 0
#endif

#ifndef TOPO_TEST_STRQUADZ
#define TOPO_TEST_STRQUADZ 0
#endif

#define TOPO_TEST_TOLERANCE 1e-6

#include "functions.h"
#include "grid_check.h"
#include "topography.h"

typedef struct {
        int x;
        int y;
        int z;
} xyz;

typedef struct {
        int use;
        _prec tol;
        _prec coef[3];
        _prec deg[3];
        _prec *out;
        _prec *velf;
        _prec *velb;
        _prec *in;
        int out_shift[3];
        int in_shift[3];
        _prec cxx[3];
        _prec cyy[3];
        _prec czz[3];
        _prec cxy[3];
        _prec cxz[3];
        _prec cyz[3];
        _prec cu1[3];
        _prec cv1[3];
        _prec cw1[3];
} topo_test_t;

topo_test_t topo_test_init(topo_t *T);
void topo_test_velfront(topo_test_t *Tt, topo_t *T);
void topo_test_velback(topo_test_t *Tt, topo_t *T);
void topo_test_velx(const topo_test_t *Tt, topo_t *T);
void topo_test_stress(const topo_test_t *Tt, topo_t *T);
void topo_test_stress_interior(const topo_test_t *Tt, topo_t *T);
void topo_test_stress_sides(const topo_test_t *Tt, topo_t *T);
int topo_test_finalize(const topo_test_t *Tt, topo_t *T);

// Tests
int topo_test_constx(const topo_test_t *Tt, const topo_t *T);
int topo_test_consty(const topo_test_t *Tt, const topo_t *T);
int topo_test_linx(const topo_test_t *Tt, const topo_t *T);
int topo_test_liny(const topo_test_t *Tt, const topo_t *T);
int topo_test_diffconstx(const topo_test_t *Tt, const topo_t *T);
int topo_test_diffconsty(const topo_test_t *Tt, const topo_t *T);
int topo_test_diffconstz(const topo_test_t *Tt, const topo_t *T);
int topo_test_difflinx(const topo_test_t *Tt, const topo_t *T);
int topo_test_diffliny(const topo_test_t *Tt, const topo_t *T);
int topo_test_difflinz(const topo_test_t *Tt, const topo_t *T);
int topo_test_diffquadx(const topo_test_t *Tt, const topo_t *T);
int topo_test_diffquady(const topo_test_t *Tt, const topo_t *T);
int topo_test_diffquadz(const topo_test_t *Tt, const topo_t *T);
int topo_test_velconst(const topo_test_t *Tt, const topo_t *T);
int topo_test_vellinx(const topo_test_t *Tt, const topo_t *T);
int topo_test_velliny(const topo_test_t *Tt, const topo_t *T);
int topo_test_vellinz(const topo_test_t *Tt, const topo_t *T);
int topo_test_velquadx(const topo_test_t *Tt, const topo_t *T);
int topo_test_velquady(const topo_test_t *Tt, const topo_t *T);
int topo_test_velquadz(const topo_test_t *Tt, const topo_t *T);
int topo_test_velfrontback(const topo_test_t *Tt, const topo_t *T);
int topo_test_strconst(const topo_test_t *Tt, const topo_t *T);
int topo_test_strlinx(const topo_test_t *Tt, const topo_t *T);
int topo_test_strliny(const topo_test_t *Tt, const topo_t *T);
int topo_test_strlinz(const topo_test_t *Tt, const topo_t *T);
int topo_test_strquadx(const topo_test_t *Tt, const topo_t *T);
int topo_test_strquady(const topo_test_t *Tt, const topo_t *T);
int topo_test_strquadz(const topo_test_t *Tt, const topo_t *T);

// Test helper functions
int topo_test_fcn(fcnp fp, const topo_t *T, const _prec *dres, const _prec tol,
                  const _prec *args, const int *regions, _prec *ferr);

int topo_test_stress_fcn(fcnp fp, check_fun check_fp,
                         const topo_t *T, const _prec *dres,
                         const _prec tol, 
                         const _prec *args,
                         const int *regions, _prec *ferr);

int topo_test_velocity_fcn(fcnp fp, check_fun check_fp,
                         const topo_t *T, const _prec *dres,
                         const _prec tol, 
                         const _prec *args,
                         const int *regions, _prec *ferr);

#endif
