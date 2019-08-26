"""usage: python3 scheme.py <debug> <use_2d>
Build scheme

debug   - Run in debug mode. Exports kernels in double precision and without
          enforcing boundary conditions. Disabled by default.

use_2d  - Generate 2D kernels. If 'xz', the y-direction is ignored. If 'yz', the
          x-direction is ignored.

use_acoustic  - Set the shear modulus to zero.

use_cartesian - Disable coordinate transform.

"""

import sys
import sympy as sp
import openfd as fd
from openfd import Expr
import numpy as np
import helper
from openfd.dev import kernelgenerator as kg

if len(sys.argv) < 6:
    print(__doc__)
    exit(0)
else:
    filename = sys.argv[1]
    prec_str = sys.argv[2]
    debug = int(sys.argv[3])
    use_2d = str(sys.argv[4])
    use_acoustic = int(sys.argv[5])
    use_cartesian = int(sys.argv[6])
    use_cubic_interpolation = int(sys.argv[7])

helper.set_precision(prec_str)
use_sponge = helper.get_use_sponge_layer(debug)
use_free_surface_bc = helper.get_use_free_surface_bc(debug)
print("Precision:", prec_str, "\n",
      "Debug:", debug, "\n",
      "Sponge layer:", use_sponge, "\n",
      "Use Cartesian version:", use_cartesian, "\n",
      "Use Acoustic version:", use_acoustic, "\n",
      "Use cubic interpolation of material parameters:", 
       use_cubic_interpolation, "\n",
      "Restrict to 2D:", use_2d, "\n",
      "Apply free surface boundary condition:", use_free_surface_bc)

def velocity(label, buf=0, debug=0, debug_ops=0, use_cartesian=0):
    """
    Generate velocity update kernels.

    Interpolate metrics to certain staggered grid positions.

    Note that f(x1,x2) is 2d, and g(x3) is 1d. Hence, interpolation is needed in
    at most two directions

    1  : Interpolate in the x1-direction
    2  : Interpolate in the x2-direction
    12 : Interpolate in both the x1 and x2-directions
    3 : Interpolate in the x3-direction 

    Arguments:
        label : Name to give the generated kernel function
        buf : Set to true if the velocity output arrays should be buffers
        debug : Enable debugging. In debug mode, the free surface boundary
            condition is not enforced so that the consistency of all operators
            can easily be checked.
        debug_ops : Only enables the interpolation operators

    """
    from helper import D, P, Pavg
    from variables import F, mem, size
    
    if use_cubic_interpolation:
        Pavg = P

    G = helper.shifts()

    f = F.f
    g = F.g

    if buf:
        u1 = F.buf_u1
        u2 = F.buf_u2
        u3 = F.buf_u3
    else:
        u1 = F.u1
        u2 = F.u2
        u3 = F.u3


    bnd_right = use_free_surface_bc

    print("Generating velocity kernels: %s. "%label)

    rho1 = fd.Variable('rho1', dtype=fd.prec, val=Pavg(Pavg(F.rho, 'y', 1), 'z', 1)) 
    rho2 = fd.Variable('rho2', dtype=fd.prec, val=Pavg(Pavg(F.rho, 'x', 1), 'z', 1)) 
    rho3 = fd.Variable('rho3', dtype=fd.prec, val=Pavg(Pavg(F.rho, 'x', 1), 'y', 1)) 

    # Jacobians
    J1 = F.f_1 * F.g3_c
    J2 = F.f_2 * F.g3_c
    J3 = F.f_c * F.g3

    J11 = F.f_c * F.g3_c
    J22 = F.f_c * F.g3_c

    J12 = F.f * F.g3_c
    J13 = F.f_1 * F.g3
    J23 = F.f_2 * F.g3

    # Derivatives of topography functions
    f1_1 = F.f1_1
    f1_2 = F.f1_2
    f1_3 = F.f1_c

    f2_1 = F.f2_1
    f2_2 = F.f2_2
    f2_3 = F.f2_c


    # This variable will contain (dt/h)/(J*rho)
    Ai1 = fd.Variable('Ai1', dtype=fd.prec, val=J1*rho1)
    Ai2 = fd.Variable('Ai2', dtype=fd.prec, val=J2*rho2)
    Ai3 = fd.Variable('Ai3', dtype=fd.prec, val=J3*rho3)

    # a: enable time stepping (FIXME: rename this parameter)
    # nu: CFL-like number: dt/h^3
    a, nu = sp.symbols('a nu')

    s11_x = D(J11 * F.s11, 'x', G.u1[0])
    s12_y = D(J12 * F.s12, 'y', G.u1[1])
    s13_z = D(F.s13, 'z', G.u1[2], bnd_right=bnd_right)

    s22_y = D(J22 * F.s22, 'y', G.u2[1])
    s12_x = D(J12 * F.s12, 'x', G.u2[0])
    s23_z = D(F.s23, 'z', G.u2[2], bnd_right=bnd_right)

    s13_x = D(J13 * F.s13, 'x', G.u3[0])
    s23_y = D(J23 * F.s23, 'y', G.u3[1])
    s33_z = D(F.s33, 'z', G.u3[2], bnd_right=bnd_right)

    w11_z = f1_1 * D(F.g_c * P(F.s11, 'x', G.u1[0]), 'z', G.u1[2],
                     interp="first", bnd_right=bnd_right)
    w12_z = f2_1 * D(F.g_c * P(F.s12, 'y', G.u1[1]), 'z', G.u1[2],
                     interp="first", bnd_right=bnd_right)

    w21_z = f1_2 * D(F.g_c * P(F.s12, 'x', G.u2[0]), 'z', G.u2[2],
                     interp="first", bnd_right=bnd_right)
    w22_z = f2_2 * D(F.g_c * P(F.s22, 'y', G.u2[1]), 'z', G.u2[2],
                     interp="first", bnd_right=bnd_right)

    w31_z = f1_3 * D(F.g * P(F.s13,'x', G.u3[0]), 'z', G.u3[2],
                     interp="first", bnd_right=bnd_right)         
    w32_z = f2_3 * D(F.g * P(F.s23,'y', G.u3[1]), 'z', G.u3[2],
                     interp="first", bnd_right=bnd_right)

    if use_2d == 'xz':
        y = 0
        x = 1
    elif use_2d == 'yz':
        y = 1
        x = 0
    else:
        x = 1
        y = 1

    # Discretization
    u1_t = a * F.u1 + Ai1 * (x * s11_x + y * s12_y + x * s13_z - x * w11_z - y * w12_z)
    u2_t = a * F.u2 + Ai2 * (x * s12_x + y * s22_y + y * s23_z - x * w21_z - y * w22_z)
    u3_t = a * F.u3 + Ai3 * (x * s13_x + y * s23_y +     s33_z - x * w31_z - y * w32_z)

    if debug_ops:
        u1_t = D(P(F.s11,'x', G.u1[0]), 'z', G.u1[2],
                 interp="first", bnd_right=bnd_right)
        u1_t += D(P(F.s12,'y', G.u1[1]), 'z', G.u1[2],
                 interp="first", bnd_right=bnd_right)
        u2_t = D(P(F.s12,'x', G.u2[0]), 'z', G.u2[2],
                 interp="first", bnd_right=bnd_right)
        u3_t = D(F.s13, 'z', G.u3[2],
                 interp="first", bnd_right=bnd_right)

    if use_cartesian:
        #s11_x = D(F.s11, 'x', G.u1[0])
        #s12_y = D(F.s12, 'y', G.u1[1])
        #s13_z = D(F.s13, 'z', G.u1[2], bnd_right=bnd_right)

        #s22_y = D(F.s22, 'y', G.u2[1])
        #s12_x = D(F.s12, 'x', G.u2[0])
        #s23_z = D(F.s23, 'z', G.u2[2], bnd_right=bnd_right)

        #s13_x = D(F.s13, 'x', G.u3[0])
        #s23_y = D(F.s23, 'y', G.u3[1])
        #s33_z = D(F.s33, 'z', G.u3[2], bnd_right=bnd_right)

        u1_t = a * F.u1 +     Ai1 * s11_x + y * Ai1 * s12_y +     Ai1 * s13_z
        u2_t = a * F.u2 + y * Ai2 * s22_y +     Ai2 * s12_x +     Ai2 * s23_z
        u3_t = a * F.u3 +     Ai3 * s33_z +     Ai3 * s13_x + y * Ai3 * s23_y   

    eqs = [] 
    eqs += [rho1, rho2, rho3]
    eqs += [Ai1, (Ai1.symbol, nu/Ai1.symbol)]
    eqs += [Ai2, (Ai2.symbol, nu/Ai2.symbol)]
    eqs += [Ai3, (Ai3.symbol, nu/Ai3.symbol)]


    if use_sponge:
        f_dcrj = apply_sponge_layer()
        eqs += [f_dcrj]
        eqs += [(u1, u1_t*f_dcrj)]
        eqs += [(u2, u2_t*f_dcrj)] 
        eqs += [(u3, u3_t*f_dcrj)]
    else:
        eqs += [(u1, u1_t)]
        eqs += [(u2, u2_t)] 
        eqs += [(u3, u3_t)]

    lhs, rhs = fd.equations(eqs)

    bounds = helper.velbounds(D(F.s11,'z'),
                              exclude_left=helper.get_exclude_left(debug))
    rj0 = sp.symbols('rj0')
    if buf:
        lhs_indices = lambda idx : (idx[0], idx[1], idx[2])
        rhs_indices = lambda idx : (idx[0], idx[1] + rj0, idx[2])
        index_bounds = (0,1,0)
    else:
        lhs_indices = None
        rhs_indices = None
        index_bounds = (1,1,0)
    grid_order = ['z', 'y', 'x']

    kernels = kg.make_kernel(label, 
                              lhs, rhs,
                              bounds, helper.gridsymbols,
                              regions=(1, 1, [0, 1, 2]),
                              debug=0, generator=helper.generator,
                              index_bounds=index_bounds,
                              extraout = [u1, u2, u3],
                              extraconst = [F.s11, F.s22, F.s33,
                                            F.s12, F.s13, F.s23, f, a,
                                            F.u1, F.u2, F.u3,
                                            F.rho,
                                            F.dcrjx, F.dcrjy, F.dcrjz,
                                            F.g,
                                            F.g3,
                                            F.g3_c,
                                            F.g_c,
                                            nu,
                                            F.f_1, F.f_2, F.f_c,
                                            f1_1, f1_2, f1_3,
                                            f2_1, f2_2, f2_3],
                              lhs_indices=lhs_indices, rhs_indices=rhs_indices,
                              grid_order=grid_order)
    return kernels

def stress(label, debug=0, debug_ops=0, use_cartesian=0):
    """
    Generate stress update kernels.
    """
    from helper import D, P, Pavg
    from variables import F, mem, size
    G = helper.shifts()

    if use_cubic_interpolation:
        Pavg = P

    print("Generating stress kernels: %s. "%label)

    a, nu = sp.symbols('a nu')
    rho1 = fd.Variable('rho1', dtype=fd.prec, val=Pavg(Pavg(F.rho, 'y', 1), 'z', 1)) 
    rho2 = fd.Variable('rho2', dtype=fd.prec, val=Pavg(Pavg(F.rho, 'x', 1), 'z', 1)) 
    rho3 = fd.Variable('rho3', dtype=fd.prec, val=Pavg(Pavg(F.rho, 'x', 1), 'y', 1)) 

    Jii = fd.Variable('Jii', dtype=fd.prec, val=F.f_c*F.g3_c)
    J12i = fd.Variable('J12i', dtype=fd.prec, val=F.f*F.g3_c)
    J13i = fd.Variable('J13i', dtype=fd.prec, val=F.f_1*F.g3)
    J23i = fd.Variable('J23i', dtype=fd.prec, val=F.f_2*F.g3)

    lam = fd.Variable('lam', dtype=fd.prec, 
                      val=nu/Pavg(
                              Pavg(
                                Pavg(F.lami, 'x', G.s11[0]), 
                                          'y', G.s11[1]),
                                          'z', G.s11[2]))

    twomu = fd.Variable('twomu', dtype=fd.prec, 
                     val=2*nu/Pavg(
                                Pavg(
                                  Pavg(F.mui, 'x', G.s11[0]), 
                                           'y', G.s11[1]),
                                           'z', G.s11[2]))

    mu12 = fd.Variable('mu12', dtype=fd.prec, val=nu/Pavg(F.mui,'z',G.s12[2]))
    mu13 = fd.Variable('mu13', dtype=fd.prec, val=nu/Pavg(F.mui,'y',G.s13[1]))
    mu23 = fd.Variable('mu23', dtype=fd.prec, val=nu/Pavg(F.mui,'x',G.s23[0]))

    vx_x = D(F.u1, 'x', G.s33[0])
    vy_y = D(F.u2, 'y', G.s33[1])    
    vz_z = D(F.u3, 'z', G.s33[2])    

    vx_y = D(F.u1, 'y', G.s12[1])    
    vx_z = D(F.u1, 'z', G.s13[2])    

    vy_x = D(F.u2, 'x', G.s12[0])    
    vy_z = D(F.u2, 'z', G.s23[2])    

    vz_x = D(F.u3, 'x', G.s13[0])    
    vz_y = D(F.u3, 'y', G.s23[1])    

    # Derivatives of topography functions
    f1_1 = F.f1_1
    f1_2 = F.f1_2
    f1_3 = F.f1_c

    f2_1 = F.f2_1
    f2_2 = F.f2_2
    f2_3 = F.f2_c

    if use_2d == 'xz':
        y = 0
        x = 1
    elif use_2d == 'yz':
        y = 1
        x = 0
    else:
        x = 1
        y = 1



    # Mixed terms due to curvilinear transformation
    wx_z = Jii * F.g_c * P(f1_1 * D(F.u1, 'z', G.s11[2], interp='last'), 'x', 
                           G.s11[0])
    wy_z = Jii * F.g_c * P(f2_2 * D(F.u2, 'z', G.s11[2], interp='last'), 'y', 
                           G.s11[1])

    q1_z = J12i * F.g_c * P(f2_1 * D(F.u1, 'z', G.s12[2], interp='last'), 'y',
                            G.s12[1])

    q2_z = J12i * F.g_c * P(f1_2 * D(F.u2, 'z', G.s12[2], interp='last'), 'x',
                            G.s12[0])

    a3_z = J13i * F.g * P(f1_3 * D(F.u3, 'z', G.s13[2], interp='last'),
                          'x', G.s13[0])

    b3_z = J23i * F.g * P(f2_3 * D(F.u3, 'z', G.u3[2], interp='last'), 'y',
                           G.s23[1])

    div_val = x * vx_x + y * vy_y + Jii * vz_z - x * wx_z - y * wy_z
    div = fd.Variable('div', dtype=fd.prec, val=div_val)

    if use_acoustic:
        mu12 = 0
        mu13 = 0
        mu23 = 0
        twomu = 0

    # Discretization
    s11_t = a * F.s11 + lam * div + x * twomu * vx_x - x * twomu * wx_z
    s22_t = a * F.s22 + lam * div + y * twomu * vy_y - y * twomu * wy_z
    s33_t = a * F.s33 + lam * div + twomu * Jii * vz_z
    s12_t = a * F.s12 + mu12 * (   y * vx_y   + x * vy_x  - y * q1_z  - x * q2_z)
    s13_t = a * F.s13 + mu13 * (J13i * vx_z   + x * vz_x  - x * a3_z)
    s23_t = a * F.s23 + mu23 * (J23i * vy_z   + y * vz_y  - y * b3_z)

    if debug_ops:
        s11_t = P(D(F.u3, 'x', G.u3[0], interp='last'), 'z', G.s11[2])
        s22_t = P(D(F.u3, 'y', G.u3[1], interp='last'), 'z', G.s11[2])
        s12_t = P(D(F.u2, 'z', G.u2[2], interp='last'), 'x', G.s12[0])
        s13_t = P(D(F.u3, 'z', G.u3[2], interp='last'), 'x', G.s13[0])
        s23_t = P(D(F.u3, 'z', G.u3[2], interp='last'), 'y', G.s23[1])
                   
    if use_cartesian:
        s11_t = a * F.s11 + lam * div + twomu * vx_x
        s22_t = a * F.s22 + lam * div + twomu * vy_y
        s33_t = a * F.s33 + lam * div + twomu * vz_z
        s12_t = a * F.s12 + mu12 * (vx_y + vy_x)
        s13_t = a * F.s13 + mu13 * (vz_x + vx_z)
        s23_t = a * F.s23 + mu23 * (vz_y + vy_z)

    eqs = [] 
    eqs += [Jii,  (Jii.symbol, 1.0 / Jii.symbol)]
    eqs += [J12i, (J12i.symbol, 1.0 / J12i.symbol)]
    eqs += [J13i, (J13i.symbol, 1.0 / J13i.symbol)]
    eqs += [J23i, (J23i.symbol, 1.0 / J23i.symbol)]
    eqs += [lam]
    if not use_acoustic:
        eqs += [twomu, mu12, mu13, mu23]
    eqs += [div]
    if use_sponge:
        f_dcrj = apply_sponge_layer()
        eqs += [f_dcrj]
        eqs += [(F.s11, s11_t * f_dcrj)]
        eqs += [(F.s22, s22_t * f_dcrj)] 
        eqs += [(F.s33, s33_t * f_dcrj)]
        eqs += [(F.s12, s12_t * f_dcrj)]
        eqs += [(F.s13, s13_t * f_dcrj)] 
        eqs += [(F.s23, s23_t * f_dcrj)]
    else:
        eqs += [(F.s11, s11_t)]
        eqs += [(F.s22, s22_t)] 
        eqs += [(F.s33, s33_t)]
        eqs += [(F.s12, s12_t)]
        eqs += [(F.s13, s13_t)] 
        eqs += [(F.s23, s23_t)]

    lhs, rhs = fd.equations(eqs)
    
    lhs_indices = None
    rhs_indices = None
    bounds = helper.strbounds(D(F.s11,'z'), F.s11,
                              exclude_left=helper.get_exclude_left(debug))
    index_bounds = (1,1,0)
    grid_order = ['z', 'y', 'x']
    kernels = kg.make_kernel(label, 
                              lhs, rhs,
                              bounds, helper.gridsymbols,
                              regions=(1, 1, [0, 1, 2]),
                              debug=0, generator=helper.generator,
                              index_bounds=index_bounds,
                              extraout = [F.u1, F.u2, F.u3],
                              extraconst = [F.s11, F.s22, F.s33,
                                            F.s12, F.s13, F.s23, a,
                                            F.u1, F.u2, F.u3, 
                                            F.lami, F.mui,
                                            F.dcrjx, F.dcrjy, F.dcrjz,
                                            F.g,
                                            F.g3,
                                            F.g3_c,
                                            F.g_c,
                                            nu,
                                            F.f_1, F.f_2, F.f_c,
                                            f1_1, f1_2, f1_3,
                                            f2_1, f2_2, f2_3],
                              lhs_indices=lhs_indices, rhs_indices=rhs_indices,
                              grid_order=grid_order)
    return kernels

def apply_sponge_layer():
        from variables import F
        return fd.Variable('f_dcrj', dtype=fd.prec, val=F.dcrjx*F.dcrjy*F.dcrjz)

def material(label, unit_material=0):
    """
    When AWP runs in double precision, the mesh reader breaks. This kernel
    overcomes this problem by manually assigning values to the material
    parameters
    """
    from variables import F

    if unit_material:
        rho0 = 1.0
        lami0 = 1.0
        mui0 = 1.0
    else:
        rho0 = 2800.00000000000000
        lami0  = 1.0 / 50400000000.00000000000000
        mui0 = 1.0 / 25200000000.00000000000000

    eqs = []
    zero =  0 * F.rho
    eqs += [(F.mrho, zero  + rho0)]
    eqs += [(F.mlami, zero + lami0)]
    eqs += [(F.mmui, zero + mui0)]

    lhs, rhs = fd.equations(eqs)

    bounds = helper.bounds()
    grid_order = ['z', 'y', 'x']

    kernels = kg.make_kernel(label, 
                              lhs, rhs,
                              bounds, helper.gridsymbols,
                              regions=(1, 1, 1),
                              debug=0, generator=helper.generator,
                              grid_order=grid_order)
    return kernels


kernels = []
kernels = velocity("dtopo_vel",debug=debug, debug_ops=0,
        use_cartesian=use_cartesian)
kernels += velocity("dtopo_buf_vel",buf=1, debug=debug,
        use_cartesian=use_cartesian)
kernels += stress("dtopo_str", debug=debug, debug_ops=0,
        use_cartesian=use_cartesian)
kernels += material("dtopo_init_material", unit_material=1)
kg.write_kernels(filename, kernels, header=True,
        source_includes=['#include <topography/kernels/%s.cuh>'%filename], 
        header_includes=['#include <awp/definitions.h>'])

