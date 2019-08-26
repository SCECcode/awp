"""usage: python3 scheme.py <debug> <use_2d>
Build scheme

debug   - Run in debug mode. Exports kernels in double precision and without
          enforcing boundary conditions. Disabled by default.

use_2d  - Generate 2D kernels. The y-direction is ignored.

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

def velocity(label, buf=0, debug=0, debug_ops=0, use_cartesian=0):
    from helper import D, P
    from variables import F, mem, size
    G = helper.shifts()

    rho1 = fd.Variable('rho1', dtype=fd.prec, val=P(P(F.rho, 'y', 1), 'z', 1)) 
    rho2 = fd.Variable('rho2', dtype=fd.prec, val=P(P(F.rho, 'x', 1), 'z', 1)) 
    rho3 = fd.Variable('rho3', dtype=fd.prec, val=P(P(F.rho, 'x', 1), 'y', 1)) 

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

    # Discretization
    u1_t = a * F.u1 + Ai1 * (s11_x + s12_y + s13_z - w11_z - w12_z)
    u2_t = a * F.u2 + Ai2 * (s12_x + s22_y + s23_z - w21_z - w22_z)
    u3_t = a * F.u3 + Ai3 * (s13_x + s23_y + s33_z - w31_z - w32_z)

    eqs = [] 
    eqs += [rho1, rho2, rho3]
    eqs += [Ai1, (Ai1.symbol, nu/Ai1.symbol)]
    eqs += [Ai2, (Ai2.symbol, nu/Ai2.symbol)]
    eqs += [Ai3, (Ai3.symbol, nu/Ai3.symbol)]
    eqs += [(u1, u1_t)]
    eqs += [(u2, u2_t)] 
    eqs += [(u3, u3_t)]

    lhs, rhs = fd.equations(eqs)

    bounds = helper.velbounds(D(F.s11,'z'),
                              exclude_left=helper.get_exclude_left(debug))

    kernels = kg.make_kernel(label, 
                              lhs, rhs,
                              bounds, helper.gridsymbols,
                              regions=(1, 1, [0, 1, 2]),
                              debug=0, generator=helper.generator,
                              index_bounds=(1, 1, 0))
    return kernels



kernels = []
kernels = velocity("velocity",debug=debug, debug_ops=0,
        use_cartesian=use_cartesian)
kg.write_kernels("demo", kernels, header=True,
header_includes=['#include "definitions.h"'])

