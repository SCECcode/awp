"""usage: python3 energy.py <prec>
Build scheme

prec    - Precision. Can be either 'float' or 'double'.

"""

import sys
import sympy as sp
import openfd as fd
from openfd import Expr
import numpy as np
import helper
from openfd.dev import kernelgenerator as kg

if len(sys.argv) < 2:
    print(__doc__)
    exit(0)
else:
    prec_str = sys.argv[1]

helper.set_precision(prec_str)

print("Precision:", prec_str)

def kinetic_energy():
    from helper import D, P, H
    from variables import F, mem, size
    G = helper.shifts()

    v1, v2, v3 = helper.fields(helper.size, helper.mem, ['v1', 'v2', 'v3']) 
    du1, du2, du3 = helper.fields(helper.size, helper.mem, ['du1', 'du2', 'du3']) 

    # Derivatives of topography functions
    f1_1 = F.f1_1
    f1_2 = F.f1_2
    f1_3 = F.f1_c

    f2_1 = F.f2_1
    f2_2 = F.f2_2
    f2_3 = F.f2_c

    J1 = F.f_1*F.g3_c
    J2 = F.f_2*F.g3_c
    J3 = F.f_c*F.g3

    rho1 = fd.Variable('rho1', dtype=fd.prec, val=P(P(F.rho, 'y', 1), 'z', 1)) 
    rho2 = fd.Variable('rho2', dtype=fd.prec, val=P(P(F.rho, 'x', 1), 'z', 1)) 
    rho3 = fd.Variable('rho3', dtype=fd.prec, val=P(P(F.rho, 'x', 1), 'y', 1)) 

    H_u1 = lambda arg : H(H(H(arg, 'z', hat=1)   , 'y', hat=1), 'x', hat=0)

    u1 = F.u1
    u2 = F.u2
    u3 = F.u3

    u1_t = 0.5 * rho1 * J1 * H(du1, 'z', G.u1[2])
    u2_t = 0.5 * rho2 * J2 * H(du2, 'z', G.u2[2])
    u3_t = 0.5 * rho3 * J3 * H(du3, 'z', G.u3[2])

    eqs = []
    eqs += [rho1, rho2, rho3]
    eqs += [(v1, u1 * u1_t)]
    eqs += [(v2, u2 * u2_t)] 
    eqs += [(v3, u3 * u3_t)]

    lhs, rhs = fd.equations(eqs)
    lhs_indices = None
    rhs_indices = None
    index_bounds = (1,1,0)
    bounds = helper.velbounds(D(F.s11,'z'),
                              exclude_left=helper.get_exclude_left(0))
    kernels = kg.make_kernel("kinetic_energy", 
                              lhs, rhs,
                              bounds, helper.gridsymbols,
                              regions=(1, 1, [0, 1, 2]),
                              debug=0, generator=helper.generator,
                              index_bounds=index_bounds,
                              extraout = [u1, u2, u3],
                              extraconst = [
                                            F.u1, F.u2, F.u3,
                                            v1, v2, v3,
                                            F.rho,
                                            F.g,
                                            F.g3,
                                            F.g3_c,
                                            F.g_c,
                                            F.f,
                                            F.f_1, F.f_2, F.f_c,
                                            f1_1, f1_2, f1_3,
                                            f2_1, f2_2, f2_3],
                              lhs_indices=lhs_indices, rhs_indices=rhs_indices)

    return kernels

def strain_energy():
    from helper import D, P, H
    from variables import F, mem, size
    G = helper.shifts()

    # initial condition input
    s11, s22, s33, s12, s13, s23 = helper.fields(helper.size, helper.mem,
            ['s11', 's22', 's33', 's12', 's13', 's23']) 

    # stress rates
    ds11, ds22, ds33, ds12, ds13, ds23 = helper.fields(helper.size, helper.mem,
            ['ds11', 'ds22', 'ds33', 'ds12', 'ds13', 'ds23']) 

    # output
    v11, v22, v33, v12, v13, v23 = helper.fields(helper.size, helper.mem,
            ['v11', 'v22', 'v33', 'v12', 'v13', 'v23']) 

    # Derivatives of topography functions
    f1_1 = F.f1_1
    f1_2 = F.f1_2
    f1_3 = F.f1_c

    f2_1 = F.f2_1
    f2_2 = F.f2_2
    f2_3 = F.f2_c

    J1 = F.f_1*F.g3_c
    J2 = F.f_2*F.g3_c
    J3 = F.f_c*F.g3

    J1c = F.f_1*F.g3_c   
    J2c = F.f_2*F.g3_c
    Jc3 = F.f_c*F.g3

    Jcc = F.f_c*F.g3_c
    J12 = F.f * F.g3_c
    J13 = F.f_1*F.g3
    J23 = F.f_2*F.g3


    lam = fd.Variable('lam', dtype=fd.prec, 
                      val=1/P(
                              P(
                                P(F.lami, 'x', G.s11[0]), 
                                          'y', G.s11[1]),
                                          'z', G.s11[2]))
    mu = fd.Variable('mu', dtype=fd.prec, 
                     val=1/P(
                                P(
                                  P(F.mui, 'x', G.s11[0]), 
                                           'y', G.s11[1]),
                                           'z', G.s11[2]))

    mu12 = fd.Variable('mu12', dtype=fd.prec, val=1 / P(F.mui, 'z', G.s12[2]))
    mu13 = fd.Variable('mu13', dtype=fd.prec, val=1 / P(F.mui, 'y', G.s13[1]))
    mu23 = fd.Variable('mu23', dtype=fd.prec, val=1 / P(F.mui, 'x', G.s23[0]))

    # Since the grid is "periodic" in the x, y directions, there is no need for
    # norms in these directions
    dskk = ds11 + ds22 + ds33
    d = lam / (2 * mu * (3 * lam + 2 * mu))
    r = 0.5 * d * Jcc * H(dskk, 'z', G.s33[2])
    e11_t = 0.25 * Jcc * H(ds11, 'z', G.s33[2]) / mu - r
    e22_t = 0.25 * Jcc * H(ds22, 'z', G.s33[2]) / mu - r
    e33_t = 0.25 * Jcc * H(ds33, 'z', G.s33[2]) / mu - r
    e12_t = 0.5 * J12 * H(ds12, 'z', G.s12[2]) / mu12
    e13_t = 0.5 * J13 * H(ds13, 'z', G.s13[2]) / mu13
    e23_t = 0.5 * J23 * H(ds23, 'z', G.s23[2]) / mu23

    eqs = []
    eqs += [lam, mu, mu12, mu13, mu23]
    eqs += [(v11, s11 * e11_t)]
    eqs += [(v22, s22 * e22_t)] 
    eqs += [(v33, s33 * e33_t)]
    eqs += [(v12, s12 * e12_t)]
    eqs += [(v13, s13 * e13_t)]
    eqs += [(v23, s23 * e23_t)]

    lhs, rhs = fd.equations(eqs)
    lhs_indices = None
    rhs_indices = None
    index_bounds = (1,1,0)
    bounds = helper.strbounds(D(F.s11,'z'), F.s11,
                              exclude_left=helper.get_exclude_left(0))
    kernels = kg.make_kernel("strain_energy", 
                              lhs, rhs,
                              bounds, helper.gridsymbols,
                              regions=(1, 1, [0, 1, 2]),
                              debug=0, generator=helper.generator,
                              index_bounds=index_bounds,
                              extraout = [ds11, ds22, ds33, ds12, ds13, ds23,
                                          s11, s22, s33, s12, s13, s23,
                                          v11, v22, v33, v12, v13, v23,
                                          ],
                              extraconst = [
                                            F.g,
                                            F.g3,
                                            F.g3_c,
                                            F.g_c,
                                            F.f,
                                            F.f_1, F.f_2, F.f_c,
                                            f1_1, f1_2, f1_3,
                                            f2_1, f2_2, f2_3],
                              lhs_indices=lhs_indices, rhs_indices=rhs_indices)

    return kernels

kernels = []
kernels += kinetic_energy()
kernels += strain_energy()

kg.write_kernels("cuenergy_kernel", kernels, header=True,
header_includes=['#include "definitions.h"'])
