"""usage: python3 simplified.py <precision>
Build simplified scheme


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
use_free_surface_bc = helper.get_use_free_surface_bc(0)
bnd_right=1

print("Precision:", prec_str,
      "Debug:", 0, 
      "Apply free surface boundary condition:", use_free_surface_bc)

def velocity(label):
    from helper import D, P
    from variables import F, mem, size
    G = helper.shifts()

    a, nu = sp.symbols('a nu')

    u1 = F.u1
    u2 = F.u2
    u3 = F.u3
 
    s13_z = D(F.s13, 'z', G.u1[2], bnd_right=bnd_right)

    u1_t = a * F.u1 + nu * s13_z

    eqs = [] 
    eqs += [(u1, u1_t)]

    lhs, rhs = fd.equations(eqs)

    bounds = helper.velbounds(D(F.s11,'z'),
                              exclude_left=helper.get_exclude_left(0))
    lhs_indices = None
    rhs_indices = None
    index_bounds = (1,1,0)

    kernels = kg.make_kernel(label, 
                              lhs, rhs,
                              bounds, helper.gridsymbols,
                              regions=(1, 1, [0, 1, 2]),
                              debug=0, generator=helper.generator,
                              index_bounds=index_bounds,
                              lhs_indices=lhs_indices, rhs_indices=rhs_indices)
    return kernels

def stress(label):
    """
    Generate stress update kernels.
    """
    from helper import D, P
    from variables import F, mem, size
    G = helper.shifts()

    print("Generating stress kernels: %s. "%label)
    a, nu = sp.symbols('a nu')

    vx_x = D(F.u1, 'x', G.s33[0])
    vy_y = D(F.u2, 'y', G.s33[1])    
    vz_z = D(F.u3, 'z', G.s33[2])    

    vx_y = D(F.u1, 'y', G.s12[1])    
    vx_z = D(F.u1, 'z', G.s13[2])    

    vy_x = D(F.u2, 'x', G.s12[0])    
    vy_z = D(F.u2, 'z', G.s23[2])    

    vz_x = D(F.u3, 'x', G.s13[0])    
    vz_y = D(F.u3, 'y', G.s23[1])    

    s13_t = a * F.s13 + nu * vx_z

    eqs = [] 
    eqs += [(F.s13, s13_t)] 

    lhs, rhs = fd.equations(eqs)
    
    lhs_indices = None
    rhs_indices = None
    bounds = helper.strbounds(D(F.s11,'z'), F.s11,
                              exclude_left=helper.get_exclude_left(0))
    index_bounds = (1,1,0)
    kernels = kg.make_kernel(label, 
                              lhs, rhs,
                              bounds, helper.gridsymbols,
                              regions=(1, 1, [0, 1, 2]),
                              debug=0, generator=helper.generator,
                              index_bounds=index_bounds,
                              lhs_indices=lhs_indices, rhs_indices=rhs_indices)
    return kernels

def kinetic_energy():
    from helper import D, P, H
    from variables import F, mem, size
    G = helper.shifts()

    v1, v2, v3 = helper.fields(helper.size, helper.mem, ['v1', 'v2', 'v3']) 
    du1, du2, du3 = helper.fields(helper.size, helper.mem, ['du1', 'du2', 'du3']) 

    H_u1 = lambda arg : H(H(H(arg, 'z', hat=1)   , 'y', hat=1), 'x', hat=0)

    u1 = F.u1
    u2 = F.u2
    u3 = F.u3

    u1_t = 0.5 * H(du1, 'z', G.u3[2])

    eqs = []
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
                              lhs_indices=lhs_indices, rhs_indices=rhs_indices)
    return kernels

def strain_energy():
    from helper import D, P, H
    from variables import F, mem, size
    G = helper.shifts()

    # initial condition input
    s11_0, s22_0, s33_0, s12_0, s13_0, s23_0 = helper.fields(helper.size, helper.mem,
            ['s11_0', 's22_0', 's33_0', 's12_0', 's13_0', 's23_0']) 

    # stress rates
    ds11, ds22, ds33, ds12, ds13, ds23 = helper.fields(helper.size, helper.mem,
            ['ds11', 'ds22', 'ds33', 'ds12', 'ds13', 'ds23']) 

    # output
    v11, v22, v33, v12, v13, v23 = helper.fields(helper.size, helper.mem,
            ['v11', 'v22', 'v33', 'v12', 'v13', 'v23']) 

    e13_t = 0.5 * H(ds13, 'z', G.s13[2])

    eqs = []
    eqs += [(v13, s13_0 * e13_t)]

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
                              lhs_indices=lhs_indices, rhs_indices=rhs_indices)

    return kernels


kernels = []
kernels = velocity("velocity")
kernels += stress("stress")
kernels += kinetic_energy()
kernels += strain_energy()
kg.write_kernels("simplified", kernels, header=False,
header_includes=['#include "definitions.h"'])
helper.file_prepend("simplified.cu", 
                    "#define align 32\n#define ngsl 8")

