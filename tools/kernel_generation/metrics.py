"""usage: python3 metrics.py <debug>
Build scheme

debug   - Run in debug mode. Exports kernels in double precision. 
          Disabled by default.

"""
import sys
import sympy as sp
import openfd as fd
from openfd import Expr
import numpy as np
import helper
from openfd.dev import kernelgenerator as kg
from helper import D, P, CGenerator

generator=CGenerator

def interpolate(label, grid):
    """
    Interpolate from the node positions, storing the metrics, to the v1, v2
    velocity grids, and cell center.

    """
    from variables import F
    G = helper.shifts()

    
    # v1
    if G.u1[0] == grid[0] and G.u1[1] == grid[1]:
        eqs = [(F.df1, P(F.f, 'y', grid[1]))]
    # v2
    elif G.u2[0] == grid[0] and G.u2[1] == grid[1]:
        eqs = [(F.df1, P(F.f, 'x', grid[0]))]
    # cell-center
    elif G.s11[0] == grid[0] and G.s11[1] == grid[1]:
        eqs = [(F.df1, P(P(F.f, 'x', grid[0]), 'y', grid[1]))]
    else:
        raise ValueError("Must specify v1, v2, or cell-centered grid")

    bounds = helper.fbounds()
    lhs, rhs = fd.equations(eqs)
    gc = helper.ghost_cells
    radius = helper.radius
    gc.visible = 0
    kernels = kg.make_kernel(label, 
              lhs, rhs,
              bounds, helper.gridsymbols,
              regions=(1, 1, 1),
              lhs_indices = lambda x : (x[0]-gc+radius, x[1]-gc+radius, x[2]),
              rhs_indices = lambda x : (x[0]-gc+radius, x[1]-gc+radius, x[2]),
              debug=0, generator=generator)

    return kernels

def differentiate(label, axis, grid):
    from variables import F
    G = helper.shifts()

    eqs = []
    hi = sp.symbols('hi')
    eqs += [(F.df1, hi*D(F.f, axis, grid))]
    lhs, rhs = fd.equations(eqs)

    bounds = helper.fbounds()
    lhs, rhs = fd.equations(eqs)
    gc = helper.ghost_cells
    radius = helper.radius
    gc.visible = 0
    kernels = kg.make_kernel(label, 
              lhs, rhs,
              bounds, helper.gridsymbols,
              regions=(1, 1, 1),
              lhs_indices = lambda x : (x[0]-gc+radius, x[1]-gc+radius, x[2]),
              rhs_indices = lambda x : (x[0]-gc+radius, x[1]-gc+radius, x[2]),
              debug=0, generator=generator)
    return kernels

def interpolate_z(label):
    from variables import F
    G = helper.shifts()

    g_3 = P(F.g, 'z', G.s11[2])

    eqs = []
    eqs += [(F.g3, g_3)]
    lhs, rhs = fd.equations(eqs)

    bounds = (fd.Bounds(1), fd.Bounds(1), g_3.bounds())
    kernels = kg.make_kernel(label, 
                             lhs, rhs,
                             bounds, helper.gridsymbols,
                             regions=(1, 1, range(3)),
                             debug=0, generator=generator)
    return kernels

def differentiate_z(label, grid):
    from variables import F
    hi = sp.symbols('hi')
    g_3 = D(F.g, 'z', grid)

    eqs = []
    eqs += [(F.g3, hi*g_3)]
    lhs, rhs = fd.equations(eqs)

    bounds = (fd.Bounds(1), fd.Bounds(1), g_3.bounds())
    kernels = kg.make_kernel(label, 
                             lhs, rhs,
                             bounds, helper.gridsymbols,
                             regions=(1, 1, range(3)),
                             debug=0, generator=generator)
    return kernels

if len(sys.argv) < 2:
    print(__doc__)
    exit(0)
else:
    prec_str = sys.argv[1]

print("Generating metric kernels. " +                                     
      "Precision: %s"%(prec_str))

kernels = []
G = helper.shifts()
helper.set_precision(prec_str)
kernels += interpolate("metrics_f_interp_1", G.u1)
kernels += interpolate("metrics_f_interp_2", G.u2)
kernels += interpolate("metrics_f_interp_c", G.s11)
kernels += differentiate("metrics_f_diff_1_1", "x", G.u1[0])
kernels += differentiate("metrics_f_diff_1_2", "x", G.u2[0])
kernels += differentiate("metrics_f_diff_2_1", "y", G.u1[1])
kernels += differentiate("metrics_f_diff_2_2", "y", G.u2[1])
kernels += interpolate_z("metrics_g_interp")
kernels += differentiate_z("metrics_g_diff_3", G.u3[2])
kernels += differentiate_z("metrics_g_diff_c", G.s11[2])
kg.write_kernels("metrics_kernel", kernels, header=True,
header_includes=['#include "definitions.h"'])

