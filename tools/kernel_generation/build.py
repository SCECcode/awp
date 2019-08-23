from openfd.dev import kernelgenerator as kg
import openfd as fd
import sympy as sp
import helper
from helper import u1, v1, w1, xx, yy, zz, out, D, F, G, nx, ny, nz, rank

def diff_x(F, out, inp, label):
    """
    Differentiate xx in the x-direction and store the result in u1
    """
    lhs = F[out]
    rhs = D(F[inp], 'x', G[out][0])
    
    bounds = helper.bounds()
    kernels = []
    kernels += kg.make_kernel(label, 
                              (lhs,), (rhs,),
                              bounds, helper.gridsymbols,
                              regions=(1, 1, 1),
                              debug=helper.debug, generator=helper.generator)
    return kernels

def diff_y(F, out, inp, label):
    """
    Differentiate yy in the y-direction and store the result in v1
    """
    lhs = F[out]
    rhs = D(F[inp], 'y', G[out][1])
    
    bounds = helper.bounds()
    kernels = []
    kernels += kg.make_kernel(label, 
                              (lhs,), (rhs,),
                              bounds, helper.gridsymbols,
                              regions=(1, 1, 1),
                              debug=helper.debug, generator=helper.generator)
    return kernels

def diff_z(F, out, inp, label):
    """
    Differentiate yy in the y-direction and store the result in v1
    """
    lhs = F[out]
    rhs = D(F[inp], 'z', G[out][2])
    
    # Adjust bounds so that theere is a boundary in the z-direction
    bndz = rhs.bounds()
    bounds = list(helper.bounds())
    bounds[2] = fd.Bounds(bndz.size, right=bndz.right)
    kernels = []
    kernels += kg.make_kernel(label, 
                              (lhs,), (rhs,),
                              bounds, helper.gridsymbols,
                              regions=(1, 1, [1, 2]),
                              debug=helper.debug, generator=helper.generator)
    return kernels

def poly(out, label):
    """
    Construct polynomial function `out = a*x^p + b*y^q + c*z^r`

    """

    a = helper.symbols('a', 3)
    p = helper.symbols('p', 3)
    s = helper.symbols('s', 3)
    grids = helper.grids('x', helper.size, helper.mem)
    lhs = out
    rhs = fd.GridFunctionExpression(sum([ai*(gi+0.5*si + rank*ni)**pi for gi,
                                         ai, si, pi, ni in 
                                     zip(grids, a, s, p, helper.size)]))
    bounds = helper.bounds()
    kernels = []
    kernels += kg.make_kernel(label, 
                              (lhs,), (rhs,),
                              bounds,
                              helper.gridsymbols,
                              regions=(1, 1, 1),
                              debug=helper.debug, generator=helper.generator)
    return kernels


kernels = []
kernels += diff_x(F, 'xx', 'u1', 'dtopo_test_diffx')
kernels += diff_y(F, 'yy', 'v1', 'dtopo_test_diffy')
kernels += diff_z(F, 'xz', 'w1', 'dtopo_test_diffz')
kernels += poly(out, 'dtopo_test_poly')
kg.write_kernels("kernels", kernels)
