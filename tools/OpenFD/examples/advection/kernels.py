from openfd import sbp_traditional as sbp, GridFunction, Bounds
from openfd import GridFunctionExpression as Expr
from openfd.dev import kernelgenerator as kg

from sympy import symbols
import sympy as sp
import helper

"""
Solves the advection equation u_t + a*u_x = 0.
"""
def kernel(language,order,debug):
    a, n, t, ak, bk, ck, du, hi, dt = symbols('a n t ak bk ck du hi dt')
    omega = symbols('omega')
    shape = (n,)
    u = GridFunction('u', shape=shape)
    du = GridFunction('du', shape=shape)
    D = sbp.Derivative('', 'x', shape=shape, order=order, gpu=True)

    # left boundary condition
    g = lambda t: sp.sin(-omega*t)

    """ SAT
     The second order SBP norm is `H = h*diag(1/2, 1 . . . 1, 1/2)`.
    """
    #FIXME: hard-coded for now
    if order == 2:
        Hi = 2.0
    elif order == 4:
        Hi = 48/17

    # kernel to compute RK4 rates to PDE
    lhs = du
    rhs = Expr(ak*lhs - a*hi*D*u)
    bounds = (D.bounds(),)
    k1 = kg.kernelgenerator(language, shape, bounds, lhs, rhs, debug=debug)
    k1s = helper.build(k1, 'pde')

    # kernel to compute RK4 rates to bc (left)
    lhs = du
    rhs = Expr(lhs - a*hi*Hi*(u - g(t + ck*dt)))
    bcbounds = (Bounds(n, left=1, right=0),)
    k2 = kg.kernelgenerator(language, shape, bcbounds, lhs, rhs)
    k2s = helper.build(k2, 'bc', regions=(0,))

    # kernel to compute RK4 update
    lhs = u
    rhs = Expr(lhs + dt*bk*du)
    bounds = (Bounds(size=n, left=0, right=0))
    k3 = kg.kernelgenerator(language, shape, bounds, lhs, rhs)
    k3s = helper.build(k3, 'update', regions=(1,))

    kernels = []
    kernels += k1s
    kernels += k2s
    kernels += k3s
    filename = 'advection.cu'
    f = open(filename,'w')
    for kl in kernels:
        f.write(kl.code)
    f.close()

# generate kernels
kernel('Cuda', 4, False)
