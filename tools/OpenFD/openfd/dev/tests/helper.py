from openfd.sbp import traditional as sbp
from openfd.base import GridFunctionExpression as GFE
from openfd.base import GridFunction, Bounds
from openfd.dev.kernelgenerator import kernelgenerator
from openfd.dev.kernelevaluator import kernelevaluator
from ..kernelgenerator import make_kernel, write_kernels
from sympy import symbols
import numpy as np
def kernel1d(generator, evaluator):
    # Test that the first derivative differentiates a linear function,
    # f(x) = x, correctly on the GPU
    nx = symbols('nx')
    u = GridFunction("u", shape=(nx + 1,))
    out = GridFunction("out", shape=(nx + 1,))
    u_x = sbp.Derivative(u, "x", order=4, gpu=True)
    expr = GFE(u_x)

    code = ''
    regions = [-1, 0, 1]
    names = {-1: 'right', 0: 'left', 1: 'interior'}
    kernelname = lambda x: 'test_%s' % x

    kg = generator(nx, u_x.bounds(), out, expr)
    kernels = [kg.kernel(kernelname(names[r]), r) for r in regions]

    nx_mem = np.int32(24)
    u_mem = np.array(range(nx_mem + 1)).astype(np.float32)
    out_mem = np.array(range(nx_mem + 1)).astype(np.float32)

    ke = evaluator(kernels,
                         inputs={nx: nx_mem, u: u_mem},
                         outputs={out: out_mem})
    ke.eval()
    ke.get_outputs()

    assert np.all(np.abs(out_mem - 1.0) < 1e-6 * nx_mem)

def kernel2d(generator, evaluator):
    # Test that the first derivative differentiates a linear function,
    # f(x, y) = x + y, correctly on the GPU
    nx, ny = symbols('nx ny')
    dim = (nx + 1, ny + 1)
    u = GridFunction("u", shape=dim)
    out = GridFunction("out", shape=dim)
    u_x = sbp.Derivative(u, "x", order=4, gpu=True)
    u_y = sbp.Derivative(u, "y", order=4, gpu=True)
    expr = GFE(u_x + u_y)

    # Kernel generation
    code = ''
    regions = [-1, 0, 1]
    regx = regions
    regy = regions
    names = {-1: 'r', 0: 'l', 1: 'i'}
    kernelname = lambda x: 'test_%s' % x

    kg = generator((nx, ny), (u_x.bounds(), u_y.bounds()), out, expr)
    kernels = []
    for rx in regx:
        for ry in regy:
            kernel = kg.kernel(kernelname(names[rx] + names[ry]), (rx, ry))
            kernels.append(kernel)

    nx_mem = np.int32(24)
    u_mem = np.zeros((nx_mem + 1, nx_mem + 1)).astype(np.float32)
    out_mem = np.zeros((nx_mem + 1, nx_mem + 1)).astype(np.float32) + 1.0
    for j in range(nx_mem + 1):
        for i in range(nx_mem + 1):
            u_mem[i, j] = i + j

    ke = evaluator(kernels,
                   inputs={nx: nx_mem, ny: nx_mem, u: u_mem},
                   outputs={out: out_mem})
    ke.eval()
    ke.get_outputs()
    assert np.all(np.abs(out_mem - 2.0) < 1e-6 * nx_mem*nx_mem)

def make_multid(nd, name='test'):
    n = symbols('n')
    dim = [10]*nd
    u = GridFunction("u", shape=dim)
    v = GridFunction("v", shape=dim)
    bounds = [Bounds(n, 0, 0)]*nd
    regions = [(0, 1)]*nd
    if nd == 1:
        regions = regions[0]
    a = make_kernel(name, u, v, bounds, (n,), regions=regions)
    return a
