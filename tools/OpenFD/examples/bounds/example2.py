from sympy import symbols
from openfd import GridFunction, GridFunctionExpression as Expr, Bounds
from openfd.dev import kernelgenerator as kg
"""
This example is a continuation of a `example1`. In this example, we generate
bounds for a grid in three dimensions of size `nx x ny x nz` and produce a
different partitioning for three different kernel functions. For each kernel,
the grid is partitioned in only direction at a time. The first kernel uses a
partitioning in the x-direction, and the second one uses a partitioning in the
y-direction, and the third, and last one uses a partitioning in the z-direction.

That is, we will setup the following bound configuration for each kernel:

bounds[0] : left     :      0 <= x < bp,      0 <= y < ny, 0 <= z < nz,
            interior :     bp <= x < nx - b,  0 <= y < ny, 0 <= z < nz,
            right    : nx - b <= x < nx,      0 <= y < ny, 0 <= z < nz,

bounds[1] : bottom   :      0 <= y < bp,      0 <= x < nx, 0 <= z < nz,
            interior :     bp <= y < ny - b,  0 <= x < nx, 0 <= z < nz,
            top      : ny - b <= y < ny,      0 <= x < nx, 0 <= z < nz,

bounds[1] : back     :      0 <= z < bp,      0 <= y < ny, 0 <= z < nz,
            interior :     bp <= z < nz - b,  0 <= y < ny, 0 <= z < nz,
            front    : nz - b <= z < nz,      0 <= y < ny, 0 <= z < nz,

"""


dim = 3
bp = 2
regions = [-1, 0, 1]
language = 'C'

nx, ny, nz = symbols('nx ny nz')
shape = (nx, ny, nz)

# Construct bounds for each kernel object
B = lambda n, b : Bounds(n, left=b, right=b)
bounds = [0]*dim

bpx = [bp, 0, 0]
bpy = [0, bp, 0]
bpz = [0, 0, bp]
for i in range(dim):
    bounds[i] = (B(nx, bpx[i]), B(ny, bpy[i]), B(nz, bpz[i]))
    bounds[i] = bounds[i][0:dim]

u = GridFunction('u', shape=shape[0:dim]) 
out = GridFunction('out', shape=shape[0:dim]) 

rhs = [0]*dim
krhs = [0]*dim
kernels = []

for i in range(dim):
    rhs[i] = Expr(u)
    # Use debug=`True` to output the region Id.
    krhs[i] = kg.kernelgenerator(language, (nx, ny, nz), bounds[i], out, rhs[i],
                                 debug=True)

    for rx in regions:
        reg = [1]*dim
        reg[i] = rx
        regionnames = {0: 'l', 1: 'i', -1: 'r'}
        kernel = krhs[i].kernel('kernel%d_%s'%(i, regionnames[rx]), tuple(reg))
        kernels.append(kernel)

for i, kernel in enumerate(kernels):
    print(kernel.code)
