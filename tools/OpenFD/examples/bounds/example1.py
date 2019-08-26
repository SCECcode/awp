"""
This example demonstrates how to specify custom bounds for controlling what
parts of the computational domain is accessed. In this example, we take a 2D
grid of size (nx, ny) and break it up into three compute regions: left,
interior, right. For each such region we generate a kernel function. 

The left region is defined by:
```
0 <= x < bp, 0 < y <= ny. 
```
The parameter `bp` controls the size of the left region. 

Similarly, the right region is
defined by:
```
nx - b <= x < nx, 0 < y <= ny. 
```

Finally, the interior region is defined by: 
```
bp <= x < nx - b, 0 < y <= ny.
```

To setup this configuration, we only need to create one bounds object per grid
dimension. For this purpose, we introduce a tuple ` bounds = (Bounds(...),
Bounds(...))`. The first component of this tuple refers to the x-direction, and
the second component refers to the y-direction. Then, the partitioning of the
x-direction into three regions is accomplished by specifying the grid size `nx`
and number of left boundary points `bp`, right boundary points `bp`.

Note that since we wish to not partition the y-direction, we simply set the
number of left and right boundary points for the corresponding bounds object to
zero.

```

"""
from sympy import symbols
from openfd import GridFunction, GridFunctionExpression as Expr, Bounds
from openfd.dev import kernelgenerator as kg


bp = 2
regions = [-1, 0, 1]
language = 'C'
nx, ny = symbols('nx ny')
shape = (nx, ny)

# Setup bounds so that the x-direction is partitioned, but do not partition in
# the y-direction
B = lambda n, b : Bounds(n, left=b, right=b)
bounds = (B(nx, bp), B(ny, 0))

u = GridFunction('u', shape=shape) 
out = GridFunction('out', shape=shape) 

rhs = Expr(u)
krhs = kg.kernelgenerator(language, shape, bounds, out, rhs, debug=False)

kernels = []
for rx in regions:
    reg = [1]*2
    reg[0] = rx
    regionnames = {0: 'left', 1: 'interior', -1: 'right'}
    kernel = krhs.kernel('%s'%regionnames[rx], tuple(reg))
    kernels.append(kernel)

for i, kernel in enumerate(kernels):
    print(kernel.code)
