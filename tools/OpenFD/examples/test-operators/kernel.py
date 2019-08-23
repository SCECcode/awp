"""
This example demonstrates an experimental version of the staggered SBP operators
that is particularly useful in actual codes. The usual operators are defined
with respect to two grids:

    x = h*[0, 1, 2, .., n-1] 
    xh = h*[0, 0.5, 1.5, ... , n - 1 - 0.5, n - 1]

    The scaling parameter `h` is the grid spacing.

The first grid `x` has `n` grid points whereas the second grid `xh` has `n+1`
grid points. 

Due to the different number of grid points in the two grids, it is difficult to
compute on them both in the same compute kernel. To remedy the
situation, we make a slight modification that introduces an additional grid
point in the first grid. That is,

    x = h*[0, 1, 2, .., n - 1, n]

The value placed in the last position does not matter because it will not be
used in practice. The presence of the extra point requires us to introduce an
extra derivative (or what whatever the operator is) stencil associated with this
point. Since we do not care about the solution here, we simply introduce a
stencil that has all of its coefficients set to zero. 

We then make the following partitioning of the two grids:

 x    o-----o-----o-|---o-----o--|---o----o-----o---*
                    |            |     
                    |            |     
 xh   o--o-----o----|o-----o-----|^----o-----o--o
                    |            |     
         Left           Interior           Right 


    * : Extra point in which no computation is performed
    ^ : Interior point moved to boundary computation

As we can see in the Figure, the same number of points in each of the three
regions are updated for both grids. For example, in the right region, we update
four points belonging to the `x` grid. Although, the last point is the extra
point that is never used. In the same region, four points are also updated for
the `xh` grid. Since the number of points is the same, the computation can take
place in the same compute kernel, meaning that the it can take place within the
same loop body.

Perhaps it seems a little strange that on the left side there are only three
points being updated per grid, whereas on the right side there are four points
being updated per grid. The reason for this discrepancy has to do with the fact
that the initial operator defined on the `x` grid has three boundary points per
side. Therefore, we must pack four points into the right boundary region
(accounting for the extra point). To ensure that the same number of points are
updated for both grids, we simply move one of the interior points originally
belonging to the interior of the `xh` grid into the right boundary region.

Another partitioning is used when the cell-centered grid does not contain any
boundary points. This setup is illustrated in the following figure.


 x    o-----o-----o---|-o-----o-----|^----o-----o-----o
                      |             |  
                      |             |  
 xh   ---o-----o----o-|----o-----o--|--o-----o-----o--*
                      |             |  
         Left           Interior           Right 

    * : Extra point in which no computation is performed
    ^ : Interior point moved to boundary computation

Hence, this partioning is the same as the previous one but with the roles
reversed.

"""

from sympy import symbols
from openfd import GridFunction, GridFunctionExpression as Expr
from openfd.dev import kernelgenerator as kg
from openfd import sbp_traditional as sbp

# Set debug to True to replace expressions to compute with region id numbers.
debug = False
language = 'C'
order = 4
sourcefile = 'kernel.c'
headerfile = 'kernel.h'
# Operator files are selected using a string formatting
# the first argument is used to read operators for the `hat` grid or not,
# the second argument is used to select the order. 
# The operators are currently searched for in the directory:
# 'openfd/sbp/resources', but expect this to change in the near future.
fmtold = lambda symbol : 'staggered_cmpl_' + symbol + '%s.json'
fmth = 'staggered_cmpl_Dhat%s.json'
fmt = lambda symbol : 'cidsg/' + symbol + '%s.json'
fmtd = lambda symbol : 'bndopt/' + symbol + '%s.json'
#fmth = 'cidsg/Dhat%s.json'
# Number of grid points and inverse of grid spacing
n, hi = symbols('n hi')
shape = (n,)
# Set to `1` if boundary points are included in the cell-centered grid
incl_bnd_pts = 0
# Regions to generate kernel functions for 
regions = (-1, 0, 1)

# Input gridfunctions
u = GridFunction('u', shape=shape)
uh = GridFunction('uh', shape=shape)

# Output gridfunctions
v = GridFunction('v', shape=shape)
vh = GridFunction('vh', shape=shape)

D = sbp.Derivative('', 'x', shape=shape, order=order, 
                   fmt=fmtd('D'), gpu=True, coef='d')
Dh = sbp.Derivative('', 'x', shape=shape, order=order, 
                   fmt=fmtd('Dhat'), gpu=True, coef='dh')

#FIXME: It is confusing that we use the derivative class to construct an
# interpolation operator. This trick will only work if the right boundary data
# is predefined. If it is not defined, then it will be obtained from the left
# boundary and multiplying this data by -1. This multiplication is consistent
# with the behavior of a derivative operator.
P = sbp.Derivative('', 'x', shape=shape, order=order, 
                   fmt=fmt('P'), gpu=True, coef='p')
Ph = sbp.Derivative('', 'x', shape=shape, order=order, 
                   fmt=fmt('Phat'), gpu=True, coef='ph')

# Derivative expressions
expr1 = Expr(hi*D*uh)
expr2 = Expr(hi*Dh*u)

diff_lhs = [v, vh]
diff_rhs = [expr1, expr2]

# Interpolation expressions
expr1 = Expr(P*uh)
expr2 = Expr(Ph*u)

interp_lhs = [v, vh]
interp_rhs = [expr1, expr2]

# Get the bounds that define the shape of the left, right, and interior compute
# regions. The bounds for the two operators are the same so it does not matter
# which one we choose the request the bounds from.
bounds = (D.bounds(), )

diff = kg.kernelgenerator(language, shape, bounds, diff_lhs, diff_rhs, 
                                 debug=debug)

interp = kg.kernelgenerator(language, shape, bounds, 
                                   interp_lhs, interp_rhs, debug=debug)

kernels = []
regionnames = {0: 'left', 1: 'interior', -1: 'right'}
for rx in regions:
    kernel = diff.kernel('diff_%s'%regionnames[rx], (rx,))
    kernels.append(kernel)
for rx in regions:
    kernel = interp.kernel('interp_%s'%regionnames[rx], (rx,))
    kernels.append(kernel)

fh = open(headerfile, 'w')
fh.write('//Auto-generated by OpenFD\n')
fh.write('#ifndef KERNEL_H\n')
fh.write('#define KERNEL_H\n')
fh.write('#define BP_LEFT %d\n'% bounds[0].left)
fh.write('#define BP_RIGHT %d\n'% bounds[0].right)
fh.write('#define INCL_BND_PTS %d\n'% incl_bnd_pts)
for kernel in kernels:
    fh.write(kernel.header)
fh.write('#endif //KERNEL_H\n')
fh.close()

fh = open(sourcefile, 'w')
fh.write('//Auto-generated by OpenFD\n')
fh.write('#include "%s"\n\n'%headerfile)
for kernel in kernels:
    fh.write(kernel.code)
fh.close()

