from openfd import sbp_traditional as sbp, GridFunction as GF, GridFunctionExpression as GFE, kernel as kl
from sympy import symbols
order=4
nx,ny = symbols('nx ny')
dim = (nx + 1, ny + 1)
u=GF('u', shape=dim)
v=GF('v', shape=dim)
out=GF('out', shape=dim)
u_x = sbp.Derivative(u, 'x', order=order)
v_y = sbp.Derivative(v, 'y', order=order)
div = GFE(u_x + v_y)
b = (u_x.bounds(), v_y.bounds())
reg = [-1,0,1]
code = ''
for i in reg:
    for j in reg:
        code += kl.kernel2d(b, (i, j), out, div)
print(kl.ckernel('divergence2d', (nx, ny), out, div, code))
