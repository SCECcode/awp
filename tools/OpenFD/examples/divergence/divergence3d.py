from sympy import symbols
from openfd import GridFunction, sbp_traditional as sbp, kernel as kl, GridFunctionExpression as GFE
# Computes the divergence of a vector field F = (u, v, w) and generates source code
# for the each region of a box (corners, sides, and interior)

nx, ny, nz = symbols('nx ny nyz')
dim = (nx+1, ny+1, nz+1)
order = 4
u = GridFunction('u', shape=dim)
v = GridFunction('v', shape=dim)
w = GridFunction('w', shape=dim)
G = GridFunction('G', shape=dim)
u_x = sbp.Derivative(u, 'x', shape=dim, order=order)
v_y = sbp.Derivative(v, 'y', shape=dim, order=order)
w_z = sbp.Derivative(w, 'z', shape=dim, order=order)
div = GFE(u_x + v_y + w_z)
b = (u_x.bounds(), v_y.bounds(), w_z.bounds())
code = ''
for i in [-1,1,0]:
    for j in [-1,1,0]:
        for k in [-1,1,0]:
            code += kl.kernel3d(b, (i, j, k), v, div)
print(kl.ckernel("divergence", (nx, ny, nz), G, div, code))
