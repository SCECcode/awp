from sympy import symbols
from openfd import GridFunction, sbp_traditional as sbp, GridFunctionExpression as GFE, kernel as kl

# Compute the divergence in 3D using central finite differences
nx, ny, nz = symbols('nx ny nz')
dim        = (nx+1, ny+1, nz+1)
order      = 2
u   = GridFunction("u", shape=dim)
u_x = sbp.Derivative(u, "x", order=order, shape=dim)
u_y = sbp.Derivative(u, "y", order=order, shape=dim)
u_z = sbp.Derivative(u, "z", order=order, shape=dim)

lhs = GridFunction("v", shape=dim)
rhs = GFE(u_x + u_y + u_z)
b = (u_x.bounds(), u_y.bounds(), u_z.bounds())
code = kl.kernel3d(b, (1, 1, 1), lhs, rhs) 
print(kl.ckernel("divergence", (nx, ny, nz), lhs, rhs, code))
