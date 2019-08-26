from sympy import symbols
from openfd import GridFunction, sbp_staggered as sbp, GridFunctionExpression as GFE, kernel as kl
from sympy.core.cache import clear_cache

order        = 4
i, n, hi, dt = symbols('i n hi dt')
rhoi, K      = symbols('rhoi K')

v            = GridFunction("v", shape=(n+1, ))
p            = GridFunction("p", shape=(n+2, ))

v_x          = sbp.Derivative(v, "x", order=order, shape=(n+2,), gridspacing=1/hi, hat = True)
p_x          = sbp.Derivative(p, "x", order=order, shape=(n+1,), gridspacing=1/hi, hat = False)

v_t          = GFE(-rhoi*p_x  ) 
p_t          = GFE(-K*v_x  ) 

rhsv         = GFE(v + dt*v_t)
rhsp         = GFE(p + dt*p_t)

# Velocity update
# 0: left, -1: right, 1: interior
vkl = ''
for i in [0, 1, -1]:
    vkl += kl.kernel1d(p_x.bounds(), i, v, rhsv) 

# Pressure update
pkl = ''
for i in [0, 1, -1]:
    pkl += kl.kernel1d(v_x.bounds(), i, p, rhsp) 

# Output to C
code = ''
code += kl.ckernel("update_velocity", n, v, rhsv, vkl)
code += kl.ckernel("update_pressure", n, p, rhsp, pkl)
print(code)

# Boundary conditions
h = symbols('h')
phat  = symbols('phat')
g0 = sbp.Restriction(GFE(rhoi*(p - phat)),"x", 0, 0, shape=(n+1,))
gn = sbp.Restriction(GFE(rhoi*(p - phat)),"x", -1, -1, shape=(n+1,))
order=4
b0 = sbp.Quadrature(g0, "x", order=order, gridspacing=1/hi, hat=True, shape=(n+1,), invert=True)
bn = sbp.Quadrature(gn, "x", order=order, gridspacing=1/hi, hat=False, shape=(n+1,), invert=True)

expr = GFE(v + dt*(b0 - bn))
vkl = ''
for i in [0, -1]:
    vkl += kl.kernel1d(g0.bounds(), i, v, expr)
code = ''
code += kl.ckernel("update_velocity_bc", n, v, rhsv[0], vkl)
print(code)
