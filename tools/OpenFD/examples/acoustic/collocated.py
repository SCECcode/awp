from openfd import Bounds, GridFunction as GF, GridFunctionExpression as GFE, sbp_traditional as sbp
from openfd import kernel as kl
from sympy import symbols

order=2
dt=0.1
hi=5
i, j, nx, ny = symbols('i j nx ny')
dim = (nx, ny)

p = GF('p', shape=dim)
u = GF('u', shape=dim)
v = GF('v', shape=dim)
K = GF('K', shape=dim)

Dx = sbp.Derivative('', 'x', order=order, shape=dim)
Dy = sbp.Derivative('', 'x', order=order, shape=dim)

pt = GFE(p - K*dt*(Dx*u + Dy*v))
ut = GFE(u - dt*Dx*p)
vt = GFE(v - dt*Dy*p)

body = ''
for i in range(-1, 2):
    for j in range(-1, 2):
        body += kl.kernel2d((Dx.bounds(), Dy.bounds()), (i,j), p, pt)
print(kl.ckernel('pressure_update', (nx, ny), p, pt, body))

body = ''
bounds = (Dx.bounds(), Bounds(size=ny))
j = 1
for i in range(-1, 2):
        body += kl.kernel2d(bounds, (i,j), ut, u)

bounds = (Bounds(size=nx), Dy.bounds())
i = 1
for j in range(-1, 2):
        body += kl.kernel2d(bounds, (i,j), v, vt)
print(kl.ckernel('velocity_update', (nx, ny), v, vt, body))
