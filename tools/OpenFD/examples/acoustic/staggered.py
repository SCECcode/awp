from openfd import GridFunction as GF, GridFunctionExpression as GFE, sbp_staggered as sbp
from openfd import kernel as kl, Bounds
from sympy import symbols, expand

order = 2
nx, ny, i, j, hi, dt = symbols('nx ny i j hi dt')
hi = 5
dt = 0.1
dimp = (nx+2, nx+2)
dimu = (nx+1, nx+2)
dimv = (nx+2, nx+1)

u = GF('u', shape=dimu)
v = GF('v', shape=dimv)
p = GF('p', shape=dimp)

ux = sbp.Derivative(u, 'x', order=order, hat=True, rgridspacing=hi, shape=dimp)
vy = sbp.Derivative(v, 'y', order=order, hat=True, rgridspacing=hi, shape=dimp)
px = sbp.Derivative(p, 'x', order=order, hat=False, rgridspacing=hi, shape=dimu)
py = sbp.Derivative(p, 'y', order=order, hat=False, rgridspacing=hi, shape=dimv)

pt = GFE(p - dt*(ux + vy) )
ut = GFE(u - dt*px)
vt = GFE(v - dt*py)

pbounds = (ux.bounds(), vy.bounds())
body = ''
for i in range(-1,2):
    for j in range(-1,2):
        body += kl.kernel2d(pbounds, (i, j), p, pt)
print(kl.ckernel('pressure_update', (nx, ny), p, pt, body)  )

ubounds = (px.bounds(), Bounds(size=ny+2))
body = ''
j = 1
for i in range(-1,2):
        body += kl.kernel2d(ubounds, (i, j), u, ut)

vbounds = (Bounds(size=nx+2), py.bounds())
i = 1
for j in range(-1,2):
        body += kl.kernel2d(vbounds, (i, j), v, vt)
print(kl.ckernel('velocity_update', (nx, ny), (u, v), (ut, vt), body) )
