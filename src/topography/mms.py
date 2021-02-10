import sympy as sp


rho, cp, cs = sp.symbols("rho cp cs")
#rho0, cp0, cs0 = sp.symbols("rho0 cp0 cs0")
#drho, dcp, dcs = sp.symbols("drho dcp dcs")

vx0, vy0, vz0 = sp.symbols("vx0 vy0 vz0")
xx0, yy0, zz0 = sp.symbols("xx0 yy0 zz0")
xy0, xz0, yz0 = sp.symbols("xy0 xz0 yz0")
dvx, dvy, dvz = sp.symbols("dvx dvy dvz")
dxx, dyy, dzz = sp.symbols("dxx dyy dzz")
dxy, dxz, dyz = sp.symbols("dxy dxz dyz")

x, y, z, t, kx, ky, kz, om_p, om_c, dt = sp.symbols("x y z t kx ky kz om_p om_c dt")
lam, mu = sp.symbols("lam mu")

S = sp.sin(kx * x) * sp.sin(ky * y) * sp.sin(kz * z)

#rho = rho0# + drho * S
#cs = cs0# + dcs * S
#cp = cp0# + dcp * S

Vp = sp.sin(kz * z) * sp.sin(om_p * t)
vx = 0
vy = 0
vz = vz0 + Vp * dvz

xx = 0
yy = 0
zz = zz0 + Vp * dzz
xy = 0
xz = 0
yz = 0



vx_t = sp.diff(vx, t)
vy_t = sp.diff(vy, t)
vz_t = sp.diff(vz, t)

vx_x = sp.diff(vx, x)
vy_x = sp.diff(vy, x)
vz_x = sp.diff(vz, x)

vx_y = sp.diff(vx, y)
vy_y = sp.diff(vy, y)
vz_y = sp.diff(vz, y)

vx_z = sp.diff(vx, z)
vy_z = sp.diff(vy, z)
vz_z = sp.diff(vz, z)

xx_x = sp.diff(xx, x)
yy_x = sp.diff(yy, x)
zz_x = sp.diff(zz, x)

xx_y = sp.diff(xx, y)
yy_y = sp.diff(yy, y)
zz_y = sp.diff(zz, y)

xx_z = sp.diff(xx, z)
yy_z = sp.diff(yy, z)
zz_z = sp.diff(zz, z)

xx_t = sp.diff(xx, t)
yy_t = sp.diff(yy, t)
zz_t = sp.diff(zz, t)

xy_x = sp.diff(xy, x)
xz_x = sp.diff(xz, x)
yz_x = sp.diff(yz, x)

xy_y = sp.diff(xy, y)
xz_y = sp.diff(xz, y)
yz_y = sp.diff(yz, y)

xy_z = sp.diff(xy, z)
xz_z = sp.diff(xz, z)
yz_z = sp.diff(yz, z)

xy_t = sp.diff(xy, t)
xz_t = sp.diff(xz, t)
yz_t = sp.diff(yz, t)

f_vx = vx_t  - (xx_x + xy_y + xz_z) / rho
f_vy = vy_t  - (xy_x + yy_y + yz_z) / rho
f_vz = vz_t  - (xz_x + yz_y + zz_z) / rho

div = vx_x + vy_y + vz_z
f_zz = zz_t - (lam * div + 2 * mu * vz_z)
print("d_vz[pos] = ", sp.ccode(vz) + ";")
print("d_zz[pos] = ", sp.ccode(zz) + ";")

print("d_vz[pos] += ", sp.ccode(dt * f_vz) + ";")
print("d_zz[pos] += ", sp.ccode(dt * f_zz) + ";")
