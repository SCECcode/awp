import pyawp
import sympy as sp

rho = 1
lam = 1
mu = 1
k = sp.symbols('k')
x = sp.symbols('x y z')
t = sp.symbols('t')

vx = sp.sin(k*x[0])*sp.sin(k*x[1])*sp.sin(k*x[2]) 
vy = sp.sin(k*x[0])*sp.sin(k*x[1])*sp.sin(k*x[2]) 
vz = sp.sin(k*x[0])*sp.sin(k*x[1])*sp.sin(k*x[2])

sxx = sp.sin(k*x[0])*sp.sin(k*x[1])*sp.sin(k*x[2])   
syy = sp.sin(k*x[0])*sp.sin(k*x[1])*sp.sin(k*x[2])   
szz = sp.sin(k*x[0])*sp.sin(k*x[1])*sp.sin(k*x[2])   
sxy = sp.sin(k*x[0])*sp.sin(k*x[1])*sp.sin(k*x[2])   
sxz = sp.sin(k*x[0])*sp.sin(k*x[1])*sp.sin(k*x[2])   
syz = sp.sin(k*x[0])*sp.sin(k*x[1])*sp.sin(k*x[2])   
                      
v = [vx, vy, vz]
s = sp.zeros(3)
s[0,0] = sxx
s[1,1] = syy
s[2,2] = szz
s[1,0] = sxy
s[0,1] = sxy
s[0,2] = sxz
s[2,0] = sxz
s[1,2] = syz
s[2,1] = syz

def mms(state, field, value):
    print("_prec mms_%s_%s(const _prec x, const _prec y, const _prec z, "\
          "const _prec *properties)"%(state, field))
    print("{")
    print("     _prec k = properties[0];");
    print("     return %s;" % value)
    print("}")
    print("")

null, v_rhs = pyawp.elastic.vel_cart(v, s, rho, x, t)
null, s_rhs = pyawp.elastic.stress_cart(v, s, lam, mu, x, t)
fields = ["vx", "vy", "vz", "sxx", "syy", "szz", "sxy", "sxz", "syz"]
values = [vx, vy, vz, sxx, syy, szz, sxy, sxz, syz]
final_values = [v_rhs[0], v_rhs[1], v_rhs[2], 
                s_rhs[0,0], s_rhs[1,1], s_rhs[2,2], 
                s_rhs[0,1], s_rhs[0,2], s_rhs[1,2]]

print('#include "mms.h"')
for field, value in zip(fields, values):
    mms("init", field, value)

for field, value in zip(fields, final_values):
    mms("final", field, value)
