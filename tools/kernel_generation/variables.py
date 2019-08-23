import openfd as fd
import sympy as sp
from helper import fields, fieldsx, fieldsy, fieldsz, fieldsxy, memory


threads = 32
nx, ny, nz = sp.symbols('nx ny nz')
size = [nx, ny, nz]
align, ngsl = sp.symbols('align ngsl')
stress_size = [nx + ngsl, ny + ngsl, nz]
# Operator radius. For fourth order operators, use radius = 2
radius = 2
metric_size = [nx + 2*ngsl - 2*radius, ny + 2*ngsl - 2*radius]
memsize = (size[0] + 4 + 2*ngsl, size[1] + 4 + 2*ngsl, 2*align + size[2])
gridsymbols = [nx, ny, nz]
ghost_cells = fd.Constant('ngsl')
rank = sp.symbols('rank')
order=4

alignment = (align, ngsl + 2, ngsl + 2)

memxy = fd.Memory((memsize[0], memsize[1] + 2*align), 
                   align=(align + alignment[1], alignment[1]), perm=(1,0))
memx = fd.Memory((memsize[0],), align=(alignment[2],))
memy = fd.Memory((memsize[1],), align=(alignment[1],))
memz = fd.Memory((memsize[2],), align=(alignment[0],))
membuf = fd.Memory((memsize[0], ngsl, memsize[2]), align=(alignment[0], 0,
                    alignment[2]), perm=(2,1,0))

mem, syms = memory(memsize)

rho, lami, mui = fields(size, mem, ['rho', 'lami', 'mui'])
# Silly hack to fill material parameters outside the compute region
mrho, mlami, mmui = fields(size, fd.Memory(size, perm=(2,1,0)), ['rho', 'lami', 'mui'])
u1, v1, w1 = fields(size, mem, ['u1', 'v1', 'w1'])
xx, yy, zz, xy, xz, yz = fields(stress_size, mem, ['xx', 'yy', 'zz',
                                            'xy', 'xz', 'yz'])
u1, u2, u3 = fields(size, mem, ['u1', 'u2', 'u3'])
s11, s22, s33, s12, s13, s23 = fields(stress_size, mem, ['s11', 's22', 's33',
                                    's12', 's13', 's23'])

# Sponge layer variables
dcrjx, = fieldsx('dcrjx')
dcrjy, = fieldsy('dcrjy')
dcrjz, = fieldsz('dcrjz')

buf_u1, buf_u2, buf_u3 = fields(size, membuf, ['buf_u1', 'buf_u2', 'buf_u3']) 
f, df1, df2 = fieldsxy('f df1 df2')
f_1, f_2, f_c = fieldsxy('f_1 f_2 f_c')
f1_1, f1_2, f1_c = fieldsxy('f1_1 f1_2 f1_c')
f2_1, f2_2, f2_c = fieldsxy('f2_1 f2_2 f2_c')

mf, mdf1, mdf2 = fieldsxy('f df1 df2', init=True)
mg, mdg = fieldsz('g dg', init=True)
g, dg = fieldsz('g dg')
g_c, = fieldsz('g_c')

g3, g3_c = fieldsz('g3 g3_c')
F = fd.Struct()
F.xx = xx
F.yy = yy
F.zz = zz
F.u1 = u1
F.v1 = v1
F.w1 = w1
F.xy = xy
F.xz = xz
F.yz = yz

# Metrics
F.g = g
F.g_c = g_c
F.dg = dg
F.g3 = g3
F.g3_c = g3_c
F.f = f
F.df1 = df1
F.df2 = df2

F.f_1 = f_1
F.f_2 = f_2
F.f_c = f_c
F.f1_1 = f1_1
F.f1_2 = f1_2
F.f1_c = f1_c
F.f2_1 = f2_1
F.f2_2 = f2_2
F.f2_c = f2_c

# Initialization of metrics
F.mf = mf
F.mdf1 = mdf1
F.mdf2 = mdf2
F.mg = mg
F.mdg = mdg

F.rho = rho
F.lami = lami
F.mui = mui
F.mrho = mrho
F.mlami = mlami
F.mmui = mmui
F.dcrjx = dcrjx
F.dcrjy = dcrjy
F.dcrjz = dcrjz
out, = fields(size, mem, ['out'])


Q = fd.Struct()
F.s11 = s11
F.s22 = s22
F.s33 = s33
F.u1 = u1
F.u2 = u2
F.u3 = u3
F.buf_u1 = buf_u1
F.buf_u2 = buf_u2
F.buf_u3 = buf_u3
F.s12 = s12
F.s13 = s13
F.s23 = s23


