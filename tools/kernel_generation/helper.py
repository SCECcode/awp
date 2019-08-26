import numpy as np
import sympy as sp
import openfd as fd
from openfd.dev.kernelgenerator import CudaGenerator, CGenerator

 
def file_prepend(filename, prepend):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(prepend + content)

def set_precision(prec_str):
    if prec_str == 'double':
        fd.prec = np.float64
    elif prec_str == 'float':
        fd.prec = np.float32
    else:
        raise ValueError('Unknown precision: %s' % prec_str)

generator = CudaGenerator
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
    
def get_exclude_left(debug):
    """
    Remove one-sided stencils at the left (bottom boundary in z-direction) when
    code is running in in production mode
    """
    if not debug:       
        exclude_left=0  
    else:               
        exclude_left=0  
    return exclude_left

def get_use_sponge_layer(debug):
    """
    Enable the sponge layer when running in debug mode.
    """
    if not debug:
        use_sponge_layer = 1
    else:
        use_sponge_layer = 0
    return use_sponge_layer

def get_use_free_surface_bc(debug):
    """
    Disable enforcement of the free surface boundary condition when running in
    debug mode.
    """
    if debug:
        bnd_right = 0
    else:
        bnd_right = 1
    return bnd_right


alignment = (align, ngsl + 2, ngsl + 2)

memxy = fd.Memory((memsize[0], memsize[1] + 2*align), 
                   align=(align + alignment[1], alignment[1]), perm=(1,0))
memx = fd.Memory((memsize[0],), align=(alignment[2],))
memy = fd.Memory((memsize[1],), align=(alignment[1],))
memz = fd.Memory((memsize[2],), align=(alignment[0],))
membuf = fd.Memory((memsize[0], ngsl, memsize[2]), align=(alignment[0], 0,
                    alignment[2]), perm=(2,1,0))



def memory(memsize):
    from openfd import Memory
    perm = (2,1,0)
    #FIXME: alignments are not permuted
    mem = Memory(memsize, align=alignment, perm=perm)
    syms = (ngsl, align)
    return mem, syms

def memoryz(memsize):
    from openfd import Memory
    mem = Memory(memsize, align=(align,), perm=perm)
    return mem

def D(expr, axis, hat=0, order=4, ops='build3_all', bnd_left=0, bnd_right=0,
        interp=None):
    if bnd_left:
        raise NotImplementedError("No support for boundary conditions on the"\
                                  " left boundary")
    D = operator('D', expr, axis, hat=hat, order=order,
            ops=ops, gpu=True, bc=bnd_right, interp=interp)
   
    # Enforce the free surface boundary condition in an energy conserving manner
    # by manually adjusting the boundary coefficients
    #if bnd_left:
    #    Hi = operator('Hi', expr, axis, hat=hat, order=order,
    #         ops=ops, gpu=True)
    #    D._coef['left'].data[0,0] += Hi._coef['left'].data[0,0]

    #if bnd_right:
    #    D = operator('D', expr, axis, hat=hat, order=order,
    #            ops=ops, gpu=True)
    #    Hi = operator('Hi', expr, axis, hat=hat, order=order,
    #         ops=ops, gpu=True)
    #    if hat:
    #        # Qh + Q = 1
    #        # Qh + Q + SAT = 0
    #        # (Qh + SAT) + Q = 0
    #        D._coef['right'].data[0,1] = \
    #            D._coef['right'].data[0,1] - Hi._coef['left'].data[0,0]
    #    else:
    #        D._coef['right'].data[1,0] = \
    #            D._coef['right'].data[1,0] - Hi._coef['left'].data[0,0]

    return D


def P(expr, axis, hat=0, order=4, ops='build3_all'):
    return operator('P', expr, axis, hat=hat, order=order,
            ops=ops, gpu=True)

def Pavg(expr, axis, hat=0, order=2, ops='build3_all'):
    return P(expr, axis, hat, order, ops)
    
def H(expr, axis, hat=0, order=4, ops='build3_all'):
    op =  operator('H', expr, axis, hat=hat, order=order,
            ops=ops, gpu=True)
    op._operator_data['idx_right'] = 0 * op._operator_data['idx_right'][0,:].T
    op._operator_data['idx_left'] = 0 * op._operator_data['idx_left'][0,:].T
    op._coef['right'].data = op._operator_data['op_right'][::-1]
    op.diagonal = True
    return op

def bounds():
    return (fd.Bounds(size[0]),
            fd.Bounds(size[1]),
            fd.Bounds(size[2]))

def fbounds():
    return (fd.Bounds(metric_size[0]),
            fd.Bounds(metric_size[1]),
            fd.Bounds(1))

def operator(op, expr, axis, order=2, hat=False, gpu=True, shape=None, 
        ops='build3', bc=False, interp=None):
    """
    Construct a staggered grid operator.

    Arguments:
        op : Operator to construct. `'D'` for derivative. `'H'` for norm
        expr : Expression to apply the derivative to
        order(optional) : Order of accuracy
        hat(optional) : Set to `True` if the derivatives are computed on the
            hat-grid (cell-centered grid).
        gpu : Generate code for the GPU. Defaults to `True`.

    """
    from openfd import sbp_traditional as sbp
    from openfd import sbp_staggered as ssbp
    #FIXME: Update this statement once proper support for the uniform grid
    # staggered operators have been implemented
    hatstr = ['p', 'm']
    hstr = ['', 'h']
    if bc:
        bcstr = "_bc"
    else:
        bcstr = ""

    if interp is "last":
        op_str = "P%s%s%s" % (hatstr[hat], op, hatstr[not hat])
        coef_str = "P%s%s%s" % (hstr[hat], op, hstr[not hat])
    elif interp is "first":
        op_str = "%s%sP%s" % (op, hatstr[hat], hatstr[not hat])
        coef_str = "%s%sP%s" % (op, hstr[hat], hstr[not hat])
    else:
        op_str = "%s%s" % (op, hatstr[hat])
        coef_str = "%s%s" % (op, hstr[hat])
    fmt = '%s/%s%s' % (ops, op_str, bcstr) + '%s.json'

    coef = '%s%s%d'  % (coef_str.lower(), axis, order)
    coef = '%s%d'  % (coef_str.lower(), order)
    return sbp.Derivative(expr, axis, order=order, fmt=fmt,
                              gpu=gpu, coef=coef, shape=shape)

def shifts():
    """
     Flags that indicate if the grid is shifted in the particular direction or
     not
     Example v = (1, 0) 
     implies that v = v(x_i+1/2, y_j)
    """
    from openfd import Struct
    G = Struct()
    G.u1 = (1, 1, 1)
    G.v1 = (0, 0, 1)
    G.w1 = (0, 1, 0)
    G.xx = (0, 1, 1)
    G.yy = (0, 1, 1)
    G.zz = (0, 1, 1)
    G.xy = (1, 0, 1)
    G.xz = (1, 1, 0)
    G.yz = (0, 0, 0)

    G.u1 = (1, 1, 1)
    G.u2 = (0, 0, 1)
    G.u3 = (0, 1, 0)
    G.s11 = (0, 1, 1)
    G.s22 = (0, 1, 1)
    G.s33 = (0, 1, 1)
    G.s12 = (1, 0, 1)
    G.s13 = (1, 1, 0)
    G.s23 = (0, 0, 0)
    G.node = (1, 0, 0)

    # Shifts chosen for AWP compatibility
    #G.u1 = (0, 0, 0)
    #G.v1 = (1, 1, 0)
    #G.w1 = (1, 0, 1)
    #G.s11 = (1, 0, 0)
    #G.s22 = (1, 0, 0)
    #G.s33 = (1, 0, 0)
    #G.s12 = (0, 1, 0)
    #G.s13 = (0, 0, 1)
    #G.s23 = (1, 1, 1)
    return G

def symbols(label, num): 
    return [sp.symbols(label +'%d'%i) for i in range(num)]

def fields(size, mem, fields=['u'], remap=None):
    v = [fd.GridFunction('%s'%fields[i], shape=size, layout=mem, macro='_',
        remap=remap) for i in
            range(len(fields))]
    return v

def fieldsxy(fields, init=False):
    if init:
        v = [fd.GridFunction('%s'%field, shape=(stress_size[0], stress_size[1]), dims=(0,1),
         layout=fd.Memory((size[0], size[1]), perm=(1,0)), macro='_') 
             for field in fields.split(' ')]
    else:
        v = [fd.GridFunction('%s'%field, shape=(stress_size[0], stress_size[1]), dims=(0,1),
         layout=memxy, macro='_') for field in fields.split(' ')]
    return v

def fieldsx(fields, init=False):
    if init:
        v = [fd.GridFunction('%s'%field, shape=(size[0]), dims=(0,), 
                         layout=fd.Memory((size[0],)), macro='_') 
                         for field in fields.split(' ')]
    else:
        v = [fd.GridFunction('%s'%field, shape=(size[0]), dims=(0,), 
                         layout=memx, macro='_') for field in fields.split(' ')]
    return v

def fieldsy(fields, init=False):
    if init:
        v = [fd.GridFunction('%s'%field, shape=(size[1]), dims=(1,), 
                         layout=fd.Memory((size[1],)), macro='_') 
                         for field in fields.split(' ')]
    else:
        v = [fd.GridFunction('%s'%field, shape=(size[1]), dims=(1,), 
                         layout=memy, macro='_') for field in fields.split(' ')]
    return v

def fieldsz(fields, init=False):
    if init:
        v = [fd.GridFunction('%s'%field, shape=(size[2]), dims=(2,), 
                         layout=fd.Memory((size[2],)), macro='_') 
                         for field in fields.split(' ')]
    else:
        v = [fd.GridFunction('%s'%field, shape=(size[2]), dims=(2,), 
                         layout=memz, macro='_') for field in fields.split(' ')]
    return v

def grids(label, sizes, layout=None):
    from grid import Grid
    x = [Grid(label, size, i, layout, interval=(0, size-1)) for
            i, size in enumerate(sizes)]
    return x

def velbounds(op, exclude_left=0):
    bndz = op.bounds()
    bnds = list(bounds())
    left = bndz.left
    if exclude_left:
        left = 1
    bnds[2] = fd.Bounds(bndz.size, left=left, right=bndz.right)
    return bnds

def strbounds(D, op, exclude_left=0):
    bndz = D.bounds()
    bnds =[0]*3
    left = bndz.left
    if exclude_left:
        left = 1
    bnds[0] = fd.Bounds(op.shape[0])
    bnds[1] = fd.Bounds(op.shape[1])
    bnds[2] = fd.Bounds(op.shape[2], left=left, right=bndz.right)
    return bnds




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

G = shifts()

