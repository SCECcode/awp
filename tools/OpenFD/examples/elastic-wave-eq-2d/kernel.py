import numpy
import openfd
import pycuda.autoinit
from openfd.cuda import make_kernel, write_kernels, load_kernels, init, \
                        copy_htod, copy_dtoh
from openfd import gridfunctions, Bounds, Struct, Memory
from helper import operator, output_vtk, init_block2d, Writer, vtk_2d, \
                   norm_coef
from openfd import grids, CArray, sponge, GridFunctionExpression
from sympy import  symbols, exp
from helper import shifts
import fs2

prec = numpy.float32
openfd.prec = prec
openfd.debug=0

def generate(order=4, ops='bndopt', het=0):

    nx, ny, hi, dt = symbols('nx ny hi dt')
    shape = (nx, ny)
    
    perm =(0, 1)
    mem = Memory(shape, perm=perm)
    v1, v2 = gridfunctions('v1 v2', shape, layout=mem)
    s11, s22, s12 = gridfunctions('s11 s22 s12', shape, layout=mem)

    D = lambda expr, axis, hat : operator('D', expr, axis, hat=hat, order=order,
            ops=ops)

    G = shifts()
    diff = D(s11, 'x', G.v1[0]) 
    s11_x = hi*D(s11, 'x', G.v1[0])
    s12_y = hi*D(s12, 'y', G.v1[1])
    s12_x = hi*D(s12, 'x', G.v2[0])
    s22_y = hi*D(s22, 'y', G.v2[1])
    v1_x = hi*D(v1, 'x', G.snn[0]) 
    v2_y = hi*D(v2, 'y', G.snn[1]) 
    v1_y = hi*D(v1, 'y', G.s12[1]) 
    v2_x = hi*D(v2, 'x', G.s12[0]) 

    if het:
        rhoi_v1, rhoi_v2, lam, mu_11, mu_12 = gridfunctions('rhoi_v1 rhoi_v2'\
                                                            ' lam mu_11 mu_12', 
                                                            shape, layout=mem)
        v1_t  = rhoi_v1*(s11_x + s12_y)
        v2_t  = rhoi_v2*(s12_x + s22_y)
        s11_t = (lam + 2*mu_11)*v1_x + lam*v2_y
        s22_t = (lam + 2*mu_11)*v2_y + lam*v1_x 
        s12_t = mu_12*(v1_y + v2_x)
    else:
        rhoi, lam, mu = symbols('rhoi lam mu')
        v1_t  = rhoi*(s11_x + s12_y)
        v2_t  = rhoi*(s12_x + s22_y)
        s11_t = (lam + 2*mu)*v1_x + lam*v2_y
        s22_t = (lam + 2*mu)*v2_y + lam*v1_x 
        s12_t = mu*(v1_y + v2_x)

    kernels = []
    center = (1, 1)
    regions = (range(3), range(3))
    bounds = (diff.bounds(), diff.bounds())

    kernels += make_kernel('update_velocity', 
                           (v1, v2), (v1 + dt*v1_t, v2 + dt*v2_t), 
                           bounds, shape, regions=regions)
    kernels += make_kernel('update_stress', 
                           (s11, s22, s12), 
                           (s11 + dt*s11_t, s22 + dt*s22_t, s12 + dt*s12_t), 
                           bounds, shape, regions=regions, debug=0)

    if ops == 'bndopt':
        write_kernels('kernels/bndopt', kernels)
    else:
        if het:
            write_kernels('kernels/elastic_heterogeneous', kernels)
        else:
            write_kernels('kernels/elastic', kernels)



def source():
    """
    Moment tensor source time function to apply.

    Generates a source using the Ricker wavelet for 

    * M12 (double couple)

    Generates an explosive source using the Ricker for
    * M11, M22
    This source is used when solving Garvin's problem

    A : Amplitude
    Vi : 1/Volume element (area: h^2 in 2D)
    fp : Peak frequency of Ricker wavelet
    t0 : timeshift

    """
    nx, ny, dt, hi = symbols('nx ny dt hi')
    t0, fp = symbols('t0 fp')
    shape = (nx, ny)
    perm =(0, 1)
    mem = Memory(shape, perm=perm)
    s11, s22, s12, v1, v2, = gridfunctions('s11 s22 s12 v1 v2', shape, layout=mem)
    xs, ys = symbols('xs ys')
    M11, M12, M22 = symbols('M11 M12 M22')
    Fx, Fz, Vi, Ti, t, rhoi = symbols('Fx Fz Vi Ti t rhoi')


    # Ricker wavelet
    f = (1 - 2*numpy.pi**2*fp**2*(t-t0)**2)*exp(-numpy.pi**2*fp**2*(t-t0)**2)
    m11 = Vi*M11*f
    m12 = Vi*M12*f
    m22 = Vi*M22*f

    # Point force applied on the boundary
    hr00i = norm_coef(order=4, hat=0)*hi
    hh00i = norm_coef(order=4, hat=1)*hi

    Fsx = hi*hh00i*Fx*f
    Fsz = hi*hr00i*Fz*f

    # Point force applied on the boundary using BNDOPT
    coef = symbols('coef')
    bndopt_hr00i = 1.0#norm_coef(order=4, hat=0, ops='bndopt')*hi
    bndopt_Fsx = f*Fx
    bndopt_Fsz = hi*bndopt_hr00i*Fz*f

    # Point force applied in the interior
    Fvz = Vi*Fz*f
    Fvx = Vi*Fx*f

    source_bounds = (Bounds(1, 0, 0), Bounds(1, 0, 0))
    bndopt_bounds = (Bounds(1, 0, 0), Bounds(4, 0, 0))
    lhs_map = lambda i : (i[0] + xs, i[1] + ys)
    rhs_map = lambda i : lhs_map(i) 
    kernels = []
    kernels += make_kernel('src_moment_tensor', 
                           (s11, s22, s12), 
                           (s11 - dt*m11, 
                            s22 - dt*m22, 
                            s12 - dt*m12), 
                           source_bounds, shape, 
                           lhs_indices=lhs_map, rhs_indices=rhs_map)
    kernels += make_kernel('src_force_sat', 
                           (v1, v2), (v1 + dt*rhoi*Fsx, 
                                      v2 + dt*rhoi*Fsz), 
                           source_bounds, shape, 
                           lhs_indices=lhs_map, rhs_indices=rhs_map)

    kernels += make_kernel('src_force_bndopt_v1', 
                           v1, v1 + dt*rhoi*bndopt_Fsx, 
                           bndopt_bounds, shape, 
                           lhs_indices=lhs_map, rhs_indices=rhs_map)
    kernels += make_kernel('src_force_bndopt_v2', 
                           v2, v2 + dt*rhoi*bndopt_Fsz, 
                           source_bounds, shape, 
                           lhs_indices=lhs_map, rhs_indices=rhs_map)
    kernels += make_kernel('src_force', 
                           (v1, v2), (v1 + dt*rhoi*Fvx, 
                                      v2 + dt*rhoi*Fvz), 
                           source_bounds, shape, 
                           lhs_indices=lhs_map, rhs_indices=rhs_map)
    # Reciprocal norm coefficient for fs2 is `2`
    kernels += make_kernel('src_force_fs2', 
                           (v1, v2), (v1 + 2.0*dt*rhoi*Fvx, 
                                      v2 + 2.0*dt*rhoi*Fvz), 
                           source_bounds, shape, 
                           lhs_indices=lhs_map, rhs_indices=rhs_map)

    write_kernels('kernels/source', kernels)

def strain_green_tensor():
    """
    Strain Green Tensor (SGT) computation for performing reciprocity test

    e_ij  = a*sigma_ij  - b*sigma_kk delta_ij, 

    a = 1/(2*mu)
    b = lam/(2*mu*(2*lam + mu))

    Note that in 3D, `b` is given by
    
    b = lam/(2*mu*(3*lam + mu))

    """
    

    nx, ny = symbols('nx ny')
    shape = (nx, ny)
    perm =(0, 1)
    mem = Memory(shape, perm=perm)
    s11, s22, s12 = gridfunctions('s11 s22 s12', shape, layout=mem)
    e11, e22, e12 = gridfunctions('e11 e22 e12', shape, layout=mem)
    lam, mu = symbols('lam mu')


    a = 1/(2*mu)
    b = lam/(2*mu*(2*lam + mu))

    eq11 = a*s11 - b*(s11 + s22)
    eq12 = a*s12
    eq22 = a*s22 - b*(s11 + s22)

    bounds = (Bounds(1, 0, 0), Bounds(1, 0, 0))
    lhs_map = lambda i : (i[0], i[1])
    rhs_map = lambda i : (i[0] + xs, i[1] + ys)
    xs, ys = symbols('xs ys')
    kernels = make_kernel('sgt', 
                           (e11, e22, e12), (eq11, eq22, eq12), 
                           bounds, shape,
                           lhs_indices=lhs_map, rhs_indices=rhs_map)

    #FIXME: Should not have to hand-code this
    

    write_kernels('kernels/sgt', kernels)

def sat(ops='cidsg', het=0):
    """
    Weakly enforced boundary condition

        s12 = 0, s22 = 0

    These are enforced by adding penalty terms to the velocity equations.

    """
    from openfd import CudaGenerator
    from openfd.dev import boundary as bnd
    dt, rhoi, nx, ny, hi, zp, zs = symbols('dt rhoi nx ny hi zp zs')
    hr00i = norm_coef(order=4, hat=0, ops=ops)*hi
    hh00i = norm_coef(order=4, hat=1, ops=ops)*hi

    # Set fields to use their numerical grid sizes
    shape = (nx, ny)
    perm =(0, 1)
    regions = (0, 1)
    memshape = (nx + 1, ny + 1)
    mem = Memory(memshape, perm=perm)
    # Input and output fields should use their actual size
    v1, = gridfunctions('v1', (nx, ny+1), layout=mem)
    v2, = gridfunctions('v2', (nx+1, ny), layout=mem)
    s11, s22 = gridfunctions('s11 s22', (nx+1, ny + 1), layout=mem)
    s12, = gridfunctions('s12', (nx, ny), layout=mem)
    grid_bounds = (Bounds(memshape[0], 0, 0), Bounds(memshape[1], 0, 0))
    kernels = []

    # Free surface
    if het:
        rhoi_v1, = gridfunctions('rhoi_v1', (nx, ny+1), layout=mem)
        rhoi_v2, = gridfunctions('rhoi_v2', (nx+1, ny), layout=mem)
        fs1 = lambda n : v1 - dt*rhoi_v1*(n[0]*hr00i*s11  + n[1]*hh00i*s12)
        fs2 = lambda n : v2 - dt*rhoi_v2*(n[0]*hh00i*s12  + n[1]*hr00i*s22)
    else:
        rhoi = symbols('rhoi')
        fs1 = lambda n : v1 - dt*rhoi*(n[0]*hr00i*s11  + n[1]*hh00i*s12)
        fs2 = lambda n : v2 - dt*rhoi*(n[0]*hh00i*s12  + n[1]*hr00i*s22)

    # Used for debugging
    dbg1 = lambda n : v1 + (abs(n[0])*s11 + abs(n[1])*s12)
    dbg2 = lambda n : v2 + (abs(n[0])*s12 + abs(n[1])*s22)

    def boundary_conditions(n):
        return (fs1(n), fs2(n))

    # left
    n = (-1, 0)
    bounds = bnd.bounds(n, grid_bounds)
    kernels += make_kernel('bnd', 
                          (v1, v2), 
                          boundary_conditions(n),
                          bounds, shape, 
                          extraconst=[hi, dt, rhoi],
                          extrain=[s11, s12, s22],
                          regions=(0, 1), 
                          debug=0)
    # right
    n = (1, 0)
    bounds = bnd.bounds(n, grid_bounds)
    kernels += make_kernel('bnd', 
                          (v1, v2), 
                          boundary_conditions(n),
                          bounds, shape, 
                          extraconst=[hi, dt, rhoi],
                          extrain=[s11, s12, s22],
                          regions=(2, 1), 
                          debug=0)
    ## bottom
    n = (0, -1)
    bounds = bnd.bounds(n, grid_bounds)
    kernels += make_kernel('bnd', 
                          (v1, v2), 
                          boundary_conditions(n),
                          bounds, shape, 
                          extraconst=[hi, dt, rhoi],
                          extrain=[s11, s12, s22],
                          regions=(1, 0), debug=0)

    ## top
    n = (0, 1)
    bounds = bnd.bounds(n, grid_bounds)
    kernels += make_kernel('bnd', 
                          (v1, v2), 
                          boundary_conditions(n),
                          bounds, shape, 
                          extraconst=[hi, dt, rhoi],
                          extrain=[s11, s12, s22],
                          regions=(1, 2), debug=0)
    if het:
        write_kernels('kernels/sat_heterogeneous', kernels)
    else:
        write_kernels('kernels/sat', kernels)

def sat_bndopt(order=4, ops='bndopt'):

    from openfd import CudaGenerator
    from openfd.dev import boundary as bnd
    dt, rhoi, nx, ny, hi, zp, zs = symbols('dt rhoi nx ny hi zp zs')
    hr00i = norm_coef(order=4, hat=0, ops=ops)*hi
    hh00i = norm_coef(order=4, hat=1, ops=ops)*hi

    # Set fields to use their numerical grid sizes
    shape = (nx, ny)
    perm =(0, 1)
    regions = (0, 1)
    memshape = (nx + 1, ny + 1)
    mem = Memory(memshape, perm=perm)
    # Input and output fields should use their actual size
    v1, = gridfunctions('v1', (nx+1, ny), layout=mem)
    v2, = gridfunctions('v2', (nx, ny+1), layout=mem)
    s11, s22 = gridfunctions('s11 s22', (nx, ny), layout=mem)
    s12, = gridfunctions('s12', (nx + 1, ny + 1), layout=mem)

    OP = lambda op, expr, axis, hat : operator(op, expr, axis, hat=hat, order=order,
            ops=ops, shape=shape)

    G = shifts()
    L1 = OP('HiL', '', 'x', G.v2[0]) 
    L2 = OP('HiL', '', 'y', G.v1[1]) 
    LT1 = OP('HiLT', '', 'x', G.v1[0]) 
    LT2 = OP('HiLT', '', 'y', G.v2[1]) 

    fs1 = lambda n : v1 + dt*hi*rhoi*(n[0]*LT1*s11 + n[1]*L2*s12)
    fs2 = lambda n : v2 + dt*hi*rhoi*(n[0]*L1*s12  + n[1]*LT2*s22)

    kernels = []
    center = (1, 1)
    regions = (range(3), range(3))


    grid_bounds = (Bounds(memshape[0], 0, 0), Bounds(memshape[1], 0, 0))

    n = (0, -1)
    bounds = (L1.bounds(), L1.bounds())
    kernels += make_kernel('v1', 
                          (v1), 
                          (fs1(n)),
                          bounds, shape, 
                          extraconst=[hi, dt, rhoi, zp, zs],
                          extrain=[s11, s12, s22],
                          regions=(1, 0), debug=0)

    bounds = (L1.bounds(), Bounds(ny+1, left=1, right=0))
    kernels += make_kernel('v2', 
                          (v2), 
                          (fs2(n)),
                          bounds, shape, 
                          extraconst=[hi, dt, rhoi, zp, zs],
                          extrain=[s11, s12, s22],
                          regions=(1, 0), debug=0)


    write_kernels('kernels/bndopt_sat', kernels)

def sponge_kernel(b=32):
    nx, ny = symbols('nx ny')
    perm =(0, 1)
    shape = (nx, ny)
    mem = Memory(shape, perm=perm)
    v1, v2 = gridfunctions('v1 v2', shape, layout=mem)
    s11, s22, s12 = gridfunctions('s11 s22 s12', shape, layout=mem)

    bounds = (Bounds(shape[0], b, b), Bounds(shape[1], b, b))
    C = lambda expr : sponge.Cerjan(expr, bounds, gam=0.98) 
    regions = (range(3), range(3))
    kernels = make_kernel('sponge_vel', 
                          (v1, v2), 
                          (C(v1), C(v2)),
                          bounds, shape, 
                          regions=regions, debug=0)
    kernels += make_kernel('sponge_stress', 
                          (s11, s22, s12), 
                          (C(s11), C(s22), C(s12)),
                          bounds, shape, 
                          regions=regions, debug=0)
    write_kernels('kernels/sponge', kernels)

het = 1
generate(ops='cidsg', het=het)
#generate(ops='bndopt')
source()
sat(ops='cidsg', het=het)
#sat_bndopt()
#sponge_kernel(b=64)
#fs2.write()
#strain_green_tensor()
