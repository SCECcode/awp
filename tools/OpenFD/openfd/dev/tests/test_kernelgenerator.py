import pytest
import numpy as np
import os
from ... import Bounds, Grid, GridFunction, GridFunctionExpression as Expr, Operator, sbp_traditional as sbp
from ... import sbp_staggered as sbps
from .. variable import Variable
from .. array import CArray
from ..kernelgenerator import kernelgenerator, CGenerator, CudaGenerator, OpenclGenerator
from ..kernelgenerator import make_kernel, write_kernels
from . helper import make_multid
from sympy import symbols
import numpy as np

@pytest.fixture(scope="module", params=["C", "Cuda", "Opencl"])
def generator(request):
    if request.param == "C":
        yield CGenerator
    if request.param == "Cuda":
        yield CudaGenerator
    if request.param == "Opencl":
        yield OpenclGenerator

def build(kg, left=None, right=None, regions=None):
    names = {-1: 'r', 0: 'l', 1: 'i'}
    kernelname = lambda x: 'test_%s' % x
    kernels = []
    if not regions:
        regions = [-1, 0, 1]
    regx = regions
    regy = regions
    for rx in regx:
        for ry in regy:
            kernel = kg.kernel(kernelname(names[rx] + names[ry]), (rx, ry),
                               left_index_mapping=left, right_index_mapping=right)
            kernels.append(kernel)
    return kernels

def test_multiline(generator):
    nx, ny = symbols('nx ny')
    dim = (nx + 1, ny + 1)
    u = GridFunction("u", shape=dim)
    out1 = GridFunction("out1", shape=dim)
    out2 = GridFunction("out2", shape=dim)
    ux = sbp.Derivative(u, "x", order=4, gpu=True)
    uy = sbp.Derivative(u, "y", order=4, gpu=True)
    u_x = Variable('u_x', ux, dtype=np.float32)
    u_y = Variable('u_y', uy, dtype=np.float32)

    # No output arguments error
    lhs = u_x.declare()
    rhs = Expr(ux + uy)
    kg = generator((nx, ny), (ux.bounds(), uy.bounds()), lhs, rhs)
    with pytest.raises(ValueError) : kernels = build(kg)
    

    # Multiline, single output: out1
    lhs = [u_x.declare(), u_y.declare(), out1]
    rhs = [u_x.val, u_y.val, Expr(u_x + u_y)]
    kg = generator((nx, ny), (ux.bounds(), uy.bounds()), lhs, rhs)
    kernels = build(kg)

    # Multiline, dual output: out1, out2
    lhs = [u_x.declare(), u_y.declare(), out1, out2]
    rhs = [u_x.val, u_y.val, Expr(u_x + u_y), Expr(u_x - u_y)]
    kg = generator((nx, ny), (ux.bounds(), uy.bounds()), lhs, rhs)
    kernels = build(kg)

def test_duplicate(generator):
    nx, ny = symbols('nx ny')
    dim = (nx + 1, ny + 1)
    u = GridFunction("u", shape=dim)
    v = GridFunction("v", shape=dim)
    out1 = GridFunction("out1", shape=dim)
    # Give warning because data arryas have the same name but
    # reference different objects
    ux = sbp.Derivative(u, "x", order=4, gpu=True)
    vx = sbp.Derivative(v, "x", order=2, gpu=True)
    lhs = out1
    rhs = Expr(ux + vx)
    kg = generator((nx, ny), (ux.bounds(), vx.bounds()), lhs, rhs)
    with pytest.warns(UserWarning) : build(kg)

def test_sorted(generator=CGenerator, ntrials=10):
    """
    Tests that the function arguments are always ordered the same way.

    Example output produced by bug found in commit:
    f2bb19081f66061839201145ec79fa66f86b72e0
    (and earlier)

    void test(float *out2, float *out1,  float c, const float *v,  float b,  int n);

    void test(float *out1, float *out2,  float b,  float c, const float *v,  int n);

    """
    a, b, c, d, n  = symbols('a b c d n')
    gridsize = (n, n)
    bounds = (Bounds(n, 0, 0), Bounds(n, 0, 0))
    dim = (10, 10)
    u = GridFunction("u", shape=dim)
    v = GridFunction("v", shape=dim)
    w = GridFunction("w", shape=dim)
    out1 = GridFunction("out1", shape=dim)
    out2 = GridFunction("out2", shape=dim)
    dout = (out1, out2)
    din = (Expr(b + c), Expr(v))
    const = []
    extrain = [w]
    extraconst = [d]
    indices = []
    loop_order = (1, 0)
    debug = False

    kg = generator(gridsize, bounds, dout, din, const, extrain,
            extraconst, indices, debug, loop_order)
    kernel = kg.kernel('test', (0, 0))
    # Regenerate kernel and make sure that output is the same each time
    for i in range(ntrials):
        kg = generator(gridsize, bounds, dout, din, const, extrain, [],
                extraconst, indices, debug, loop_order)
        kernel = kg.kernel('test', (0, 0))
        code = kernel.header
        expected = 'void test(float *out1, float *out2, const float *v,'\
                   ' const float *w, const float b, const float c,'\
                   ' const float d, const int n);\n'
        assert code == expected

def test_gridsize(generator=CGenerator):

    gridsize = (1, 1)
    n  = symbols('n')
    dim = (10, 10)
    bounds = (Bounds(n, 0, 0), Bounds(n, 0, 0))
    din = GridFunction("in", shape=dim)
    dout = GridFunction("out", shape=dim)
    with pytest.raises(TypeError) : generator(gridsize, 
                                                    bounds, dout, din
                                                    )

    # Discard duplicates
    gridsize = (n, n)
    kg = generator(gridsize, bounds, dout, din)
    kernel = kg.kernel('test', (0, 0))
    assert kernel.header == 'void test(float *out, const float *in, '\
                            'const int n);\n'
        
def test_index_mapping(generator):
    from sympy import ccode
    nx, ny = symbols('nx ny')
    dim = (nx + 1, ny + 1)
    u = GridFunction("u", shape=dim)
    v = GridFunction("v", shape=dim)
    a, b, c = symbols('a b c')
    idx = CArray('idx', data=np.array([[0, 1, 2], [2, 2, 2]]))
    idy = CArray('idy', data=np.array([[2, 1, 2], [1, 1, 1]]))
    lmap = lambda x : (idx[x[0],1], idy[x[1],1], x[2])
    rmap = lambda x : (x[0], x[1], x[2])
    ux = sbp.Derivative(u, "x", order=4, gpu=True)
    lhs = u
    rhs = Expr(v)

    kg = generator((nx, ny), (ux.bounds(), ux.bounds()), lhs, rhs)
    kernels = build(kg, lmap, rmap)

    # Use GridFunction instead of array
    # In this case, idx, idy should show up in the argument list
    idx = GridFunction('idx', shape=dim)
    idy = GridFunction('idy', shape=dim)
    lmap = lambda x : (idx[x[0], 0], 0, x[2])
    rmap = lambda x : (idy[x[0], 0], 0, x[2])
    
    kg = generator((nx, ny), (ux.bounds(), ux.bounds()), lhs, rhs)
    kernels = build(kg, lmap, rmap, regions=[0])

def test_loop_bounds(generator):

    #FIXME: Add index bounds to CGenerator
    if generator == CGenerator:
        return

    nx, ny = symbols('nx ny')
    dim = (nx + 1, ny + 1)
    u = GridFunction("u", shape=dim)
    out1 = GridFunction("out1", shape=dim)
    ux = sbp.Derivative(u, "x", order=4, gpu=True)
    uy = sbp.Derivative(u, "y", order=4, gpu=True)

    lhs = out1
    rhs = Expr(ux)
    kg = generator((nx, ny), (ux.bounds(), uy.bounds()), lhs, rhs, 
                   index_bounds=(1,1))
    kernels = build(kg)
    assert 'bi' in kernels[0].header
    assert 'i >= ei' in kernels[0].code


def test_header(generator=CGenerator):
    gridsize=(1, 1)
    a, b, c, d, n  = symbols('a b c d n')
    gridsize = (n, n)
    bounds = (Bounds(n, 0, 0), Bounds(n, 0, 0))
    dim = (10, 10)
    u = GridFunction("u", shape=dim)
    dout = u
    din = Expr(b)
    kg = generator(gridsize, bounds, dout, din)
    kernel = kg.kernel('test', (0, 0))
    assert kernel.header == 'void test(float *u, const float b, const int n);\n'

def test_add_kernel(generator):
    gridsize = (1, 1)
    n  = symbols('n')
    dim = (10, )
    u = GridFunction("u", shape=dim)
    v = GridFunction("v", shape=dim)
    bounds = (Bounds(n, 0, 0), Bounds(n, 0, 0))
    a = make_kernel('test', u, v, bounds[0], (n,))
    assert a[0].name == 'test'                   
    a = make_kernel('test', u, v, bounds[0], (n,), regions=(0, 1))
    assert a[0].name == 'test_0'
    assert a[1].name == 'test_1'
    a = make_kernel('test', u, v, bounds[0], (n,), regions=0)
    assert a[0].name == 'test_0'                 
    a = make_kernel('test', u, v, bounds[0], (n,), regions=1)
    assert a[0].name == 'test_1'                 
    a = make_kernel('test', u, v, bounds[0], (n,), regions=2)
    assert a[0].name == 'test_2'                 
    a = make_kernel('test', u, v, bounds[0], (n,), regions=-1)
    assert a[0].name == 'test_2'
    
    a = make_multid(1)
    assert a[0].name == 'test_0'
    assert a[1].name == 'test_1'
    
    a = make_multid(2)
    assert a[0].name == 'test_00'
    assert a[1].name == 'test_01'

    a = make_multid(3)
    assert a[0].name == 'test_000'
    assert a[1].name == 'test_001'

    with pytest.raises(NotImplementedError) : make_multid(4)

def test_precision(generator):
    gridsize = (1, 1)
    n  = symbols('n')
    dim = (10, )
    u = GridFunction("u", shape=dim, dtype=np.float64)
    v = GridFunction("v", shape=dim, dtype=np.float64)
    bounds = Bounds(n)
    a = make_kernel('test', u, v, bounds, (n,))
    assert 'double' in a[0].header
    u = GridFunction("u", shape=dim, dtype=np.float32)
    v = GridFunction("v", shape=dim, dtype=np.float32)
    a = make_kernel('test', u, v, bounds, (n,))
    assert 'float' in a[0].header

def test_write_kernel():

    n = symbols('n')
    shape = (n,)
    dout = GridFunction('out', shape)
    din = GridFunction('in', shape)
    
    # write source
    kl = make_kernel('test', dout, din, Bounds(n), shape)
    write_kernels('test', kl)
    os.remove('test.c')

    # Also write header
    write_kernels('test', kl, header=True)
    os.remove('test.c')

    # Add include statement
    write_kernels('test', kl, header=True, 
                      header_includes='//test',
                      source_includes='#include <assert.h>')
    os.remove('test.c')

    # Give warning if using mixed language
    kl = make_kernel('test_c', dout, din, Bounds(n), shape, 
                generator=CGenerator)

    kl += make_kernel('test_cuda', dout, din, Bounds(n), shape, 
                generator=CudaGenerator)
    kl += make_kernel('test_opencl', dout, din, Bounds(n), shape, 
                generator=OpenclGenerator)
    with pytest.warns(UserWarning) : write_kernels('test', kl, header=True)
    os.remove('test.c')

