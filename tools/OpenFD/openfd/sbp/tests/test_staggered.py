from ... import Grid, StaggeredGrid, GridFunction, Operator, OperatorException, GridFunctionException,GridFunctionExpression
from ... import Polynomial
from ... import sbp_staggered as sbp
from sympy import symbols, simplify
from sympy.core.cache import clear_cache
import pytest

def test_infer_shape():
    n   = symbols('n')
    u   = GridFunction('u', shape=(n+2,))
    u_x = sbp.Derivative(u, "x", order=2)
    assert u_x.shape == (n+1,)
    u   = GridFunction('u', shape=(n+1,))
    u_x = sbp.Derivative(u, "x", order=2, hat=True)
    assert u_x.shape == (n+2,)

def test_derivative21sym():

    n   = symbols('n')
    u   = GridFunction('u', shape=(n+2,))
    u_x = sbp.Derivative(u, "x", order=2, shape=(n+1,))

    a0 = -2.13608682062236798770982204587199
    b0 = 2.20413023093355198156473306880798
    c0 = -0.06804341031118396609933540730708
    a1 = -0.15853599588482414350920635115472
    b1 = -0.76219600617276384024734170452575
    c1 = 0.92073200205758798375654805568047

    assert u_x[0] == a0*u[0] + b0*u[1] + c0*u[2]
    assert u_x[1] == a1*u[0] + b1*u[1] + c1*u[2]

    assert u_x[n] == -a0*u[n+1] - b0*u[n] - c0*u[n-1]
    assert u_x[-1] == -a0*u[n+1] - b0*u[n] - c0*u[n-1]
    
    assert u_x[n-1] == -a1*u[n+1] - b1*u[n] - c1*u[n-1]
    assert u_x[-2] == -a1*u[n+1] - b1*u[n] - c1*u[n-1]

    assert u_x[u_x.right(0)] == -a0*u[n+1] - b0*u[n] - c0*u[n-1]
    assert u_x[u_x.right(1)] == -a1*u[n+1] - b1*u[n] - c1*u[n-1]


def test_derivative21():
    from sympy import expand
    clear_cache()

    i = symbols("i", integer=True)
    n = 20
    x   = Grid("x", size=n+1, interval=(0, 1.0))
    xhat = StaggeredGrid("xhat", size=n+2, interval=(0, 1.0))

    phat0 = Polynomial(xhat, degree=0)
    phat1 = Polynomial(xhat, degree=1)
    phat2 = Polynomial(xhat, degree=2)
    
    p0 = Polynomial(x, degree=0)
    p1 = Polynomial(x, degree=1)
    p2 = Polynomial(x, degree=2)

    # Interior
    u_x    = sbp.Derivative(phat2, "x", order=2, gridspacing = x.gridspacing)
    
    assert abs(expand(u_x[i] - p1[i])).subs(i,u_x.num_bnd_pts_left) < 1e-14
    assert u_x.order(i) == 2
    
    # Left boundary
    u_x    = sbp.Derivative(phat1, "x",shape=(n+1,),  order=2, gridspacing = x.gridspacing)
    for j in u_x.range(0):
        assert abs(expand(u_x[j] - p0[j])/n) <  1e-15
        assert u_x.order(j) == 1
    
    ## Right boundary
    u = GridFunction("u", shape=(n+2,))
    u_x    = sbp.Derivative(phat1, "x", shape=(n+1,), order=2, gridspacing = x.gridspacing)
    for j in u_x.range(-1):
        assert abs(expand(u_x[j] - p0[j])/n) <  1e-15
        assert u_x.order(j) == 1

    # Test hat version
    
    # Interior
    uhat_x    = sbp.Derivative(p2, "x", order=2, gridspacing = x.gridspacing, hat=True)
    assert abs(expand(uhat_x[i] - phat1[i])).subs(i,u_x.num_bnd_pts_left) < 1e-14
    
    # Left boundary
    uhat_x    = sbp.Derivative(p1, "x", shape=(n+2,), order=2, gridspacing = x.gridspacing, hat=True)

    left = u_x.bounds(0)
    for j in u_x.range(0):
        assert abs(expand(uhat_x[j] - phat0[j])/n) <  1e-15
    
    ## Right boundary
    uhat_x    = sbp.Derivative(p1, "x", shape=(n+2,), order=2, gridspacing = x.gridspacing, hat=True)
    for j in u_x.range(-1):
        assert abs(expand(uhat_x[uhat_x.right(j)] - phat0[uhat_x.right(j)])/n) <  1e-15

def test_derivative_periodic2():
    n = 20
    u = GridFunction('u', shape=(n,), periodic=True)
    u_x = sbp.Derivative(u, "x", order=2)
    u_x = sbp.Derivative(u, "x", order=2, periodic=True)
    assert u_x[0] == 1.0*u[1] - 1.0*u[0]
    assert u_x[-1] == 1.0*u[0] - 1.0*u[-1]
    u_x = sbp.Derivative(u, "x", order=2, periodic=True, hat=True)
    assert u_x[0]  == 1.0*u[0] - 1.0*u[-1]
    assert u_x[-1] == 1.0*u[-1] - 1.0*u[-2]

def test_derivative42():
    from sympy import expand
    clear_cache()

    i = symbols("i", integer=True)
    n = 20
    x   = Grid("x", size=n+1, interval=(0, 1.0))
    xhat = StaggeredGrid("xhat", size=n+2, interval=(0, 1.0))

    phat0 = Polynomial(xhat, degree=0)
    phat1 = Polynomial(xhat, degree=1)
    phat2 = Polynomial(xhat, degree=2)
    phat3 = Polynomial(xhat, degree=3)
    phat4 = Polynomial(xhat, degree=4)
    
    p0 = Polynomial(x, degree=0)
    p1 = Polynomial(x, degree=1)
    p2 = Polynomial(x, degree=2)
    p3 = Polynomial(x, degree=3)
    p4 = Polynomial(x, degree=4)


    # Interior
    u_x    = sbp.Derivative(phat4, "x", order=4, gridspacing = x.gridspacing)
    u = GridFunction("u", shape=(n, ))
    ux    = sbp.Derivative(u, "x", order=4, gridspacing = x.gridspacing)

    assert abs(expand(u_x[i] - p3[i])).subs(i,u_x.num_bnd_pts_left) < 1e-14
    assert u_x.order(i) == 4
    
    # Left boundary
    u_x    = sbp.Derivative(phat2, "x",shape=(n+1,),  order=4, gridspacing = x.gridspacing)
    for j in u_x.range(0):
        assert abs(expand(u_x[j] - p1[j])/n) <  1e-15
        assert u_x.order(j) == 2
    
    ## Right boundary
    u_x    = sbp.Derivative(phat2, "x", shape=(n+1,), order=4, gridspacing = x.gridspacing)
    for j in u_x.range(-1):
        assert abs(expand(u_x[u_x.right(j)] - p1[u_x.right(j)])/n) <  1e-15
        assert u_x.order(j) == 2

    # Test hat version

    # Interior
    uhat_x    = sbp.Derivative(p4, "x", order=4, gridspacing = x.gridspacing, hat=True)
    assert abs(expand(uhat_x[i] - phat3[i])).subs(i,u_x.num_bnd_pts_left) < 1e-14
    
    # Left boundary
    u = GridFunction("u", shape=(n+1,))
    uhat_x    = sbp.Derivative(p2, "x", shape=(n+2,), order=4, gridspacing = x.gridspacing, hat=True)
    for j in u_x.range(0):
        assert abs(expand(uhat_x[j] - phat1[j])/n) <  1e-15
    
    # Right boundary
    uhat_x    = sbp.Derivative(p2, "x", shape=(n+2,), order=4, gridspacing = x.gridspacing, hat=True)
    for j in u_x.range(-1):
        assert abs(expand(uhat_x[uhat_x.right(j)] - phat1[uhat_x.right(j)])/n) <  1e-15

def test_derivative63():
    from sympy import expand
    clear_cache()

    i = symbols("i", integer=True)
    n = 20
    x   = Grid("x", size=n+1, interval=(0, 1.0))
    xhat = StaggeredGrid("xhat", size=n+2, interval=(0, 1.0))

    phat0 = Polynomial(xhat, degree=0)
    phat1 = Polynomial(xhat, degree=1)
    phat2 = Polynomial(xhat, degree=2)
    phat3 = Polynomial(xhat, degree=3)
    phat4 = Polynomial(xhat, degree=4)
    phat5 = Polynomial(xhat, degree=5)
    phat6 = Polynomial(xhat, degree=6)
    
    p0 = Polynomial(x, degree=0)
    p1 = Polynomial(x, degree=1)
    p2 = Polynomial(x, degree=2)
    p3 = Polynomial(x, degree=3)
    p4 = Polynomial(x, degree=4)
    p5 = Polynomial(x, degree=5)
    p6 = Polynomial(x, degree=6)


    # Interior
    u_x    = sbp.Derivative(phat6, "x", order=6, gridspacing = x.gridspacing)
    assert abs(expand(u_x[i] - p5[i])).subs(i,u_x.num_bnd_pts_left) < 1e-14
    assert u_x.order(i) == 6
    
    # Left boundary
    u_x    = sbp.Derivative(phat3, "x",shape=(n+1,),  order=6, gridspacing = x.gridspacing)
    for j in u_x.range(0):
        assert abs(expand(u_x[j] - p2[j])/n) <  1e-15
        assert u_x.order(j) == 3
    
    ## Right boundary
    u_x    = sbp.Derivative(phat3, "x", shape=(n+1,), order=6, gridspacing = x.gridspacing)
    for j in u_x.range(-1):
        assert abs(expand(u_x[u_x.right(j)] - p2[u_x.right(j)])/n) <  1e-15
        assert u_x.order(j) == 3

    # Test hat version

    # Interior
    uhat_x    = sbp.Derivative(p6, "x", order=6, gridspacing = x.gridspacing, hat=True)
    
    assert abs(expand(uhat_x[i] - phat5[i])).subs(i,u_x.num_bnd_pts_left) < 1e-14
    
    # Left boundary
    u = GridFunction("u", shape=(n+1,))
    uhat_x    = sbp.Derivative(p3, "x", shape=(n+2,), order=6, gridspacing = x.gridspacing, hat=True)
    for j in u_x.range(0):
        assert abs(expand(uhat_x[j] - phat2[j])/n) <  1e-15
    
    # Right boundary
    uhat_x    = sbp.Derivative(p3, "x", shape=(n+2,), order=6, gridspacing = x.gridspacing, hat=True)
    for j in u_x.range(-1):
        assert abs(expand(uhat_x[uhat_x.right(j)] - phat2[uhat_x.right(j)])/n) <  1e-15

def test_derivative84():
    from sympy import expand
    clear_cache()

    i = symbols("i", integer=True)
    n = 20
    x   = Grid("x", size=n+1, interval=(0, 1.0))
    xhat = StaggeredGrid("xhat", size=n+2, interval=(0, 1.0))

    phat0 = Polynomial(xhat, degree=0)
    phat1 = Polynomial(xhat, degree=1)
    phat2 = Polynomial(xhat, degree=2)
    phat3 = Polynomial(xhat, degree=3)
    phat4 = Polynomial(xhat, degree=4)
    phat5 = Polynomial(xhat, degree=5)
    phat6 = Polynomial(xhat, degree=6)
    phat7 = Polynomial(xhat, degree=7)
    phat8 = Polynomial(xhat, degree=8)
    
    p0 = Polynomial(x, degree=0)
    p1 = Polynomial(x, degree=1)
    p2 = Polynomial(x, degree=2)
    p3 = Polynomial(x, degree=3)
    p4 = Polynomial(x, degree=4)
    p5 = Polynomial(x, degree=5)
    p6 = Polynomial(x, degree=6)
    p7 = Polynomial(x, degree=7)
    p8 = Polynomial(x, degree=8)


    # Interior
    
    u_x    = sbp.Derivative(phat8, "x", order=8, gridspacing = x.gridspacing)
    assert abs(expand(u_x[i] - p7[i])).subs(i,u_x.num_bnd_pts_left) < 1e-14
    assert u_x.order(i) == 8
    
    # Left boundary
    u_x    = sbp.Derivative(phat4, "x",shape=(n+1,),  order=8, gridspacing = x.gridspacing)
    for j in u_x.range(0):
        assert abs(expand(u_x[j] - p3[j])/n) <  1e-15
        assert u_x.order(j) == 4
    
    ## Right boundary
    u_x    = sbp.Derivative(phat4, "x", shape=(n+1,), order=8, gridspacing = x.gridspacing)
    for j in u_x.range(-1):
        assert abs(expand(u_x[u_x.right(j)] - p3[u_x.right(j)])/n) <  1e-14
        assert u_x.order(j) == 4

    # Test hat version

    # Interior
    uhat_x    = sbp.Derivative(p8, "x", order=8, gridspacing = x.gridspacing, hat=True)
    
    assert abs(expand(uhat_x[i] - phat7[i])).subs(i,u_x.num_bnd_pts_left) < 1e-14
    
    # Left boundary
    u = GridFunction("u", shape=(n+1,))
    uhat_x    = sbp.Derivative(p4, "x", shape=(n+2,), order=8, gridspacing = x.gridspacing, hat=True)
    for j in u_x.range(0):
        assert abs(expand(uhat_x[j] - phat3[j])/n) <  1e-15
    
    # Right boundary
    uhat_x    = sbp.Derivative(p4, "x", shape=(n+2,), order=8, gridspacing = x.gridspacing, hat=True)
    for j in u_x.range(-1):
        assert abs(expand(uhat_x[uhat_x.right(j)] - phat3[uhat_x.right(j)])/n) <  1e-14

def test_quadrature21():
    from sympy import expand
    from ... base import GridFunctionExpression as GFE
    clear_cache()

    i = symbols("i", integer=True)
    n = 20
    x   = Grid("x", size=n+1, interval=(0, 1.0))
    xhat = StaggeredGrid("xhat", size=n+2, interval=(0, 1.0))

    phat0 = Polynomial(xhat, degree=0)
    phat1 = Polynomial(xhat, degree=1)
    phat2 = Polynomial(xhat, degree=2)
    
    p0 = Polynomial(x, degree=0)
    p1 = Polynomial(x, degree=1)
    p2 = Polynomial(x, degree=2)

    Iu    = sbp.Quadrature(GFE(p0), "x", order=2, gridspacing = x.gridspacing, shape=(n+1,))
    assert abs(expand(Iu[i] - x.gridspacing)).subs(i, Iu.num_bnd_pts_left) < 1e-14
    assert Iu.order() == 1
    I = sum([Iu[i] for i in range(n+1)])
    assert abs(I - 1.0) < 1e-14

    # Check inverse operator
    Iuinv = sbp.Quadrature(p0, "x", order=2, gridspacing = x.gridspacing, shape=(n+1,), invert=True)
    assert abs(Iuinv[0] - 1.0/Iu[0]) < 1e-14

def test_quadrature21hat():
    from sympy import expand
    clear_cache()

    i = symbols("i", integer=True)
    n = 20
    x   = Grid("x", size=n+1, interval=(0, 1.0))
    xhat = StaggeredGrid("xhat", size=n+2, interval=(0, 1.0))

    phat0 = Polynomial(xhat, degree=0)
    phat1 = Polynomial(xhat, degree=1)
    phat2 = Polynomial(xhat, degree=2)
    
    p0 = Polynomial(x, degree=0)
    p1 = Polynomial(x, degree=1)
    p2 = Polynomial(x, degree=2)

    Iu    = sbp.Quadrature(phat0, "x", order=2, gridspacing = xhat.gridspacing, shape=(n+2,), hat = True)
    assert abs(expand(Iu[i] - xhat.gridspacing)).subs(i, Iu.num_bnd_pts_left) < 1e-14
    assert Iu.order() == 1
    I = sum([Iu[i] for i in range(n+2)])
    assert abs(I - 1.0) < 1e-14

def test_interpolation21():

    from sympy import expand
    clear_cache()

    i = symbols("i", integer=True)
    n = 20
    x   = Grid("x", size=n+1, interval=(0, 1.0))
    xhat = StaggeredGrid("xhat", size=n+2, interval=(0, 1.0))

    phat0 = Polynomial(xhat, degree=0)
    phat1 = Polynomial(xhat, degree=1)
    phat2 = Polynomial(xhat, degree=2)
    
    p0 = Polynomial(x, degree=0)
    p1 = Polynomial(x, degree=1)
    p2 = Polynomial(x, degree=2)

    # Interior
    Pu    = sbp.Interpolation(phat1, "x", order=2)
    assert abs(expand(Pu[i] - p1[i])).subs(i,Pu.num_bnd_pts_left) < 1e-14
    assert Pu.order(i) == 1

def test_interpolation21hat():

    from sympy import expand
    clear_cache()

    i = symbols("i", integer=True)
    n = 20
    x   = Grid("x", size=n+1, interval=(0, 1.0))
    xhat = StaggeredGrid("xhat", size=n+2, interval=(0, 1.0))

    phat0 = Polynomial(xhat, degree=0)
    phat1 = Polynomial(xhat, degree=1)
    phat2 = Polynomial(xhat, degree=2)
    
    p0 = Polynomial(x, degree=0)
    p1 = Polynomial(x, degree=1)
    p2 = Polynomial(x, degree=2)

    # Interior
    Pu    = sbp.Interpolation(p1, "x", hat=True, order=2)
    assert abs(expand(Pu[i] - phat1[i])).subs(i,Pu.num_bnd_pts_left) < 1e-14
    assert Pu.order(i) == 1

def test_interpolation2():

    from sympy import expand
    clear_cache()

    i = symbols("i", integer=True)
    n = 20
    x   = Grid("x", size=n+1, interval=(0, 1.0))
    xhat = StaggeredGrid("xhat", size=n+2, interval=(0, 1.0))

    phat0 = Polynomial(xhat, degree=0)
    phat1 = Polynomial(xhat, degree=1)
    phat2 = Polynomial(xhat, degree=2)
    
    p0 = Polynomial(x, degree=0)
    p1 = Polynomial(x, degree=1)
    p2 = Polynomial(x, degree=2)

    # Interior
    u_x    = sbp.Interpolation(phat1, "x", order=2)
    
    assert abs(expand(u_x[i] - p1[i])).subs(i,u_x.num_bnd_pts_left) < 1e-14
    assert u_x.order(i) == 1

def test_custom_operator():

    # Load another operator and check that it is different from the default one
    n = symbols("n")
    u = GridFunction("u", shape=(n+1,))
    u_x    = sbp.Derivative(u, "x", order=2, fmt='staggered_P%s%s.json')
    v_x    = sbp.Derivative(u, "x", order=2)
    assert u_x[0] != v_x[0]

