import pytest
from sympy import symbols
from sympy.core.cache import clear_cache
from ... import GridFunction, GridFunctionExpression
from .. import restriction as r

def test_restriction():

    clear_cache()

    nx = symbols('nx')
    ny = symbols('ny')
    u = GridFunction("u", shape=(nx+1))
    v = GridFunction("v", shape=(nx+1))
    e0 = r.Restriction(u,"x", 0, 0, shape=(nx+1))
    en = r.Restriction(u, "x",-1, -1, shape=(nx+1))

    
    assert e0[0] == u[0]
    assert e0[1] == 0.0
    assert en[-1] == u[-1]
    assert en[nx] == u[-1]

    a = GridFunctionExpression(u*u + e0)
    b = GridFunctionExpression(u + en)
    c = GridFunctionExpression(u*u + en)
    d = GridFunctionExpression(e0 + en)

    assert a[0]  == u[0]**2 + u[0]
    assert b[-1] == u[-1] + u[-1]
    assert c[-1] == u[-1] + u[-1]**2
    assert d[0] == u[0]
    assert d[-1] == u[-1]

    e0 = r.Restriction(GridFunctionExpression(u + v) ,"x", 0, 0, shape=(nx+1))
    en = r.Restriction(GridFunctionExpression(u - v), "x",-1, -1, shape=(nx+1))

    assert e0[0] == u[0] + v[0]
    assert e0[1] == 0.0
    assert en[-1] == u[nx] - v[nx]
    assert e0[0] + en[0] == u[0] + v[0]
    
    v  = symbols('v')
    e0 = r.Restriction(GridFunctionExpression(u + v) ,"x", 0, 0, shape=(nx+1))
    en = r.Restriction(GridFunctionExpression(u - v), "x",-1, -1, shape=(nx+1))

    assert e0[0] == u[0] + v
    assert e0[1] == 0.0
    assert en[-1] == u[-1] - v
    
    v = GridFunction("v", shape=(nx+1, ny+1))
    v[0,0]
    ex0 = r.Restriction(v, "x", 0,  0,  shape=(nx+1, ny+1))
    exn = r.Restriction(v, "x", -1, -1, shape=(nx+1, ny+1))
    ey0 = r.Restriction(v, "y", 0,  0,  shape=(nx+1, ny+1))
    eyn = r.Restriction(v, "y", -1, -1, shape=(nx+1, ny+1))


    ax = GridFunctionExpression(v*v + ex0)
    bx = GridFunctionExpression(v*v + exn)
    ay = GridFunctionExpression(v*v + ey0)
    by = GridFunctionExpression(v*v + eyn)

    x, y = symbols('x y')
    assert ax[0, y] == v[0, y]**2 + v[0, y]
    assert bx[-1, y] == v[-1, y]**2 + v[-1, y]
    assert ay[x, 0] == v[x, 0]**2 + v[x, 0]
    assert by[x, -1] == v[x, -1]**2 + v[x, -1]

