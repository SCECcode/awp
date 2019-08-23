from ... import Grid, GridFunction, Operator, OperatorException, GridFunctionException
from ...base import gridfunction as gridfunc 
from ... import Polynomial, GridFunctionExpression
from ... import traditional as sbp
from ... import staggered as sbps
from sympy import symbols, simplify
from sympy.core.cache import clear_cache
import pytest

def test_mixed_derivative():
    nx, ny = symbols('nx ny')
    dim = (nx, ny)
    dim = (10, 10)
    u = GridFunction('u', shape=dim)
    p = 4
    u_x = sbp.Derivative(u, 'x', order=p, shape=dim)
    u_y = sbp.Derivative(u, 'y', order=p, shape=dim)
    u_xy = sbp.Derivative(u_x, 'y', order=p, shape=dim)
    u_yx = sbp.Derivative(u_y, 'x', order=p, shape=dim)

    u_xy[0,0]
    u_xy[0,-1]
    u_xy[-1,-1]
    u_xy[-1,0]
    u_yx[0,0]
    u_yx[0,-1]
    u_yx[-1,-1]
    u_yx[-1,0]

def test_mixed_orders():
    dim = (10, )
    u = GridFunction('u', shape=dim)
    u_x1 = sbp.Derivative(u, 'x', order=2, shape=dim)
    u_x2 = sbp.Derivative(u, 'x', order=4, shape=dim)

    assert u_x1[0] + u_x2[0] != 2*u_x2[0]

def test_mixed_orders_staggered():
    dim = (10, )
    u = GridFunction('u', shape=dim)
    u_x1 = sbps.Derivative(u, 'x', order=2, shape=dim)
    u_x2 = sbps.Derivative(u, 'x', order=4, shape=dim)

    assert u_x1[0] + u_x2[0] != 2*u_x2[0]
