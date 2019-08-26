from ... import Grid, GridFunction, Operator, OperatorException, GridFunctionException
from ...base import gridfunction as gridfunc 
from ... import Polynomial, GridFunctionExpression
from ... import traditional as sbp
from sympy import symbols, simplify
from sympy.core.cache import clear_cache
import pytest

class Polynomial_xy(GridFunctionExpression):

    def __init__(self, gx, gy):
        """
        Defines a polynomial f(x,y) = g(x) + g(y)
        """
        self.gx = gx
        self.gy = gy

    def __getitem__(self, indices):
        return self.gx[indices[0]] + self.gy[indices[1]]

    @property
    def shape(self):
        return self.gx.shape

def test_Derivative21():
    clear_cache()

    n,i = symbols("n i", integer=True)
    n = 10
    x   = Grid("x", size=n+1, interval=(0, n))

    p0 = Polynomial(x, degree=0)
    p1 = Polynomial(x, degree=1)
    p2 = Polynomial(x, degree=2)

    # Interior
    u_x    = sbp.Derivative(p2, "x", order=2)
    assert simplify(u_x[i] - p1[i]) == 0
    assert u_x.order(i) == 2

    # Left boundary
    u_x    = sbp.Derivative(p1, "x", order=2)
    for i in u_x.range(0):
        assert simplify(u_x[i] - p0[i]) == 0
        assert u_x.order(i) == 1
    
    # Right boundary
    u_x    = sbp.Derivative(p1, "x", order=2)
    for i in u_x.range(-1):
        assert simplify(u_x[i] - p0[i]) == 0
        assert u_x.order(i) == 1

def test_Derivative42():
    clear_cache()

    n,i = symbols("n i",integer=True)
    n = 12
    x   = Grid("x", size=n+1, interval=(0,n))

    p0 = Polynomial(x, degree=0)
    p1 = Polynomial(x, degree=1)
    p2 = Polynomial(x, degree=2)
    p3 = Polynomial(x, degree=3)
    p4 = Polynomial(x, degree=4)

    # Interior
    u_x    = sbp.Derivative(p4,"x", order=4)
    i = 5
    assert abs(simplify(u_x[i] - p3[i])) < 1e-13
    assert u_x.order(i) == 4

    # Left boundary
    u_x    = sbp.Derivative(p2,"x", order=4)
    for i in u_x.range(0):
        assert abs(simplify(u_x[i] - p1[i])) < 1e-13
        assert u_x.order(i) == 2
    
    # Right boundary
    u = GridFunction('u', shape=(n+1,))
    u_x    = sbp.Derivative(p2, "x", order=4)

    for i in u_x.range(-1):
        assert abs(simplify(u_x[i] - p1[i])) < 1e-13
        assert abs(simplify(u_x[u_x.right(i)] - p1[u_x.right(i)])) < 1e-13
        assert u_x.order(i) == 2

# 2D

def test_Derivative42_2D():
    clear_cache()

    nx,ny,i,j = symbols("nx ny i j", integer=True)
    nx = 12
    ny = 12

    x   = Grid("x", size=nx+1, interval=(0,nx))
    y   = Grid("y", size=nx+1, interval=(0,ny))
    
    xy = Polynomial_xy(x, y)
    p1 = Polynomial(xy, degree=1)
    p2 = Polynomial(xy, degree=2)
    p3 = Polynomial(xy, degree=3)
    p4 = Polynomial(xy, degree=4)
    
    # Interior
    u_x    = sbp.Derivative(p4, "x", shape=(nx+1, ny+1), order=4)
    i = 5
    j = 6
    assert abs(simplify(u_x[i, j] - p3[i,j])) < 1e-13
    
    # Left boundary
    u_x    = sbp.Derivative(p2, "x", shape=(nx+1, ny+1), order=4)
    for i in u_x.range(0):
        assert abs(simplify(u_x[i, j] - p1[i,j])) < 1e-13
    
    # Right boundary
    u = GridFunction('u', shape=(nx+1, ny+1))
    u_x    = sbp.Derivative(p2, "x", shape=(nx+1, ny+1), order=4)
    for i in u_x.range(-1):
        assert abs(simplify(u_x[i, j] - p1[i, j])) <  1e-13
 
