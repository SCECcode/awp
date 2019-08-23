from .. import Grid, GridFunction, Operator, OperatorException, GridFunctionException, GridFunctionExpression
from .. import Polynomial, PolynomialException
from sympy import symbols, Rational
from sympy.core.cache import clear_cache
import pytest

def test_polynomial_new():
    clear_cache()

    nx = symbols('nx')
    x   = Grid("g",size=nx,interval=(0,1))
    p = Polynomial(x,degree=2)


    p = Polynomial(x,degree=2)
    assert p.coeffs == [1,1,Rational(1,2)]
    p = Polynomial(x,coeffs=[1,2,3])
    assert p.coeffs == [1,2,3]
    assert p.degree == 2
    p = Polynomial(x,coeffs=[0])
    assert p.degree == 0
    p = Polynomial(x,coeffs=[0,1])
    assert p.degree == 1

    with pytest.raises(PolynomialException): Polynomial(x) 
    with pytest.raises(PolynomialException): Polynomial(x,degree=-2) 
    with pytest.raises(PolynomialException): Polynomial(x,degree=2,coeffs=[1,0]) 


def test_polynomial_taylor():
    clear_cache()

    nx = symbols('nx')
    x   = Grid("x",size=nx,interval=(0,1))
    p = Polynomial(x,degree=2)

def test_polynomial_getitem():
    pass
