import pytest
from sympy.printing import ccode
from openfd import GridFunction, GridFunctionExpression as Expr
import openfd
from ..variable import Variable
import numpy

def test_variable():

    u_x = Variable('u_x', dtype=numpy.float32)

    # Only show type when called by ccode
    lhs = u_x.declare()
    assert str(u_x) == 'u_x'
    assert str(lhs) == 'u_x'
    assert ccode(u_x) == 'u_x'
    assert ccode(lhs) == 'float u_x'
    assert ccode(lhs[0]) == 'float u_x'

    ## Raise exception when no type is specified
    u_x = Variable('u_x')
    with pytest.raises(ValueError) : u_x.declare()
    
    u_x = Variable('u_x', dtype='float')
    assert ccode(u_x) == 'u_x'
    assert ccode(u_x[0]) == 'u_x'
    assert ccode(u_x[0,0]) == 'u_x'

    # Try more involved expressions
    u = GridFunction('u', shape=(10,))
    rhs = Expr(u + u_x)
    assert str(rhs) == 'u + u_x'

def test_value():

    vy = Variable('v_y')
    u_x = Variable('u_x', val=vy, dtype=openfd.prec)
    assert ccode(u_x.declare()) == 'float u_x'
