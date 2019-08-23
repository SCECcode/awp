from ... import GridFunction, GridFunctionExpression as Expr
from .. import Constant
from .. import manipulators 
from sympy import symbols

def test_pack():

    a = GridFunction('a', shape=(10,10))
    c = Constant('c')
    
    # Test label and default packing
    d = GridFunction('e', shape=(10,))
    expr = Expr(a*c**2*0.3 + 0.2*c)
    out, data = manipulators.pack(expr, label='e')
    assert out == Expr(d[0]*a + d[1])
    assert data[0] == c**2*0.3
    assert data[1] == 0.2*c

    # Test multi-index packing
    d = GridFunction('d', shape=(10,10))
    i, j = symbols('i j')
    out, data = manipulators.pack(expr, index = lambda i : (i, j))
    assert out == Expr(d[0, j]*a + d[1, j])
    assert data[0] == c**2*0.3
    assert data[1] == 0.2*c
    
    # Test output dictionary
    d = GridFunction('d', shape=(10,))
    i, j = symbols('i j')
    out, data = manipulators.pack(expr, dict=True)
    assert out == Expr(d[0]*a + d[1])
    assert data[d[0]] == c**2*0.3
    assert data[d[1]] == 0.2*c

def test_constant():

    a = GridFunction('a', shape=(10,10))
    b = GridFunction('b', shape=(10,10))
    c = Constant('c')
    d = Constant('d')
    
    expr = Expr(a*c)
    const = manipulators.constants(expr)
    assert len(const) == 1
    assert c in const

    expr = Expr(a/c)
    const = manipulators.constants(expr)
    assert len(const) == 1
    assert 1/c in const

    expr = Expr(a+c)
    const = manipulators.constants(expr)
    assert len(const) == 1
    assert c in const
    
    expr = Expr(a*c**2)
    const = manipulators.constants(expr)
    assert len(const) == 1
    assert c**2 in const
    
    expr = Expr(a+c**2)
    const = manipulators.constants(expr)
    assert len(const) == 1
    assert c**2 in const
    
    expr = Expr(a+0.2*c)
    const = manipulators.constants(expr)
    assert len(const) == 1
    assert 0.2*c in const
    
    expr = Expr(a+0.2*c**2)
    const = manipulators.constants(expr)
    assert len(const) == 1
    assert 0.2*c**2 in const

    expr = Expr(a*c*0.3 + 0.2)
    const = manipulators.constants(expr)
    assert 0.2 in const
    assert c*0.3 in const
    
    expr = Expr(a*c**2*0.3 + 0.2*c)
    const = manipulators.constants(expr)
    assert 0.2*c in const
    assert c**2*0.3 in const
    
    expr = Expr(a*c**2*d*0.3 + 0.2*c)
    const = manipulators.constants(expr)
    assert 0.2*c in const
    assert c**2*d*0.3 in const
    
    # Check index expressions
    expr = Expr(a*c**2*0.3 + 0.2*c)
    const = manipulators.constants(expr[0,1])
    assert 0.2*c in const
    assert c**2*0.3 in const

def test_isconstant():

    c = Constant('c')
    u = GridFunction('u', shape=(10,10))
    assert manipulators.isconstant(c)
    assert manipulators.isconstant(0.2*c)
    assert manipulators.isconstant(0.2*c**2)
    assert manipulators.isconstant(0.2*c**2+1)
    assert not manipulators.isconstant(u)
    assert not manipulators.isconstant(0.2*u)
    assert not manipulators.isconstant(0.2*u**2)
    assert not manipulators.isconstant(0.2*u**2+1)
