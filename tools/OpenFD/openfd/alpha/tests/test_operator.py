from .. import Operator, GridFunction, Expression as Expr
from .. import operator
from sympy import symbols

class Shift(Operator):

    def __getitem__(self, indices):
        return self.args[indices+1]

class Difference(Operator):

    def __getitem__(self, indices):
        return self.args[indices+1] - self.args[indices]
    
def test_noncommutative():
    
    op1 = Operator('F1')
    op2 = Operator('F2')

    a = op1*op2
    b = op2*op1
    assert a != b

def test_distributive_plus():
    S = Shift('S')
    u = GridFunction('u', shape=(10,))
    v = GridFunction('v', shape=(10,))

    expr1 = Expr(S * (u + v))
    expr2 = Expr((S * u) + (S * v))
    assert expr1[0] == expr2[0]

def test_getitem():
    
    a = GridFunction('a', shape=(10,))
    b = GridFunction('b', shape=(10,))
    c = GridFunction('c', shape=(10,))
    d = symbols('d')

    # Start with testing the default operator that does nothing
    I = Operator('O')
    assert Expr(I*a)[0]       ==   a[0]
    assert Expr(I*b)[0]       ==   b[0]
    assert Expr(I*a + d)[0]   ==   a[0] + d
    assert Expr(- d)[0]        == - d
    assert Expr(I*(a+b))[0]   ==   a[0] + b[0]
    assert Expr(I*(a-b))[0]   ==   a[0] - b[0]
    assert Expr(I*a+b)[0]     ==   a[0] + b[0]
    assert Expr(-I*a)[0]      == - a[0]
    assert Expr((I*a)**2)[0]  ==   a[0]**2

    ## Test shift operator
    S = Shift('S')
    assert Expr(S*a)[0]           ==   a[1]
    assert Expr(S*a*b)[0]         ==   a[1]*b[1]
    assert Expr(S*(a + b))[0]     ==   a[1] + b[1]
    assert Expr(S*a + b)[0]       ==   a[1] + b[0]
    assert Expr(S*S*a + b)[0]     ==   a[2] + b[0]
    assert Expr(S*(S*a + b))[0]   ==   a[2] + b[1]
    assert Expr(S*S*S*a + b)[0]   ==   a[3] + b[0]
    assert Expr(b*S*a)[0]         ==   b[0]*a[1]
    assert Expr(b*S*a + a)[0]     ==   b[0]*a[1] + a[0]
    assert Expr(-b*S*a + a)[0]    == - b[0]*a[1] + a[0]
    assert Expr(-a*b*S*a)[0]      == - a[0]*b[0]*a[1]
    assert Expr(-a*b*S*a*b)[0]    == - a[0]*b[0]*a[1]*b[1]
    assert Expr(S*a*a)[0]         ==   a[1]*a[1]
    assert Expr(S*a**2)[0]        ==   a[1]*a[1]

    # Test difference operator
    D = Difference('D')
    assert Expr(D*a)[0]             ==   a[1] - a[0]
    assert Expr(D*(a + b))[0]       ==   a[1] + b[1] - a[0] - b[0]
    assert Expr(D*D*a)[0]           ==   a[2] - 2*a[1] + a[0]
    assert Expr(D*D*(a + b))[0]     ==   a[2] - 2*a[1] + a[0] + b[2] - 2*b[1] + b[0]
    assert Expr(D*(b + D*a))[0]     ==   b[1] - b[0] + a[2] - 2*a[1] + a[0]

def test_str():
    a = GridFunction('a', shape=(10,))
    b = GridFunction('b', shape=(10,))
    c = GridFunction('c', shape=(10,))
    S = Shift('S')

    assert str(S(a*b)) == 'S(a*b)'
    assert str(S) == 'S'


def test_call():
    a = GridFunction('a', shape=(10,))
    S = Shift('S')

    assert Expr(S*a)[0]       == a[1]
    assert Expr(S(a))[0]      == a[1]
    assert Expr(S*S(a))[0]    == a[2]
    assert Expr(S(S(a)))[0]   == a[2]
    assert Expr(S*S(S(a)))[0] == a[3]


