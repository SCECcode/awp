from .. import Grid, GridFunction, Operator, OperatorException, GridFunctionException, GridFunctionExpression, Constant
from ... import sbp_traditional as sbp
from sympy import symbols
from sympy.core.cache import clear_cache
import pytest

def default_operatordata():
    import numpy as np
    data = {'op_left' : np.array([-1.0, 1.0]), 
            'idx_left' : np.array([0, 1]),
            'op_right' : np.array([-1.0, 1.0]),
            'idx_right' : np.array([0, 1]),
            'op_interior' : np.array([-0.5, 0.5]),
            'idx_interior' : np.array([-1, 1])}
    return data


def test_operator_new():
    clear_cache()

    nx,ny,nz = symbols("nx ny nz")

    u   = GridFunction("u",shape=(nx,))
    u_x = Operator(u,"x",shape=(nx,), operator_data = default_operatordata()) 

    with pytest.raises(ValueError): Operator(u,"y",shape=(nx,))
    with pytest.raises(IndexError): Operator(u,"z",shape=(nx,ny))
    
def test_operator_shape():
    clear_cache()

    nx,ny,nz = symbols("nx ny nz")

    u      = GridFunction("u",shape=(nx,))
    v      = GridFunction("v",shape=(nx,))

    expr   = u + v
    expr_x = Operator(expr,"x", operator_data = default_operatordata()) 

    assert expr_x.shape == (nx,)

    # Infer shape from expression
    expr_x = Operator(expr,"x", operator_data = default_operatordata()) 

    assert expr_x.shape == (nx,)

def test_operator_getitem():
    clear_cache()

    from sympy import simplify, expand

    i,j,k,nx,ny,nz = symbols("i j k nx ny nz",integer=True)

    u      = GridFunction("u",shape=(nx+1,))
    u_x    = sbp.Derivative(u, "x", order=2)
    u_xx   = sbp.Derivative(u_x, "x", order=2)

    assert expand(u_x[0] - (u[1] - u[0])) == 0
    assert expand(u_x[1]        - ( 0.5*u[2] - 0.5*u[0] )) == 0
    assert expand(u_x[i]        - ( -0.5*u[i-1] + 0.5*u[i+1] ))  == 0
    assert expand(u_x[nx]       - ( u[nx] - u[nx-1])) == 0
    assert expand(u_x[-1]       - ( u[nx] - u[nx-1])) == 0
    assert expand(u_xx[i]- (0.25*u[i+2] - 0.5*u[i] + 0.25*u[i-2])) == 0
    
    u      = GridFunction("u",shape=(nx+1,ny+1))
    u_x    = sbp.Derivative(u,"x", order=2)
    u_xx   = sbp.Derivative(u_x,"x", order=2)
    
    assert u_x[0,j]
    assert expand(u_x[0,j] - (u[1,j] - u[0,j])) == 0
    assert expand(u_xx[i,j]- (0.25*u[i+2,j] - 0.5*u[i,j] + 0.25*u[i-2,j])) == 0

    with pytest.raises(ValueError): u_x[2]

    u_x    = sbp.Derivative(Constant(1, shape=(nx+1,)), "x", shape=(nx+1,), order=2)
    assert u_x[0] == 0
    assert u_x[j] == 0

def test_operator_index_mapping():
    clear_cache()
    
    i,j,k,nx,ny,nz = symbols("i j k nx ny nz",integer=True)

    u      = GridFunction("u",shape=(nx,))
    u_x    = sbp.Derivative(u,"x", order=2)

    assert u_x.index_mapping(0)     == u_x._idx_left
    assert u_x.index_mapping(1)     == u_x._idx_interior
    assert u_x.index_mapping(nx-1)  == u_x._idx_right
    assert u_x.index_mapping(nx-2)  == u_x._idx_interior

    with pytest.raises(IndexError): u_x.index_mapping(-2)
    with pytest.raises(IndexError): u_x.index_mapping(nx)
    with pytest.raises(IndexError): u_x.index_mapping(2*nx)

    u      = GridFunction("u",shape=(nx+1,))
    u_x    = sbp.Derivative(u,"x", order=2)
    
    assert u_x.index_mapping(0)     == u_x._idx_left
    assert u_x.index_mapping(1)     == u_x._idx_interior
    assert u_x.index_mapping(nx)    == u_x._idx_right
    assert u_x.index_mapping(nx-1)  == u_x._idx_interior
    
    with pytest.raises(IndexError): u_x.index_mapping(nx+1)
    with pytest.raises(IndexError): u_x.index_mapping(2*nx)

    assert u_x.index_mapping(i)     == u_x._idx_interior
    assert u_x.index_mapping(i+1)   == u_x._idx_interior
    assert u_x.index_mapping(i-1)   == u_x._idx_interior
    assert u_x.index_mapping(j+1)   == u_x._idx_interior
    assert u_x.index_mapping(j-1)   == u_x._idx_interior
    
    u      = GridFunction("u",shape=(10,))
    u_x    = sbp.Derivative(u,"x", order=2)
    
    assert u_x.index_mapping(0)  == u_x._idx_left
    assert u_x.index_mapping(9)  == u_x._idx_right
    assert u_x.index_mapping(8)  == u_x._idx_interior
    
    u      = GridFunction("u",shape=(nx+1,ny+1))
    u_x    = sbp.Derivative(u,"x", order=2)
    u_y    = sbp.Derivative(u,"y", order=2)
    
    assert u_x.index_mapping((0,j))     == u_x._idx_left
    assert u_x.index_mapping((1,j))     == u_x._idx_interior
    assert u_y.index_mapping((i,0))     == u_y._idx_left
    assert u_y.index_mapping((i,1))     == u_y._idx_interior
    
    u      = GridFunction("u",shape=(nx+1,ny+1,nz+1))
    u_x    = sbp.Derivative(u,"x", order=2)
    u_y    = sbp.Derivative(u,"y", order=2)
    u_z    = sbp.Derivative(u,"z", order=2)
    
    assert u_x.index_mapping((0,j,k+2))     == u_x._idx_left
    assert u_x.index_mapping((1,j,k+3))     == u_x._idx_interior
    assert u_z.index_mapping((1,j,k+3))     == u_z._idx_interior

@pytest.mark.skip(reason="Grid is broken and needs fixing.")
def test_operator_numerical_verification():
    clear_cache()
    
    i,j,k,nx,ny,nz = symbols("i j k nx ny nz",integer=True)
    u   = GridFunction("u",shape=(nx,))

    x   = Grid("x",size=nx+1,interval=(0,nx))

    p   = lambda k: GridFunctionExpression(x**k)
    p_x = lambda k: GridFunctionExpression(k*x**max([k-1,0]))

    e = GridFunctionExpression(0*x)
    u_x = sbp.Derivative(e,"x", order=2)

    # left,      interior,  right
    # 1st order, 2nd order, 1st order
    assert p_x(1)[0]  == sbp.Derivative(p(1),"x", order=2)[0] 
    assert p_x(2)[1]  == sbp.Derivative(p(2),"x", order=2)[1]
    assert p_x(1)[nx] == sbp.Derivative(p(1),"x", order=2)[nx]

def test_operator_order():
    clear_cache()
    
    i,j,k,nx,ny,nz = symbols("i j k nx ny nz",integer=True)
    u   = GridFunction("u",shape=(nx+1,))
    u_x = sbp.Derivative(u,"x", order=2)

    assert u_x.order(0)  == 1
    assert u_x.order(1)  == 2
    assert u_x.order(nx) == 1

def test_operator_gridspacing():
    clear_cache()
    
    i,j,k,nx,ny,nz = symbols("i j k nx ny nz",integer=True)
    hi  = symbols('hi')
    u   = GridFunction("u",shape=(nx+1,))
    u_x = sbp.Derivative(u,"x",gridspacing=1/hi, order=2)
    assert u_x[0] == (1.0*u[1] - 1.0*u[0])*hi

def test_operator_right():
    clear_cache()

    nx = symbols('nx')
    u   = GridFunction("u",shape=(nx+1,))
    u_x = sbp.Derivative(u, "x", order=2)
    assert u_x.right(0) == nx
    assert u_x.right(-1) == nx
    assert u_x.right(nx-1) == nx-1

def test_operator_is_matrix():
    u_x = sbp.Derivative('', "x", order=2)
    assert u_x.is_matrix

def test_operator_matrixform():
    from ... import Matrix
    nx = symbols('nx')
    u   = GridFunction("u",shape=(nx+1,))
    v   = GridFunction("v",shape=(nx+1,))
    ux = sbp.Derivative(u, "x", shape=(nx+1,), order=2)
    expr1 = GridFunctionExpression(ux+v)
    ux = sbp.Derivative(u + v, "x", shape=(nx+1,), order=2)
    expr2 = GridFunctionExpression(ux+v)
    ux = sbp.Derivative(u + v, "x", shape=(nx+1,), order=2)
    expr3 = GridFunctionExpression(v*ux+v)
    ux = sbp.Derivative(u*v, "x", shape=(nx+1,), order=2)
    expr4 = GridFunctionExpression(v*ux+v)
    vx = sbp.Derivative(v, "x", shape=(nx+1,), order=2)
    vxx = sbp.Derivative(vx, "x", shape=(nx+1,), order=2)
    expr5 = GridFunctionExpression(vx)
    expr6 = GridFunctionExpression(vxx)

    assert str(expr1) == 'Dx(u) + v'
    assert str(expr2) == 'Dx(u + v) + v'
    assert str(expr3) == 'v + v*Dx(u + v)'
    assert str(expr4) == 'v + v*Dx(u*v)'
    assert str(expr5) == 'Dx(v)'
    assert str(expr6) == 'Dx(Dx(v))'

    D = sbp.Derivative("", "x", shape=(nx+1,), order=2)
    mexpr1 = GridFunctionExpression(D*u + v)
    mexpr2 = GridFunctionExpression(D*(u + v) + v)
    mexpr3 = GridFunctionExpression(v*D*(u + v) + v)
    mexpr4 = GridFunctionExpression(v*D*(u*v) + v)
    mexpr5 = GridFunctionExpression(D*v)
    mexpr6 = GridFunctionExpression(v*D + v)

    assert str(mexpr1) == 'Dx*u + v'
    assert str(mexpr2) == 'Dx*(u + v) + v'
    assert str(mexpr3) == 'v + v*Dx*(u + v)'
    assert str(mexpr4) == 'v + v*Dx*u*v'
    assert str(mexpr5) == 'Dx*v'
    assert str(mexpr6) == 'v + v*Dx'

    assert expr1[0] == mexpr1[0]
    assert expr2[0] == mexpr2[0]
    assert expr3[0] == mexpr3[0]
    assert expr4[0] == mexpr4[0]
    assert expr5[0] == mexpr5[0]
    
    with pytest.raises(OperatorException) : mexpr6[0]

    # Test systems
    A = Matrix([[0,D],[D,0]])
    q = Matrix([[u],[v]])
    b = A.T*q
    assert str(b[0]) == 'Dx^T*v'
    assert str(b[1]) == 'Dx^T*u'

    # Test half-nesting
    D = sbp.Derivative("", "x", shape=(nx+1,), order=2)
    Dxx = sbp.Derivative(D*u, "x", shape=(nx+1,), order=2)
    ux = sbp.Derivative(u, "x", shape=(nx+1,), order=2)
    uxx = sbp.Derivative(ux, "x", shape=(nx+1,), order=2)

    assert str(Dxx) == 'Dx(Dx*u)'
    assert Dxx[0] == uxx[0]

    # Test nesting
    #FIXME: Fix the operator mode so that nesting works
    #assert str(D*D*u) == 'Dx**2*u'
    #assert GridFunctionExpression(D*D*u)[0] == uxx[0]


def test_operator_transpose():
    from sympy import Matrix
    clear_cache()

    nx = symbols('nx')
    u   = GridFunction("u",shape=(nx+1,))
    u_x = sbp.Derivative(u, "x", order=2)

    d = u_x.T
    assert d._transpose
    assert d.is_transpose
    assert not u_x.is_transpose

    A = Matrix([[0,u_x],[-u_x,0]])
    B = A.T
    assert B[0,1] == A[1,0]
    assert B[1,0] == A[0,1]

    # Try with matrix-based operator
    Dx = sbp.Derivative("", "x", shape=(10,10), order=2)
    Dy = sbp.Derivative("", "y", shape=(10,10), order=2)

    assert str( Dx ) == 'Dx'
    assert str( Dx.T ) == 'Dx^T'
    assert str( Dy.T ) == 'Dy^T'
    assert str( GridFunctionExpression(u*Dx).T ) == 'Dx^T*u'
    assert str( GridFunctionExpression(Dx*Dy).T ) == 'Dy^T*Dx^T'
    assert str( GridFunctionExpression(Dy*Dy).T ) == 'Dy^T**2'
    assert str( GridFunctionExpression(Dx*Dy).T ) == 'Dy^T*Dx^T'

def test_operator_coef():
    clear_cache()
    from ...dev.array import CArray
    import numpy as np
    data = np.zeros((4,))
    nx = symbols('nx')
    u   = GridFunction("u",shape=(nx+1,))
    d   = CArray("d", data)
    u_x = sbp.Derivative(u, "x", order=2, coef='d', periodic=True)
    assert u_x[1] == d[0]*u[0] + d[1]*u[2]
    u_x = sbp.Derivative(u, "x", order=4, coef='d', periodic=False)
    data = np.zeros((4,4))
    d   = CArray("dl", data)
    assert u_x[0] == d[0,0]*u[0] + d[1,0]*u[1] + d[2,0]*u[2] + d[3,0]*u[3] + d[4,0]*u[4] + d[5,0]*u[5]    
    assert u_x[1] == d[0,1]*u[0] + d[1,1]*u[1] + d[2,1]*u[2] + d[3,1]*u[3] + d[4,1]*u[4] + d[5,1]*u[5]    

    data = np.zeros((4,4))
    d   = CArray("dr", data)
    assert u_x[-1] == d[0,0]*u[-1] + d[1,0]*u[-2] + d[2,0]*u[-3] + d[3,0]*u[-4] + d[4,0]*u[-5] + d[5,0]*u[-6]    
    assert u_x[-2] == d[0,1]*u[-1] + d[1,1]*u[-2] + d[2,1]*u[-3] + d[3,1]*u[-4] + d[4,1]*u[-5] + d[5,1]*u[-6]    

    label = str(u_x.coef(0))
    assert label == 'dl'
    label = str(u_x.coef(1))
    assert label == 'd'
    label = str(u_x.coef(2))
    assert label == 'dr'

    with pytest.raises(ValueError) : u_x.coef(3)

def test_operator_gpu():
    from ...dev.array import CArray
    import numpy as np
    data = np.zeros((4,4))
    i, nx = symbols('i nx')
    u   = GridFunction("u",shape=(nx+1,))
    l   = CArray("dl",data)
    r   = CArray("dr", data)
    u_x = sbp.Derivative(u, "x", order=2, coef='d', gpu=True)
    assert u_x[0] == l[0,i]*u[0] + l[1,i]*u[1] 
    assert u_x[-1] == r[0,i]*u[-1] + r[1,i]*u[-2] 
