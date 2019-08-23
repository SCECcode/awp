from .. import GridFunction
from sympy import symbols

def test_getitem():
    u = GridFunction('u', shape=(10, 10))
    i, j = symbols('i j')
    u[0,0] + u[1,1] 
    u[(0,0)] + u[(1,1)] 
    u[[0,0]] + u[[1,1]]
    u[i+1,j] + u[i,j]
    u[(i+1,j)] + u[(i,j)]
    u[[i+1,j]] + u[[i,j]]

    # Test wrap-around
    u = GridFunction('u', shape=(10, 10), lower_out_of_bounds='wrap-around')
    assert u[-1,0] == u[9,0]
    assert u[-1,-1] == u[9,9]

    # Test no-action
    u = GridFunction('u', shape=(10, 10), lower_out_of_bounds='no action')
    assert u[-1,0] == u[-1,0]
    u = GridFunction('u', shape=(10, 10), upper_out_of_bounds='no action')
    assert u[11,0] == u[11,0]

def test_call():

    u = GridFunction('u', shape=(10, 10))
    i, j = symbols('i j')
    u(0,0) + u(1,1)
    u(i+1,j) + u(i,j)
    u((i+1,j)) + u(i,j)
