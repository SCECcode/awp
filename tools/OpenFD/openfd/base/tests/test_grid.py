from .. import Grid, GridException, GridFunctionExpression, StaggeredGrid
from sympy import symbols
from sympy.core.cache import clear_cache
import pytest

def test_grid_new():
    clear_cache()

    nx,ny,nz = symbols("nx ny nz")

    g   = Grid("g",size=nx)
    g   = Grid("g",size=nx,interval=(0,1))


def test_grid_getitem():
    clear_cache()

    a,b,nx,ny,nz = symbols("a b nx ny nz")

    x   = Grid("x",size=nx+1,interval=(0,1))

    assert x[0] == 0
    assert x[nx] == 1
    
    x   = Grid("x",size=nx+1,interval=(a,b))
    
    assert x[0] == a
    assert x[nx] == b
    
    with pytest.raises(IndexError): x[nx+1]


def test_grid2d_getitem():

    clear_cache()

    a,b,nx,ny,nz = symbols("a b nx ny nz")

    x   = Grid("x",size=nx,interval=(0,nx),axis=1)

    assert x[0,0] == 0
    assert x[0,nx-1] == nx

    


    
def test_staggered_grid_getitem():
    from sympy import simplify
    clear_cache()

    a,b,nx,ny,nz = symbols("a b nx ny nz")

    # Staggered grid including boundary points
    g = StaggeredGrid("g", size=nx+2, interval=(0, 1))
    
    assert g[0] == 0
    assert g[nx+1] == 1
    h = 1/nx
    assert g[1] == h/2
    assert simplify(g[nx] - (1 - h/2)) == 0
    
