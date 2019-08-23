import pytest
from .. import boundary 
from ... import Bounds, GridFunction
from .. import kernelgenerator as kg
from sympy import symbols


def test_kernels():
    nx, ny = symbols('nx ny')
    language = 'C'
    gridsize = (nx, ny)
    normal = (1, 0)
    bounds = (Bounds(nx, 3, 3), Bounds(ny, 4, 4))
    u = GridFunction('u', shape=gridsize)
    v = GridFunction('v', shape=gridsize)
    lhs = u 
    rhs = v
    bounds = boundary.bounds(normal, bounds)
    ka = kg.kernelgenerator(language, gridsize, bounds, lhs, rhs)
    kernels = boundary.kernels('test', normal, ka)

def test_region():
    with pytest.raises(IndexError) : boundary.region((1,), (1,0))
    with pytest.raises(IndexError) : boundary.region((1,0), (1,0))

    assert boundary.region((1,)) == (-1,)
    assert boundary.region((1,), (0,)) == (-1,)
    assert boundary.region((1,0), (1,)) == (-1,1)
    assert boundary.region((0,1), (1,)) == (1,-1)
    assert boundary.region((0,0,1), (1,0)) == (1,0, -1)
    assert boundary.region((0,1,0), (1,1)) == (1,-1, 1)

def test_get_regions():
    with pytest.raises(NotImplementedError) : boundary.get_regions(4)
    assert boundary.get_regions(2) == [(0,), (1,), (-1,)]
    assert boundary.get_regions(2, ((0,),)) == [(0,)]
    assert boundary.get_regions(3) == [
            (0,0), (0,1), (0,-1),
            (1,0), (1,1), (1,-1),
            (-1,0), (-1,1), (-1,-1),
            ]

def test_bounds():

    grid = (Bounds(10, 2, 2), Bounds(10, 2, 2))
    expected = (Bounds(10, 1, 1), Bounds(10, 2, 2))      
    normal = (1, 0, 0)
    actual = boundary.bounds(normal, grid)
    check_bounds(expected, actual)

def check_bounds(expected, actual):
    for e, a in zip(expected, actual):
        assert e.size == a.size
        assert e.left == a.left
        assert e.right == a.right

def test_label():
    assert boundary.label((-1,)) == '00'
    assert boundary.label((1,)) == '01'
    assert boundary.label((0, -1,)) == '10'
    assert boundary.label((0, 1,)) == '11'

def test_component():
    with pytest.raises(ValueError) : boundary.component((0,0))
    with pytest.raises(ValueError) : boundary.component((1.1,0))
    with pytest.raises(ValueError) : boundary.component((0,0,0))
    assert boundary.component((1,)) == 0
    assert boundary.component((-1,)) == 0
    assert boundary.component((0, -1)) == 1
    assert boundary.component((0, 0, 1)) == 2
    assert boundary.component((0, 0, 0, 1)) == 3

def test_side():
    assert boundary.side((1,)) == 1
    assert boundary.side((-1,0)) == 0
    assert boundary.side((0,-1,0)) == 0
