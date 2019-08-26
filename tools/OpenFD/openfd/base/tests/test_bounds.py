from .. import Bounds
from openfd import Left, Right
from sympy import symbols
from sympy.core.cache import clear_cache
import pytest

def test_bounds_new():
    clear_cache()

    nx = symbols("nx")
    b   = Bounds(size=nx+1, left=3, right=3)

    with pytest.raises(ValueError): Bounds(size=1, left=1, right=1)
    with pytest.raises(ValueError): Bounds(size=nx, left=-1, right=0)
    with pytest.raises(ValueError): Bounds(size=nx, left=0, right=-1)

def test_bounds_left():
    clear_cache()
    
    nx, j  = symbols("nx j")
    b   = Bounds(size=nx+1, left=3, right=3)
    
    assert b.is_left(Left(0))
    assert b.is_left(0)
    assert b.is_left(1)
    assert b.is_left(2)
    assert not b.is_left(3)
    assert not b.is_left(-1)
    assert not b.is_left(j)
    
    b   = Bounds(size=nx+1, left=0, right=0)
    assert not b.is_left(0)
    assert not b.is_right(-1)

def test_is_right():
    clear_cache()
    
    nx, i = symbols("nx i")
    b   = Bounds(size=nx+1, left=3, right=3)
    
    assert b.is_right(Right(0))
    assert not b.is_right(0)
    assert not b.is_right(1)
    assert not b.is_right(2)
    assert not b.is_right(i+1)
    assert not b.is_right(i-1)
    assert b.is_right(nx)
    assert b.is_right(nx-1)
    assert b.is_right(nx-2)
    assert not b.is_right(nx-3)
    assert not b.is_right(nx+1)
    assert b.is_right(-1)
    assert b.is_right(-2)
    assert b.is_right(-3)
    assert not b.is_right(-4)

    b   = Bounds(size=20, left=1, right=1)
    assert not b.is_right(i+1)
    assert not b.is_right(i-1)
    
    nx = symbols("nx")
    b   = Bounds(size=nx+2, left=3, right=3)
    
    assert b.is_right(nx+1)
    assert not b.is_right(nx+2)

def test_is_interior():
    clear_cache()
    
    i, nx   = symbols("i nx")
    b       = Bounds(size=nx+1, left=3, right=3)

    assert b.is_interior(i)
    assert b.is_interior(nx-3)
    assert b.is_interior(3)

def test_getitem():
    clear_cache()
    
    nx, j  = symbols("nx j")
    b   = Bounds(size=nx+1, left=3, right=3)
    
    with pytest.raises(IndexError): b[-4]
    with pytest.raises(IndexError): b[nx+1]
    
    assert b[j]  == b.tag_interior
    assert b[0]  == b.tag_left
    assert b[nx] == b.tag_right
    assert b[-1] == b.tag_right

def test_in_bounds():
    clear_cache()
    
    nx = symbols("nx")
    b   = Bounds(size=nx+1, left=3, right=3)

    assert b.inbounds(1)
    assert b.inbounds(nx)
    assert not b.inbounds(nx+2)
    assert not b.inbounds(-4)

def test_is_overflow():
    clear_cache()
    
    nx,i = symbols("nx i")
    b   = Bounds(size=nx+1, left=3, right=3)
    
    assert not b.is_overflow(1)
    assert not b.is_overflow(nx)
    assert not b.is_overflow(-1)
    assert b.is_overflow(nx+2)
    assert b.is_overflow(nx+3)
    assert not b.is_overflow(i)

def test_is_underflow():
    clear_cache()
    
    nx,i = symbols("nx i")
    b   = Bounds(size=nx+1, left=3, right=3)
    
    assert b.is_underflow(-4)
    assert b.is_underflow(-5)
    assert not b.is_underflow(1)
    assert not b.is_overflow(i)
    assert not b.is_underflow(i-1)

def test_range():
    clear_cache()
    
    nx,i = symbols("nx i")
    b = Bounds(size=3, left=1, right=1)
    assert b.left_range     == (0, 1)
    assert b.right_range    == (2, 3)
    assert b.interior_range == (1, 2)

    b = Bounds(size=4, left=1, right=1)
    assert b.left_range     == (0, 1)
    assert b.interior_range == (1, 3)
    
    b = Bounds(size=nx+1, left=1, right=1)
    assert b.right_range    == (nx, nx+1)
    assert b.interior_range == (1, nx)

    b = Bounds(size=nx+1, left=0, right=0)
    b = Bounds(size=2, left=1, right=1)

    b = Bounds(size=4, left=1, right=1)
    assert b.range(0) == b.left_range
    assert b.range(1) == b.interior_range
    assert b.range(-1) == b.right_range

def test_periodic():

    nx,i = symbols("nx i")
    b = Bounds(size=3, periodic=True)
    assert b[-1]   == b.tag_interior
    assert b[0]    == b.tag_interior
    assert b[nx+1] == b.tag_interior

def test_cycle():
    nx,i = symbols("nx i")
    b = Bounds(size=nx, periodic=True)
    assert b.cycle(-1) == nx - 1 
    assert b.cycle(-2) == nx - 2
    assert b.cycle(nx) == 0
    b = Bounds(size=nx+1, periodic=True)
    assert b.cycle(-1) == nx 
    assert b.cycle(-2) == nx - 1
    assert b.cycle(nx+1) == 0
