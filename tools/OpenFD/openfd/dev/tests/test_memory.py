from sympy import symbols
import pytest

from .. import memory

def test_init():
    shape = (10, 10, 10)
    align = (1, 1, 1)
    mem = memory.Memory(shape, align=align)
    assert mem.shape == shape
    assert mem.align == align

    # Wrong number of items
    with pytest.raises(AssertionError) : memory.Memory((1, 1), (1,)) 
    with pytest.raises(AssertionError) : memory.Memory((1, 1), (1,1), (1,)) 

    # Not a valid permutation
    with pytest.raises(AssertionError) : memory.Memory((1, 1), (1,1), (0, 0)) 
    with pytest.raises(AssertionError) : memory.Memory((1, 1), (1,1), (2, 1)) 
    with pytest.raises(AssertionError) : memory.Memory((1, 1), (1,1), (3, 2)) 

def test_get_c():

    # Test regular access
    mem = memory.Memory((10,))
    assert mem.get_c(0) == 0
    assert mem.get_c(1) == 1
    
    mem = memory.Memory((10, 10))
    assert mem.get_c(9, 1) == 19
    assert mem.get_c(9, 2) == 29

    i, j, k, nx, ny, nz = symbols('i j k nx ny nz')
    mem = memory.Memory((nx, ny))
    assert mem.get_c(i, j) == i + nx*j
    mem = memory.Memory((nx, ny, nz))
    assert mem.get_c(i, j, k) == i + nx*j + nx*ny*k
    
    mem = memory.Memory((nx, ny))
    with pytest.raises(IndexError) : mem.get_c(1, 1, 1) 
    
    # Test access with alignment
    mem = memory.Memory((10,), align=(1,))
    assert mem.get_c(0) == 1
    assert mem.get_c(1) == 2
    
    mem = memory.Memory((10, 10), align=(1,1))
    assert mem.get_c(0, 0) == 11
    assert mem.get_c(9, 1) == 30
    i, j, k, nx, ny, nz = symbols('i j k nx ny nz')
    mem = memory.Memory((nx, ny), align=(1, 1))
    assert mem.get_c(i, j) == (i + 1) + nx*(j + 1)
    mem = memory.Memory((nx, ny, nz), align=(1, 1, 1))
    assert mem.get_c(i, j, k) == (i + 1) + nx*(j + 1) + nx*ny*(k + 1)

    # Test access with permutation
    i, j, k, nx, ny, nz = symbols('i j k nx ny nz')
    mem = memory.Memory((nx, ny), perm=(1, 0))
    assert mem.get_c(i, j) == j + ny*i
    mem = memory.Memory((nx, ny, nz), perm=(2, 1, 0))
    assert mem.get_c(i, j, k) == k + nz*j + ny*nz*i
