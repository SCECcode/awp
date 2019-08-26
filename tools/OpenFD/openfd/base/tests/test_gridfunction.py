from .. import gridfunctions, GridFunction, GridFunctionException, \
               GridFunctionExpression
from .. import gridfunction as gridfunc
from sympy import symbols
from sympy.core.cache import clear_cache
import pytest

def test_gridfunctions():
    from openfd import Memory
    from sympy import ccode, symbols

    nx, ny = symbols('nx ny')
    shape = (nx, ny)
    mem = Memory(shape=shape, perm=(1,0))
    u, v = gridfunctions('u v', shape=shape, layout=mem)
    u_ = GridFunction('u', shape=shape, layout=mem)
    v_ = GridFunction('v', shape=shape, layout=mem)
    u_ = GridFunction('u', shape=shape, layout=mem)
    assert u == u_
    assert ccode(u[1,1]) == ccode(u_[1,1])
    assert v == v_

    remap = lambda idx : (idx[0] + 1, idx[1] + 1)
    u, v = gridfunctions('u v', shape=shape, layout=mem, remap=remap)
    assert str(u[0, 0]) == 'u[1, 1]'

    out = gridfunctions('u v', shape=shape, struct=1)
    assert out.u == u_
    assert out.v == v_

    out = gridfunctions('u v', shape=shape, struct=1, remap=remap)
    assert str(out.u[0, 0]) == 'u[1, 1]'

def test_gridfunction_new():
    clear_cache()
    nx,ny,nz = symbols("nx ny nz")

    u = GridFunction("u",shape=(nx,))
    v = GridFunction("u",shape=(nx,))
    assert u == v
    
    u = GridFunction("u",shape=(nx,ny))
    v = GridFunction("v",shape=(nx,ny))
    assert u != v

def test_gridfunction_shape():
    clear_cache()
    nx,ny,nz = symbols("nx ny nz")

    u = GridFunction("u",shape=(nx,))
    assert u.shape == (nx,)

    u = GridFunction("u",shape=(nx,ny))
    assert u.shape == (nx,ny)
    
    u = GridFunction("u",shape=(nx,ny,nz))
    assert u.shape == (nx,ny,nz)

def test_gridfunction_getitem():
    clear_cache()
    i,j,k,nx,ny,nz = symbols("i j k nx ny nz")
    
    u = GridFunction("u",shape=(nx,))
    v = GridFunction("v",shape=(nx,))
    assert u[i]     != v[j]
    assert u.shape  == (nx,)
    with pytest.raises(IndexError): u[nx+1]
    u[0]
    u[-1]
    u[i]

    u = GridFunction("u",shape=(nx,ny))
    v = GridFunction("v",shape=(nx,ny))
    assert u[i,j]   != v[i,j]
    assert u.shape  == (nx,ny)
    with pytest.raises(IndexError): u[i]
    with pytest.raises(IndexError): u[nx+1, ny+1]

    u = GridFunction("u",shape=(nx,ny,nz))
    v = GridFunction("v",shape=(nx,ny,nz))
    assert u[i,j,k] != v[i,j,k]
    assert u.shape  == (nx,ny,nz)
    with pytest.raises(IndexError): u[i]
    with pytest.raises(IndexError): u[i,j]

    u[0,0,0]
    u[i,0,0]
    u[i+j,0,i]
    u[-1,0,j]
    u[(i,j,k)]
    
    # Test remapping
    remap = lambda idx : (idx[0] + 1, idx[1], idx[2])
    u = GridFunction("u",shape=(nx,ny,nz), remap=remap)
    assert str(u[i,j,k]) == 'u[i + 1, j, k]'


    # Test smaller dimensions that index dimension
    u = GridFunction("u",shape=(nx,ny), dims=(0,2))
    assert str(u[i,j,k]) == 'u[i, k]'
    
         
def test_gridfunc_eval():
    clear_cache()
    i,j,k,nx,ny,nz = symbols("i j k nx ny nz")
    u = GridFunction("u",shape=(nx,))
    v = GridFunction("v",shape=(nx,))
    gridfunc.eval(u+v,i)
    assert gridfunc.eval(u + v,i) == u[i] + v[i]
    assert gridfunc.eval(u + u,i) == 2*u[i]
    
    p = GridFunction("p",shape=(nx,ny))
    q = GridFunction("q",shape=(nx,ny))
    assert gridfunc.eval(p + q,(i,j))== p[i,j] + q[i,j]

def test_gridfunction_periodic():
    clear_cache()
    i, nx = symbols("i nx")
    u = GridFunction("u",shape=(nx,), periodic=True)
    assert u[-1] == u[nx-1]
    assert u[nx] == u[0]
    assert u[nx+1] == u[1]
    u = GridFunction("u",shape=(nx+1,), periodic=True)
    assert u[-1] == u[nx]
    assert u[nx+1] == u[0]
    assert u[nx+2] == u[1]
    
def test_gridfunc_expression_new():
    clear_cache()
    i,j,k,nx,ny,nz = symbols("i j k nx ny nz")

    u = GridFunction("u",shape=(nx,))
    expr = GridFunctionExpression(u)
    expr = GridFunctionExpression(0*u)

def test_gridfunc_matrix():

    from ... import sbp_traditional as sbp 
    from .. import Matrix

    u1   = GridFunction("u1", shape=(10,))
    u2   = GridFunction("u2", shape=(10,))
    u3   = GridFunction("u3", shape=(10,))
    u = GridFunctionExpression(u1*u2*u3 + 2)
    uT = GridFunctionExpression(u3*u2*u1 + 2)
    assert u.T == uT

    A1 = Matrix([[0,u1],[0,0]])
    A2 = Matrix([[0,2],[u2,0]])
    A3 = Matrix([[0,3],[0,3]])
    C = (A1*A2*A3).T
    D = A3.T*A2.T*A1.T
    assert C[0,0] == D[0,0]
    assert C[1,0] == D[1,0]
    assert C[0,1] == D[0,1]
    assert C[1,1] == D[1,1]

def test_gridfunc_memory():
    from ...dev import memory
    from sympy.printing import ccode

    i,j,k = symbols("i j k")
    nx,ny,nz = symbols("nx ny nz")
    mx,my,mz = symbols("mx my mz")
    mem = memory.Memory((mx, my, mz))

    u = GridFunction("u", shape=(nx, ny, nz), layout=mem)
    assert ccode(u[i, j, k]) == "u[i + j*mx + k*mx*my]"
    
    with pytest.raises(TypeError) : GridFunction("u", shape=(nx, ny), layout=i)

def test_gridfunc_ccode():
    from sympy.printing import ccode

    i, j, nx, ny = symbols("i j nx ny")
    u = GridFunction("u", shape=(nx, ny))
    assert ccode(u[i, j]) == "u[i + j*nx]"

def test_gridfunc_macro():
    i, j, nx, ny = symbols("i j nx ny")
    u = GridFunction("u", shape=(nx, ny), macro='_')
    M = u.macro(i,j)
    assert M.define() == '#define _u(i,j) (i) + (j)*nx' 
    assert M.undefine() == '#undef _u' 


