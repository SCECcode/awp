from .. import kernel as k
from .. import Bounds, GridFunction, GridFunctionExpression
from ... import sbp_traditional as sbp
from sympy import symbols
from sympy.core.cache import clear_cache
import numpy as np

def test_kernel1d():
    clear_cache()
    
    nx = symbols("nx")
    u = GridFunction("u", shape=(nx+1,))
    v = GridFunction("v", shape=(nx+1,))

    v_x    = sbp.Derivative(v, "x", shape=(nx+1,), order=4)
    b = v_x.bounds()

    lhs = [u]
    rhs = [v_x]
    code = ''
    code = k.kernel1d(b, 0, lhs, rhs)
    code += k.kernel1d(b, -1, lhs, rhs)
    code += k.kernel1d(b, 1, lhs, rhs)

    code = ''
    code = k.kernel1d(b, 0, u, v)

def test_kernel2d():
    clear_cache()
    
    nx = symbols("nx")
    ny = symbols("ny")

    u = GridFunction("u", shape=(nx+1, ny+1))
    v = GridFunction("v", shape=(nx+1, ny+1))

    v_x = sbp.Derivative(v, "x", shape=(nx+1, ny+1), order=4)
    v_y = sbp.Derivative(v, "y", shape=(nx+1, ny+1), order=4)

    b = (v_x.bounds(), v_y.bounds())

    lhs = [u]
    rhs = [GridFunctionExpression(v_x + v_y)]
    code = ''
    code += k.kernel2d(b, (1, 1), lhs, rhs) + '\n\n'
    code += k.kernel2d(b, (0, 1), lhs, rhs) + '\n\n'

def test_kernel3d():
    clear_cache()
    
    nx = symbols("nx")
    ny = symbols("ny")
    nz = symbols("nz")

    u = GridFunction("u", shape=(nx+1,  ny+1, nz+1))
    v = GridFunction("v", shape=(nx+1,  ny+1, nz+1))

    v_x = sbp.Derivative(v, "x", shape=(nx+1, ny+1, nz+1), order=2)
    v_y = sbp.Derivative(v, "y", shape=(nx+1, ny+1, nz+1), order=2)
    v_z = sbp.Derivative(v, "z", shape=(nx+1, ny+1, nz+1), order=2)

    b = (v_x.bounds(), v_y.bounds(), v_z.bounds())

    lhs = [u]
    rhs = [GridFunctionExpression(v_x + v_y + v_z)]
    code = ''
    code += k.kernel3d(b, (1, 1, 1), lhs, rhs) + '\n\n'
    code += k.kernel3d(b, (0, 1, 0), lhs, rhs) + '\n\n'

def test_ckernel():
    clear_cache()

    nx = symbols("nx")
    ny = symbols("ny")
    nz = symbols("nz")

    b = (nx, ny, nz)

    u = GridFunction("u", shape=(nx+1,  ny+1, nz+1))
    v = GridFunction("v", shape=(nx+1,  ny+1, nz+1))
    
    v_x = sbp.Derivative(v, "x", shape=(nx+1, ny+1, nz+1), order=2)
    v_y = sbp.Derivative(v, "y", shape=(nx+1, ny+1, nz+1), order=2)
    v_z = sbp.Derivative(v, "z", shape=(nx+1, ny+1, nz+1), order=2)

    lhs = [u]
    rhs = [GridFunctionExpression(v_x + v_y + v_z)]
    bounds = (v_x.bounds(), v_y.bounds(), v_z.bounds())
    body = k.kernel3d(bounds, (1, 1, 1), lhs, rhs)

    k.ckernel("test", b, lhs, rhs, body, header=True)
    k.ckernel("test", b, lhs, rhs, body)

def test_dshape():
    clear_cache()

    nx = symbols("nx")
    ny = symbols("ny")
    nz = symbols("nz")

    v = GridFunction("v", shape=(nx+1,  ny+1, nz+1))

    v_x = sbp.Derivative(v, "x", shape=(nx+1, ny+1, nz+1), order=2)
    v_y = sbp.Derivative(v, "y", shape=(nx+1, ny+1, nz+1), order=2)
    v_z = sbp.Derivative(v, "z", shape=(nx+1, ny+1, nz+1), order=2)

    b = ( (0, nx), (2, ny), (3, nz) )
    assert k._dshape(b) == (nx, ny, nz)
    b = ( (0, 1), (2, 3), (3, 4) )
    assert k._dshape(b) == ()

def test_isptr():
    v = GridFunction("v", shape=(1,  1, 1))
    assert k._isptr(v)

def test_array():
    a = [1.1, 2.2, 3.3]
    code = k.array('a', a)
    assert code == 'const float a[3] = {1.1, 2.2, 3.3};\n'
    b = np.array([[0.0, 1.0],[2.0, 3.0]])
    code = k.array('b', b.flatten())
    assert code == 'const float b[4] = {0, 1, 2, 3};\n'
