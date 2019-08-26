import pytest
import json
from .. import tools

def load_op_data(filename=None):
    if not filename:
        filename = 'resources/staggered_D42.json'
    res = open(filename).read()
    op = json.loads(res)
    return op

def test_accuracy():
    x = tools.grid_x(11) 
    xh = tools.grid_xshift_bnd(12)
    op = load_op_data()
    tools.accuracy(x, xh, op)

def test_accuracy_shift():
    x = tools.grid_x(11) 
    xh = tools.grid_xshift_bnd(12)
    op = load_op_data('resources/staggered_Dhat42.json')
    tools.accuracy_shift(x, xh, op)

def test_add_zeros():
    op = load_op_data()
    tools.add_zeros(op, 'left', 0)
    assert op['op_left'][0][0] == 0
    assert op['op_left'][0][1] == 0
    tools.add_zeros(op, 'right', -1)
    assert op['op_right'][-1][0] == 0
    assert op['op_right'][-1][1] == 0

def test_add_interior():
    op = load_op_data()
    m = len(op['op_left'])
    tools.add_interior(op, 'left')
    assert m + 1 == len(op['op_left'])
    assert m + 1 == len(op['idx_left'])
    assert op['idx_left'][m][m] == m
    assert op['idx_left'][m][m+1] == m+1

def test_add_zero_point():
    op = load_op_data()
    tools.add_zeros(op, 'right', 0)
    x = tools.grid_xghost1(12) 
    xh = tools.grid_xshift_bnd(12)
    tools.accuracy(x, xh, op, is_outside=True)


@pytest.mark.skip(reason="Run test using code in examples/staggered_derivative\
        instead")
def test_add_interior_point():
    op = load_op_data('resources/staggered_Dhat42.json')
    tools.add_interior(op, 'right')
    x = tools.grid_x(11) 
    xh = tools.grid_xshift_bnd(12)
    tools.accuracy_shift(x, xh, op, is_interior=True)

def test_grid_x():
    n = 11
    x = tools.grid_x(n)
    tol = 1e-6
    assert abs(x[0] - 0.0) < tol
    assert abs(x[1] - 0.1) < tol

def test_grid_xghost1():
    n = 11
    x = tools.grid_xghost1(n)
    tol = 1e-6
    h = x[1] - x[0]
    assert abs(x[0] - 0.0) < tol
    assert abs(x[1] - h) < tol
    assert abs(x[-2] - 1.0) < tol
    assert abs(x[-1] - 1.0 - h) < tol

def test_grid_xshift():
    n = 11
    x = tools.grid_xshift(n)
    tol = 1e-6
    h = x[1] - x[0]
    assert abs(x[0] - 0.5*h) < tol

def test_grid_xshiftbnd():
    n = 11
    x = tools.grid_xshift_bnd(n)
    tol = 1e-6
    h = x[2] - x[1]
    assert abs(x[0]) < tol
    assert abs(x[1]-0.5*h) < tol
    assert abs(x[-1]-1.0) < tol
    assert abs(x[-2]-1.0+0.5*h) < tol

