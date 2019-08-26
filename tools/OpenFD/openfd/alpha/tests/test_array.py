from .. import array
from sympy import symbols
from sympy.printing import ccode
import numpy as np
import pytest
i, j = symbols('i j')

def test_carray_getitem():
    a = array.CArray('a', np.array([0]))
    assert str(a[i]) == 'a[i]'

    with pytest.raises(IndexError): a[i,j]

def test_array_ccode():
    a = array.CArray('a', np.array([0]))
    assert ccode(a[i]) == 'a[i]'

def test_array2d_getitem():
    a = array.CArray('a', np.array([[0], [0]]))
    with pytest.raises(IndexError): a[i]

def test_array2d_sympystr():
    a = array.CArray('a', np.array([[0],[0]]))
    str(a) == 'a'
    assert str(a[i,j]) == 'a[i][j]'

def test_array2d_getitem_ccode():
    a = array.CArray('a', np.array([[0],[0]]))
    assert ccode(a[i,j]) == 'a[i][j]'

def test_array2d_ccode():
    a = array.CArray('a', np.array([[0, 1],[2, 3]]))
    astr = 'const float a[2][2] = {{0, 1}, {2, 3}};'
    assert ccode(a) == astr

def test_shape():
    a = array.CArray('a', np.array([[0, 1]]))
    assert a.shape[0] == 1
    assert a.shape[1] == 2

def test_array_list_to_array():
    import numpy as np
    a = array.CArray('a', [[0, 1]])
    assert a.shape[0] == 1
    assert a.shape[1] == 2

