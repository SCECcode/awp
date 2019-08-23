import numpy
from .. import grids

def test_regular():
    n = 11
    x = grids.regular(n)
    assert len(x) == n
    assert numpy.isclose(x[0], 0)
    assert numpy.isclose(x[1], 0.1)
    x = grids.regular(n,extend=1)
    assert len(x) == n + 1
    assert numpy.isclose(x[-1], 0)

def test_cellcentered():
    n = 11
    x = grids.cellcentered(n)
    assert len(x) == n + 1
    assert numpy.isclose(x[0], 0)
    assert numpy.isclose(x[1], 0.05)
    assert numpy.isclose(x[-1], 1)

def test_x():
    n = 11
    assert numpy.all(numpy.isclose(grids.x(n), grids.regular(n)))
    assert numpy.all(numpy.isclose(grids.x(n,shift=1), grids.cellcentered(n)))

def test_xy():
    shape = (11, 12)
    shift = (0, 0)
    X, Y = grids.xy(shape, shift)
    assert X.shape[0] == shape[1]
    assert Y.shape[0] == shape[1]
    assert X.shape[1] == shape[0]
    assert Y.shape[1] == shape[0]

    X, Y = grids.xy(shape, shift, extend=True)
    assert X.shape[0] == shape[1] + 1
    assert Y.shape[0] == shape[1] + 1
    assert X.shape[1] == shape[0] + 1
    assert Y.shape[1] == shape[0] + 1
