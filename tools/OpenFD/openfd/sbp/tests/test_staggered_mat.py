import pytest
from .. import staggered_mat as sm
import numpy as np
import mpmath as mp
np.set_printoptions(linewidth=200)

def ncmp(a):
    return np.abs(a) < 1e-12

def accuracy(D, n, order, hat=False):
    for p in range(int(order/2)+1):
        assert ncmp(D.dot(f(n, p, hat=not hat)) - p*f(n, max(p-1, 0), hat=hat)).all()

def quadratureaccuracy(H, n, order, hat=False):
    for p in range(int(order/2)+1):
        fp = f(n, max(p+1, 0), hat=hat)/(p+1)      
        v = H.diagonal().dot(f(n, p, hat=hat))
        assert ncmp(v - (fp[-1] - fp[0]))

def f(n, p, hat=False):
    x = sm.grid(n, hat)
    x = np.array([xi**p for xi in x], dtype=np.float64)
    return x

def test_gridspacing():
    assert ncmp(sm.gridspacing(10) - 0.2)
    assert ncmp(sm.gridspacing(10, 0, 1) - 0.1)
    assert ncmp(sm.gridspacing(10, -1, 0) - 0.1)

def test_grid():
    assert ncmp(len(sm.grid(10)) - 11)
    assert ncmp(sm.grid(10)[0] + 1)
    assert ncmp(sm.grid(10)[1] - sm.gridspacing(10) + 1)
    assert ncmp(sm.grid(10)[-1] - 1)

    assert ncmp(len(sm.grid(10, hat=True)) - 12)
    assert ncmp(sm.grid(10, hat=True)[0] + 1)
    assert ncmp(sm.grid(10, hat=True)[1] - 0.5*sm.gridspacing(10) + 1)
    assert ncmp(sm.grid(10, hat=True)[-1] - 1)
    assert ncmp(sm.grid(10, hat=True)[-2] + 0.5*sm.gridspacing(10) - 1)

def test_derivative():
    n = 18
    h = sm.gridspacing(n)
    for order in range(2,8,2):
        D = sm.derivative(n, h, order)
        accuracy(D, n, order)
    for order in range(2,8,2):
        Dh = sm.derivative(n, h, order, hat=True)
        accuracy(Dh, n, order, hat=True)

def test_quadrature():
    n = 18
    h = sm.gridspacing(n)
    for order in range(2,8,2):
        Hi = sm.quadrature(n, h, order, invert=False)
        quadratureaccuracy(Hi, n, order)
    for order in range(2,8,2):
        Hih = sm.quadrature(n, h, order, hat=True, invert=False)
        quadratureaccuracy(Hih, n, order, hat=True)

def test_restrict():
   e0 = sm.restrict(10, 0)
   en = sm.restrict(10, 1)
   assert ncmp(e0[0,0] - 1)
   assert ncmp(en[-1,0] - 1)
   e0 = sm.restrict(10, 0, hat = True)
   en = sm.restrict(10, 1, hat = True)
   assert ncmp(e0[0,0] - 1)
   assert ncmp(en[-1,0] - 1)
