from openfd import Bounds, gridfunctions, Left, Right
from .. import sponge
from sympy import symbols

def test_cerjan():
    bounds = (Bounds(10, 4, 4), Bounds(10, 4, 4))
    damp = sponge.cerjan(0, 0, bounds)
    i, j = symbols('i j')
    assert damp.subs(i, 3) == 1
    
    damp = sponge.cerjan(0, 1, bounds)
    assert damp.subs(i, 6) == 1
    damp = sponge.cerjan(1, 1, bounds)
    assert damp.subs(j, 6) == 1

def test_new_cerjan():
    bounds = (Bounds(10, 4, 4), Bounds(10, 4, 4))
    u, v = gridfunctions('u v', shape=(10, 10))
    Cu = sponge.Cerjan(u, bounds)
    Cv = sponge.Cerjan(v, bounds)
    i = Left('i')
    j = Right('j')
    print(Cu[i,j])
    assert 0




