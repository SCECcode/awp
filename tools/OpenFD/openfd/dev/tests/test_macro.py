from ..macro import Macro
import sympy as sp

def test_guard():
    x, y = sp.symbols('x y')
    expr = x + y
    M = Macro('m',[x, y], expr)
    mg = M.guard()
    assert str(mg) == '(x) + (y)'
