from .. equations import Equation
from .. variable import Variable
import openfd


def test_init():
    eq = Equation('a', 'b')
    assert eq.lhs == 'a'
    assert eq.rhs == 'b'
    var = Variable('a', dtype=openfd.prec, val='b')
    eq = Equation(var)
    assert str(eq.lhs) == 'a'
    assert str(eq.rhs) == 'b'
