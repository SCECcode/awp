from .. import tensoroperator as to
from .. import Operator
from .. import GridFunction
from .. import Expression as Expr
from .. import types 
from .. import array
from .. import stencil
from sympy import srepr
import pytest
import numpy as np

from sympy.printing import ccode

def test_call():
    coef = array.CArray('d', np.array((-0.5, 0.5)))
    st = stencil.ConvolutionalStencil(coef, (-1, 1))
    D = to.TensorOperator('D', data=[st])
    d = Operator('A', test=2, x=1)
    u = GridFunction('u', shape=(10,10))



    # No axis
    with pytest.raises(ValueError) : Expr(D*u)[0,0]

    # 
    # -2, -1, 0, 1
    # -1, 0, 1, 2

    expr = Expr(D.x*u)
    print(ccode(st.T.coef))
    print(ccode(st.coef))

    i = types.Index('i', region=0)
    j = types.Index('j', region=0)
    print(expr[i,j])
    print(Expr(D.x.T*u)[i,j])
    assert 0
