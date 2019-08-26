from .. import Left, Right
from sympy import symbols
def test_left():

    l = Left('i')

    lstr = str(l)
    assert lstr == 'i'

def test_right():

    r = Right('i')
    rstr = str(r)
    assert rstr == 'i'
