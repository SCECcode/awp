from .. import GridFunction, Expression

def test_expression_getitem():
    u = GridFunction('u', shape=(10,))
    v = GridFunction('v', shape=(10,))
    assert Expression(u + v)[0] == u[0] + v[0]

def test_expression_call():
    u = GridFunction('u', shape=(10,))
    v = GridFunction('v', shape=(10,))
    assert Expression(u + v)(0) == u[0] + v[0]

