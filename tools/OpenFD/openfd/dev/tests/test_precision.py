def test_double():
    """
    Check that arrays are outputted in double precision.

    """
    from openfd import sbp_staggered as sbp
    from sympy import  ccode
    D = sbp.Derivative('', 'x', shape=(10, ), order=4, gpu=True)
    coef = D._coef['left']
    assert '-2.3959440875028211' in (ccode(coef, precision=16))

