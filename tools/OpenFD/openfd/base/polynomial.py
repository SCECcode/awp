from sympy.core.compatibility import is_sequence
from . import GridFunctionExpression, GridFunction, Constant

class PolynomialException(Exception):
    pass

class Polynomial(GridFunctionExpression):

    """
        The polynomial class is used to build polynomial expressions that can be for instance be used to verify accuracy
        conditions.

    """

    def __new__(cls, x, degree=None, coeffs=None, *arg, **kw_args):
        """ 
          Create a polynomial expression using a Grid object. The polynomial is defined by either specifying the degree
          or the coefficients. If only the degree is specified, then a Taylor polynomial is produced.

        >>> from sympy import symbols
        >>> from openfd import Grid, Polynomial 
        >>> from sympy.core.cache import clear_cache
        >>> clear_cache()
        >>> nx = symbols('nx')
        >>> x = Grid("x",size=nx,interval=(0,1))
        >>> p = Polynomial(x,degree=2)
        >>> p
        1 + x + x**2/2
        >>> p[1]
        1 + 1/(nx - 1) + 1/(2*(nx - 1)**2)

        >>> p = Polynomial(x,coeffs=[0,2,0])
        >>> p
        2*x

        >>> p.degree
        1

        """


        # Polynomials needs to be defined by either specifying degree or coeffs
        # The number of coefficients is equal to degree + 1
        # degree 1 polynomial: a*x + b (2 coefficients)
        
        if degree is None and coeffs is None:
            raise PolynomialException("Ambiguous polynomial. Check degree or coefficients.")

        elif (coeffs is not None and degree is not None) and degree+1 != len(coeffs):
            raise PolynomialException("Degree could not be determined.")
        # Use coeffs to define polynomial
        elif coeffs is not None:
            _degree = 0
            for c in coeffs[1:]:
                if c != 0:
                    _degree = _degree + 1
            _coeffs = coeffs
        elif degree < 0:
            raise PolynomialException("Polynomial degree must non-negative.") 
        # Use degree to define the polynomial (Taylor polynomial)
        else:
            _degree = degree
            _coeffs = taylor_coefficients(degree)

        # Build expression
        expr = 0
        k = 0
        for c in _coeffs:
            if k == 0:
                expr += c*Constant(1, shape=x.shape)
            else:
                expr = expr + c*x**k
            k = k + 1

        obj = GridFunctionExpression.__new__(cls, expr, *arg, **kw_args)
        obj._degree = _degree
        obj._coeffs = _coeffs
        return obj


    @property
    def degree(self):
        return self._degree
    
    @property
    def coeffs(self):
        return self._coeffs


def taylor_coefficients(p):
    from sympy import Rational
    from math import factorial
    coeffs = [1]
    for i in range(1,p+1):
        coeffs.append(Rational(1,factorial(i)))
    return coeffs




