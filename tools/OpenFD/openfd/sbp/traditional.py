from ..base import Operator
from . import DifferenceOperator
from sympy import Matrix
from . import differenceoperator as do

class Derivative(DifferenceOperator):
    """ This class is used to construct standard SBP(2p, p) difference operators 
    that use a 2p order accurate approximation in the interior and a p order accurate 
    approximation at the boundary. 
    
    Example demonstrating SBP(2, 1) assuming that the grid spacing is set
    to unity.
    >>> from sympy import symbols
    >>> from openfd import GridFunction, Operator, sbp_traditional as sbp
    >>> nx,j = symbols('nx j')
    >>> u    = GridFunction("u",shape=(nx+1,))
    >>> u_x  = sbp.Derivative(u,"x", order=2)
    >>> u_x[0]
    -1.0*u[0] + 1.0*u[1]
    >>> u_x[nx]
    1.0*u[nx] - 1.0*u[nx - 1]
    >>> u_x[j]
    -0.5*u[j - 1] + 0.5*u[j + 1]

    To include the grid spacing in the computation, we can introduce a new symbol 'h' 
    for it. 
    >>> nx,j,h = symbols('nx j h')
    >>> u    = GridFunction("u",shape=(nx+1,))
    >>> u_x  = sbp.Derivative(u,"x",gridspacing=h, order=2)
    >>> u_x[0]
    (-1.0*u[0] + 1.0*u[1])/h

    
    For computational performance reasons, floating point divisions should be avoid 
    whenever possible. To avoid division we
    introduce its reciprocal 'hi' (hi = 1/h) instead.
    >>> nx,j,hi = symbols('nx j hi')
    >>> u    = GridFunction("u",shape=(nx+1,))
    >>> u_x  = sbp.Derivative(u,"x", rgridspacing=hi, order=2)
    >>> u_x[0]
    hi*(-1.0*u[0] + 1.0*u[1])

    """
    
    def __new__(cls, expr, axis, order=2, shape=None, gridspacing=1, rgridspacing=None, 
                periodic=False,
                fmt='traditional_D%s.json', **kw_args):

        """
        Differentiate a symbolic expression using a high order 
        finite difference approximation on a staggered grid.

        Parameters

        expr : GridFunctionExpression,
               symbolic expression to differentiate
        axis : string,
               the direction in which to compute the derivative 'x', 'y', or 'z'
        order : int,
                the order of accuracy of the approximation in the interior. 
        gridspacing : Symbol, optional,
                      the grid spacing to use in the evaluation of the derivative. 
                      After evaluation, the final expression will be divided by `gridspacing`.
                      To avoid divisions, use `rgridspacing`.
        rgridspacing : Symbol, optional,
                       this is the reciprocal grid spacing (1/gridspacing). When
                       specified, it causes the final expression to be multiplied by `rgridspacing`.
        periodic : bool, optional
                   set `periodic = True` to repeat the interior stencil in a periodic manner at the boundary.
        hat : bool, optional,
              set `hat = True` to apply the operator on the shifted grid.
              Otherwise, the operator is applied on the regular grid. 
        fmt : string, optional,
              specifies the formatting string that is used to load .json data.
              Change this value to load experimental operator implementations.

        Returns

        out : Derivative,
              a symbolic object representing the differentation of the expression.

        """

        operatorname = fmt % (_implementation(order))
        d = do.readderivative(operatorname)

        return DifferenceOperator.__new__(cls, expr, axis, d['label'], shape, 
                                          gridspacing = gridspacing, operator_data = d, 
                                          rgridspacing = rgridspacing, 
                                          periodic = periodic, **kw_args)

def _implementation(order):
    impl = {2 : '21',  4 : '42',  6 : '63'}
    if order not in impl:
        raise ValueError('Traditional Derivative `order = %d` not implemented' % order)
    return impl[order]
