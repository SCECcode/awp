import numpy as np
from ..base import Operator
from . import DifferenceOperator
from sympy import Matrix
from . import Restriction
from six import iteritems
from . import differenceoperator as do

"""
This module contains summation-by-parts finite difference operators on staggered grids. 

There are two types of operators depending on which type of grid it acts on. 
The parameter `hat` is used to specify the grid.
Use `hat = False` to construct a difference operator on the regular grid.
Use `hat = True` to construct a difference operator on the shifted grid.

The order of accuracy of the operators is determined by the parameter `order` 
and corresponds to the interior order of accuracy. 
The boundary accuracy is always half of the interior accuracy.


"""

class Derivative(DifferenceOperator):
    """ 
    The Derivative class is used to construct first derivatives on a staggered grid. 

    """
    def __new__(cls, expr, axis, order=2, shape=None, gridspacing=1, rgridspacing=None, 
                hat=False, periodic=False, fmt='staggered_D%s%s.json',**kw_args):
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
        hat : bool, optional,
              set `hat = True` to apply the operator on the shifted grid.
              Otherwise, the operator is applied on the regular grid. 
        coef : str, optional,
                Print coefficient index instead of coefficient data using `coef` as the label for the coefficient data
                holder.
        periodic : bool, optional,
                   set `periodic = True` to use periodic boundary stencils.
               
        fmt : string, optional,
              specifies the formatting string that is used to load .json data.
              Change this value to load experimental operator implementations.

        Returns

        out : Derivative,
              a symbolic object representing the differentation of the expression.

        """

        d = readderivative(order, hat, fmt=fmt)
        shape = _infershape(expr, axis, shape, hat, periodic)


        return DifferenceOperator.__new__(cls, expr, axis, d['label'], shape, gridspacing=gridspacing, 
                                 operator_data = d, rgridspacing = rgridspacing, periodic=periodic, 
                                 **kw_args)

class Interpolation(DifferenceOperator):
    """ 
    The Interpolation class is used to construct interpolation operators on a staggered grid. 

    """
    def __new__(cls, expr, axis, order=2, shape=None, gridspacing=1, rgridspacing=None, 
                hat=False, periodic=False, fmt='staggered_P%s%s.json', **kw_args):
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
        hat : bool, optional,
              set `hat = True` to apply the operator on the shifted grid.
              Otherwise, the operator is applied on the regular grid. 
        periodic : bool, optional,
                   set `periodic = True` to use periodic boundary stencils.
               
        fmt : string, optional,
              specifies the formatting string that is used to load .json data.
              Change this value to load experimental operator implementations.

        Returns

        out : Interpolation,
              a symbolic object representing the interpolation of the expression.

        """

        d = readinterpolation(order, hat)
        shape = _infershape(expr, axis, shape, hat, periodic)


        return DifferenceOperator.__new__(cls, expr, axis, d['label'], shape, gridspacing=gridspacing, 
                                 operator_data = d, rgridspacing = rgridspacing, periodic=periodic, **kw_args)
    
    @property
    def symbol(self):
        return 'P'

class Quadrature(DifferenceOperator):
    """ 
    The Quadrature class is used to construct quadrature operators (norm operators) on a staggered grid. 

    """
    def __new__(cls, expr, axis, order=2, shape=None, gridspacing=1, rgridspacing=None, 
                hat=False, invert=False, fmt='staggered_H%s%s.json', **kw_args):
        """
        Multiply a symbolic expression by the weights in a quadrature rule.

        Parameters

        expr : GridFunctionExpression,
               Symbolic expression to multiply by quadrature weights.
        axis : string,
               The direction in which to compute to apply the quadrature rule.
        order : int,
                The order of accuracy of the associated difference operator.
                The degree of the quadrature rule is `order - 1`.
        gridspacing : Symbol, optional,
                      the grid spacing to use in the evaluation of the derivative. 
                      After evaluation, the final expression will be divided by `gridspacing`.
                      To avoid divisions, use `rgridspacing`.
        rgridspacing : Symbol, optional,
                       this is the reciprocal grid spacing (1/gridspacing). When
                       specified, it causes the final expression to be multiplied by `rgridspacing`.
        hat : bool, optional,
              set `hat = True` to use the quadrature weights on the shifted grid.
              Otherwise, the quadrature weights are associated with the regular grid. 
        invert : bool, optional,
                set `invert = True` to invert that weights in the quadrature rule.

        Returns

        out : Derivative,
              A symbolic object representing the differentation of the expression.

        """

        d = readquadrature(order, hat, invert, fmt)
        shape = _infershape(expr, axis, shape, hat, False)

        # Reverse the roles of the gridspacing and reciprocal gridspacing 
        # because the difference operator is defined as D/h but it should be P/h for a quadrature
        if not invert:
            temp = gridspacing
            gridspacing = rgridspacing
            rgridspacing = temp

        return DifferenceOperator.__new__(cls, expr, axis, d['label'], shape, gridspacing=gridspacing, rgridspacing=rgridspacing,
                                          operator_data=d, **kw_args)

    @property
    def symbol(self):
        return 'H'

    def order(self):
        return self._operator_data['order']


def readderivative(order, hat, fmt='staggered_D%s%s.json'):
    """
    Loads the data for first derivative operator stored in a json file.

    Parameters:

    order : int,
            Interior order of accuracy
    hat  : bool,
           set to `True` if the operator for the hat-grid should be loaded.
    fmt  : string, optional
           format string that specifies filename of the operator

    Returns:

    d : dict,
        contains the data read from the .json file.

    """
    
    impl = _implementation(order)
    if hat:
        hat = 'hat'
    else:
        hat = ''

    operatorname = fmt % (hat, impl)
    d = do.readderivative(operatorname)

    return d

def readinterpolation(order, hat, fmt='staggered_P%s%s.json'):
    """
    Loads the data for interpolation operator stored in a json file.

    Parameters:

    order : int,
            Interior order of accuracy
    hat  : bool,
           set to `True` if the operator for the hat-grid should be loaded.
    fmt  : string, optional
           format string that specifies filename of the operator

    Returns:

    d : dict,
        contains the data read from the .json file.

    """
    
    impl = _implementation(order)
    if hat:
        hat = 'hat'
    else:
        hat = ''

    operatorname = fmt % (hat, impl)
    d = do.readinterpolation(operatorname)

    return d

def readquadrature(order, hat, invert=False, fmt='staggered_H%s%s.json'):
    """
    Loads the data for quadrature operator (or norm operator) stored in a json file.

    Parameters:

    order : int,
            Interior order of accuracy
    hat  : bool,
           set to `True` if the operator for the hat-grid should be loaded.
    invert : bool,
             set to `True` to invert the operator.
    fmt  : string, optional
           format string that specifies filename of the operator

    Returns:

    d : dict,
        contains the data read from the .json file.

    """

    impl = _implementation(order)
    if hat:
        hat = 'hat'
    else:
        hat = ''

    operatorname = fmt % (hat, impl)
    d = do.readquadrature(operatorname, invert)

    return d

def _infershape(expr, idx, shape, hat, periodic):
    from ..base import Axis, gridfunction
    # Sets the correct shape for a staggered grid operator
    if shape is None:
        shape = gridfunction.infer_shape(expr)

        a = Axis(idx, shape)
        # When the operator is periodic, both grids have the same number of grid points
        if periodic:
            shape = a.add(shape, 1)
        elif hat:
            shape = a.add(shape, 1)
        else:
            shape = a.add(shape, -1)
    return shape

def _implementation(order):
    impl = {2 : '21',  4 : '42',  6 : '63', 8 : '84'}
    if order not in impl:
        raise ValueError('DifferenceOperator not implemented')
    return impl[order]
