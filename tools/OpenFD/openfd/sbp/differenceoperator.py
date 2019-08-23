from sympy import Expr, Tuple, Matrix, sympify
from sympy.core.compatibility import NotIterable, is_sequence
from ..base import gridfunction as gridfunc
from ..base import Operator, GridFunctionExpression
from six import iteritems
import numpy as np

class DifferenceOperator(Operator):
    """
        The DifferenceOperator class extends the Operator class by providing additional concepts such
        as the order of accuracy, and grid spacing. 
    
    """

    def __new__(cls, expr, axis, label, shape=None, gridspacing=1, rgridspacing = None, order_left=None,
            order_interior=None, order_right=None, 
                op_left=None, idx_left=None, op_right=None, idx_right=None, op_interior=None, idx_interior=None,
                operator_data=None, periodic=False, coef=None, gpu=False, **kw_arg):
        """
        Differentiate a symbolic expression using a high order 
        finite difference approximation.

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

        if is_sequence(order_interior):
            raise ValueError("Only one interior stencil supported.")

        obj = Operator.__new__(cls, expr, axis, label, shape=shape, op_left=op_left, idx_left=idx_left, 
                               op_right=op_right, idx_right=idx_right, op_interior=op_interior, 
                               idx_interior=idx_interior, operator_data=operator_data, 
                               periodic=periodic, coef=coef, gpu=gpu, **kw_arg)
        obj._gridspacing    = gridspacing
        obj._rgridspacing   = rgridspacing

        if operator_data:
            obj._order_left     = operator_data['order_left']
            obj._order_right    = operator_data['order_right']
            obj._order_interior = operator_data['order_interior']
        else:
            obj._order_left     = order_left
            obj._order_right    = order_right
            obj._order_interior = order_interior

        return obj

    @property
    def gridspacing(self):
        return self._gridspacing
    
    @property
    def rgridspacing(self):
        return self._rgridspacing

    def order(self, index):
        """
            Returns the order of a stencil near or on the boundary, or in the interior

            >>> from sympy import symbols
            >>> from openfd import GridFunction, sbp_traditional as sbp
            >>> nx,i = symbols('n i')
            >>> a  = GridFunction("a",shape=(nx+1,))
            >>> a_x  = sbp.Derivative(a, "x", order=2)
            >>> a_x.order(0)
            1
            >>> a_x.order(1)
            2
            >>> a_x.order(nx)
            1

        """
        if self._bounds[index] == self._bounds.tag_left:
            if is_sequence(self._order_left):
                return int(self._order_left[index])
            else:
                return int(self._order_left)
        if self._bounds[index] == self._bounds.tag_right:
            if is_sequence(self._order_right):
                return int(self._order_right[self.right(index)])
            else:
                return int(self._order_right)
        if self._bounds[index] == self._bounds.tag_interior:
                return int(self._order_interior)
        raise IndexError("Index out of bounds")
    
    def __getitem__(self, indices, **kw_args):
        expr = Operator.__getitem__(self, indices, **kw_args)
        if self.rgridspacing is None:
            return expr/self.gridspacing
        else:
            return expr*self.rgridspacing

    @property
    def symbol(self):
        return 'D'

def readderivative(operatorname):
    """ 
    Loads a derivative operator from a json file.
    If the right boundary is unspecified, then the left boundary is mirrored.

    Parameters

    operatorname : string,
                   filename of the operator to load

    Returns

    d : dict,
        the operator data loaded.
    """
    d = readoperator(operatorname)

    if 'op_right' not in d:
        d['op_right'] = -d['op_left']
    if 'idx_right' not in d:
        d['idx_right'] = d['idx_left']
    if 'order_right' not in d:
        d['order_right'] = d['order_left']

    return d

def readinterpolation(operatorname):
    """ 
    Loads an interpolation operator from a json file.
    If the right boundary is unspecified, then the left boundary is mirrored.

    Parameters

    operatorname : string,
                   filename of the operator to load

    Returns

    d : dict,
        the operator data loaded.
    """
    d = readoperator(operatorname)

    if 'op_right' not in d:
        d['op_right'] = d['op_left']
    if 'idx_right' not in d:
        d['idx_right'] = d['idx_left']
    if 'order_right' not in d:
        d['order_right'] = d['order_left']

    return d


def readquadrature(operatorname, invert=False):

    d = readoperator(operatorname)

    if 'op_right' not in d:
        d['op_right'] = d['op_left']
    if 'idx_right' not in d:
        d['idx_right'] = d['idx_left']
    if 'order_left' not in d:
        d['order_left'] = None
    if 'order_right' not in d:
        d['order_right'] = None
    if 'order_interior' not in d:
        d['order_interior'] = None

    if invert:
        d['op_left'] = 1.0/d['op_left']
        d['op_right'] = 1.0/d['op_right']

    return d

def readoperator(operatorname):
    import json
    from sympy import Matrix

    res = open(_file(operatorname)).read()
    d = json.loads(res)
    dout = {}

    # Store arrays as numpy arrays
    arrays = {'op_left', 'op_right', 'op_interior',
              'order_left', 'order_right', 'order_interior'}
    idx = {'idx_interior'}
    bug = {'idx_left', 'idx_right'}
    for k, v in iteritems(d):
        if k in arrays:
            dout[k] = np.array(v)
        elif k in idx:
            dout[k] = np.array(v,dtype=np.int32)
        elif k in bug:
        #FIXME: Causes Python 3 to get the wrong answer for right boundary computations
        #dout[k] = np.array(v,dtype=np.int32)
            dout[k] = Matrix(v)
        else:
            dout[k] = v

    return dout

def _file(filename):
    import pkg_resources
    resource_package = __name__  # Could be any module/package name
    resource_path = '/'.join(('resources', filename))  # Do not use os.path.join(), see below
    return pkg_resources.resource_filename(resource_package, resource_path)

