from sympy import Expr, Symbol, Tuple, sympify
from sympy.core.compatibility import NotIterable , is_sequence
from sympy import preorder_traversal
from sympy.core.cache import clear_cache
from sympy.core.mul import Mul
from . import Bounds
from .. dev import memory
from .. dev.macro import Macro
import openfd

string_types=str

def gridfunctions(label, shape, dtype=None, layout=None, struct=False, 
                  remap=None, macro='_'):
    """
    Construct gridfunctions from string.

    Arguments:
        label : A string of gridfunctions labels that are each separated by a
            space.
        shape : Grid dimensionality.
        dtype (optional) : data type.
        layout : Memory layout object.
        struct(optional) : Return a struct of grid functions instead of a list.
            Defaults to `True`.
        remap : Function that remaps the indices when accessed. This function
            takes the indices as input arguments (tuple) and returns a new
            tuple. 
        macro : Emit C-macro code. This option improves the readability of the
            generated code.

    Returns:
        A list of grid functions constructed from the label.

    Example:

        >>> u, v = gridfuntions('u v', shape=(10, 1))

    """
    from openfd import Struct
    fields = label.split(' ')

    if struct:
        out = Struct(**{field : GridFunction(field, shape, dtype=dtype, 
                                             layout=layout, 
                                             remap=remap, macro=macro) 
                        for field in fields})
    else:
        out = [GridFunction(field, shape, layout=layout, dtype=dtype,
                            remap=remap, macro=macro) 
                        for field in fields]

    return out

class GridFunctionException(Exception):
    pass

class GridFunctionBase(Expr):


    def __new__(cls, base, *args, **kw_args):

        if not args:
            raise GridFunctionException("GridFunctionBase needs at least one index.")
        if isinstance(base, (string_types, Symbol)):
            base = GridFunction(base)
        args = list(map(sympify, args))
        return Expr.__new__(cls, base, *args, **kw_args)
    
    @property
    def base(self):
        return self.args[0]

    @property
    def shape(self):
        return self.base.shape

    @property
    def indices(self):
        return self.args[1:] 

    def _sympystr(self, p):
        indices = list(map(p.doprint, self.indices))
        return "%s[%s]" % (p.doprint(self.base), ", ".join(indices))
    
    def _ccode(self, p):
        from sympy.printing import ccode
        mac = self.base.macro(*self.indices, rearr=0)
        if mac:
            return mac.str_base()
        else:
            index   = self.base.layout.get_c(*self.indices)
            return "%s[%s]" % (self.base.label, ccode(index))




class GridFunction(Expr, NotIterable):

    """
    Symbolic vector that is mapped to a grid.

    Attributes:
        label : The label that is displayed in symbolic expressions.
        shape : A tuple defining the shape of the grid.
        periodic : An undocumented feature that probably does not work.
        dims : A tuple defining what indices can be used to access elements of
            the grid function. For example, if `dims=(0,2)` then `u[i,j,k]`
            would be the same as accessing `u[i,k]`.
        layout : Memory layout to use when accessing the entries of the grid.
                    Should be an object of type `Memory` (see dev.memory for
                    details).
        remap : Function that remaps the indices when accessed. This function
            takes the indices as input arguments (tuple) and returns a new
            tuple. 
        visible : Make the gridfunction visible in the kernel function header
            declaration.

    """
    def __new__(cls, label, shape, 
                periodic = False, 
                dims = None,
                layout = None, 
                remap = None,
                dtype = None,
                visible = 1,
                macro = '_',
                **kw_args):

        if isinstance(label, string_types):
            label = Symbol(label)
        elif isinstance(label, Symbol):
            pass
        else:
            label = sympify(label)

        if is_sequence(shape):
            shape = Tuple(*shape)
        else:
            shape = Tuple(shape)

        if not layout:
            layout = memory.Memory(shape) 

        # Set data type based on precision
        if not dtype:
            dtype = openfd.prec

        if not dims:
            dims = range(len(shape))



        if not isinstance(layout, memory.Memory):
            raise TypeError("Wrong type <%s>. Expected type <%s>."
                            % (type(layout), memory.Memory))

        b = lambda n: Bounds(size=n, left=0, right=0, periodic=periodic) 

        obj = Expr.__new__(cls, label, **kw_args)
        obj._periodic = periodic
        obj._bounds = list(map(b, shape))
        obj._remap = remap
        obj._shape = shape
        obj.visible = visible
        obj._layout = layout
        obj._dims = dims
        obj._macro = macro
        obj.dtype = dtype
        obj.ptr = True
        return obj

    def __getitem__(self, indices, **kw_args):

        if not is_sequence(indices):
            indices =[indices]

        if self._remap:
            indices = self._remap(indices)

        if is_sequence(indices):

            indices = self._rearrange(indices)

            if self.shape and len(self.shape) > len(indices):  
                raise IndexError("Dimension mismatch.")
            check_bounds = [self._bounds[i].inbounds(indices[i]) for i in
                            range(len(self.shape))]
            if not all(check_bounds):
                raise IndexError("At least one index in ``%s`` is out of bounds." % str(indices))
            if self.periodic:
                indices = [self._bounds[i].cycle(idx) for i, idx in enumerate(indices)]
            else:
                indices = map(right, indices, self.shape)
            return GridFunctionBase(self, *indices, **kw_args)
        else:
            if not self._bounds[0].inbounds(indices):
                raise IndexError("Index ``%s`` is out of bounds." % str(indices))
            if self.shape and len(self.shape) != 1:
                raise GridFunctionException("Dimension mismatch.")
            indices = right(indices, self.shape[0])
            return GridFunctionBase(self, indices, **kw_args)

    def _rearrange(self, indices):
        # Rearrange indices 
        return [indices[dimi] for dimi in self._dims]

    def macro(self, *indices, rearr=1):
        if not self._macro:
            return None

        if rearr:
            indices = self._rearrange(indices)
        str_indices = [str(idx) for idx in indices]
        return Macro(self._macro, str(self.label), str_indices,
                     self.layout.get_c(*indices))



    @property
    def label(self):
        return self.args[0]

    @property
    def T(self):
        return self

    @property
    def shape(self):
        return self._shape

    @property
    def layout(self):
        return self._layout
    
    def _sympystr(self, p):
        return p.doprint(self.label)
    
    @property
    def free_symbols(self):
        return {self}

    @property
    def periodic(self):
        return self._periodic

class GridFunctionExpression(Expr):

    def __new__(cls, expr, *arg, **kw_args):
        clear_cache()
        obj = Expr.__new__(cls, expr, *arg, **kw_args)
        obj._expr = expr
        return obj

    def __getitem__(self, indices, **kw_args):
        return eval(self._expr, indices)

    @property 
    def T(self):
        return GridFunctionExpression(transpose(self._expr))

    def _sympystr(self, p):
        return self._expr


def transpose(expr):
    from .operator import Operator
    if expr is not None:
        if hasattr(expr,'T'):
            val = expr.T
            return val
        elif not hasattr(expr,'args'):
            return expr
        elif isinstance(expr, Mul):
            new_args = (transpose(arg) for arg in expr.args[::-1])
            return expr.func(*new_args)
        elif isinstance(expr, Operator):
            newexpr = expr.T
            return newexpr.func(*expr.args[::-1])
        elif not expr.args:
            return expr
    new_args = (transpose(arg) for arg in expr.args)
    return expr.func(*new_args)


def eval(expr, indices, **kw_args):
    """

    Evaluates a grid function expression. Same as calling [i] on the grid function 
    expression itself.

    Parameters

    expr : GridFunctionExpression,
           expression to evaluate

    indices : tuple,
              the indices to evaluate the expression at

    Example

    >>> u = GridFunction('u', shape=(10,))
    >>> v = GridFunction('v', shape=(10,))
    >>> expr = GridFunctionExpression(u + v)
    >>> eval(expr, 0)
    u[0] + v[0]

    """
    from .operator import Operator, OperatorException
    if expr is not None:
        args = [a for a in expr.args]
        if hasattr(expr,'__getitem__'):
            val = expr[indices]
            return val
        elif not hasattr(expr,'args'):
            return expr
        # Handle operators that behave as matrices
        # D*(a*D*v)
        elif isinstance(expr, Mul):
            
            for i, arg in enumerate(expr.args):
                if isinstance(arg, Operator):
                    if not arg.is_matrix:
                        continue
                    # The operator is not acting on anything
                    if arg == expr.args[-1]:
                        raise OperatorException('No right operand found') 

                    # Fetch the operator and change the expression it acts on to
                    # the arguments to the right
                    op = arg
                    op_args = [a for a in expr.args[i+1:]]
                    op._expr = Mul(*op_args)

                    # Build new argument list that has removed the arguments to the left 
                    # of the operator since they are now built into the operator itself
                    before_op = expr.args[0:i]
                    new_args = []
                    if before_op:
                        new_args = [a for a in before_op]
                    new_args.append(op)

                    args = new_args
                    break

        elif not expr.args:
            return expr
    new_args = (eval(arg, indices, **kw_args) for arg in args)
    return expr.func(*new_args)



def infer_shape(expr, **kw_args):
    for arg in preorder_traversal(expr):
        if arg is not None:
            if isinstance(arg,GridFunction):
                return arg.shape
    return None

def infer_dims(expr, **kw_args):
    for arg in preorder_traversal(expr):
        if arg is not None:
            if isinstance(arg,GridFunction):
                return arg._dims
    return None

def right(index, n):
    from . import Right

    # Treat special indices that can access the right boundary
    if isinstance(index, Right):
        return n - index - 1

     # Do not do anything if the index format is symbolic
    index       = sympify(index)
    index_dict  = index.as_coefficients_dict()
    if len(index_dict) > 1 or (len(index_dict) == 1 and index_dict[1] == 0 and index != 0):
        return index

    if index < 0:
        return n + index
    else:
        return index

def periodic(index, n):
     # Do not do anything if the index format is symbolic
    index       = sympify(index)
    index_dict  = index.as_coefficients_dict()
    if len(index_dict) > 1 or (len(index_dict) == 1 and index_dict[1] == 0 and index != 0):
        return index

    if index < 0:
        return n + index
    else:
        return index

class Constant(GridFunction):
    """
    This class represents a constant grid function. 

    """


    def __new__(cls, expr, shape, visible = 0, *arg, **kw_args):
        from sympy import sympify
        return GridFunction.__new__(cls, expr, shape, *arg, **kw_args)

    def __getitem__(self, indices):
        return self.label

    @property
    def shape(self):
        return self._shape

from sympy import Matrix as Mat
class Matrix(Mat):
    pass

    @property
    def T(self):
        """
        Transposes the matrix.

        Example

        >>> A = Matrix([[0, 1], [0, 0]])
        >>> A.T
        Matrix([
        [0, 0],
        [1, 0]])

        """
        AT = super(Matrix, self).T
        for i in range(AT.shape[0]):
            for j in range(AT.shape[1]):
                AT[i,j] = transpose(AT[i,j])
        return AT

