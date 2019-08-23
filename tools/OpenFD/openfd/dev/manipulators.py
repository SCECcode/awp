"""
The manipulators module is responsible for altering expressions in one way or another.
"""

from . import Constant
from sympy import Float, Integer

# List that defines what classes are considered as constants
lconstants = [Float, Integer, Constant, float, int, bool]

def pack(expr, label='d', index= lambda i : (i,), dict=False, accepted=lconstants):
    """
    Packs constants and symbols into arrays and modifies the expression by 
    replacing the constants and symbols with these arrays.

    Parameters

    expr : Expression,
           symbolic expression to manipulate.
    label : string, optional,
            name of the array to put constants in. 
    index : lambda, optional,
            indexing function used to specify array accesses.
            This function should return a tuple whose length determines the dimensionality
            of the array.
    dict : bool, optional,
           store replaced values in a dictionary. `False` by default.
    accepted : list, optional,
               list of classes that define constants.

    Returns

    newexpr: Expression,
             symbolic expression obtained by replacing in constants in `expr`.
    data: list, 
          list of replaced values. If `dict = True`, then this list is replaced by 
          a dictionary.

    Example

    ```python
    >>> from .gridfunction import GridFunction
    >>> from .expressions import Expression
    >>> from .manipulators import pack
    >>> u = GridFunction('u', shape=(10,10))
    >>> expr = Expression(0.2*u + 0.3)
    >>> out, data = pack(expr)
    >>> out
    d[0]*u + d[1]
    >>> data
    [0.200000000000000, 0.300000000000000]

    >>> from sympy import symbols
    >>> i, j =  symbols('i j')
    >>> out, data = pack(expr, index= lambda j : (i, j))
    >>> out
    d[i, 0]*u + d[i, 1]
    >>> data
    [0.200000000000000, 0.300000000000000]
    >>> out, data = pack(expr, index= lambda i : (i, j))
    >>> out
    d[0, j]*u + d[1, j]
    >>> data
    [0.200000000000000, 0.300000000000000]

    ```

    """
    from .. import GridFunction
    from sympy import symbols

    # Determine shape of output array.
    #FIXME: remove hard-coded value that sets the maximum number of 
    # elements that can be stored along each array dimension.
    shape = [1e3]*len(index(0))
    outputarray = GridFunction(label, shape=shape)

    # Get constants in this expression
    const = constants(expr)

    # dictionary of constants that will be substituted for array values
    subs = {}

    if dict:
        data = {}
    else:
        data = []

    for k, c in enumerate(const):
        # Pack constant into substitution array
        val = outputarray[index(k)]
        subs[c] = val

        # Pack constant into output list/dict
        if dict:
            data[val] = c
        else:
            data.append(c)

    return expr.subs(subs), data

def constants(expr, accepted=lconstants, useset=False):
    """
    Returns a list of unique constants found in the expression `expr`.
    If no constants are found, an empty list is returned.

    Parameters

    expr : Expression,
           symbolic expression to search for constants
    accepted : list, optional,
               list of classes that define constants.

    See also

    isconstant

    """

    return list(set(_constants(expr, const=[], accepted=accepted)))

def _constants(expr, const=[], accepted=lconstants):
    # Recursive function for finding constants in the expression `expr`.
    # A "constant" is only added if it is declared as constant by `isconstant`.
    from .. import GridFunctionBase, GridFunction
    from sympy import Mul, Pow
    
    if isinstance(expr, GridFunction):
        return const, expr.func(*expr.args)

    # Stores any constants found in the argument list of the current
    # `expr` node.
    data = []
    for arg in expr.args:

        # Avoid parsing GridFunction indices, e.g., u[i, j].
        if isinstance(arg, GridFunctionBase):
            continue

        if isconstant(arg, accepted):
            data.append(arg)
        else:
            # Traverse nested expressions
            try:
                if arg.args:
                    const = _constants(arg, const, accepted)
            except:
                pass

    if data:
        const.append(expr.func(*data))

    return const


def isconstant(expr, accepted=lconstants):
    """
    Returns `True` if the expression `expr` is a constant. 
    Otherwise, returns `False`.

    Parameters

    expr : Expression,
           Symbolic expression to check if constant.
    accepted : list, optional,
               list of classes that define constants.

    Examples

    >>> isconstant(0.2)
    True
    >>> from . import Constant
    >>> c = Constant('c')
    >>> isconstant(0.2*c)
    True

    """
    from .. import GridFunctionExpression
    from . import Constant
    from sympy import Float, Integer, Pow

    try:
        if expr.args:
            return all((isconstant(arg) for arg in expr.args))
    except:
        pass

    for a in accepted:
        if isinstance(expr, a):
            return True
    return False
