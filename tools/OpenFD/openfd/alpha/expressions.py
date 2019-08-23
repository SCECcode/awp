"""Module for constructing and manipulating symbolic expressions.

Expressions extend standard symbolic algebra to support additional
operations and conventions. There are few ways in which can construct
expressions, and the different ways have different implications for the
evaluation and usage of an expression.

The standard form of an expression is defined by having variables and grid
functions present in it. There can be operators that act on the grid functions
that yield a new result once evaluated, which is a new grid function (or some
combination of grid functions). A typical example would be an Operator `D`
acting on a `GridFunction` such as:

```python
>>> from . import GridFunction, Operator, Expression
>>> u = GridFunction('u', shape=(10,))
>>> D = Operator('D')
>>> expr = Expression(D(u))

```

When the expressions are written in this way, it is clear that D acts on `u` and
we can evaluate the expression by calling `expr[0]`. In this case, we obtain

```python
>>> expr[0]
u[0]

```

This result should not come as a surprise because the default operator is the
identity operator.

There is an alternative way in which we can define the operation above. Consider

```python
>>> expr = Expression(D*u)
>>> expr[0]
u[0]

```

The disadvantage with this way of writing is that the operator will act on
any products to the right of it, for example

```python
>>> class Shift(Operator):
...     def __getitem__(self, index):
...         return self.args[index+1]
>>> v = GridFunction('v', shape=(10,))
>>> S = Shift('S')
>>> expr = Expression(S*u*v)
>>> expr[0]
u[1]*v[1]
>>> expr = Expression(S*(u)*v) # Parenthesis has no effect
>>> expr[0]
u[1]*v[1]
>>> expr = Expression(S(u)*v)
>>> expr[0]
u[1]*v[0]

```

Note that to demonstrate the effect we had to use a different operator than the
Identity operator.

However, it has some great advantages as it enables two forms of expressions:
one form that only contains operator expressions, and the type of expressions we
have seen above. When the expressions only contain operators it possible to
define a new operators, and also perform special operations such as taking the
transpose of an operator.

The functions and classes in this module is what makes it possible to write
expressions such as `D*u**2`.


"""
from sympy import Expr
global OPERATOR_COUNT
OPERATOR_COUNT = 0

class Expression(Expr):
    """
    Expression handler for OpenFD data types.

    Any expression that contains the OpenFD data types such as
    `GridFunction`, `Operator` must be wrapped in a `Expression(..)`
    to ensure that operations such as `expr[i]` perform as expected.

    """

    def __new__(cls, expr):
        """
        Constructor for Expression.

        Arguments:

            expr: Expression that

        """
        obj = Expr.__new__(cls, expr)
        obj._expr = expr
        return obj

    def region(self, is_nonoverlapping=True):
        """
        Call this method on a symbolic expression of type `Expression` to
        obtain all of its compute regions.

        Arguments:

            is_nonoverlapping : An optional `bool` that should be set to `True` to
                to convert any overlapping regions into non-overlapping regions.
                When this option is enabled, all of the regions returned are
                guaranteed to be non-overlapping. If there are overlapping
                regions present in the expression, then the overlapping regions
                are split into new regions by finding their intersection, and
                their respective complements. Defaults to `True`.

        See also:

            * More details about regions can be found in the module `regions`.

        Returns:

            `list` : Each element in this list is of type `Region`. If no
            regions are found, then `None` is returned.

        Note:

            A call to this method will recursively traverse the expression tree
            and query each node for regions.
        """
        #FIXME: add implementation
        pass

    def _call(self, method, *arg, **kw_args):
        """
        This method is used to call any user-specified method on all objects of
        a symbolic expression. If an object does not have the specific method,
        then no method is called for the particular object.

        This method will be used to perform the `__getitem__` call and also the
        `regions` call.

        Arguments:

            method(str): The name of the method to call.

        Notes:

            To call the specific method on each object in an expression, the
            expression tree is traversed. This traversal is currently done by 
            the function `_getitem` and that will be replaced soon.

            This method has been made private to discourage direct calls by
            users. Instead, for each method that needs to be called, it should
            be implemented as a separate method and call `_invoke` to perform
            its task(s). 

            If we want some new method to be callable on expressions, then we
            add it to this class and call `_call`. For
            example,

            ```python
            def new_method(self, arg1, arg2):
                return self._call('new_method', arg1, arg2)
            ```

        """
        #FIXME: Add implementation
        pass

    def __getitem__(self, indices):
        """
        Accesses `expr[indices]` (symbolic access)

        Arguments:

            indices : tuple,
                      indices to access

        Returns:

            expr : Expr,
                   resulting expression after evaluation

        Example:

            >>> from . import GridFunction
            >>> u = GridFunction('u', shape=(10,))
            >>> v = GridFunction('v', shape=(10,))
            >>> expr = Expression(u + v)
            >>> expr[0]
            u[0] + v[0]

        """
        out = _function(self._expr, indices, method_name='__getitem__', 
                        function=lambda x, indices : x[indices])
        return out

    def __call__(self, indices):
        """
        Accesses `expr(indices)` (value access)

        Arguments:

            indices : tuple,
                      indices to access

        Returns:

            expr : Expr,
                   resulting expression after evaluation

        Example:

            #TODO: Enable this example once array has been properly merged.
            #>>> from .array import CArray
            #>>> a = CArray('a', data=[1.0, 0.0])
            #>>> b = CArray('b', data=[-1.0, 1.0])
            #>>> expr = Expression(a + b)
            #>>> expr(0)
            #0.0
        """
        out = _function(self._expr, indices, method_name='__call__', 
                        function=lambda x, indices : x(indices))
        return out



    def _sympystr(self, printer):
        return printer.doprint(self._expr)

def _function(expr, indices, method_name='', function=None, *args, **kw_args):
    """
    Calls a function `function` on all terms in an expression that supports it.
    If any term in the expression does not have this function implemented, then
    the call is ignored.

    Arguments:

        expr : The symbolic expression to recursively apply the function to.
        indices : A tuple that contains the indices to pass to the function.
        function_name : A string that specifies the name of the method to call
            on the expression. For example, `'__getitem__'`.
        function : A function handler that invokes the method. 

    Examples:

        The following example calls the `__getitem__` method on a expression.
        >>> from . import GridFunction, Operator, Expression
        >>> u = GridFunction('u', shape=(10,))
        >>> expr = Expression(u)
        >>> # Get the zeroth index.
        >>> _function(expr, 0, '__getitem__', lambda x, indices : x[indices])
        u[0]

    """
    from . import Operator
    from sympy import Mul

    if expr is not None:

        expr = set_operator_args(expr)

        if hasattr(expr, method_name):
            val = function(expr, indices)
            return val
        if not hasattr(expr, 'args'):
            return expr
        if not expr.args:
            return expr

    current_args = [a for a in expr.args]
    new_args = (_function(arg, indices, method_name, function, args, kw_args)
                for arg in current_args)
    return expr.func(*new_args)

def set_operator_args(expr):
    """
    When there are nested expressions such as `D*D*op` the expression needs to
    be rewritten to something like `D(D(op))`. This rewrite is taken care by
    this function.

    Arguments:

        expr : Symbolic expression to act on.

    Returns:

        Expression : The updated symbolic expression.

    Notes:

        These notes provide a more detailed description of what this function
        does and the problems it solves.

        Search through a sympy expression of the form `Mul(*, *, op, .. , op, *,
        *)` each '*' is some other expression or symbol, and op is the operator
        that should be acting on what is located to the right of it.

        This function does nothing if the expression `expr` does not start with
        a `Mul` object.

        Once the search begins there are two major problems to solve:

        1. If the same operator is applied onto itself then Sympy will convert
        something like `Mul(op, op)` to `Pow(op, 2)`.  If left untreated the end
        result will most likely be incorrect.  To overcome this problem, each
        Pow expression is reconverted to a `Mul` but `evaluate=False` is passed
        so that it stays as `Mul(op, op)` etc.  Otherwise, sympy would simply
        recompute `Pow(op, 2)`.

        2. When multiple operators are nested, it is important that they appears
        as unique to ensure correct output. However, the same operator may be
        referenced over and over again. Therefore, each time an operator is
        encountered a 'copy' is made and the label for this copy is set by a
        global counter (`OPERATOR_COUNTER`) that ensures it gets a unique ID.

    """
    from . import Operator
    from sympy import Mul, Pow

    if not isinstance(expr, Mul):
        return expr

    expr = pow2mul(expr)

    global OPERATOR_COUNT

    # Suppose we have `Mul(a, op, op, b)` then the new expression should become:
    # `Mul(a, op(op(b)))` The parenthesis indicate that for `op(b)` has set its
    # arguments to `b` The arguments before the operator(s) are stored in the
    # list `prevargs`.

    prevargs = []

    for i, arg in enumerate(expr.args):

        if isinstance(arg, Operator):
            # Make a copy of the operator and assign its arguments via recursion
            operator = arg.copy(str(OPERATOR_COUNT))
            OPERATOR_COUNT += 1

            # Do nothing if the operator already has arguments assigned to it
            if operator.args is not None:
                prevargs.append(arg)
                continue

            nextarg = set_operator_args(Mul(*expr.args[i+1:]))

            operator._args = Expression(nextarg)
            newargs = prevargs + [operator]
            return Mul(*newargs, evaluate=False)
        else:
            prevargs.append(arg)
            continue

    # No operator found
    return expr

def pow2mul(expr):
    """
    Converts any `Pow` into `Mul`.

    Arguments:

        expr (`Expression`) : The expression to replace `Pow` in.

    Returns:

        Expr: Updated symbolic expression with Pows removed.

    Example:

        ```python
        >>> from sympy import Pow, Mul, Integer, srepr
        >>> from . import Operator, GridFunction
        >>> I = Operator('I')
        >>> u = GridFunction('u', shape=(10,))
        >>> I**2*u
        I**2*u
        >>> pow2mul(I**2*u)
        I*I*u

        ```

    """
    from . import Operator
    from sympy import Mul
    from sympy import Pow

    newargs = []
    for arg in expr.args:
        if isinstance(arg, Pow):
            operator = arg.args[0]
            if isinstance(operator, Operator):
                power = arg.args[1]
                for _ in range(power):
                    newargs.append(operator)
            else:
                power = arg.args[1]
                for _ in range(power):
                    newargs.append(operator)

        else:
            newargs.append(arg)

    expr = Mul(*newargs, evaluate=False)
    return expr
