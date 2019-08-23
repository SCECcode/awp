from ..base import GridFunctionExpression, GridFunction
from sympy.core.compatibility import NotIterable, string_types
from sympy import Expr, Symbol, sympify
from . types import C

class Variable(Expr, NotIterable):

    def __new__(cls, label, val=None, dtype=None, 
                isdeclared=0, 
                declarable=1,
                visible=0, **kw_args):
        """
        Variables are scalar objects objects that store symbolic expressions.

        A simple use case of a variable is to break up one or more expressions into 
        multiple ones by identifying the parts that perform the same computations.
        This practice can improve performance because it increases the reuse of already 
        computed results and also improves the readability of the generated source code.

        Parameters 

        label : string, 
                variable name.
        val : Expr, optional,
              Symbolic expression to assign to the value
        dtype : str, optional,
                Type to declare when called by code generators.
                Usually, dtype is omitted on construction. Instead, use 
                the function `declare` once it is time to declare the variable.

        """

        if isinstance(label, string_types):
            label = Symbol(label)
        else:
            label = sympify(label)
        obj = Expr.__new__(cls, label, **kw_args)
        obj._label = label
        obj._expr = val
        obj._isdeclared = isdeclared
        obj._declarable = declarable
        obj._visible = visible
        if dtype:
            obj._hastype = True
        else:
            obj._hastype = False
        obj._dtype = dtype
        return obj

    def __str__(self):
        return str(self.label)

    def _sympystr(self, p):
        return p.doprint(Variable(self._label, val=self._expr, dtype=self.dtype,
                                 isdeclared=0))

    def declare(self):
        """
        Constructs a new variable by specifying a type for the current variable.
        If no type is specified, the already existing type is reused. If no existing
        type is found, an exception is raised. Variables that have a type defined will
        output this type when using code printer.


        Returns 

        var : Variable,
              a copy of the old variable but with its type set to `dtype` (if specified).

        Example

        ```python
        >>> from .variable import Variable
        >>> u = Variable('u')
        >>> from sympy.printing import ccode
        >>> ccode(u)
        'u'
        >>> ccode(u.declare('float'))
        'float u'

        ```

        """                   
        if self.hastype:
            dtype = self.dtype
        else:
            raise ValueError('No type specified.')
        return Variable(self.label, val=self._expr, dtype=dtype, isdeclared=1)

    def _ccode(self, p):
        pr = ''
        if self.isdeclared:
            pr += str(C.get_type(self.dtype)) + ' '
        return pr + str(self.label)

    def __getitem__(self, indices, **kw_args):
        return self

    @property
    def symbol(self):
        return Variable(self.label, val=self._expr, dtype=self.dtype, 
                        isdeclared=0, declarable=0)

    @property
    def free_symbols(self):
        return {self}

    @property
    def label(self):
        """
        Returns the label.

        Example

        ```python
        >>> var = Variable('u')
        >>> var.label
        'u'

        ```

        """
        return str(self._label)
    
    @property
    def dtype(self):
        """
        Returns the type. If not type has been assigned, `None` is returned.

        Example 

        ```python
        >>> var = Variable('u', dtype='float')
        >>> var.dtype
        'float'

        ```

        """
        return self._dtype

    @property
    def hastype(self):
        """
        Returns `True` if a type has been assigned. Otherwise, returns `False`.
            
        Example 
        
        ```python
        >>> var = Variable('u', dtype='float')
        >>> var.hastype
        True

        ```
        """

        return self._hastype

    @property
    def val(self):
        """
        Returns the value that has been assigned. 
        If no value has been assigned, `None` is returned.
            
        Example 
        
        ```python
        >>> var = Variable('u', val=1)
        >>> var.val
        1

        ```
        """
        return self._expr

    @property
    def value(self):
        """
        Same as `val`.
        """
        return self.val
    
    @property
    def declarable(self):
        return self._declarable

    @property
    def isdeclared(self):
        """
        Returns `True` if the variable was initialized by calling declare()
        """
        return self._isdeclared
    
    @property
    def visible(self):
        """
        Returns `True` if the variable was initialized by calling declare()
        """
        return self._visible

