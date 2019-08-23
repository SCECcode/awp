"""
In OpenFD, Operators are composite operations that act on gridfunctions or other operators.
The algebraic rules that govern operators are similar to matrix and vector
multiplication.

By default, the `Operator` class implements an identity operator. Custom operators can be 
built by inheriting from the `Operator` class and overloading its `__getitem__`
function to specify actions that take place when an operator acts on a
gridfunction.
An example of a custom operator is presented later.

The following examples demonstrates how to build expressions involving the
identity operator
```python
>>> from . import Expression, GridFunction
>>> I = Operator('I')
>>> u = GridFunction('u', shape=(10,))
>>> expr = Expression(I*u)
>>> expr
I*u
>>> expr[0]
u[0]

```

This example implements a custom operator that shifts its operand one step when
accessed.

```python
>>> class Shift(Operator):
...     def __getitem__(self, index):
...         return self.args[index+1]
>>> u = GridFunction('u', shape=(10,))
>>> S = Shift('S')
>>> expr = Expression(S*u)
>>> expr
S*u
>>> expr[0]
u[1]

```

The following examples demonstrates the basic rules that operators satisfy. 
The shift operator from the previous example is reused.

```python
>>> from . import Expression, GridFunction
>>> a = GridFunction('a', shape=(10,))
>>> b = GridFunction('b', shape=(10,))
>>> Expression(S*a)[0]
a[1]
>>> Expression(a*S*b)[0]
a[0]*b[1]
>>> Expression(a*S*S*b)[0]
a[0]*b[2]
>>> Expression(S*(a + b))[0]
a[1] + b[1]
>>> Expression(S*(a + S*b))[0]
a[1] + b[2]
>>> Expression(S*a*b)[0]
a[1]*b[1]

```

See also:

* gridfunction
* expressions:Expression

"""
from sympy import Expr, Symbol
from .expressions import Expression, OPERATOR_COUNT 

class Operator(Symbol):
    """
    This is the base class for constructing operators.

    Attributes:
        label : The label that is displayed when printing the operator.


    """
    """
    Attributes:
        data : A dictionary that contains all of the data arrays used by this operator.
    """


    def __new__(cls, label='Op', idx='', args=None):
        identifier = label + idx
        obj = Symbol.__new__(cls, identifier, commutative=False)
        obj._label = label
        obj._args = args
        obj._data = {}
        obj.initialize()

        return obj

    def initialize(self):
        """
        Override this method to specify any additional instructions that should
        be executed during the initialization of this operator.
        """
        pass


    def copy(self, idx=''):
        """
        Returns a copy of the operator

        Parameters:

        idx : str, optional,
              id to append to operator label. 

        """
        return self.__class__(self.label, idx, args=self.args)

    def __getitem__(self, indices):
        return self.args[indices]

    def __call__(self, args=None):
        """

        Applies the operator to an expression

        Example

        >>> from . import Expression, GridFunction
        >>> I = Operator('I')
        >>> u = GridFunction('u', shape=(10,))
        >>> expr = I(u)
        >>> expr
        I(u)

        """
        #FIXME: the call function must return a copy of the operator
        # to make sure that the arguments are not overwritten in the old state.
        # This solution requires assigning a unique ID to the copy to prevent
        # it from being identified as the original. This ID is a global variable
        # Ideally, this global variable should be removed because it is probably
        # only a question of time before it starts causing problems.
        # Part of the problem lies in the fact that this class inherits from
        # symbol, which identifies different symbols based on their label.
        # If this class could inherit from `Expr` instead then maybe this problem
        # could be resolved without having to use global variables.
        global OPERATOR_COUNT
        OPERATOR_COUNT+=1
        oldargs = self.args
        self._args = Expression(args)
        obj = self.copy(str(OPERATOR_COUNT))
        self._args = oldargs
        return obj

    def regions(self):
        """
        Returns the regions defined by this operator. The regions specifies the
        part of the computational domain where the operator either performs a
        convolutional operation (interior), or a matrix-vector multiplication
        (boundary).

        There should typically not be any need to call this method directly.
        Instead, this method is called by first calling `region` on an
        expression of type `Expression`.

        Returns:

            list : Each item in this list is of type `Region`.

        """
        #FIXME: Add implementation
        pass


    @property
    def args(self):
        """
        Returns the expression this operator acts on.
        """
        return self._args

    @property
    def label(self):
        """
        Returns the operator label.

        >>> I = Operator('I')
        >>> I.label
        'I'
        """
        return self._label

    def __str__(self):
        if self.args is not None:
            return '%s(%s)' %(self.label, str(self.args))
        else:
            return self.label

    def _sympystr(self, p):
        return str(self)
         
    # Setter and getter methods for data (should already be implemented by base class)
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
