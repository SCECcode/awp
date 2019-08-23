"""
See the notebook `gridfunction.ipynb` for usage.

"""
from sympy import Symbol, Expr
from sympy.core.compatibility import is_sequence
from . import bounds as bnd

class GridFunction(Symbol):
    """
    Baseclass for grid functions.

    Attributes:
        label (`str`) : The label used to distinguish the grid function.
            The label will appear in symbolic expressions.
        shape (`tuple`) : A tuple that specifies the size of the grid
            function in each dimension. The sizes can be symbolic.

    """

    def __new__(cls, label, shape, lower_out_of_bounds='wrap-around',
                upper_out_of_bounds='raise exception', bounds=None):
        """

        Arguments:

            lower_out_of_bounds (optional) : Action to take when an index in the
                grid function is less than its lower bound. Defaults to
                `wrap-around`.
            upper_out_of_bounds (optional) : Action to take when an index in the
                grid function is greater than or equal to its upper bound.
                Defaults to `raise exception`.
            bounds (optional) : A tuple that contains `Bounds` objects, one per
                grid dimension. Defaults to `None` (which initializes a tuple
                containing bounds with the default settings).

        Notes:

            For additional details about out of bounds handling, see the
            `bounds` module.

        """

        if bounds is None:
            _bounds = tuple([bnd.Bounds(0, upper, lower_out_of_bounds,
                                        upper_out_of_bounds)
                             for upper in shape])
        else:
            assert len(bounds) == len(shape)
            _bounds = bounds

        obj = Symbol.__new__(cls, label, commutative=False)
        obj._shape = shape
        obj._label = label
        obj._bounds = _bounds
        return obj

    def __getitem__(self, indices):
        """
        Accesses an element of the gridfunction.

        Arguments:
            indices : An `int` or `Index` that determines the indices to access
                of the gridfunction. Use a tuple for multi-dimensional
                gridfunctions.

        """
        try:
            len(indices)
        except:
            indices = [indices]
        try:
            return GridFunctionBase(self, [self.bounds[dim][idx] for dim, idx in
                                           enumerate(indices)])
        except IndexError:
            return GridFunctionBase(self, indices)

    def __call__(self, *indices):
        """
        Accesses an element via get value. However, GridFunctions cannot hold
        values and therefore calling this method has the same effect as
        accessing an element via symbol (`__getitem__`). 

        Arguments:
            indices : An `int` or `Index` that determines the indices to access
                of the gridfunction. Use a tuple for multi-dimensional
                gridfunctions.

        """
        try:
            len(indices)
            return self.__getitem__(indices)
        except:
            return self.__getitem__(*indices)


    @property
    def label(self):
        """
        Returns the label.

        Example

        ```python
        >>> u = GridFunction('u', shape=(10,))
        >>> u.label
        'u'

        ```

        """

        return self._label

    @property
    def shape(self):
        """
        Returns the shape.

        Example

        ```python
        >>> u = GridFunction('u', shape=(10,))
        >>> u.shape
        (10,)

        ```

        """

        return self._shape

    @property
    def bounds(self):
        """
        Returns the bounds for each grid dimension.


        Example:

        ```python
        >>> u = GridFunction('u', shape=(10,))
        >>> bounds = u.bounds
        >>> bounds[0].upper
        10
        >>> bounds[0].lower
        0

        ```

        """

        return self._bounds

class GridFunctionBase(Expr):
    """
    An object of type `GridFunctionBase` is constructed when accessing
    a `GridFunction` object using `u[indices]`.
    This class provides printers for the result of accessing a grid function.

    """
    def __new__(cls, base, indices):
        if not is_sequence(indices):
            indices = tuple([indices])
        obj = Expr.__new__(cls, base, *indices)
        obj._base = base
        obj._indices = indices
        return obj

    def _sympystr(self, printer):
        # Convert _indices = [1,2,3] to string '1, 2, 3'
        indices = ', '.join([str(s) for s in self._indices[0:len(self._base.shape)]])
        return printer.doprint('%s[%s]' % (self._base.label, indices))
