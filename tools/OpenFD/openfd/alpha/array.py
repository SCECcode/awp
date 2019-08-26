"""
See the notebook `array.ipynb` for usage.
"""
from sympy.core.compatibility import is_sequence
from sympy import Expr, Symbol
import numpy as np

class Array(Expr):
    """
    Base class for symbolic arrays. Symbolic arrays can hold data and generate
    code that outputs the array access and initialization for the target
    language.

    Attributes:
        label (`str`): The label that identifies the array in symbolic expressions.
        data  : The data stored in the array. Can be accessed via the special
            method `__call__`. This attribute should be an array (e.g.,
            `numpy.array`).
        rettype (`str`): The return type used when initializing the array
        format (`str`): The formatting string used to output each array value
            during initialization.

    """

    def __new__(cls, label, data):
        """
        Initializes the symbolic array.
        """
        symbol = Symbol(label)
        obj = Expr.__new__(cls, symbol)
        obj._label = label
        obj._symbol = symbol

        # Convert lists to numpy arrays
        if isinstance(data, list):
            data = np.array(data)


        obj.data = data
        obj.dec_str()
        return obj

    def dec_str(self):
        """
        Defines the type and formatting options for the array.
        """

        self.rettype = 'const float'
        self.format = '%g'

    def __getitem__(self, indices):
        """
        Accesses an element of the array.

        Arguments:

        indices : A tuple that specifies the indices of the element.
                 The indices can be either in symbolic form or ints.

        Returns:
            ArrayElement : The requested element in symbolic form.

        """
        return ArrayElement(self, indices)

    def __call__(self, *indices):
        """
        Accesses a value of the array.

        Arguments:

        indices : A tuple that specifies the indices of the element to access.
            The indices can be either in symbolic form or ints.

        Returns:
           The value for the given array index.

        """
        return self.data[indices]

    def _sympystr(self, printer):
        return printer.doprint(self.label)

    def _ccode(self, _):
        return carray(self.label, self.data, self.rettype, self.format)

    @property
    def label(self):
        """
        Returns the label of the array
        """
        return self._label

    @property
    def symbol(self):
        """
        Returns the symbol of the array.
        """
        return self._symbol

    @property
    def shape(self):
        """
        Returns the shape of the array.
        """
        return self.data.shape


class CArray(Array):
    """
    Symbolic C-style array.
    """

    language = "C"

class ArrayElement(Expr):
    """
    Base class for displaying an element of an array in symbolic form.
    """
    def __new__(cls, base, indices):
        """
        This constructor is intended to be called by the `_getitem__` method of
        an instance of `Array`.

        Arguments:
            base : A reference to the array object that was called.
            indices : A tuple that specifies the indices of the element to
                access.

        """
        if not is_sequence(indices):
            indices = [indices]

        if len(indices) != len(base.data.shape):
            raise IndexError('Incorrect dimensionality. Expected `%s`. Got `%s`'
                             % (len(base.data.shape), len(indices)))
        obj = Expr.__new__(cls, base)
        obj._base = base
        obj._indices = indices
        return obj

    @property
    def base(self):
        """
        Returns the base (the array that constructed this object)
        """
        return self._base

    @property
    def indices(self):
        """
        Returns the indices used to access the array.
        """
        return self._indices

    def _sympystr(self, printer):
        """
        Displays the array indices as `a[i][j]`.
        Override this function to change it to something like `a[i, j]`.
        """
        indices = [printer.doprint(idx) for idx in self.indices]
        idxstr = ']['.join(indices)
        return printer.doprint("%s[%s]" % (self.base.label, idxstr))

    def _ccode(self, _):
        indices = [str(idx) for idx in self.indices]
        idxstr = ']['.join(indices)
        return "%s[%s]" % (self.base.label, idxstr)

def carray(name, array, dtype='const float', fmt='%f'):
    """
    Builds a string for initializing a C array.

    Arguments:

        array : A list of values to put in the Carray string.
        dtype : An optional string that specifies the data type of the array.
        fmt : An optional formatting string for the array values.

    Returns:

        String containing the array initialization instruction.

    Example:

    ```python
    >>> import numpy as np
    >>> a = np.array([1.1, 2.2, 3.3])
    >>> code = carray('a', a)
    >>> code
    'const float a[3] = {1.100000, 2.200000, 3.300000};\\n'

    ```

    """

    values = []
    if len(array.shape) == 1:
        for element in np.nditer(array.copy(order='C')):
            values.append(fmt % element)
    elif len(array.shape) == 2:
        for i in range(array.shape[0]):
            values.append('{' +  ', '.join([fmt % x for x in array[i, :]]) + '}')
    else:
        raise NotImplementedError("Array dimension larger than two is not supported.")
    shape = '[' + ']['.join([str(element) for element in array.shape]) + ']'
    code = '%s %s%s = {%s};\n' % (dtype, name, shape, ', '.join(values))
    return code
