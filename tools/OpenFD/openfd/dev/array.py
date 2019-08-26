from sympy.core.compatibility import is_sequence
from sympy import Expr, Symbol


class Array(Expr):
    """ 
    Base class for symbolic arrays. Arrays are displayed as 
    `a[i][j]` in symbolic expressions and hold numerical data. These arrays are 
    intended for storing small amounts of data that fit onto the stack, or into 
    GPU registers.
    Use Sympy's `ccode()` function to generate the c code for intializing the 
    array.


    Attributes:
        label : A string naming this array. This label is displayed when
                printing an expression containing the array.  
        data : A numpy.array that contains the data for the array.  
        dtype : A string, optional, that specifies the C type of the
                array. 
        format : A string, optional, that specifies the C formating string
                 for the array values.
        init : Append array initialization to beginning of compute kernel.

    """

    def __new__(cls, label, data):
        """
        Initialize array.


        """
        label = Symbol(label)
        obj = Expr.__new__(cls, label)
        obj._label = label
        
        obj.data = data
        obj.dec_str()
        return obj

    def dec_str(self):
        """
        Defines the type and formatting options for the array.
        """

        self.dtype = 'const float'
        self.format = '%.16f'

    def __getitem__(self, indices):
        """
        Returns the requested element in symbolic form. 

        :param : tuple that specifies the indices of the element. 
                 The indices can be either in symbolic form form or ints.

        """
        return ArrayElement(self, indices)

    def _sympystr(self, p):
        return p.doprint(self.label)

    def _ccode(self, p):
        return carray(self.label, self.data, self.dtype, self.format)
        
    @property
    def label(self):
        return self._label

class CArray(Array):

    language = "C"

class ArrayElement(Expr):
    """

    Base class for displaying an element of an array in symbolic form.

    """
    def __new__(cls, base, indices):
        """
        This constructor is intended to be called by the _getitem__ function of an 
        array. 

        :param base : A reference to the array object that was called.
        :param indices : A tuple that specifies the indices of the element to 
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
        return self._base
    
    @property
    def indices(self):
        """
        Returns the indices used to access the array.
        """
        return self._indices

    def _sympystr(self, p):
        """
        Displays the array indices as `a[i][j]`. 
        Override this function to change to something like `a[i,j]`.
        """
        indices = map(p.doprint, self.indices)
        idxstr = ']['.join(indices)
        return p.doprint("%s[%s]" % (self.base.label, idxstr))

    def _ccode(self, p):
        indices = map(str, self.indices)
        idxstr = ']['.join(indices)
        return "%s[%s]" % (self.base.label, idxstr)

def carray(name, array, dtype='const float', fmt='%.16f'):
    """
    Builds a string for initializing a C array.

    :param array : A list of values to put in the Carray string.
    :param dtype : An optional string that specifies the data type of the array.
    :param fmt : An optional formatting string for the array values.
    :returns : String containing the array initialization instruction.

    Example

    >>> import numpy as np
    >>> a = np.array([1.1, 2.2, 3.3])
    >>> code = carray('a', a)
    >>> code
    'const float a[3] = {1.100000, 2.200000, 3.300000};\\n'

    """
    import numpy as np 

    values = []
    if len(array.shape) == 1:
        for a in np.nditer(array.copy(order='C')):
            values.append(fmt % a)
    elif len(array.shape) == 2:
        for i in range(array.shape[0]):
            values.append('{' +  ', '.join(map(lambda x : fmt % x, array[i,:])) + '}')
    else:
        raise NotImplementedError("Array dimension larger than two is not supported.")
    shape = '[' + ']['.join(map(str, array.shape)) + ']'
    code = '%s %s%s = {%s};\n' % (dtype, name, shape, ', '.join(values))
    return code
