from sympy import sympify
from sympy.core.compatibility import is_sequence

class Axis:
    """ 
    This class is used to select a pre-defined component from indices such as (x, y, z).

    Example:
    ```python
    >>> c = Axis("x", shape=(10, 11, 12))
    >>> c.val((1,2,3))
    1
    
    >>> c.len
    10

    >>> c.id
    0
    
    >>> c.axis
    'x'

    ```

    """

    def __init__(self, axis, shape, dims=None):

        if not dims:
            dims = list(range(len(shape)))
        if len(shape) != len(dims):
            raise IndexError("Dimensions and shape mismatch")

        self.axis_dict = {"x": 0, "y": 1, "z": 2}
        idx = self.axis_dict[axis]
        # Rearrange shape so that it matches dim, and determine the local id
        max_dim = max(dims)+1
        new_shape = [0]*max_dim
        local_id = idx 
        for i, ai in enumerate(dims):
            new_shape[dims[i]] = shape[i]
            if idx == dims[i]:
                local_id = i
        shape = new_shape

        is_valid_axis = False
        self._local_id = local_id
        self._axis = axis
        self.shape = shape

    @property
    def id(self):
        return self.axis_dict[self._axis]

    @property
    def local_id(self):
        """
        This index is the same as `id` unless the argument `dims` has been
        specified. `dims` make it possible to apply select say the "z" axis for
        a 1D object if `dims = (2,)`. However, this property can cause problem
        for objects that want to axes say `shape[axis.id]` because now that
        element is out of bounds (length of shape is 1 one, but
        axis.id = 2, in this example). In this case, we define `local_id` as the
        index obtained by taking the position of the value in `dims` that
        matches the global axis id.

        """
        return self._local_id

    @property
    def axis(self):
        return self._axis

    def val(self, indices):
        if is_sequence(indices): 
            if self.shape and len(self.shape) > len(indices):
                raise ValueError("Dimension mismatch.")
            index = sympify(indices[self.id])
        else:
            if self.shape and len(self.shape) != 1:
                raise ValueError("Dimension mismatch.")
            index = sympify(indices)
        return index

    def add(self, indices, value, overwrite=False):
        """
        Add a value to the axis component.
        
        Parameters

        indices : tuple,
                  is the indices to a add a value to
        value : Expr,
                is the value to add
        overwrite: bool,
                   Overwrites the value for the axis component if true

        Returns

        out : tuple,
              is the updated values.


        # Example

        >>> a = Axis('x', shape=(1,1))
        >>> a.add((1,1), 1)
        (2, 1)
        >>> a.add((1,1), -1, overwrite=True)
        (-1, 1)
                
        """
        return _add(self.id, indices, value, overwrite)


    @property
    def len(self):
        n     = self.shape[self.id]
        return n

def _add(idx, indices, value, overwrite=False):
    if is_sequence(indices):
        if overwrite:
            new_index = indices[:idx] + (sympify(value),) \
                        + indices[idx+1:]
        else:
            new_index = indices[:idx] + (sympify(indices[idx]) + sympify(value),) \
                        + indices[idx+1:]
    else:
        if overwrite:
            new_index = sympify(value)
        else:
            new_index = sympify(indices) + sympify(value)
    return new_index
