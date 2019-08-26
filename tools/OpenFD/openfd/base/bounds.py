from sympy import sympify
from sympy.core.compatibility import is_sequence

class Bounds:


    """
    The bounds class is used to define the bounds of an Operator, or GridFunction.
    There are three properties that determine the bounds, the number of left and right boundary points,
    and the total number of points. Once initialized, a bounds object can be queried to determine
    if a expression is in bounds or not, and if it is associated with the boundary or not. 
    Keep in mind that the bounds checking is in no way smart, meaning that if the bounds of the expression does not
    match the bounds of the object it is likely to to be considered valid and viewed as indexing the interior.

    # Examples

    The following examples set up a Bounds object with ```nx+1``` grid points (symbolic) and one left and one right boundary
    point.
    ```python
    >>> from sympy import symbols
    >>> nx = symbols('nx')
    >>> b = Bounds(size=nx+1,left=1,right=1)
    >>> b.is_left(0)
    True
    >>> b.is_left(1)
    False
    >>> b.is_interior(1)
    True
    >>> b.is_right(nx)
    True
    >>> b.is_right(nx+1)
    False

    ```
    """

    def __init__(self, size, left=0, right=0, periodic=False ):
        if size is None:
            size = (1,)

        if periodic:
            left=0
            right=0

        # Check that only the size is symbolic
        sym_size = sympify(size)
        if isinstance(size, int):
        
            if size < 0:
                raise ValueError("'size' must be non-negative.")

            if size < left + right:
                raise ValueError("Boundary size cannot exceed total size.")

        if not isinstance(left, int) or left < 0:
            raise ValueError("'left' must be a non-negative integer.") 
        
        if not isinstance(right, int) or right < 0:
            raise ValueError("'right' must be a non-negative integer.") 

        self._size  = size
        self._left  = left
        self._right = right
        self._periodic = periodic

        # enums returned when accessing the corresponding field
        self._tag_left      = 0
        self._tag_right     = -1
        self._tag_interior  = 1

    def is_left(self, index):
       from . import Left
       if isinstance(index, Left):
           return True
       if self.is_overflow(index) or self.is_underflow(index) or self.left == 0:
          return False

       d, dn, n = self.parse_index(index)

       # Assume that it there is anything symbolic in the expression except
       # the shape, then the index maps to the interior
       tmp = d.copy()
       interior_value = index - tmp[1] - n*tmp[n]
       if not interior_value == 0:
           return False

       if self.size == 1:
           return False

       if d[0] == 1:
           return True
       if d[1] > 0 and d[1] < self.left and (not n in d or n == 0):
           return True

       return False

    def is_right(self, index):
       from . import Right
       if isinstance(index, Right):
           return True
       if self.is_overflow(index) or self.is_underflow(index):
          return False
       
       d, dn, n = self.parse_index(index)
       
       if self.size == 1:
           return False

       # symbolic
       if dn[0] == 0: 
           # Within bounds
           if len(d) == 1 and d[1] < 0 and -d[1] <= self.right:
               return True
           if d[n] > 0 and -d[1] < 1-dn[1]+self.right:
               return True
       # numeric
       if n == 0:
           # Within bounds
           if len(d) == 1 and d[1] < 0 and -d[1] <= self.right:
               return True
           if d[1] < dn[1] and d[1] >= dn[1] - self.right:
               return True

       return False

    def is_interior(self, index):
       if self.periodic:
           return true

       d, dn, n = self.parse_index(index)

       if self.is_left(index):
        return False

       if self.is_right(index):
        return False

       if self.is_overflow(index):
        return False

       if self.is_underflow(index):
        return False

       return True

    def cycle(self, index):
        d, dn, n = self.parse_index(index)
        
        # symbolic
        if dn[0] == 0: 
            if len(d) == 1 and d[1] < 0:
                return self.size + d[1]
            if d[n] > 0 and d[1] >= self.size - n :
                return n - self.size  + d[1]
        return index

    def range(self, idx):
        import warnings 
        if idx == 0:
            return self.left_range
        if idx == 1:
            return self.interior_range
        if idx == 2:
            return self.right_range
        if idx == -1:
            #TODO: Deprecate me
            warnings.warn('Region `-1` will be deprecated in the near future.'\
                         ' Please use `region=2` instead.', 
                          PendingDeprecationWarning)
            return self.right_range

    def __getitem__(self, index):
       from . import Left, Right

       if self.periodic:
           return self._tag_interior

       if isinstance(index, Left):
           return self._tag_left
       
       if isinstance(index, Right):
           return self._tag_right

       if self.is_underflow(index):
           raise IndexError("Index out of bounds (underflow).")
       if self.is_overflow(index):
           raise IndexError("Index out of bounds (overflow).")

       if self.is_left(index):
           return self._tag_left
       
       if self.is_right(index):
           return self._tag_right

       return self._tag_interior

    def is_underflow(self, index):
       d, dn, n = self.parse_index(index)

       if len(d) == 1 and -d[1] > self.right and not self.right == 0:
           return True
       else:
           return False

    def is_overflow(self, index):
       d, dn, n = self.parse_index(index)

       # numeric
       if n == 0:
           # Overflow
           if d[1] >= dn[1]:
               return True
       
       # symbolic
       if dn[0] == 0: 
           # Overflow
           if d[n] > 0 and d[1] >= dn[1] or d[n] > 1:
               return True

       return False

    def inbounds(self, index):
        if self.periodic:
            return True
        if self.is_underflow(index):
            return False

        if self.is_overflow(index):
            return False
        
        return True

    def parse_index(self, index): 
       index       = sympify(index)
       index_dict  = index.as_coefficients_dict()

       # Determine if the shape is symbolic or numeric
       n  = sympify(self.size)
       size_dict = n.as_coefficients_dict()
       # Remove constant part
       size_no_constant = n - size_dict[1]
       return index_dict, size_dict, size_no_constant

    @property
    def left(self):
        return self._left
    
    @property
    def right(self):
        return self._right
    
    @property
    def interior(self):
        return self.size - self.left - self.right
    
    @property
    def size(self):
        return self._size

    @property
    def left_range(self):
        """ 
        Returns the bounds of the left indices as a tuple (first, last). 
        'first' is the index of the first left point
        'last' is the index of the last left point + 1 

        # Examples

        One interior point
        ```python
        >>> b = Bounds(size=3, left=1, right=1)
        >>> b.left_range
        (0, 1)

        ```
        """
        return (0, self.left)

    @property
    def right_range(self):
        """ 
        Returns the bounds of the right indices as a tuple (first, last). 
        'first' is the index of the first right point
        'last' is the index of the last right point + 1 

        # Examples

        One interior point

        ```python

        >>> b = Bounds(size=3, left=1, right=1)
        >>> b.right_range
        (2, 3)

        ```
        """
        return (self.size - self.right, self.size)

    @property
    def interior_range(self):
        """ 
        Returns the bounds of the interior indices as tuple (first, last). 
        'first' is the index of the first interior point
        'last' is the index of the last interior point + 1 

        # Examples

        one interior point

        ```python
        >>> b = Bounds(size=3, left=1, right=1)
        >>> b.interior_range
        (1, 2)

        ```
        """
        return (self.left, self.size - self.right)

    @property
    def tag_left(self):
        return self._tag_left
    
    @property
    def tag_right(self):
        return self._tag_right
    
    @property
    def tag_interior(self):
        return self._tag_interior

    @property
    def periodic(self):
        return self._periodic



