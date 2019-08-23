from sympy.core.compatibility import is_sequence
from . import GridFunction, GridFunctionExpression
from . import Bounds


class GridException(Exception):
    pass

class Grid(GridFunctionExpression):
    """
        The Grid class is used to build numerical grid functions that can be used to verify the implementation of
        Operators. By default the Grid class implements a regular Cartesian grid but other grids can be constructed
        by deriving from this class.

        For example, suppose we want evaluate the polynomial expression: x^2 + 2 somewhere on the interval: (a,b) with
        grid spacing h = 1/nx (i.e., the grid is constructed using nx + 1 grid points).
        Then we can do as follows. First we construct the grid, and then we use the grid to build the expression and
        finally we evaluate the expression at the desired grid point. 

        >>> from openfd import Grid, GridFunctionExpression
        >>> from sympy import symbols
        >>> a, b, nx = symbols('a b nx')
        >>> x   = Grid("x", size=nx+1, interval=(a, b))
        >>> x[0]
        a

        To build the expression it is important that we encapsulate it using **GridFunctionExpression**, or otherwise the
        evaluation of the expression will fail
        >>> expr = GridFunctionExpression(x**2 + 2)
        >>> expr[0]
        a**2 + 2

    """

    def __new__(cls, label, size=None, axis=None, interval=None, left=None, right=None, gridspacing=None, **kw_args):

        obj = GridFunctionExpression.__new__(cls, label, size, **kw_args)

        # Defaults
        if size is None:
            size = 2

        if interval is None:
            interval = (0, 1)

        if left is None:
            left = 1
        
        if right is None:
            right = 1

        if axis is None:
            axis = 0
        
        obj._interval = interval
        obj._bounds   = Bounds(size=size, left=left, right=right)
        obj._size     = size
        obj._axis     = axis

        if gridspacing is None:
            obj._h        = obj._gridspacing()
        else:
            obj._h = gridspacing



        return obj
    
    def coordinate(self, idx):
        """ 
            Defines the location of grid points x[idx] = a + h*idx, 
            where idx is the grid point index and h is the grid spacing.
        """
        return self.interval[0] + self.gridspacing*idx

    def __getitem__(self, idx, coordinate=None,**kw_args):
        """ 
            Returns the nodal value of a grid point indexed by 'idx'.
        """
        if is_sequence(idx):
            if len(idx) < self.axis:
                raise IndexError("Grid dimensions are larger than index access.")
            else:
                idx = idx[self.axis]

        if not self.inbounds(idx):
            raise IndexError("Index out of bounds.")
        return self.coordinate(idx)
    
    def inbounds(self, idx):
        return self._bounds.inbounds(idx)

    def is_left(self, idx):
        return self._bounds.is_left(idx)

    def is_right(self, idx):
        return self._bounds.is_right(idx)

    @property
    def gridspacing(self):
        return self._h

    @property
    def interval(self):
        return self._interval

    @property
    def range_left(self):
        return self._bounds.range_left
    
    @property
    def range_right(self):
        return self._bounds.range_right
    
    @property
    def range_interior(self):
        return self._bounds.range_interior

    @property
    def left(self):
        return self._bounds.left

    @property
    def right(self):
        return self._bounds.right

    """ 
        Returns the number needed such that grid spacing is correctly computed.
        If the grid has n + 1 grid points, then this function returns n.

    """
    def n(self):
        return self._size - 1

    @property
    def interior(self):
        return self._bounds.interior

    @property
    def axis(self):
        return self._axis

    def _gridspacing(self):
        """
            Defines the grid spacing of the grid using the interior grid points.
        """
        h = (self.interval[1] - self.interval[0])/(self.n())
        return h

from sympy import Rational
class StaggeredGrid(Grid):
    """
        Implements a staggered grid by shifting the standard grid by +h/2. 
        Let the staggered grid be defined on the interval 
        a <= x <= b, then
        the grid points are defined as
        x_j = a + (j - 1/2)*h for j = 1,2,..
        x_0 = a (left boundary point)
        for grid spacing h.

    """

    def __new__(cls, label, size=2, interval=None, **kw_args):

        obj = Grid.__new__(cls, label, size, interval, left=1, right=1, **kw_args)

        return obj
    
    def n(self):
        return self._size-2

    def coordinate(self, idx):
        a = self.interval[0]
        b = self.interval[1]
        h = self.gridspacing

        if self.is_left(idx):
            return a
        elif self.is_right(idx):
            return b
        else:
            return a + (idx-Rational(1,2))*h
