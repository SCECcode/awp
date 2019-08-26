"""
See the notebook bounds.ipynb for usage.

"""
from . import index

class Bounds(object):
    """
    Base class for `Bounds`. This class handles symbolic bounds of the form
    `[a, b)`, where `a` is and `b` are symbolic expressions. The upper bound `b`
    is non-inclusive. The only possible values in the set are integers. Thus,
    the least upper bound is set to `b-1`.

    The bounds object makes it possible to check if an index is in bounds by
    calling its special function `__getitem__` and passing in an index. This
    index can be a symbolic expression. If it is possible to determine if the
    index is out of bounds, then a specific action is taken for the bound that
    was violoated. The possible actions to take are specified via the
    `lower_out_of_bounds` and `upper_out_of_bounds` attributes

    Attributes:

        lower : The expression for the lower (inclusive) bound.
        upper : The expression for the upper (non-inclusive) bound.
        lower_out_of_bounds : Determines what action to take when a lower
            bounds violation occurs. Defaults to `'wrap-around'`. See the notes
            for a discussion of the different options.
        upper_out_of_bounds : Determines what action to take when a lower
            bounds violation occurs. Defaults to `'raise exception'`. See the
            notes for a discussion of the different options.

    Notes:

        The possible actions that can be taken during a out-of-bounds violation
        are the following:
        * `'wrap-around'` : Causes the index to wrap-around from below if the
            least upper bound is exceeded. Similarly, wrap-around from above
            occurs when the lower bound is exceeded.
        * `'raise-exception'` : Raises an exception that explains if the bounds
            violation is due to under-, or overflow.
        * `'no action'` : No action is taken in this case. In other words, the
            input index is returned as output.

    """

    def __init__(self, lower, upper, lower_out_of_bounds='wrap-around',
                 upper_out_of_bounds='raise exception'):

        self.lower_out_of_bounds = lower_out_of_bounds
        self.upper_out_of_bounds = upper_out_of_bounds

        lower_actions = {'wrap-around' : self._lower_wrap_around,
                         'raise exception' : _lower_raise_exception,
                         'no action' : _no_action}

        upper_actions = {'wrap-around' : self._upper_wrap_around,
                         'raise exception' : _upper_raise_exception,
                         'no action' : _no_action}

        if self.lower_out_of_bounds in lower_actions:
            self._lower_out_of_bounds_action = lower_actions[lower_out_of_bounds]
        else:
            raise NotImplementedError('Undefined action `%s`.' %
                                      lower_out_of_bounds)

        if self.upper_out_of_bounds in upper_actions:
            self._upper_out_of_bounds_action = upper_actions[upper_out_of_bounds]
        else:
            raise NotImplementedError('Undefined action `%s`.' %
                                      upper_out_of_bounds)

        self.lower = lower
        self.upper = upper

    def is_in_bounds(self, idx):
        """
        Determines if an index is in bounds or not.

        Arguments:

            idx : The index to check. The index can be symbolic.

        Returns:

            bool : `True` is returned if the index is in bounds. Otherwise,
                `False` is returned.

        Raises:

            IndexError: This exception is raised if it is not possible to
                determine if the index is in bounds or not.

        """
        self._check_ambiguity(idx)
        return self._is_in_lower_bound(idx) and self._is_in_upper_bound(idx)

    def lower_out_of_bounds_action(self, idx, *args, **kw_args):
        """
        Called when the index is below the lower bound.
        Override to change default behavior.

        Arguments:

            idx : Index that triggered the call of this method.

        """
        return self._lower_out_of_bounds_action(idx)

    def upper_out_of_bounds_action(self, idx, *args, **kw_args):
        """
        Called when the index is above the upper bound.
        Override to change default behavior.

        Arguments:

            idx : Index that triggered the call of this method.

        """
        return self._upper_out_of_bounds_action(idx)

    def __getitem__(self, idx):
        """
        Performs a specific action to an index if it is less than
        or great than the given lower or upper bounds. See the class
        attributes for a discussion of the possible actions to take.

        Arguments:

            idx : The index under consideration.

        Returns:

            The index itself if it is in bound. Otherwise, calls the action
            function for the specific out of bounds (below lower bounds, or
            above upper bounds).

        Raises:

            LowerOutOfBoundsException : This exception is raised if the
                attribute `lower_out_of_bounds` is set to `'raise exception'`
                and the index under consideration is less than the lower bound
                attribute `lower`.

            UpperOutOfBoundsException : This exception is raised if the
                attribute `upper_out_of_bounds` is set to `'raise exception'`
                and the index under consideration is greater than or equal to
                the upper bound attribute `upper`.

        """

        self._check_ambiguity(idx)
        if not self._is_in_lower_bound(idx):
            return self.lower_out_of_bounds_action(idx)
        if not self._is_in_upper_bound(idx):
            return self.upper_out_of_bounds_action(idx)
        return idx

    def intersection(self, other):
        """
        Determines the intersection of this bounds object and another bounds
        object. The intersection is a new Bounds object given by the bounds that
        is common to both objects (overlap). The intersection can only be
        computed for two objects at a time.

        Arguments:

            other(`Bounds`): The other object that intersects this object.
            assumptions(`dict`, optional) : Assumptions needed to resolve any
                ambiguities when any of the lower of upper bounds are symbolic.
                See the notes for further details.

        Returns:

            `Bounds` : A new bounds object with upper and lower bounds defined
                by the overlap of the two intersecting objects. If there is no
                overlap, then `None` is returned.

        Raises:

            ValueError: This exeception is raised if the intersection cannot be
            computed because it is not possible to evaluate the bounds.

        Examples:

            The following example computes the intersection between the Bounds
            object that partially overlap.
            ```python
            >>> A = Bounds(0, 4)
            >>> B = Bounds(2, 6)
            >>> C = A.intersect(B)
            >>> C
            (2, 4)
            >>> D = B.intersect(A)
            >>> D
            (2, 4)

            ```

        Notes:

            If any of the lower and upper bounds are symbolic, then there
            currently is no way to evaluate them. Some possible solutions to this
            problem are:

            1. Add attributes to this class that determine the rank of the lower
                and upper bounds. The lowest rank determines the minimum and the
                highest rank determines the maximum.

            2. Modify the class `Index` that can be used to describe bounds as
                symbolic expressions. This modification would introduce the
                option to assign a value to each object of type `Index` and this
                value will be used when such that the lower and upper bounds can
                be evaluated.

            3. Add a new class `Assumptions` that handles inequality, or
               equality-based assumptions for each symbol. Each such assumption
               would be of the form `assumptions.add_greater_than(a, b)` that
               translates into saying that the value of Index `a` is greater
               than the value of Index `b`, and so forth.

        """
        #FIXME: Add implementation
        pass


    def _is_in_lower_bound(self, idx):
        constant = index.index_to_constant(idx)
        symbol = index.index_to_symbol(idx)
        return not (symbol == index.index_to_symbol(self.lower) and
                    constant < index.index_to_constant(self.lower))

    def _is_in_upper_bound(self, idx):
        constant = index.index_to_constant(idx)
        symbol = index.index_to_symbol(idx)
        return not (symbol == index.index_to_symbol(self.upper) and
                    constant >= index.index_to_constant(self.upper))

    def _check_ambiguity(self, idx):
        symbol = index.index_to_symbol(idx)
        if (symbol - index.index_to_symbol(self.lower) != 0 and
                symbol - index.index_to_symbol(self.upper) != 0):
            raise IndexError("""Unable to determine if index: `%s` is in bounds
            or not.""" % idx)

    def _lower_wrap_around(self, idx):
        """
        Causes wrap-around for lower bounds.

        Arguments:

            idx : The index to wrap-around.

        Returns:

            The index after wrap-around has been performed.
        """
        return self.upper + idx  - self.lower

    def _upper_wrap_around(self, idx):
        """
        Causes wrap-around for upper bounds.

        Arguments:

            index : The index to wrap-around.

        Returns:

            The index after wrap-around has been performed.
        """
        return idx - self.upper + self.lower

def _lower_raise_exception(idx):
    """
    Raises an out of bounds exception.
    """
    raise LowerOutOfBoundsException("""The index `%s` is out of bounds
                                    (underflow)""" % idx)

def _upper_raise_exception(idx):
    """
    Raises an out of bounds exception.
    """
    raise UpperOutOfBoundsException('The index `%s` is out of bounds (overflow)'
                                    % idx)

def _no_action(idx):
    """
    Returns the index unchanged.
    """
    return idx

class LowerOutOfBoundsException(Exception):
    """
    This exception is raised when a lower out of bounds error occurs.
    In this case, an index is less than `Bounds.lower`.
    """
    pass

class UpperOutOfBoundsException(Exception):
    """
    This exception is raised when a upper out of bounds error occurs.
    In this case, an index is greater than or equal to `Bounds.upper`.
    """
    pass
