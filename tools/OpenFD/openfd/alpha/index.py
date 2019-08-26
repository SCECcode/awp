"""
Module for manipulating indices.

"""
from sympy import Symbol
class Index(Symbol):
    pass


class IndexTarget(object):
    """
    Class that lets you target a specific index in a list for manipulation. Only
    the selected index can be changed in the list.

    Attributes:

        component : The component that is mutable, can be either `int` or `string`.
                    All other components are immutable.
        base : A list of indices to modify.

    """

    def __init__(self, component, base):
        """
        """
        self.component = get_component_id(component)
        self.base = base

    def add(self, value):
        """
        Adds some value to the non-locked component.

        Arguments:
            value : The value to add.

        Returns:
            tuple : A tuple containing the updated result.

        Example:
            >>> index = IndexTarget('x', (2, 1))
            >>> index.add(10)
            (12, 1)
        """
        out = list(self.base)
        out[self.component] += value
        return tuple(out)

    def mul(self, value):
        """
        Adds some value to the non-locked component.

        Arguments:
            value : The value to multiply with.

        Returns:
            tuple : A tuple containing the updated result.

        Example:
            >>> index = IndexTarget('x', (2, 1))
            >>> index.mul(10)
            (20, 1)
        """
        out = list(self.base)
        out[self.component] *= value
        return tuple(out)

    def set(self, value):
        """
        Assigns some value to the non-locked component.

        Arguments:
            value : The value to assign.

        Returns:
            tuple : A tuple containing the updated result.

        Example:
            >>> index = IndexTarget('x', (1, 1))
            >>> index.set(10)
            (10, 1)
        """
        out = list(self.base)
        out[self.component] *= value
        return tuple(out)


def get_component_id(component_id, default_components=None):
    """
    Converts a component id in string form to integer form.

    Returns:
        `int` : The component id
        default_components : A dictionary that contains the components labels as
            keys and with non-negative integers as values.

    Example:

        >>> get_component_id('x')
        0

    """
    if not default_components:
        labels = 'xyztuvw'
        components = {}
        for index, label in enumerate(labels):
            components[label] = index
    else:
        components = default_components

    if isinstance(component_id, int):
        if component_id < 0:
            raise ValueError('Only non-negative components allowed.')
        else:
            return component_id
    else:
        return components[component_id]

def indices(string):
    """
    Reads in a string of labels that will be converted to indices. Each
    label must be separated by a ` ` (space).

    Arguments:

        string : The string containing the labels to convert to indices. 

    Returns:

        tuple : This tuple contains the indices.

    Example:
        ```python
        >>> a, b = indices('a b')
        >>> a
        a
        

    """
    labels = string.split(' ')
    symbols = []
    for label in labels:
        symbols.append(Index(label))
    if len(symbols) == 1:
        return symbols[0]
    else:
        return tuple(symbols)


def index_to_constant(index_expr):
    """
    Returns the constant part of an index expression.

    Arguments: 
        index : The index to parse. This index can contain both symbols and
            constants.

    Returns:
        int : The constant part. Returns `0` if there is no constant.
        
        
    """
    from sympy import sympify

    index = sympify(index_expr)
    index_dict = index.as_coefficients_dict()
    constant = index_dict[1]
    return constant

def index_to_symbol(index_expr):
    """
    Returns the symbolic part of an index expression.

    Arguments: 
        index : The index to parse. This index can contain both symbols and
            constants.

    Returns:
        int : The symbolic part. Returns `0` if there is no symbolic part.
        
    """
    from sympy import sympify
    index = sympify(index_expr)
    index_dict = index.as_coefficients_dict()
    index_dict[1] = 0
    symbol = sum([k*index_dict[k] for k in index_dict])
    return symbol
