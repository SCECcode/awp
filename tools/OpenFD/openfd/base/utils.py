"""
Module that contains utility functions and classes.

"""

def to_atom(arg):
    """
    Remove sequence behavior for a single item. If the input argument is a
    sequence with more than one item, it is left unchanged.
    """
    try:
        len(arg)
    except:
        return

    if len(arg) == 1:
            return arg[0]
    else:
            return arg

def to_tuple(arg):
    """
    Convert a single item to a tuple and leave it unchanged if it is a sequence.
    """
    if is_seq(arg):
        return arg
    else:
        return (arg,)

def is_seq(arg):
    """
    Check if an argument is a sequence or not.
    """

    if hasattr(arg, '__iter__') and not isinstance(arg, str):
        return True
    else:
        return False
    
class Struct(dict):
    """
    Make a dict behave as a struct.

    Example:
    
        test = Struct(a=1, b=2, c=3)

    """
    def __init__(self,**kw):
        dict.__init__(self,kw)
        self.__dict__ = self
