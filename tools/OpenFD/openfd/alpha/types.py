import numpy as np
from sympy import Symbol

class Type(Symbol):

    def __new__(cls, label, value=None, dtype=None, **kw_args):
        obj = Symbol.__new__(cls, label)
        obj.value = value
        obj.dtype = dtype
        return obj


class Index(Type):

    def __new__(cls, label, value=None, dtype=np.int32, region=0, **kw_args):
        obj = Type.__new__(cls, label, value, dtype, **kw_args)
        obj._region = region
        return obj

    @property
    def region(self):
        return self._region


