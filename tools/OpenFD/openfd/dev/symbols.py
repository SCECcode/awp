from sympy import Symbol

class Constant(Symbol):

    def __new__(cls, label, visible = 0, **kw_args):
        obj = Symbol.__new__(cls, label, **kw_args)
        obj.visible = visible
        return obj


