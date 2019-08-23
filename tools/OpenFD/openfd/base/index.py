from sympy import Symbol, Expr, sympify
from . import GridFunctionBase


#TODO: Remove this solution for accessing compute regions because it  
# prevents property evaluation of index expressions.
class Index(Expr):

    def __new__(cls, *args):
        args = list(map(sympify, args))
        return Expr.__new__(cls, *args)

    def _ccode(self, p):
        from sympy import ccode
        return ccode(self.base)

    @property
    def base(self):
        return self.args[0]

    def _sympystr(self, p):
        return p.doprint(self.base)


class Left(Index):
    pass

class Right(Index):
    pass
