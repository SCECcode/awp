from ..base import GridFunctionExpression, Axis, Bounds, Operator, gridfunction as gridfunc
from sympy import sympify, Tuple
from sympy.core.compatibility import is_sequence

class Restriction(GridFunctionExpression):

    def __new__(cls, expr, axis, opidx, expridx, shape=None, **kw_args):
        if shape is None:
            shape = gridfunc.infer_shape(expr)

        if shape is None:
            shape = (1,)

        if is_sequence(shape):
            shape = Tuple(*shape)
        else:
            shape = Tuple(shape)


        obj = GridFunctionExpression.__new__(cls, expr, axis, opidx, **kw_args)

        obj._opidx = opidx
        obj._expridx = expridx
        obj._axis = Axis(axis, shape)
        return obj

    def __getitem__(self, indices, **kw_args):
        index = self._axis.val(indices)
        n     = self._axis.len

        if not is_sequence(indices):
            idx = [indices]
        else:
            idx = [a for a in indices]

        idx[self._axis.id] = self.expridx

        if index == self.opidx or (index == n - 1 and self.opidx == -1):
            return self._expr[tuple(idx)]
        else:
            return sympify(0.0)

    def bounds(self):
        n =  self._axis.shape[self._axis.id]
        from ..base import Bounds
        b = Bounds(size=n, left=1, right=1)
        return b

    @property
    def opidx(self):
        return self._opidx
    
    @property
    def expridx(self):
        return self._expridx

    def _sympystr(self, p):
        return p.doprint(self.label)
    
    @property
    def label(self):
        return "Restriction(%s, %s, %s, %s)" % (self.args[0], self.args[1], self.opidx, self.expridx)
