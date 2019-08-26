"""

    Sponge layers

    cerjan : Cerjan sponge layer

"""
from openfd import GridFunctionExpression


#def make_cerjan(label, fields, bounds, regions, *args):
#    from openfd import make_kernel
#    make_kernel(
#    pass

class Cerjan(GridFunctionExpression):

    def __new__(cls, expr, bounds, gam=0.92, **kw_arg):

        obj =  GridFunctionExpression.__new__(cls, expr, 'C', **kw_arg)
        obj.gam = gam
        obj.bounds = bounds
        obj.expr = expr

        return obj

    def __getitem__(self, indices, **kw_args):
        from openfd import Left, Right
        axis = 0
        damp = 1
        for idx in indices:
            if isinstance(idx, Left):
                side = 0
            elif isinstance(idx, Right):
                side = 1
            else:
                side = -1

            if side != -1:
                damp *= cerjan(axis, side, self.bounds, 
                               gam=self.gam)

            axis += 1
        return damp*self.expr[indices] 



def cerjan(axis, side, bounds, gam=0.92):
    from sympy import exp, symbols
    import numpy
    
    if side == 0:
        size = 0 
        b = bounds[axis].left - 1
    elif side == 1:
        size = bounds[axis].size
        b = bounds[axis].right
    else:
        raise ValueError('side must be 0 (left) or 1 (right)')
    
    indices = symbols('i j k')
    idx = indices[axis]

    alpha = numpy.sqrt(-numpy.log(gam))/b

    damp = exp(-alpha**2*(idx - b)**2)
    return damp

