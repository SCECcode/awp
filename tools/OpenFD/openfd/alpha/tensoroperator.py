from . import Operator
from . import Axis






class TensorOperator(Operator):
    pass

    def __new__(cls, label, idx='', data=None, T=0, **kw_args):
        obj = Operator.__new__(cls, label, idx, **kw_args)
        obj._data = data
        obj._T = T
        return obj

    def copy(self, idx=''):
        obj = Operator.copy(self, idx)
        if hasattr(self, '_T'):
            obj._T = self._T
        if hasattr(self, '_axis'):
            obj._axis = self._axis
        if hasattr(self, '_data'):
            obj._data = self._data
        return obj
      
    def __getitem__(self, indices):
        from . utils import to_tuple
        from . import index

        if not hasattr(self, '_axis'):
            no_axis()

        indices = to_tuple(indices)

        ax = self._axis[0]
        idx = indices[ax]

        if hasattr(idx, 'region'):
            reg = idx.region
        else:
            reg = 0

        stencil = self._data[reg]

        if self._T:
            return stencil.T(self.args, ax, indices)
        else:
            return stencil(self.args, ax, indices)

    @property
    def label(self):
        lbl = self._label
        if hasattr(self, '_axis'):
            lbl = lbl + '.' + self._axis[1]
        if self._T:
            lbl = lbl + '.T'
        return lbl

    def new_axis(self, idx, label, update=None):
        from . operators import new_state
        if not update:
            update = lambda x : no_nesting()
        return new_state(self, '_axis', (idx, label), update)

    @property
    def x(self):
        return self.new_axis(0, 'x')

    @property
    def y(self):
        return self.new_axis(1, 'y')

    @property
    def z(self):
        return self.new_axis(2, 'z')

    @property
    def T(self):
        from . operators import new_state
        return new_state(self, '_T', True, lambda x : not x)

# Exceptions
def no_nesting():
    raise NotImplementedError('Operator nesting is not supported.')

def no_axis():
    raise ValueError('No operator axis selected')
