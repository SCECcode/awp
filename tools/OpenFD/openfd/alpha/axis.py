class Axis(object):

    def __init__(self, axis, lookup=None, inverse_lookup=None):

        if not lookup:
            self.lookup = default_lookup

        if not inverse_lookup:
            self.inverse_lookup = default_inverse_lookup

        if isinstance(axis, str):
            self._label = axis
            self._idx = self.lookup(axis)
        elif isinstance(axis, int):
            self._label = self.inverse_lookup(axis)
            self._idx = axis
        else:
            raise ValuError('Axis must be string or int')


    @property
    def idx(self):
        return self._idx

    @property
    def label(self):
        return self._label


def default_lookup(key):
    opts = {'x' : 0, 'y' : 1, 'z' : 2}
    return opts[key]

def default_inverse_lookup(idx):
    opts = ['x', 'y', 'z']
    return opts[idx]


