import openfd as fd

class Grid(fd.GridFunction):

    def __new__(cls, label, size, axis=None, layout=None, interval=None):

        if not axis:
            axis = 0
        if not interval:
            interval = (0, 1)

        h = (interval[1] - interval[0])/(size-1)

        obj = fd.GridFunction.__new__(cls, label, (size,), layout=layout)
        obj.size = size
        obj.axis = axis
        obj.h = h
        obj.interval = interval
        return obj

    def __getitem__(self, indices, **kw_args):
        if len(indices) < self.axis:
            raise IndexError('Grid dimension exceeds index dimension')
        return self.gridpoint(indices[self.axis])

    def gridpoint(self, index):
        return index*self.h

    @property
    def gridspacing(self):
        return self.h
