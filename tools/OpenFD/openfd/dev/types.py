import numpy as np
class C(object):
    _const = 'const'
    _void = 'void'
    _float = 'float'
    _double = 'double'
    _int = 'int'
    _uint = 'size_t'
    _ptr = '*  __restrict__ '

    @staticmethod
    def get_type(nptype):
        """
        Return C type given numpy type

        Arguments:
            nptype : Numpy type (e.g., np.float32)

        """

        out = None
        if nptype == np.int32:
            out = C._int
        elif nptype == np.uint32:
            out = C._uint
        elif nptype == np.float32:
            out = C._float
        elif nptype == np.float64:
            out = C._double
        else:
            raise ValueError('Unknown type')
        return out

    @staticmethod
    def get_ptr(nptype):
        """
        Define as pointer using '*'

        Arguments:
            nptype : Numpy type (e.g., np.float32)

        """
        return C.get_type(nptype) + ' ' + C._ptr

