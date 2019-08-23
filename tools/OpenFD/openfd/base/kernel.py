from sympy.printing import ccode
from sympy import symbols, sympify
from . import Constant
import numpy as np

def kernel1d(bounds, block, lhs, rhs, header=True):
    lhs         = _to_seq(lhs)
    rhs         = _to_seq(rhs)
    bounds      = (bounds, )
    block       = (block, )
    disp        = _displacements(bounds, block)
    bnds        = _bounds(bounds, block)
    isym        = _symbols(('i', ))

    loop = Loop(bnds, block, ('i', )) 
    
    code = ''
    code += loop.header(0)

    for l, r in zip(lhs, rhs):
        for i in loop.range(0):
            gidx  = _grid_indices(block, (i, ), isym)
            code += loop.indent(1) + _body(l, r, *_displace(gidx, disp))

    code += loop.footer(0)

    return code

def kernel2d(bounds, block, lhs, rhs, header=True):
    lhs         = _to_seq(lhs)
    rhs         = _to_seq(rhs)
    disp        = _displacements(bounds, block)
    bnds        = _bounds(bounds, block)
    isym        = _symbols(('i', 'j'))

    loop = Loop(bnds, block, ('i', 'j')) 
    
    code = ''
    code += loop.header(0)
    code += loop.header(1)

    for l, r in zip(lhs, rhs):
        for i in loop.range(0):
                for j in loop.range(1):
                    gidx  = _grid_indices(block, (i, j), isym)
                    code += loop.indent(2) + _body(l, r, _displace(gidx, disp))

    code += loop.footer(1)
    code += loop.footer(0)

    return code

def kernel3d(bounds, block, lhs, rhs, header=True):
    lhs         = _to_seq(lhs)
    rhs         = _to_seq(rhs)
    disp        = _displacements(bounds, block)
    bnds        = _bounds(bounds, block)
    isym        = _symbols(('i', 'j', 'k'))

    loop = Loop(bnds, block, ('i', 'j', 'k')) 
    
    code = ''
    code += loop.header(0)
    code += loop.header(1)
    code += loop.header(2)

    for l, r in zip(lhs, rhs):
        for i in loop.range(0):
                for j in loop.range(1):
                    for k in loop.range(2):
                        gidx  = _grid_indices(block, (i, j, k), isym)
                        code += loop.indent(3) + _body(l, r, _displace(gidx, disp))

    code += loop.footer(2)
    code += loop.footer(1)
    code += loop.footer(0)

    return code

def ckernel(name, bounds, dout, din, body, const='', cin = 'const float *', cout = 'float *', 
            cint = 'const int', cconst = 'const float', cret = 'void', header=False, extrain = [], extraconst = []):

    bounds = _to_seq(bounds)
    dout   = _freesymbols(_to_seq(dout)) + extraconst
    din    = _freesymbols(_to_seq(din)) + extrain

    din = [ i for i in din if i not in dout]

    infun  = lambda x: '%s%s' %(cin, x) if _isptr(x) else '%s %s' % (cconst, x)
    outfun = lambda x: '%s%s' %(cout, x) if _isptr(x) else '%s %s' % (cconst, x)
    insym  = sorted(map(infun, din))
    outsym = sorted(map(outfun, dout))

    str_cout   = ''.join(['%s, ' %(d) for d in set(outsym)])[0:-2]
    str_cin    = ''.join([', %s' %(d) for d in set(insym)])
    str_cint   = ''.join([', %s %s' %(cint, d) for d in set(bounds)])
    str_cconst = ''.join([', %s %s' %(cconst, d) for d in set(const)])

    code = ''
    if header:
        head_char = ';'
    else:
        head_char = ' {'
    code += '%s %s(%s%s%s%s)%s\n' %(cret, name, str_cout, str_cin, str_cint, str_cconst, head_char) 

    if not header:
        code += ''.join(['%s %s\n' % (_tab(1), b) for b in body.split('\n')])
        code += '}\n\n'

    return code

def array(name, array, dtype='const float', fmt='%g'):
    """
    Initializes a C array using the values in `array`.
    
    Parameters

    array : list,
            values to write
    dtype : str, optional,
            data type
    fmt : str, optional,
          formatting string.

    Example

    >>> a = [1.1, 2.2, 3.3]
    >>> code = array('a', a)
    >>> code
    'const float a[3] = {1.1, 2.2, 3.3};\\n'

    """
    values = []
    if len(array.shape) == 1:
        for a in np.nditer(array.copy(order='C')):
            values.append(fmt % a)
    elif len(array.shape) == 2:
        for i in range(array.shape[0]):
            values.append('{' +  ', '.join(map(lambda x : '%f' % x, array[i,:])) + '}')
    else:
        raise NotImplementedError("CArray dimension above larger than two is not supported.")
    shape = '[' + ']['.join(map(str, array.shape)) + ']'
    code = '%s %s%s = {%s};\n' % (dtype, name, shape, ', '.join(values))

    return code

class Loop:

    def __init__(self, bounds, block, syms, use_header=True):
        self.use_header = use_header
        self.block = block
        self.bounds = bounds
        self.syms = syms

    def header(self, i):
     if not self.active(i) or not self.use_header:
         return ''
     bounds = self.bounds[i]
     return self.indent(i) + 'for (int %s = %s; %s < %s; ++%s) {\n' % \
            (self.syms[i], bounds[0], self.syms[i], bounds[1], self.syms[i])

    def active(self, i):
        if self.block[i] == 1:
            return True
        else:
            return False

    def indent(self, i):
        a = 0
        for k in range(i):
            if self.active(k):
                a += 1
        return self.tab(a)

    def range(self, i):
        return range(*_loop_bounds(self.bounds[i], self.block[i]))

    def footer(self, i):
        if not self.active(i) or not self.use_header:
            return ''
        return self.indent(i) + '}\n'

    def tab(self, i, space=8):
        return _tab(i, space)

def _grid_indices(blockidx, loopidx, isym):
    i_ = [_idx(idx, ki, isymi) for idx, ki, isymi in zip(blockidx, loopidx, isym)]
    return i_

def _displace(idx, disp):
    return tuple(i + j for i, j in zip(idx, disp))

def _symbols(syms):
    return [symbols(s) for s in syms]

def _to_seq(u):
    try:
        len(u) > 0
    except TypeError:
        return [u]
    else:
        return u

def _displacements(bounds, block):
    assert len(bounds) == len(block)
    br = lambda b, idx  : b.range(idx)
    return [_displacement(br(b, idx), idx) for b, idx in zip(bounds, block)]

def _bounds(bounds, block):
    assert len(bounds) == len(block)
    br = lambda b, idx  : b.range(idx)
    return [_set_bounds(br(b, idx), idx) for b, idx in zip(bounds, block)]

def _tab(i, space=8):
    return ''.join([' ' for k in range(space*i)])

def _body(lhs, rhs, indices):
    from sympy import expand
    return '%s = %s;\n' % (ccode(expand(lhs[indices])), ccode(expand(rhs[indices])))

def _loop_bounds(bounds, idx):
    if idx == 0 or idx == -1:
        return bounds
    else:
        return (0, 1)

def _idx(idx, i, isym):
    if idx == 0 or idx == -1:
        return i
    else:
        return isym

def _set_bounds(bounds, idx):
    if idx == 0 or idx == -1:
        a = bounds[1] - bounds[0]
        return (0, a)
    else:
        return (bounds[0], bounds[1])

def _displacement(bounds, idx):
    if idx == 0 or idx == -1:
        return bounds[0]
    else:
        return 0

def _freesymbols(obj):
    ds = []
    for o in obj:
        fs = sympify(o).free_symbols
        for f in fs:
            if not isinstance(f, Constant):
                ds.append(f)
    return ds

def _dshape(bounds):
    """ 
    Returns the symbols used to defined the bounds of a grid function for each dimension
    """
    ds = []
    for b in bounds:
        fs = sympify(b[0]).free_symbols
        for f in fs:
            ds.append(f)
        fs = sympify(b[1]).free_symbols
        for f in fs:
            ds.append(f)
    return tuple(ds)

def _isptr(u):
    from .. import GridFunction
    if isinstance(u, GridFunction):
        return True
