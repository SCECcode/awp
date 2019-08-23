"""
The computation of an expression is partitioned into 3 x 1 regions in 1D, 
3 x 3 regions in 2D, or 3 x 3 x 3 regions in 3D. These regions correspond to the left, 
center, and right parts of the computational domain for a given coordinate direction. 
There are three ways in which these regions can be referenced. We either define a coordinate 
`(0, 0, 0)` , index  `0` or label `left bottom back`. The conversions between in each format
is listed below in the following tables.

**1D**

| Label    | Coordinate | Index |
|----------|------------|-------|
| `left`   | `(0,)`     | `0`   |
| `center` | `(1,)`     | `1`   |
| `right`  | `(2,)`     | `2`   |


**2D**

| Label           | Coordinate | Index |
|-----------------|------------|-------|
| `left bottom`   | `(0, 0)`   | `0`   |
| `center bottom` | `(1, 0)`   | `1`   |
| `right bottom`  | `(2, 0)`   | `2`   |
| `left center`   | `(0, 1)`   | `3`   |
| `center center` | `(1, 1)`   | `4`   |
| `right center`  | `(2, 1)`   | `5`   |
| `left top`      | `(0, 2)`   | `6`   |
| `center top`    | `(1, 2)`   | `7`   |
| `right top`     | `(2, 2)`   | `8`   |

**3D**

| Label                  | Coordinate  | Index |
|------------------------|-------------|-------|
| `left bottom back`     | `(0, 0, 0)` | `0`   |
| `center bottom back`   | `(1, 0, 0)` | `1`   |
| `right bottom back`    | `(2, 0, 0)` | `2`   |
| `left center back`     | `(0, 1, 0)` | `3`   |
| `center center back`   | `(1, 1, 0)` | `4`   |
| `right center back`    | `(2, 1, 0)` | `5`   |
| `left top back`        | `(0, 2, 0)` | `6`   |
| `center top back`      | `(1, 2, 0)` | `7`   |
| `right top back`       | `(2, 2, 0)` | `8`   |
| `left bottom center`   | `(0, 0, 1)` | `9`   |
| `center bottom center` | `(1, 0, 1)` | `10`  |
| `right bottom center`  | `(2, 0, 1)` | `11`  |
| `left center center`   | `(0, 1, 1)` | `12`  |
| `center center center` | `(1, 1, 1)` | `13`  |
| `right center center`  | `(2, 1, 1)` | `14`  |
| `left top center`      | `(0, 2, 1)` | `15`  |
| `center top center`    | `(1, 2, 1)` | `16`  |
| `right top center`     | `(2, 2, 1)` | `17`  |
| `left bottom front`    | `(0, 0, 2)` | `18`  |
| `center bottom front`  | `(1, 0, 2)` | `19`  |
| `right bottom front`   | `(2, 0, 2)` | `20`  |
| `left center front`    | `(0, 1, 2)` | `21`  |
| `center center front`  | `(1, 1, 2)` | `22`  |
| `right center front`   | `(2, 1, 2)` | `23`  |
| `left top front`       | `(0, 2, 2)` | `24`  |
| `center top front`     | `(1, 2, 2)` | `25`  |
| `right top front`      | `(2, 2, 2)` | `26`  |

For the coordinates, there are two formats supported. 
These differ only for the last value that can be either `2` or `-1`. 
The formats are labelled as `c` and `py`, where `py` refers  to Python's wrap around 
feature for accessing the last element in an array using `-1`.

| Label    | Coordinate (*c*) | Coordinate (*py*) |
|----------|------------------|-------------------|
| `left`   | `(0,)`           | `(0,)`            |
| `center` | `(1,)`           | `(1,)`            |
| `right`  | `(2,)`           | `(-1,)`           |

The index can be changed from starting at `0` (zero-based) to `1` (one-based).


"""

# Labels that can be used to select regions in 1D, 2D, and 3D.
_labels1d = ['left',                   'center',                   'right'] 
_labels2d = ['left bottom',            'center bottom',            'right bottom',
             'left center',            'center center',            'right center',
             'left top',               'center top',               'right top']
_labels3d = ['left bottom back',       'center bottom back',       'right bottom back',
             'left center back',       'center center back',       'right center back',
             'left top back',          'center top back',          'right top back',
             'left bottom center',     'center bottom center',     'right bottom center',
             'left center center',     'center center center',     'right center center',
             'left top center',        'center top center',        'right top center', 
             'left bottom front',      'center bottom front',      'right bottom front',
             'left center front',      'center center front',      'right center front',
             'left top front',         'center top front',         'right top front']
_labels = {1 : _labels1d, 2 : _labels2d, 3 : _labels3d}
_onebased = 0

# Reverse lookup tables for labels
_dlabels1d = { ai : i + _onebased for i, ai in enumerate(_labels1d)}
_dlabels2d = { ai : i + _onebased for i, ai in enumerate(_labels2d)}
_dlabels3d = { ai : i + _onebased for i, ai in enumerate(_labels3d)}
_dlabels = {1 : _dlabels1d, 2 : _dlabels2d, 3 : _dlabels3d}

# Coordinate formats that specify the indices to use to denote left, center, and right
# for a given axis 
# The naming convention `py` stands for `python` and refers to the fact that `-1` is used
# to the access the last element of an array in Python.
_coordfmt = {'c' : (0, 1, 2), 'py' : (0, 1, -1)}

# Default coordinate format to use by objects and functions that require knowing 
# the coordinate format (typically passed as an optional parameter).
_defcoordfmt = 'c'
# Default priority list for auto-detecting and converting input to a either 
# a coordinate or index. The lowest index is given the highest priority 
_defclspriority = {'c' : 1, 'py' : 2, 'index' : 3}

class Coordinate:
    def __init__(self, coord, fmt=_defcoordfmt):
        """
        The Coordinate class is used to represent coordinates (i, j), (i, j, k) 
        and facilitates the conversion to other formats (labels, indices).

        Parameters

        coord : tuple(int),
                coordinate data.
        fmt : str, optional.
              coordinate format, can be either 'c' or 'py'. 

        """

        # Check that coord and fmt are correctly specified
        _fmterr(fmt)
        _coorderr(coord, fmt)

        try:
            len(coord)
        except:
            coord = tuple([coord])

        self._coord = coord
        self._fmt = fmt
        self._dim = len(coord)

    def index(self):
        """
        Converts coordinate to index.

        Returns
        
        indexout : Index,
                   index associated with the coordinate.
        """
        indexout = Index(coordtoindex(self.coord, self.format), dim=self.dim)
        return indexout

    def label(self):
        """
        Converts coordinate to label.

        Returns

        labelout : Label,
                   label associated with the coordinate
        """
        labelout = Label(coordtolabel(self.coord, fmt=self.format))
        return labelout

    def toc(self):
        """
        Returns the coordinate in C format.
        """
        return Coordinate(coordtocfmt(self.coord), fmt='c')

    def todef(self):
        """
        Returns the coordinate in the default format.
        """
        return Coordinate(coordtodeffmt(self.coord))

    def topy(self):
        """
        Returns the coordinate in the Python format.
        """
        return Coordinate(coordtopyfmt(self.coord), fmt='py')

    @property
    def coord(self):
        """
        Get coordinate data.

        Returns  

        coordout : tuple(int),
                   coordinate data.
        """
        coordout = self._coord
        return coordout

    @property
    def format(self):
        """
        Get coordinate format.

        Returns

        format : str,
                 coordinate format.

        """
        fmtout = self._fmt
        return fmtout

    @property
    def dim(self):
        """
        Get coordinate dimensionality

        Returns

        dimout : int,
                 coordinate dimensionality.
        """
        dimout = self._dim
        return dimout

class Index:
    def __init__(self, index, dim=1):
        """
        The index class is used to represent the indices that access different parts of the compute regions.
        Indices can be converted to either labels or coordinates.

        Parameters

        index : int,
                integer

        dim : int, optional,
              dimensionality of the problem. For 1D, use `dim = 1`, and so forth. 

        """

        _dimerr(dim)
        _indexerr(index, dim)

        self._index = index
        self._dim = dim

    def coord(self, fmt=_defcoordfmt):
        """
        Converts index to coordinate.

        Parameters 

        fmt : str, optional,
              coordinate format.

        Returns

        coordout : Coordinate,
                coordinate associated with the index.

        """
        coordout = Coordinate(indextocoord(self.index, self.dim, fmt))
        return coordout

    def label(self):
        """
        Converts index to label.

        Returns

        labelout : Label,
                   label associated with the index.

        """
        labelout =  Label(indextolabel(self.index, self.dim))
        return labelout

    @property
    def index(self):
        """
        Get index data.

        Returns

        indexout : int,
                   index.

        """
        indexout = self._index
        return indexout

    @property
    def dim(self):
        """
        Get dimensionality.

        Returns

        dimout : int,
                 dimensionality.

        """
        dimout = self._dim
        return dimout

class Label:
    def __init__(self, label):
        """
        Class for representing the labels that access different parts of the compute regions.
        Labels can be converted to either indices or coordinates.

        Parameters

        label : str,
                label is specified as 'left', 'center', 'right'.

        """

        _labelerr(label)
        self._label = label
    
    def coord(self, fmt=_defcoordfmt):
        """
        Converts label to coordinate.

        Parameters

        fmt : str, optional.
              coordinate format.

        Returns

        outcoord : Coordinate,
                   coordinate corresponding to the label.

        """
        outcoord = Coordinate(labeltocoord(self.label, fmt=fmt))
        return outcoord

    def index(self):
        """
        Converts label to index.

        Returns

        outindex : Index,
                   index corresponding to the label.

        """
        outindex = Index(labeltoindex(self.label), dim=labeltodim(self.label))
        return outindex
    
    @property
    def label(self):
        """
        Get label.

        Returns

        outlabel : str,
                   the label.

        """
        outlabel = self._label
        return outlabel

def autoconvert(arg, dim=None, fmt=_defcoordfmt):
    """
    Attempts to automatically convert the input argument `arg` to either an index, a coordinate, or label.
    Index conversion only takes place if `dim` is specified.

    Parameters

    arg : str, tuple(int), or int,
          input to convert.
    dim : int, optional,
          number of dimensions. This argument only needs to be specified if the input is an index. 
    fmt : str, optional,
          coordinate format. This argument only needs to be specified if the format of the input
          is a coordinate in the non-default coordinate format.
    """

    if isindex(arg, dim):
        out = Index(arg, dim)
    elif islabel(arg):
        out = Label(arg)
    elif iscoord(arg, fmt):
        out = Coordinate(arg, fmt)
    else:
        raise ValueError("Unable to determine type for auto conversion. Got: `%s`" % str(arg))

    return out

def iscoordfmt(fmt):
    """
    Determines if the selected coordinate format exists.

    Parameters

    fmt : string,
          coordinate format.

    Returns

    out : bool,
          `True` if the coordinate format is valid.

    Example

    ```python
    >>> iscoordfmt('c')
    True
    >>> iscoordfmt('py')
    True

    ```
    """

    if any((f == fmt for f in _coordfmt)):
        return True
    else:
        return False

def iscoord(coord, fmt=_defcoordfmt):
    """
    Determines if the input argument `coord` is a coordinate or not.

    Parameters

    coord : tuple,
            coordinate specified using the format `fmt`.
    fmt : tuple, optional,
          coordinate format.

    Returns

    out : bool,
          `True` if the coordinate is specified
          according to the coordinate format `fmt`.

    Example

    ```python

    >>> iscoord((0, 1), fmt='c')
    True
    >>> iscoord((0, 1, 2), fmt='c')
    True
    >>> iscoord((0, 1, -1), fmt='py')
    True

    ```
    """

    try:
        len(coord)
    except:
        coord = tuple([coord])

    dim = len(coord)

    if not isdim(dim):
        out = False
    else:
        coordfmt = _coordfmt[fmt]
        chk = lambda x : x in coordfmt
        out = all(map(chk, coord))

    return out

def isindex(index, dim=1):
    """
    Determines if the input argument `index` is an index or not. 
    Indices must be greater than zero and bounded by the number of compute
    regions for the given dimension.

    Parameters

    index : int,
            the index to check. The range of acceptable values depends on the dimension.

    dim : int,
          dimension.  

    Returns

    out : bool,
          `True` if the index is within range for the given dimension.

    Example

    ```python
    >>> isindex(1, dim=1)
    True
    >>> isindex(3, dim=1)
    False
    >>> isindex(3, dim=2)
    True

    ```
    """
    
    if not isdim(dim):
        out = False
    elif not isinstance(index, int):
        out = False
    elif index < _onebased:
        out = False
    elif index >= 3**dim + _onebased:
        out = False
    else:
        out = True
    return out

def islabel(label):
    """
    Determines if the input argument `label` is a label or not.
    A label must use any of the keywords `left`, `right`, etc., that reference
    different compute regions.

    Parameters

    label : str,
            label to check.

    Returns 

    out : bool,
          `True` if `label` is any acceptable keyword for referencing a compute region.

    """

    out = False
    for key in _labels:
        if label in _labels[key]:
            out = True
            break
    return out

def isdim(dim):
    """
    Determines if the input argument `dim` specifies a valid number of dimensions or not.

    Parameters

    dim : int,
          dimensionality. The dimensionality cannot exceed `3` (3D).

    Returns

    out : bool,
          `True` if `dim` satisfies the criteria discussed.

    """

    if not isinstance(dim, int):
        out = False
    elif dim > 3 or dim <= 0:
        out = False
    else:
        out = True

    return out

def coordtoindex(coord, fmt=_defcoordfmt):
    """
    Converts the coordinate `coord` to an index. 
    The indices start at 1 and increment by one as follows:

    Coordinate       Index

    (0, 0)      ->   1
    (1, 0)      ->   2
    (2, 0)      ->   3
    (0, 1)      ->   4
    (1, 1)      ->   5
    (2, 1)      ->   6

    Parameters

    coord : tuple(int),
            the coordinate to convert.
    fmt : tuple(int), optional,
          the format specification for the coordinate.

    Returns

    index : int,
            the index associated with the coordinate. 

    Example

    ```python
    >>> coordtoindex((0, 1))
    3

    ```
    """

    try:
        len(coord)
    except:
        coord = [coord]

    _dimerr(len(coord))

    coord = coordtocfmt(coord)
        
    return _onebased + sum([3**i*c for i, c in enumerate(coord)])

def coordtolabel(coord, fmt=_defcoordfmt):
    """
    Converts the coordinate `coord` to a label.

    Parameters

    coord : tuple(int),
            the coordinate to convert.
    fmt : tuple(int), optional,
          the format specification for the coordinate.

    Returns

    label : str,
            the label associated with the coordinate. 

    Example

    ```python
    >>> coordtolabel((0, 1))
    'left center'

    """

    try:
        len(coord)
    except:
        coord = tuple([coord])

    dim = len(coord)
    _dimerr(dim)
    label = _labels[dim][coordtoindex(coord, fmt)-_onebased]
    return label

def coordtocfmt(coord):
    """
    Converts the coordinate `coord` to the 'C' format (0, 1, 2).

    """
    try:
        len(coord)
    except:
        coord = tuple([coord])

    _dimerr(len(coord))

    change = lambda x : 2 if x == -1 else x  
    return tuple(map(change, coord))

def coordtodeffmt(coord):
    """
    Converts the coordinate `coord` to its default format.

    """

    if _defcoordfmt == 'c':
        return coordtocfmt(coord)
    elif _defcoordfmt == 'py':
        return coordtopyfmt(coord)
    else:
        raise _fmterr(_defcoordfmt)

def coordtopyfmt(coord):
    """
    Converts the coordinate `coord` to the 'python' format (0, 1, -1).

    """
    try:
        len(coord)
    except:
        coord = tuple([coord])
    
    _dimerr(len(coord))

    change = lambda x : -1 if x == 2 else x  
    return tuple(map(change, coord))

def coordtofmt(coord, fmt=_defcoordfmt):
    """
    Converts the coordinate `coord` to the format `fmt`.

    Parameters

    coord : tuple(int),
            coordinate to convert.
    fmt : str, optional,
          coordinate format.

    Returns 

    newcoord : tuple(int),
               updated coordinate (no update takes place if the format is the same as 
               the coordinate is already using).

    """

    if fmt == 'c':
        return coordtocfmt(coord)
    elif fmt == 'py':
        return coordtopyfmt(coord)
    else:
        return _fmterr(fmt)

def indextocoord(index, dim=1, fmt=_defcoordfmt):
    """
    Converts the index `index` to its corresponding coordinate.

    Parameters

    index : int,
            index to convert.
    dim : int, optional,
          dimensionality of the coordinate (`1` by default).
    fmt : str, optional,
          coordinate format to use.

    """

    # Convert to zero-based indexing if it is not already in this format
    index = index - _onebased

    if dim == 1:
        coord = index
    elif dim == 2:
        i = index % 3
        j = int(index / 3)
        coord = (i, j)
    elif dim == 3:
        i = index % 3
        j = int(index / 3) % 3
        k = int(index / 9)
        coord = (i, j, k)
    else:
        _dimerr(dim)

    return coordtofmt(coord, fmt)

def indextolabel(index, dim=1):
    """
    Converts the index `index` to its corresponding label.

    Parameters

    index : int,
            index to convert.
    dim : int, optional,
          dimensionality of the coordinate (`1` by default).

    Returns

    label : str,
            the label corresponding to the index.

    """

    # Convert to zero-based indexing if it is not already in this format
    index = index - _onebased
        
    _dimerr(dim)
    label = _labels[dim][index]

    return label

def labeltoindex(label):
    """
    Converts the label `label` to an index.

    Parameters
    
    label : str,
            the label to convert. 

    Returns

    index : int,
            the index associated with the label.

    Example

    ```python
    >>> labeltoindex('left')
    0
    >>> labeltoindex('center')
    1
    >>> labeltoindex('right')
    2
    >>> labeltoindex('left bottom')
    0

    ```

    """

    _labelerr(label)

    for key in _dlabels:
        if label in _dlabels[key]:
            index = _dlabels[key][label]
            break

    return index

def labeltocoord(label, fmt=_defcoordfmt):
    """
    Converts the label `label` to a coordinate.

    Parameters

    label : str,
            the label to convert.

    fmt : tuple(int),
          coordinate format.

    Returns

    coord : tuple(int),
            the coordinate associated with the label.

    Example

    ```python
    >>> labeltocoord('left')
    (0,)
    >>> labeltocoord('center')
    (1,)
    >>> labeltocoord('right')
    (2,)
    >>> labeltocoord('left bottom')
    (0, 0)

    ```
    """

    index = labeltoindex(label)

    _labelerr(label)

    for key in _labels:
        if label in _labels[key]:
            dim = key
            break

    coord = indextocoord(index, dim)

    return coord

def labeltodim(label):
    """
    Determines the dimensionality given a label.

    Parameters

    label : str,
            the label to process.

    Returns

    dim : int,
          dimensionality.

    """

    _labelerr(label)

    for key in _dlabels:
        if label in _dlabels[key]:
            dim = key
            break

    return dim

def _fmterr(fmt):
    # is not a valid coordinate format error
    if not fmt in _coordfmt:
        fmts = ', '.join(_coordfmt)
        raise ValueError('Invalid coordinate format specified. Expected: %s. Got: %s' 
                         % (fmts, str(fmt)))

def _coorderr(coord, fmt):
    # is not a coordinate error
    if not iscoord(coord, fmt):
        raise ValueError('Invalid coordinate specified. Got: %s' % str(coord))

def _labelerr(label):
    # is not a label error
    if not islabel(label):
        raise ValueError('Unknown label specified. Got: %s' % label)

def _dimerr(dim):
    # invalid dimension error
    if not isdim(dim):
        raise IndexError('No more than three coordinates supported. Got dimension: %d' % dim)

def _indexerr(index, dim):
    # invalid index error
    if not isindex(index, dim):
        raise IndexError('Index is invalid. Got: `index: %d` and `dimension: %d`' % (index, dim))

