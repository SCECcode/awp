"""

Construct different types of grids

"""
import openfd
import numpy


def x(n, shift=False, prec=None, extend=False):
    """
    Construct a regular grid `x` or a cell-centered grid `xc`. The cell-centered
    grid includes grid points at the boundary.

    Regular:
        x_j = j*h, j = range(n), h = 1.0/(n - 1)

    Cell-centered:
        xc_j = (j-0.5*h), j = range(n)
        xc_0 = 0, xc_n = 1

    Arguments:
        n : Number of grid points in the regular grid
        prec : Floating-point precision to use. Defaults to `np.float64`.
        extend : If `True`, then the regular is constructed to be of the same
            size as the shifted grid. An additional zero grid points is included
            at the end of the regular grid.

    Returns:
        x : Regular grid, or cell-centered grid

    See also:
        * regular
        * cellcentered
    """

    if shift:
        return cellcentered(n, prec)
    else:
        return regular(n, prec, extend)

def regular(n, prec=None, extend=False):
    """
    Construct a regular grid for the unit interval.

    Arguments:
        n : Number of cells in the regular grid
        prec : Floating-point precision to use. Defaults to `np.float64`.
        extend : If `True`, then the regular is constructed to be of the same
            size as the shifted grid. An additional zero grid points is included
            at the end of the regular grid.

    Returns:
        x : Regular grid.

    """
    if not prec:
        prec = openfd.prec

    h = gridspacing(n)
    if extend:
        xv = numpy.array([prec(j*h) for j in range(n)] + [prec(0)])
    else:
        xv = numpy.array([prec(j*h) for j in range(n)])

    return xv

def cellcentered(n, prec=None):
    """
    Construct a regular grid for the unit interval.

    Arguments:
        n : Number of grid points in the regular grid
        prec : Floating-point precision to use. Defaults to `np.float64`.
        extend : If `True`, then the regular is constructed to be of the same
            size as the shifted grid. An additional zero grid points is included
            at the end of the regular grid.

    Returns:
        x : Cell-centered grid
    """
    if not prec:
        prec = openfd.prec
    h = gridspacing(n)
    xv = regular(n, prec, extend=True)
    xp = prec(xv - 0.5*h)
    xp[0] = 0.0
    xp[-1]= 1.0

    return xp

def xy(shape, shift, prec=None, extend=False):
    """
    Construct a particular grid type in 2D. 

    Arguments:
        shape : Number of grid points in each direction.
        shift : Shift in each direction. See below.
        prec : Floating-point precision to use. Defaults to `np.float64`.
        extend : If `True`, then the regular is constructed to be of the same
            size as the shifted grid. An additional zero grid points is included
            at the end of the regular grid.

    Shifts:
        (0, 0) : Regular grid in each direction (solution is stored at the
            nodes)
        (1, 1) : Cell-centered grid in each direction (solution is stored at the
            cell-center)
        (0, 1) : Regular grid in the x-direction, cell-centered grid in the
            y-direction (solution is stored at the center of the left and right
            edges)

    Returns:
        X, Y : Coordinates for the x, y and grid directions. Arrays of size
            `shape[1] x shape[0]`.

    """

    assert len(shape) == 2
    assert len(shift) == 2
    nx = shape[0]
    ny = shape[1]
    _x = x(nx, shift=shift[0], prec=prec, extend=extend)
    _y = x(ny, shift=shift[1], prec=prec, extend=extend)
    X, Y = numpy.meshgrid(_x, _y)

    return X, Y

def gridspacing(n, prec=None):
    """
    Compute the grid spacing.

    Arguments:
        n : Number of cells in the regular grid.
        prec : Floating-point precision to use. Defaults to `np.float64`.

    """

    if not prec:
        prec = openfd.prec
    return prec(1.0/(n-1))
