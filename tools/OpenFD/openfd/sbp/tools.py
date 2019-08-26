"""
The tools are designed to help out with making operators compliant with OpenFD
operator restrictions. 
"""
import numpy as np

def add_zeros(op, side, pos=0):
    """
    Pads the boundary part of an operator with zeros in either the first or last
    row.
    
    Arguments:
        op : A dictionary that contains the data for the staggered grid operator
            to modify.
        side : The side to modify. Either `left` or `right`.
        pos : Either `0` for first row or `-1` for last row.

    """
    assert side == 'left' or side == 'right'
    assert pos == 0 or pos == -1

    op_label = 'op_%s'%side
    idx_label = 'idx_%s'%side

    op_data = op[op_label]
    idx = op[idx_label]

    cols = len(op_data[0])
    
    zeros = [0]*cols

    # Add to first row
    if pos==0:
        op[op_label] = [zeros] + op_data
        op[idx_label] = [[*range(cols)]] + idx
    # Add to last row
    else:
        op[op_label] = op_data + [zeros]
        op[idx_label] = idx + [[*range(cols)]]

def add_interior(op, side, sign=-1):
    """
    Pads the boundary part of an operator with a interior stencil in the last
    row (closest to the interior)
    
    Arguments:
        op : A dictionary that contains the data for the staggered grid operator
            to modify.
        side : The side to modify. Either `left` or `right`.
    """
    assert side == 'left' or side == 'right'

    op_label = 'op_%s'%side
    idx_label = 'idx_%s'%side

    op_data = op[op_label]
    idx = op[idx_label]

    rows = len(op_data)
    cols = len(op_data[0])
    
    op_interior = op['op_interior']
    idx_interior = op['idx_interior']

    # Determine new padding, and pad all boundary stencils
    padding = idx_interior[-1]
    zeros = [0.0]*padding 
    for i in range(rows):
        op_data[i] = [0] + op_data[i] + zeros
    for i in range(len(idx)):
        idx[i] = list(range(0, cols+padding+1))

    # Pad interior stencil so that it is of the same length as the boundary
    # stencils
    stencil = [0.0]*len(idx[0])
    if side == 'left':
        for i in range(len(op_interior)):
            stencil[i+cols+idx_interior[0]] = op_interior[i]
    else:
        for i in range(len(op_interior)):
            stencil[i+cols+idx_interior[0]] = sign*op_interior[i]

    # Place interior stencil after the last row and fill with zeros ahead of it
    # so that the interior stencil ends up in the correct position
    #assert len(stencil) == len(op_data[0])
    op_data += [stencil]

    # Add to last row
    op[op_label] = op_data
    op[idx_label] = idx + [idx[0]]

def apply_stencil(stencil, f, side):
    """
    Tests that a stencil is computed correctly by checking what it evaluates to
    when applied to a polynomial test function.

    Arguments:
        stencil : A list of stencil coefficients
        nx : Number of grid points of the grid that the operator is defined on.
        i : The grid point at which the stencil is applied.
        f : An array that contains the projection of the polynomial test
            function onto the grid.
        side : The side at which the stencil is applied. Can be either `'left'`
            or `'right'`.

    Returns:
        The error in approximation evaluated at the grid point `i`. 

    """
    n = len(stencil)
    nf = len(f)
    if side == 'left':
        return sum([stencil[j]*f[j] for j in range(n)])
    else:
        return sum([stencil[-(j+1)]*f[nf - n +  j] for j in range(n)])

def accuracy(x, xh, op, is_outside=False, tol=1e-6):
    """
    Tests the boundary accuracy for the operator defined on the regular grid.

    Arguments:
        x : regular grid
        xh : shifted grid
        op : A dictionary defining the operator.
        is_outside (optional) : This optional flag should be set to `True` when
            an extra point has been added to the operator that is placed outside the
            boundary (grid is generated by calling `grid_xghost1` in this case).
        tol (optional) : Tolerance threshold for floating point comparison.

    """
    h = x[2] - x[1]
    for i in range(len(op['op_left'])):
        exact = 2*x[i]
        assert abs(1.0/h*apply_stencil(op['op_left'][i], xh**2, 'left')
                   - exact ) < tol
        exact = 2.0 - 2*x[i]
        # Take extra point outside the boundary into account in the computation
        # of the exact solution. Force the exact solution to be zero for the
        # extra point since no computation is performed there.
        if is_outside:
            exact = 2.0 - 2*x[i] + 2*h
        if i == 0 and is_outside:
            exact = 0.0
        assert abs(1.0/h*apply_stencil(op['op_right'][i], xh**2, 'right')
                   - exact ) < tol

def accuracy_shift(x, xh, op, is_interior=False, tol=1e-6):
    """
    Tests the boundary accuracy for the operator defined on the shifted grid.

    Arguments:
        x : regular grid
        xh : shifted grid
        op : A dictionary defining the operator.
        is_outside (optional) : This optional flag should be set to `True` when
            an extra point has been added to the operator that is placed outside the
            boundary (grid is generated by calling `grid_xghost1` in this case).
        tol (optional) : Tolerance threshold for floating point comparison.

    """
    h = x[2] - x[1]
    for i in range(len(op['op_left'])):
        exact = 2*xh[i]
        assert abs(1.0/h*apply_stencil(op['op_left'][i], x**2, 'left')
                   - exact ) < tol

    for i in range(len(op['op_right'])):
        exact = 2.0 - 2*xh[i]
        assert abs(1.0/h*apply_stencil(op['op_right'][i], x**2, 'right')
                   - exact ) < tol

def grid_x(n):
    """
    Equidistant grid defined from 0 to 1 using `n` grid points.
    """
    return np.linspace(0, 1, n)

def grid_xghost1(n):
    """
    Equidistant grid defined from 0 to 1+h using `n` grid points, 
    where `h` denotes the grid spacing.

    Arguments:
        n : The number of grid points.

    Returns:
        An array of length `n` that contains the grid point values.

    """
    x = np.zeros((n,1))
    h = 1.0/(n-2)
    for i in range(n-1):
        x[i] = i*h
    x[-1] = 1.0 + h
    return np.array(x)

def grid_xshift(n):
    """
    Equidistant grid defined from 0 to 1 using `n` grid points, but shifted half
    a step. That is, `x[0] = h/2`, where `h` is the grid spacing.

    Arguments:
        n : The number of grid points.

    Returns:
        An array of length `n` that contains the grid point values.

    """
    x = grid_x(n)
    return x + 0.5*x[1]

def grid_xshift_bnd(n):
    """
    Equidistant grid defined from 0 to 1 using `n` grid points, but shifted half
    a step. Grid includes boundary points. That is, `x[0] = h/2`, where `h` is
    the grid spacing.

    Arguments:
        n : The number of grid points.

    Returns:
        An array of length `n` that contains the grid point values.

    """

    x = [0]*n
    x[0] = 0.0
    h = 1.0/(n-2)
    for i in range(1, n):
        x[i] = h*(i - 0.5)
    x[n-1] = 1.0
    return np.array(x)
