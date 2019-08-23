"""
Module that uses the normal to generate regions and bounds for the boundary.

The normal is defined as a tuple of ints. For example, `normal = (0, 0, 1)`.
Only one component of the normal must be non-zero and take on either the value
`+1` or `-1`. The non-zero component and its sign uniquely defines the side of a
n-dimensional cube. These two properties are used to label each side.

In two dimensions, the sides are labelled as shown in the following figure:

      | Label | Normal   |
      |-------|----------|
      | 00    | (-1,  0) |
      | 01    | ( 1,  0) |
      | 10    | ( 0, -1) |
      | 11    | ( 0,  1) |

              11
        --------------                
        |            |                
        |            |                
    00  |            |  01            
        |            |                
        |            |                
 x1     --------------
 ^            10
 |
 |                    
 --------------------> x0

 In three dimensions, the sides are labelled as:

      | Label | Normal       |
      |-------|--------------|
      | 00    | (-1,  0,  0) |
      | 01    | ( 1,  0,  0) |
      | 10    | ( 0, -1,  0) |
      | 11    | ( 0,  1,  0) |
      | 20    | ( 0,  0, -1) |
      | 21    | ( 0,  0,  1) |

"""

def kernels(name, normal, generator, regions=None):
    """

    """
    names = {-1: '2', 0: '0', 1: '1'}
    
    kernels = []

    # Do not output the region label if the normal defined in 1D 
    if len(normal) == 1:
        kernelname = '%s_%s' %(name, label(normal))
        kernel = generator.kernel(kernelname, region(normal, (0,)))
        kernels.append(kernel)
        return kernels
    else:
        kernelname = lambda reg : '%s_%s_%s' %(name, label(normal), reg)

    if not regions:
        regions = get_regions(len(normal), [[1]]*len(normal))

    for reg in regions:
        # Convert region say (-1, -1) to '22'
        region_name = ''.join([names[r] for r in reg])
        kernel = generator.kernel(kernelname(region_name), region(normal, reg))
        kernels.append(kernel)

    return kernels

def region(normal, surface=None): 
    """
    Determine the region in `n`-dimensional space given the normal and a
    region on the side for which the normal is defined. The region on this side
    is defined by a `n-1`-dimensional tuple. 

    Arguments:
        normal : The normal with respect to a boundary (tuple of length `n`).
        surface : The region parameterization that defines a specific region for
            this side (length `n-1`).

    Returns:
        A tuple of length `n` that gives the region in `n`-dimensional space.
        
    Example:

    Obtain the interior region of the left boundary in 2D
    ```python
    >>> region((-1, 0), (1,))
    >>> (0, 1)


    ```
    Obtain the bottom interior region of the left boundary in 3D
    ```python
    >>> region((1, 0, 0), (1, 0))
    >>> (-1, 1, 0)

    >>> region((0, 1, 0), (1, 0))
    >>> (1, -1, 0)

    """
    if not surface:
        pass
    elif len(normal) == 1:
        if len(normal) != len(surface):
            raise IndexError('Inconsistent dimensions for normal and surface')
    elif len(normal) != len(surface) + 1:
        print(len(normal), len(surface))
        raise IndexError('Inconsistent dimensions for normal and surface')

    out = [1]*len(normal)
    comp = component(normal)
    out[comp] = region_id(side(normal))
    k = 0
    for i in range(len(normal)):
        if i != comp:
            out[i] = surface[k]
            k += 1

    return tuple(out)

def get_regions(dim, region_list=None):
    """
    Define regions for a boundary given the dimension of the ambient space.
    The dimension of the boundary is n - 1  where `n` is the dimension of the
    ambient space. 

    Arguments:
        dim : Dimensionality of the ambient space
        region_list (optional) : The regions to use in each grid dimension.
            Defaults to `0, 1, -1`.

    Returns:
        A list of possible regions for the given dimension. The length is
        `m^(n-1)`, if `m` regions are defined in each direction.

    """

    if not region_list:
        region_list = []
        for di in range(dim):
            region_list.append((0, 1, -1))

    regs = []
    # Define regions in 3D
    if dim == 3:
        for ireg in region_list[0]:
            for jreg in region_list[1]:
                regs.append((ireg, jreg))
    # Define regions in 2D
    elif dim == 2:
        for ireg in region_list[0]:
                regs.append((ireg,))
    else:
        raise NotImplementedError('Region generation in other dimensions '\
                                  'than 2D, and 3D is currently not supported.')
    return regs

def bounds(normal, grid, size=1):
    """
    Define bounds for a boundary given the non-zero normal component with
    respect to the boundary and bounds for the entire grid.

    Arguments:
        normal : The normal with respect to a boundary (tuple of length `n`).
        grid : Tuple that defines the grid bounds in each direction.
        size (optional) : The number of points in the normal direction. 

    Returns:
        A tuple of `Bounds`.  

    """
    from openfd.base import Bounds

    comp = component(normal)

    bnds = [0]*len(grid)
    for i in range(len(grid)):
        bnds[i] = Bounds(grid[i].size, left=grid[i].left, right=grid[i].right)

    bnds[comp] = Bounds(grid[comp].size, left=size, right=size)

    return bnds

def region_id(side):
    """
    Convert from the normal `side` property to the region id number.

    Arguments:
        side (`int`) : Side obtained by calling `side()`.

    Returns:
        Region id number. Either `0` (left boundary)` or `-1` (right boundary).

    """
    #FIXME: Change this function once the region labelling convention changes
    if side == 0:
        return 0
    if side == 1:
        return -1


def label(normal):
    """
    Use a normal to produce a label formatted as: '%s%s` where the first
    argument is the normal component and second argument is the "side" (see
    `side(..)`. 

    Arguments:
        normal : The normal with respect to a boundary (tuple of length `n`).

    Returns:
        The label.

    Example:

    ```python
    >>> label((0, 0, -1))
    '20'

    ```

    """
    return '%s%s'%(component(normal), side(normal))

def component(normal):
    """
    Gives the non-zero component of the normal. The normal must be of unit
    length and defined with respect to some side of a n-dimensional cube.

    Arguments:
        normal : The normal with respect to a boundary (tuple of length `n`).

    Returns:
        int : The component. The first component is `0`, and the second
        component is `1`, etc.

    Example:
    ```python
    >>> component((0, 0, -1))
    2

    ```
    """
    # Check that only one component has been specified
    if (sum([abs(ni) for ni in normal])) != 1:
        raise ValueError('Invalid normal')

    comp = 0
    for ni in normal:
        if abs(ni) == 1:
            break
        else:
            comp += 1
    return comp

def side(normal):
    """
    Gives the sign of the non-zero component of the normal. A negative sign -1
    is converted to 0 and a positive sign +1 is unchanged.

    Arguments:
        normal : The normal with respect to a boundary (tuple of length `n`).

    Returns:
        int : The side. If the non-zero component of the normal is positive,
        then `1` is returned. Otherwise, `0` is returned. 
    
    Example:
    ```python
    >>> side((0, 0, -1))
    0
    >>> side((0, 0, 1))
    1

    ```
    """

    sign = normal[component(normal)]
    if sign == -1:
        return 0
    else:
        return 1
