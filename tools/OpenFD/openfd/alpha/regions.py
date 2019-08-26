"""
The region module is responsible for constructing new compute regions and for
providing different ways for working with regions.  A compute region is the
bounds of some rectangular part of the computational domain. This information is
needed by the kernel generators and executors to determine where certain types
of compute need to take place.  Unlike `Bounds` that are one dimensional
objects, a `Region` object can be viewed as a tuple of many bounds objects, with
one such object per direction.

In most cases, regions are defined by operators of type `Operator`. For these
operators, the `Region` objects are used to define where the operators perform a
either convolutional or dense matrix-vector product computations. 
"""

class Region(object):
    """
    This class defines a new compute region. As previously mentioned, a compute
    region is defined by bounds for each grid direction that determine its
    location on the computational grid. In addition to keeping track of the bounds
    for each grid direction, the region class also keeps track of what type of
    computations can be performed in a given region. 

    Attributes:

        bounds : A tuple that contains objects of type `Bound` that determine the
            bounds for each grid direction.
        paddding : A tuple that contains padding information for each grid
            direction. See the notes for more details.
        compute_types : A tuple that contains `str` objects that specify the
            types of computation. There is one such object for each grid
            direction. See the notes for a definition of the different compute
            types.

    Notes:

    **Padding**
    The attribute `padding` is used to define a region with two layers: an inner
    layer and an outer layer. The inner layer is typically viewed as the area to
    which data can be *written* to during computation. The size of the inner
    layer is always determined by the attribute `bounds`. The outer layer is a
    few points larger than the inner layer depending on the value of the
    attribute `padding`. See the schematic for visual representation of the
    inner and outer layer. If the padding is zero, then the inner and outer
    layer have the same dimensions. When the padding is some positive integer,
    then the outer layer is this many points larger than the inner layer. 
    
    The padding attribute is specified for each grid direction at a time and
    holds two parameters. The first parameter is an `int` that describes the
    amount of padding to the left. The second parameter is an `int` that
    describes the amount of padding to the right. The use of left and right is
    defined as the right direction being in the increasing axis direction.

    |                   |                   |                    |
    <-- left padding --> <-- inner layer --> <-- right padding --> 
    <-----------             outer layer               ---------->                 

    ---------->
    increasing axis direction

    **Compute types**
    The attribute `compute_types` is a tuple that specifies what type of
    computation is performed in a given grid direction. The purpose of the
    compute type is to flag what kind of computation is being performed.  This
    flag will be used by other methods and classes to resolve issues with
    operators that perform different computations in regions that overlap.
    
    The following options are available for the compute type:
        * 'convolution' : Use this option if this region performs a
            convolutional computation. 
        * 'dense' : Use this option if this region performs a dense
            matrix-vector computation.


    """

    def __init__(self, bounds=None, compute_types=None):
        """
        Initializes a region. 
        """
        #FIXME: add implementation
        pass

    def is_convolution(self, index):
        """
        Returns `True` if the compute type of this region is set to
        `'convolution'` in the grid direction `index`. Otherwise, `False` is
        returned.

        Arguments:

            index(`int`) : The grid direction to return the bounds for.
        """
        #FIXME: add implementation
        pass

    def is_dense(self, index):
        """
        Returns `True` if the compute type of this region is set to
        `'convolution'` in the grid direction `index`. Otherwise, `False` is
        returned.

        Arguments:

            index(`int`) : The grid direction to return the bounds for.
        """
        #FIXME: add implementation
        pass

    @property
    def bounds(self):
        """
        Returns the attribute `bounds`.
        """
        #FIXME: add implementation
        pass

    @property
    def padding(self):
        """
        Returns the attribute `padding`.
        """
        #FIXME: add implementation
        pass

    @property
    def compute_types(self):
        """
        Returns the attribute `compute_types`.
        """
        #FIXME: add implementation
        pass
